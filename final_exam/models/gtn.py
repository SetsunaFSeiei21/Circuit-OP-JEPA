"""Device-level adaptation of Fan et al.'s DAC 2024 GTN, equations 1--4.

This is an independent paper-based implementation, not imported author code.
Neighbor attention, separate attention within each minimum-basis circuit loop,
learned neighbor/loop merging, and the mean-pool/64/16 MLP head are retained.
The original device/net bipartite graph is used ONLY to find loops. Learned
tokens are whole devices, with terminal-pair edges and named-port features.
See gtn_source.json for the precise adaptations and source-search limitations.
"""
from __future__ import annotations

from collections import OrderedDict
import math

import torch
from torch import nn
from torch.nn import functional as F

from .common import DeviceEmbedding, graph_memory, model_config


def _device_cycles(edge_roles, live):
    """Recover net incidence before finding cycles; same-net cliques are not loops.

    edge_roles[i,j,a*T+b] says terminal a on i and terminal b on j are
    on the SAME net. In particular the diagonal marks existing terminals.
    Minimum-cycle-basis discovery is preprocessing, never a learned feature
    derived from OP values or performance labels.
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("GTN loop preprocessing requires networkx>=3.0") from exc
    n, _, pairs = edge_roles.shape
    terminals = math.isqrt(pairs)
    if terminals * terminals != pairs or terminals < 1:
        raise ValueError("GTN edge_roles must contain all terminal-pair channels")
    parent = list(range(n * terminals))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    present = set()
    for i, j, pair in edge_roles.nonzero(as_tuple=False).tolist():
        if not live[i] or not live[j]:
            continue
        a, b = divmod(pair, terminals)
        first, second = i * terminals + a, j * terminals + b
        present.update((first, second))
        union(first, second)
    graph = nx.Graph()
    graph.add_nodes_from(("device", i) for i in range(n) if live[i])
    for endpoint in sorted(present):
        graph.add_edge(("device", endpoint // terminals), ("net", find(endpoint)))
    # A simple device/net graph intentionally ignores an intra-device terminal
    # tie as a one-device loop. Distinct loops with the same device projection
    # have identical token sets and are deduplicated after projection.
    cycles = {
        tuple(sorted(index for kind, index in cycle if kind == "device"))
        for cycle in nx.minimum_cycle_basis(graph)
    }
    return sorted(cycle for cycle in cycles if len(cycle) >= 2)


def build_loop_membership(edge_roles, device_mask):
    """Return CPU bool [topologies, loops, devices] with zero-padded loops.

    Call once on ResidentDataset.static, then gather by graph_ids on the GPU.
    Do not discover loops from device adjacency: three devices on one net form
    a clique but no circuit loop. Terminal equivalence reconstructs that net.
    """
    if edge_roles.ndim != 4 or device_mask.ndim != 2:
        raise ValueError("Expected edge_roles [B,N,N,T*T] and device_mask [B,N]")
    b, n = device_mask.shape
    if edge_roles.shape[:3] != (b, n, n):
        raise ValueError("GTN graph tensor shapes disagree")
    edges = edge_roles.detach().to(device="cpu", dtype=torch.bool)
    live = device_mask.detach().to(device="cpu", dtype=torch.bool)
    rows = [_device_cycles(edges[i], live[i].tolist()) for i in range(b)]
    result = torch.zeros(b, max(1, max(map(len, rows), default=0)), n, dtype=torch.bool)
    for i, cycles in enumerate(rows):
        for j, members in enumerate(cycles):
            result[i, j, list(members)] = True
    return result


def add_resident_loop_membership(store, indices, batch):
    """Attach already precomputed topology masks without CPU graph work."""
    if not hasattr(store, "gtn_loop_membership"):
        store.gtn_loop_membership = build_loop_membership(
            store.static["edge_roles"], store.static["device_mask"]).to(store.device)
    topology = store.graph_ids.index_select(0, indices)
    batch["gtn_loop_membership"] = store.gtn_loop_membership.index_select(0, topology)[
        :, :, :batch["device_mask"].shape[1]]
    return batch


class _MaskedAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.head_dim = cfg.heads, cfg.dim // cfg.heads
        self.q = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.k = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.v = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.output = nn.Linear(cfg.dim, cfg.dim)
        self.dropout = cfg.dropout

    def project(self, nodes):
        def split(layer):
            return layer(nodes).reshape(*nodes.shape[:2], self.heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.q), split(self.k), split(self.v)
        return q @ k.transpose(-2, -1) / math.sqrt(self.head_dim), v

    def forward(self, nodes, allowed, bias=None):
        scores, values = self.project(nodes)
        if bias is not None:
            scores = scores + bias
        weights = scores.masked_fill(~allowed[:, None], -torch.inf).softmax(-1)
        attended = F.dropout(weights, self.dropout, self.training) @ values
        return self.output(attended.transpose(1, 2).flatten(2))

    def loops(self, nodes, memberships, valid):
        scores, values = self.project(nodes)
        # Empty padded loops are given real keys only for numerical stability;
        # their candidate masks below guarantee they never contribute.
        keys = torch.where(memberships.any(-1, keepdim=True), memberships, valid[:, None])
        weights = scores[:, None].masked_fill(~keys[:, :, None, None], -torch.inf).softmax(-1)
        attended = F.dropout(weights, self.dropout, self.training) @ values[:, None]
        return self.output(attended.permute(0, 1, 3, 2, 4).flatten(3))


class GTNLayer(nn.Module):
    def __init__(self, cfg, terminal_pairs):
        super().__init__()
        self.neighbor = _MaskedAttention(cfg)
        self.loop = _MaskedAttention(cfg)
        # Device graph adaptation retains D/G/S/B (and other terminal) identity
        # as neighbor attention bias instead of adding learned net tokens.
        self.edge_bias = nn.Linear(terminal_pairs, cfg.heads, bias=False)
        self.merge_query = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.merge_key = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.scale = math.sqrt(cfg.dim)

    def forward(self, nodes, batch, memberships, loops_enabled):
        valid = batch["device_mask"]
        n = valid.shape[1]
        connected = batch["adjacency"] | torch.eye(n, dtype=torch.bool, device=nodes.device)[None]
        allowed = (connected | ~valid[:, :, None]) & valid[:, None, :]
        edge_bias = self.edge_bias(batch["edge_roles"]).permute(0, 3, 1, 2)
        neighbor = self.neighbor(nodes, allowed, edge_bias)
        if not loops_enabled:
            return F.relu(neighbor).masked_fill(~valid[..., None], 0.)
        loops = self.loop.loops(nodes, memberships, valid)
        candidates = torch.cat((neighbor[:, None], loops), dim=1)
        available = torch.cat((valid[:, None], memberships & valid[:, None]), dim=1)
        # Equation 4 scores each candidate with its own projected q and k;
        # softmax is over the neighbor candidate and incident loop candidates.
        scores = (self.merge_query(candidates) * self.merge_key(candidates)).sum(-1) / self.scale
        # Padding receives a dummy neighbor candidate and is zeroed afterwards.
        available[:, 0] = True
        weights = scores.masked_fill(~available, -torch.inf).softmax(dim=1)
        merged = (weights[..., None] * candidates).sum(dim=1)
        return F.relu(merged).masked_fill(~valid[..., None], 0.)


class GTNEncoder(nn.Module):
    def __init__(self, config, prep):
        super().__init__()
        self.cfg = model_config(config)
        self.embedding = DeviceEmbedding(self.cfg, prep)
        terminal_pairs = len(prep["feature_spec"]["terminal_names"]) ** 2
        self.layers = nn.ModuleList(GTNLayer(self.cfg, terminal_pairs)
                                    for _ in range(self.cfg.encoder_layers))
        self.register_buffer("loops_enabled", torch.tensor(True, dtype=torch.bool))
        self._loop_cache = OrderedDict()

    @staticmethod
    def warmup_epochs(total_epochs):
        """Paper's 450/700 neighbor-only schedule, scaled to the chosen budget."""
        if int(total_epochs) < 1:
            raise ValueError("total_epochs must be positive")
        return min(int(total_epochs) - 1, int(total_epochs) * 450 // 700)

    def set_epoch(self, epoch, total_epochs):
        """One-based epoch; freeze early-stop counters during this warmup."""
        if int(epoch) < 1:
            raise ValueError("epoch must be one-based")
        self.loops_enabled.fill_(int(epoch) > self.warmup_epochs(total_epochs))

    def prepare_resident(self, store):
        if not hasattr(store, "gtn_loop_membership"):
            store.gtn_loop_membership = build_loop_membership(
                store.static["edge_roles"], store.static["device_mask"]).to(store.device)
        # Static fields are automatically gathered by ResidentDataset.batches
        # and sliced on dimension 1 (device axis). Store this orientation so
        # the ordinary resident path never transfers graph tensors to the CPU.
        store.static["gtn_loop_membership_by_device"] = store.gtn_loop_membership.transpose(1, 2)
        return store.gtn_loop_membership

    def _memberships(self, batch):
        if "gtn_loop_membership_by_device" in batch:
            value = batch["gtn_loop_membership_by_device"]
            if value.ndim != 3 or value.shape[:2] != batch["device_mask"].shape:
                raise ValueError("Device-oriented GTN loop memberships must have shape [B,N,L]")
            return value.transpose(1, 2).to(dtype=torch.bool)
        if "gtn_loop_membership" in batch:
            memberships = batch["gtn_loop_membership"]
            if (memberships.ndim != 3 or memberships.shape[0] != batch["device_mask"].shape[0]
                    or memberships.shape[2] != batch["device_mask"].shape[1]):
                raise ValueError("GTN loop memberships must have shape [B,L,N]")
            return memberships.to(device=batch["device_mask"].device, dtype=torch.bool)
        # Standalone/collate fallback. The resident runner uses the explicit
        # precomputation path above, so no host graph work occurs per minibatch.
        edges = batch["edge_roles"].detach().to(device="cpu", dtype=torch.bool)
        valid = batch["device_mask"].detach().cpu()
        rows = []
        for edge, live in zip(edges, valid):
            key = (tuple(edge.shape), edge.numpy().tobytes(), live.numpy().tobytes())
            if key not in self._loop_cache:
                self._loop_cache[key] = build_loop_membership(edge[None], live[None])[0]
                if len(self._loop_cache) > 64:
                    self._loop_cache.popitem(last=False)
            else:
                self._loop_cache.move_to_end(key)
            rows.append(self._loop_cache[key])
        count = max(row.shape[0] for row in rows)
        return torch.stack([F.pad(row, (0, 0, 0, count-row.shape[0])) for row in rows]).to(
            batch["device_mask"].device)

    def forward(self, batch):
        nodes = self.embedding(batch)
        memberships = self._memberships(batch)
        loops_enabled = bool(self.loops_enabled)
        for layer in self.layers:
            nodes = layer(nodes, batch, memberships, loops_enabled)
        return graph_memory(nodes, batch["device_mask"])


class GTNDecoder(nn.Module):
    """Original GTN mean-pooling readout and two hidden layers (64, 16)."""
    def __init__(self, config, num_metrics):
        super().__init__()
        cfg = model_config(config)
        self.output = nn.Sequential(nn.Linear(cfg.dim, 64), nn.ReLU(),
                                    nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, num_metrics))

    def forward(self, memory, padding):
        return self.output(memory[:, 0])
