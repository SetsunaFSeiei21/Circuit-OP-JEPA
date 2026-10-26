"""Device-resident numeric tables and graph construction.

Python decodes strings/nulls once. Normalization, graph relations, pin feature
gathers and minibatch selection execute on the requested torch device. Topology
is stored once per distinct graph, never as a dense graph for every sizing point.
"""
import math
import os
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import (canonical_model_name, device_category, is_op_target,
                   parameter_applicability)
from .tensors import (_identity_vocabularies, _model_identity, _named_port_nets,
                      _performance_metric_mask)


PIPELINE = "device_resident_v1"


def deterministic_workspace():
    workspace = os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if workspace not in {':4096:8', ':16:8'}:
        raise ValueError('Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 before CUDA initialization')


def require_device(device):
    device = torch.device(device)
    if device.type not in {'cpu', 'cuda'}:
        raise ValueError('The resident pipeline supports cpu or cuda devices')
    if device.type == "cuda":
        deterministic_workspace()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; refusing CPU fallback")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        elif device.index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device {device.index} is not visible')
    return device


def _asinh(values, scale):
    ratio = values / scale
    large = torch.copysign(values.abs().log() - scale.log() + math.log(2.), values)
    return torch.where(torch.isfinite(ratio), torch.asinh(ratio), large)


def _prefix_sum(values):
    # CUDA float cumsum has no deterministic implementation in older supported
    # torch versions (including 2.6). A fixed addition tree keeps this GPU-only
    # without relaxing deterministic_algorithms or changing the weighted median.
    result, stride = values.clone(), 1
    while stride < result.numel():
        result = torch.cat((result[:stride], result[stride:] + result[:-stride]))
        stride *= 2
    return result


def normalize(values, mask, stats, names, dtype=torch.float32):
    """Float64 physical transforms; inactive entries never enter arithmetic."""
    values = torch.where(mask, values.to(torch.float64), 0.)
    kwargs = dict(dtype=torch.float64, device=values.device)
    mean, std = (torch.tensor(stats[k], **kwargs) for k in ("mean", "std"))
    kinds = stats.get("kind", ["linear"] * len(names))
    use_asinh = torch.tensor([k == "asinh" for k in kinds], device=values.device)
    scale = torch.tensor(stats.get("asinh_scale", [1.] * len(names)), **kwargs)
    physical = torch.where(use_asinh, _asinh(values, scale), values)
    difference = physical - mean
    result = torch.where(torch.isfinite(difference), difference / std, physical / std - mean / std)
    result = torch.where(mask, result, 0.).to(dtype)
    bad = ~torch.isfinite(result)
    if bool(bad.any()):
        column = int(bad.nonzero()[0, -1])
        raise ValueError(f"Non-finite standardized feature {names[column]}; inspect units/transforms")
    return result


def device_statistics(rows, names, rules, weighting, device):
    """Train-only, circuit-balanced statistics using float64 device reductions."""
    device = require_device(device)
    circuits = {name: i for i, name in enumerate(sorted({c for c, _ in rows}))}
    values = np.array([[row.get(name) if row.get(name) is not None else 0.
                        for name in names] for _, row in rows], dtype=np.float64).reshape(len(rows), len(names))
    observed = np.array([[row.get(name) is not None for name in names]
                         for _, row in rows], dtype=np.bool_).reshape(values.shape)
    x, mask = torch.as_tensor(values, device=device), torch.as_tensor(observed, device=device)
    ids = torch.tensor([circuits[c] for c, _ in rows], dtype=torch.long, device=device)
    result = {"mean": [], "std": [], "count": [], "kind": [], "asinh_scale": [],
              "circuit_count": [], "version": 2, "weighting": weighting}

    def weights(selected):
        counts = torch.bincount(selected, minlength=len(circuits))
        active = (counts > 0).sum()
        if weighting == "equal_circuit":
            return 1. / (active * counts[selected].to(torch.float64)), active
        return torch.full(selected.shape, 1. / max(1, selected.numel()),
                          dtype=torch.float64, device=device), active

    for j, name in enumerate(names):
        v, c = x[mask[:, j], j], ids[mask[:, j]]
        w, active = weights(c)
        rule, scale = rules[name], x.new_tensor(1.)
        if rule["kind"] == "asinh":
            if rule["scale"] == "fit":
                nz = v != 0
                if bool(nz.any()):
                    absolute, order = v[nz].abs().sort(stable=True)
                    nz_w, _ = weights(c[nz])
                    ordered_w = nz_w[order]
                    index = (_prefix_sum(ordered_w) >= ordered_w.sum() * .5 - 1e-15).nonzero()[0, 0]
                    scale = absolute[index]
            else:
                scale = x.new_tensor(float(rule["scale"]))
            v = _asinh(v, scale)
        mean, std = x.new_tensor(0.), x.new_tensor(1.)
        if v.numel():
            magnitude = v.abs().amax()
            if bool(magnitude > 0):
                relative = v / magnitude
                center = (relative * w).sum() / w.sum()
                mean = center * magnitude
                std = (((relative - center).square() * w).sum() / w.sum()).sqrt() * magnitude
                std = torch.where(std == 0, 1., std)
        result["mean"].append(float(mean))
        result["std"].append(float(std))
        result["count"].append(v.numel())
        result["kind"].append(rule["kind"])
        result["asinh_scale"].append(float(scale))
        result["circuit_count"].append(int(active))
    return result


def _topology_key(sample, devices):
    return (tuple(sorted(sample.get("ports", {}).items())),
            tuple((d["device_id"], d["device_type"], canonical_model_name(d),
                   tuple(sorted(d["terminals"].items()))) for d in devices))


def _graph(sample, devices, prep, device, pin_vocab):
    """Construct adjacency and directional terminal roles with device kernels."""
    spec, (models, ports) = prep["feature_spec"], _identity_vocabularies(prep)
    terminals = spec["terminal_names"]
    t = len(terminals)
    nets = sorted({net for d in devices for net in d["terminals"].values()}
                  | set(sample.get("ports", {}).values()))
    net_id = {net: i for i, net in enumerate(nets)}
    named = _named_port_nets(sample, ports) if ports is not None else {}
    if ports is not None:
        net_roles = torch.tensor([[int(role in named.get(net, ())) for role in range(len(ports))]
                                  for net in nets], dtype=torch.float32, device=device)
    if pin_vocab is not None:
        from .baselines import _pin_tokens
        tokens = _pin_tokens(sample, terminals, prep.get("input_contract_version", 1))
        parent_id = {d["device_id"]: i for i, d in enumerate(devices)}
        parents = [parent_id.get(token["device_id"], -1) for token in tokens]
        types = [pin_vocab.get(token["key"], pin_vocab["<UNK>"]) for token in tokens]
        categories = [prep.get("device_category_vocab", {}).get(
            device_category(token["device"]) if token["device"] is not None else "port", 0) for token in tokens]
        n = len(tokens)
        ns = torch.tensor([net_id[token["net"]] for token in tokens], device=device)
        ts = torch.tensor([terminals.index(token["terminal"]) if token["terminal"] is not None else -1
                           for token in tokens], device=device)
        parent = torch.tensor(parents, device=device)
        pins = ts >= 0
        adjacency = ((parent[:, None] == parent[None]) & pins[:, None] & pins[None])
        adjacency |= (ns[:, None] == ns[None]) & (pins[:, None] | pins[None])
        adjacency |= torch.eye(n, dtype=torch.bool, device=device)
        one_hot = F.one_hot(ts.clamp_min(0), t).bool() & pins[:, None]
        roles = ((one_hot[:, None, :, None] & one_hot[None, :, None, :])
                 & adjacency[:, :, None, None]).reshape(n, n, t*t)
        graph = {"port_roles": torch.zeros(n, t, device=device),
                 "node_parent": parent, "device_ids": [token["token_id"] for token in tokens]}
        if ports is not None:
            graph["named_port_roles"] = (one_hot[:, :, None] * net_roles[ns, None]).reshape(n, t*len(ports))
        if models is not None:
            graph["device_models"] = torch.tensor([_model_identity(token["device"], models) for token in tokens], device=device)
    else:
        n = len(devices)
        types = [prep["device_vocab"].get(d["device_type"], prep["device_vocab"]["<UNK>"]) for d in devices]
        categories = [prep.get("device_category_vocab", {}).get(device_category(d), 0) for d in devices]
        ns = torch.tensor([[net_id[d["terminals"][terminal]] if terminal in d["terminals"] else -1
                            for terminal in terminals] for d in devices], device=device)
        roles = ((ns[:, None, :, None] == ns[None, :, None, :])
                 & (ns[:, None, :, None] >= 0) & (ns[None, :, None, :] >= 0)).reshape(n, n, t*t)
        external = torch.tensor([net_id[net] for net in sample.get("ports", {}).values()], dtype=torch.long, device=device)
        graph = {"port_roles": (torch.isin(ns, external) & (ns >= 0)).float(),
                 "node_parent": torch.arange(n, device=device),
                 "device_ids": [d["device_id"] for d in devices]}
        adjacency = roles.any(-1)
        if ports is not None:
            graph["named_port_roles"] = (net_roles[ns.clamp_min(0)] * (ns >= 0).unsqueeze(-1)).reshape(n, t*len(ports))
        if models is not None:
            graph["device_models"] = torch.tensor([_model_identity(d, models) for d in devices], device=device)
    graph.update(device_types=torch.tensor(types, dtype=torch.long, device=device),
                 device_categories=torch.tensor(categories, dtype=torch.long, device=device),
                 device_mask=torch.ones(n, dtype=torch.bool, device=device),
                 adjacency=adjacency, edge_roles=roles.float())
    return graph


def _reserve(device, needed):
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        # Leave working space for minibatches, activations and optimizer state.
        if needed > free * .7:
            raise RuntimeError(f"GPU resident cache needs about {needed/2**30:.2f} GiB, but only "
                               f"{free/2**30:.2f} GiB is free; leave training workspace or use a larger GPU. "
                               "Refusing CPU fallback.")


class ResidentDataset:
    def __init__(self, samples, preprocessing, device, *, include_op=False,
                 include_performance=False, include_op_states=False, pin_vocab=None,
                 max_tokens=None, performance_metric_mask=None, performance_dtype=torch.float32,
                 label="data", verbose=True):
        if not samples:
            raise ValueError("Cannot cache an empty dataset")
        started = time.perf_counter()
        self.device = require_device(device)
        self.rows, self.prep = samples, preprocessing
        self.label, self.verbose = label, verbose
        spec = preprocessing["feature_spec"]
        pn, on, mn = (spec[k] for k in ("parameter_names", "op_names", "metric_names"))
        selection = _performance_metric_mask(performance_metric_mask, len(mn), include_performance)
        states = preprocessing.get("op_state_features", []) if include_op_states else []
        ordered = [sorted(s["devices"], key=lambda d: d["device_id"]) for s in samples]
        b, d = len(samples), max(map(len, ordered))
        needed = b*d*(len(pn)*11 + (len(on)*13 if include_op else 0) + len(states)*5)
        needed += b*len(mn)*21 if include_performance else 0
        _reserve(self.device, needed)
        if verbose:
            print(f"{label}: preparing {b} samples on {self.device} (resident numeric cache)", flush=True)
        raw_p = np.zeros((b, d, len(pn)), dtype=np.float64)
        mask_p = np.zeros(raw_p.shape, dtype=np.bool_)
        app_p = np.zeros(raw_p.shape, dtype=np.bool_)
        if include_op:
            raw_o = np.zeros((b, d, len(on)), dtype=np.float64)
            mask_o = np.zeros(raw_o.shape, dtype=np.bool_)
            eligible = np.zeros((b, d), dtype=np.bool_)
        if states:
            raw_s = np.zeros((b, d, len(states)), dtype=np.float32)
            mask_s = np.zeros(raw_s.shape, dtype=np.bool_)
        if include_performance:
            raw_y = np.zeros((b, len(mn)), dtype=np.float64)
            mask_y = np.zeros(raw_y.shape, dtype=np.bool_)
        graph_ids, lookup, graphs = [], {}, []
        for bi, (sample, devices) in enumerate(zip(samples, ordered)):
            key = _topology_key(sample, devices)
            if key not in lookup:
                graph = _graph(sample, devices, preprocessing, self.device, pin_vocab)
                n = len(graph["device_ids"])
                if max_tokens is not None and n > max_tokens:
                    raise ValueError(f"{n} device/pin/port tokens exceed max_tokens={max_tokens}; no truncation is allowed")
                lookup[key] = len(graphs)
                graphs.append(graph)
            graph_ids.append(lookup[key])
            for ni, dev in enumerate(devices):
                params = dev.get("parameters", {})
                raw_p[bi, ni] = [params.get(name) if params.get(name) is not None else 0. for name in pn]
                mask_p[bi, ni] = [params.get(name) is not None for name in pn]
                app_p[bi, ni] = parameter_applicability(dev, pn)
                if include_op and is_op_target(dev):
                    eligible[bi, ni] = True
                    op = dev.get("op", {})
                    raw_o[bi, ni] = [op.get(name) if op.get(name) is not None else 0. for name in on]
                    mask_o[bi, ni] = [op.get(name) is not None for name in on]
                    for si, name in enumerate(states):
                        value = dev.get("op_state", {}).get(name)
                        if value is not None:
                            if value not in (0, 1):
                                raise ValueError(f"OP state {name} must be categorical 0/1/null")
                            raw_s[bi, ni, si], mask_s[bi, ni, si] = value, True
            if include_performance:
                target = sample.get("performance", {})
                for j, name in enumerate(mn):
                    if selection is None or selection[j]:
                        value = target.get(name)
                        if value is not None:
                            raw_y[bi, j], mask_y[bi, j] = value, True
            if verbose and (bi + 1) % 5000 == 0:
                print(f"{label}: decoded {bi+1}/{b} samples | {len(graphs)} unique topologies", flush=True)
        self.tables = {}
        for name, raw, mask, stats, names in (
                ("parameters", raw_p, mask_p, preprocessing["parameters"], pn),
                *(([("op", raw_o, mask_o, preprocessing["op"], on)]) if include_op else []),
                *(([("performance", raw_y, mask_y, preprocessing["performance"], mn)]) if include_performance else [])):
            tensor, observed = torch.as_tensor(raw, device=self.device), torch.as_tensor(mask, device=self.device)
            self.tables[name] = normalize(tensor, observed, stats, names,
                                          performance_dtype if name == "performance" else torch.float32)
            self.tables['parameter_mask' if name == 'parameters' else name + '_mask'] = observed
            if name == "performance" and performance_dtype == torch.float64:
                self.tables["physical_performance"] = tensor
        self.tables["parameter_applicability"] = torch.as_tensor(app_p, device=self.device)
        if include_op:
            self.tables["op_target_mask"] = torch.as_tensor(eligible, device=self.device)
        if states:
            self.tables["op_state"] = torch.as_tensor(raw_s, device=self.device)
            self.tables["op_state_mask"] = torch.as_tensor(mask_s, device=self.device)
        self.host_graph_ids, self.graph_ids = graph_ids, torch.tensor(graph_ids, device=self.device)
        self.graphs = [{"device_ids": graph["device_ids"]} for graph in graphs]
        self.max_nodes = max(len(g["device_ids"]) for g in graphs)
        self.static = {}
        for name in graphs[0]:
            if name == "device_ids":
                continue
            values = []
            for graph in graphs:
                value = graph[name]
                shape = list(value.shape)
                shape[0] = self.max_nodes
                if name in ("adjacency", "edge_roles"):
                    shape[1] = self.max_nodes
                padded = value.new_zeros(shape)
                padded[tuple(slice(0, size) for size in value.shape)] = value
                values.append(padded)
            self.static[name] = torch.stack(values)
        self.encoded = None
        if verbose:
            print(f"{label}: ready | {len(graphs)} topologies | {self.nbytes/2**20:.1f} MiB resident "
                  f"on {self.device} | {time.perf_counter()-started:.1f}s", flush=True)

    @property
    def nbytes(self):
        return sum(v.numel()*v.element_size() for v in (*self.tables.values(), *self.static.values(), self.graph_ids))

    def batches(self, batch_size, *, indices=None, seed=None, metadata=False):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        order = list(range(len(self.rows))) if indices is None else list(indices)
        if seed is not None:
            random.Random(seed).shuffle(order)
        gpu_order = torch.tensor(order, dtype=torch.long, device=self.device)
        for start in range(0, len(order), batch_size):
            host = order[start:start+batch_size]
            idx = gpu_order[start:start+batch_size]
            topology = self.graph_ids.index_select(0, idx)
            n = max(len(self.graphs[self.host_graph_ids[i]]["device_ids"]) for i in host)
            if self.encoded is not None:
                batch = {"encoder_memory": self.encoded.index_select(0, idx)[:, :n+1],
                         "encoder_padding": ~self.static["device_mask"].index_select(0, topology)[:, :n]}
                batch["encoder_padding"] = F.pad(batch["encoder_padding"], (1, 0), value=False)
                for key in ("performance", "performance_mask"):
                    if key in self.tables:
                        batch[key] = self.tables[key].index_select(0, idx)
            else:
                batch = {k: v.index_select(0, topology)[:, :n, :n] if k in ("adjacency", "edge_roles")
                         else v.index_select(0, topology)[:, :n] for k, v in self.static.items() if k != "node_parent"}
                parent = self.static["node_parent"].index_select(0, topology)[:, :n]
                real = (parent >= 0) & batch["device_mask"]
                for key, table in self.tables.items():
                    value = table.index_select(0, idx)
                    if key in ("performance", "performance_mask", "physical_performance"):
                        batch[key] = value
                        continue
                    select = parent.clamp_min(0)
                    if value.ndim == 3:
                        select = select.unsqueeze(-1).expand(-1, -1, value.shape[-1])
                    value = value.gather(1, select)
                    live = real.unsqueeze(-1) if value.ndim == 3 else real
                    batch[key] = value.masked_fill(~live, 0)
            if metadata:
                batch["sample_ids"] = [self.rows[i]["sample_id"] for i in host]
                batch["device_ids"] = [self.graphs[self.host_graph_ids[i]]["device_ids"] for i in host]
            yield idx, batch

    def cache_encoder(self, encoder, batch_size, progress_every=100):
        if any(p.requires_grad for p in encoder.parameters()) or encoder.training:
            raise ValueError("Only a frozen eval encoder may have a resident feature cache")
        needed = len(self.rows) * (self.max_nodes+1) * encoder.cfg.dim * 4
        _reserve(self.device, needed)
        memory = torch.empty(len(self.rows), self.max_nodes+1, encoder.cfg.dim, device=self.device)
        progress = BatchProgress("frozen encoder cache", math.ceil(len(self.rows)/batch_size), progress_every, self.device)
        with torch.no_grad():
            for step, (indices, batch) in enumerate(self.batches(batch_size), 1):
                value, _ = encoder(batch)
                value = F.pad(value, (0, 0, 0, self.max_nodes+1-value.shape[1]))
                memory.index_copy_(0, indices, value)
                progress.update(step)
        self.encoded = memory


class BatchProgress:
    """Plain lines work in both interactive terminals and redirected Slurm logs."""
    def __init__(self, label, total, every=100, device=None):
        self.label, self.total, self.every = label, total, every
        self.started = time.perf_counter()
        self.device = torch.device(device) if device is not None else None
        print(f"{label}: 0/{total} batches", flush=True)

    def update(self, step, loss=None):
        if step != 1 and step != self.total and step % self.every:
            return
        if loss is None and self.device is not None and self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        detail = f" loss={float(loss):.6g}" if loss is not None else ""
        elapsed = time.perf_counter() - self.started
        eta = elapsed * (self.total-step) / max(1, step)
        print(f"{self.label}: {step}/{self.total} ({100*step/max(1,self.total):.1f}%)"
              f"{detail} elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)
