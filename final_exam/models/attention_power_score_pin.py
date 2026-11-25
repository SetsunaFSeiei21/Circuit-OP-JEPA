"""Detached, per-head powers of one PE's original masked attention matrix.

Keep self connections when multiplying; exclude the diagonal only when reading
R/T from each power. Never reapply the one-step topology mask to a matrix power.
Process one graph and one row chunk at a time, retaining just one full A per
graph (all heads), not K full matrices or a batch-sized powers/autograd stack.
"""
import torch

from .attention_score_pin import AttentionScoreExtractorPin, _chunk_attention_weights_pin


def attention_power_scores_pin(weights, valid, distance, chunk_size=128):
    """[B,H,N,N] -> [B,N,K,H,2]; K=distance, independent batches/heads.

This small pure-tensor primitive is also usable with gradients for verification.
The v7 observer calls it under no_grad because scores are observations only.
"""
    if (weights.ndim != 4 or weights.shape[-2] != weights.shape[-1]
            or not weights.is_floating_point()):
        raise ValueError('Attention powers require floating [B,H,N,N] weights')
    b, h, n, _ = weights.shape
    if (valid.shape != (b, n) or valid.dtype != torch.bool or valid.device != weights.device):
        raise ValueError('Attention powers require aligned boolean node validity')
    if type(distance) is not int or distance < 1 or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('Aggregation distance and chunk size must be positive integers')
    if not b or not n or not h:
        raise ValueError('Attention powers require nonempty batch/head/node axes')
    work = weights.float() if weights.dtype in (torch.float16, torch.bfloat16) else weights
    result = []
    with torch.autocast(device_type=weights.device.type, enabled=False):
        for graph in range(b):
            keep = valid[graph, None, :, None] & valid[graph, None, None, :]
            base = work[graph].masked_fill(~keep, 0.)
            receives, transmits = [[] for _ in range(distance)], [None] * distance
            columns = torch.arange(n, device=weights.device)
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                power_rows = base[:, start:end]
                diagonal = columns[start:end, None] == columns[None]
                for order in range(distance):
                    if order:
                        # Same PE, same head; not elementwise squaring or cross-PE rollout.
                        power_rows = power_rows @ base
                    observed = power_rows.masked_fill(diagonal[None], 0.)
                    receives[order].append(observed.sum(-1))
                    part = observed.sum(-2)
                    transmits[order] = part if transmits[order] is None else transmits[order] + part
            orders = [torch.stack((torch.cat(receives[k], -1), transmits[k]), -1)
                      for k in range(distance)]  # K x H,N,2
            result.append(torch.stack(orders, 1).permute(2, 1, 0, 3))  # N,K,H,2
    return torch.stack(result).to(weights.dtype)


class AttentionPowerScoreExtractorPin(AttentionScoreExtractorPin):
    """v7-only observer. Stop score gradients without detaching the PE main path."""
    def __init__(self, distance, chunk_size=128, checkpoint_scores=True):
        super().__init__(chunk_size, checkpoint_scores)
        if type(distance) is not int or distance < 1:
            raise ValueError('Aggregation distance must be a positive integer')
        self.distance = distance

    @torch.no_grad()
    def forward(self, attention, x, mask, valid):
        if self.distance == 1:
            # Exact legacy arithmetic (including chunk reduction order) for the v3 control.
            return super().forward(attention, x, mask, valid).unsqueeze(2)
        q, k, mask = self.prepare_attention(attention, x, mask, valid)
        b, n = valid.shape
        graphs = []
        with torch.autocast(device_type=x.device.type, enabled=False):
            for graph in range(b):
                rows = [_chunk_attention_weights_pin(
                    q[graph:graph+1, :, start:start+self.chunk_size], k[graph:graph+1],
                    mask[graph:graph+1, :, start:start+self.chunk_size],
                    valid[graph:graph+1, start:start+self.chunk_size], valid[graph:graph+1])
                    for start in range(0, n, self.chunk_size)]
                base = torch.cat(rows, dim=-2)
                del rows
                graphs.append(attention_power_scores_pin(
                    base, valid[graph:graph+1], self.distance, self.chunk_size)[0])
                del base
        return torch.stack(graphs).to(x.dtype)
