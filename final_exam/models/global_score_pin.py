"""Per-graph, per-PE/head descriptive statistics of already extracted R/T scores.

No distribution is fitted and no attention is renormalized. The valid mask
includes the native GRAPH token and excludes padding. All operations remain
differentiable; the consuming model owns any stop-gradient boundary.
"""
import math

import torch
from torch import nn


class GlobalScoreStatisticsPin(nn.Module):
    """[B,N,L,H,2] -> graph context [B,6LH], statistics/deltas [B,L,H,3].

    Context order: layer, head, [mean_R, std_R, std_T, dmean_R, dstd_R, dstd_T].
    Population variance uses N; std = sqrt(variance + epsilon). Mean T is
    needed only to center T and is deliberately absent from the output.
    First-layer differences are boundary zeros. Empty graphs return zeros;
    native circuit batches always contain at least the GRAPH token.
    """
    def __init__(self, epsilon=1e-8):
        super().__init__()
        if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('Global score std epsilon must be finite and positive')
        self.epsilon = epsilon

    def forward(self, scores, valid):
        if (scores.ndim != 5 or scores.shape[-1] != 2
                or scores.shape[2] < 1 or scores.shape[3] < 1
                or not scores.is_floating_point()
                or valid.shape != scores.shape[:2] or valid.dtype != torch.bool
                or valid.device != scores.device):
            raise ValueError('Global scores require [B,N,L,H,2] and aligned boolean validity')
        # Preserve double precision for diagnostics/gradcheck. Half reductions
        # accumulate in float32, as in the existing attention-score extractor.
        values = scores.float() if scores.dtype in (torch.float16, torch.bfloat16) else scores
        invalid = ~valid[:, :, None, None, None]
        values = values.masked_fill(invalid, 0.)
        count = valid.sum(1).to(values.dtype)[:, None, None, None]
        denominator = count.clamp_min(1)
        means = values.sum(1) / denominator
        centered = (values - means[:, None]).masked_fill(invalid, 0.)
        variance = centered.square().sum(1) / denominator
        std = (variance + self.epsilon).sqrt()
        statistics = torch.stack((means[..., 0], std[..., 0], std[..., 1]), dim=-1)
        statistics = statistics.masked_fill(count == 0, 0.)
        differences = torch.cat((torch.zeros_like(statistics[:, :1]),
                                 statistics[:, 1:] - statistics[:, :-1]), dim=1)
        features = torch.cat((statistics, differences), dim=-1).flatten(1)
        return features, statistics, differences
