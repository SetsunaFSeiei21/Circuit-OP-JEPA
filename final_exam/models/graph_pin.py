"""Pin-node GAT and DeepGEN adapters with architecture-native readouts.

The kernels retain the source-attributed implementation from graph.py. Their
learned node axis now contains terminals and named ports, never whole devices.
"""
from .common_pin import PinEmbedding
from .graph import GATEncoder, DeepGENEncoder, MeanReadout, DeepGENReadout


class GATEncoderPin(GATEncoder):
    def __init__(self, config, prep):
        super().__init__(config, prep)
        self.embedding = PinEmbedding(self.cfg, prep)


class MeanReadoutPin(MeanReadout):
    """Native mean pooling over all valid terminal and external-port tokens."""


class DeepGENEncoderPin(DeepGENEncoder):
    def __init__(self, config, prep):
        super().__init__(config, prep)
        self.embedding = PinEmbedding(self.cfg, prep)


class DeepGENReadoutPin(DeepGENReadout):
    """Native learned query pooling reads pin tokens rather than device tokens."""
