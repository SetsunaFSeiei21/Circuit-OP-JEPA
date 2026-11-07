"""ACCFormer_CR pin adapter; source and native decoder are unchanged.

Topology attention now runs on observed terminals plus named external ports.
Parent physical-model identity is repeated per terminal, never fitted on test.
"""
from .accformer import ACCFormerEncoder, ACCFormerDecoder
from .common_pin import PinEmbedding


class ACCFormerEncoderPin(ACCFormerEncoder):
    def __init__(self, config, prep):
        super().__init__(config, prep)
        self.embedding = PinEmbedding(self.cfg, prep)


class ACCFormerDecoderPin(ACCFormerDecoder):
    """CR per-metric queries cross-attend to the complete valid pin memory."""
