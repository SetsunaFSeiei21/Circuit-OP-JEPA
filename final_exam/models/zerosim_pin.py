"""ZeroSim-inspired pin encoder, sharing the existing attention kernels.

Pin-vocabulary IDs and a true pin graph are required. Parent design values are
injected into each owned terminal; external port tokens have no design values.
"""
from device_jepa.model import DeviceEncoder, MetricQueryDecoder, ModelConfig
from .common_pin import require_pin_batch


class ZeroSimEncoderPin(DeviceEncoder):
    def __init__(self, config, prep):
        spec = prep['feature_spec']
        cfg = config if isinstance(config,ModelConfig) else ModelConfig(**config)
        super().__init__(cfg,len(prep['device_vocab']),len(spec['parameter_names']),
            len(spec['terminal_names']),prep.get('input_contract_version',1),
            len(prep.get('device_category_vocab',{})),len(prep.get('model_vocab',{})),
            len(prep.get('port_role_vocab',{})))

    def forward(self, batch):
        return super().forward(require_pin_batch(batch))


class MetricQueryDecoderPin(MetricQueryDecoder):
    """Matched ZeroSim-family metric queries over graph plus pin memory."""


def make_decoder_pin(config, prep):
    cfg = config if isinstance(config,ModelConfig) else ModelConfig(**config)
    return MetricQueryDecoderPin(cfg,len(prep['feature_spec']['metric_names']))
