"""Explicit pin-level baseline factories; device-only factories remain unchanged."""
import torch
from experiments.op_transfer.models import ReadoutModel
from .common import model_config
from .common_pin import MLPPin,design_input_pin
from .graph_pin import GATEncoderPin,MeanReadoutPin,DeepGENEncoderPin,DeepGENReadoutPin
from .accformer_pin import ACCFormerEncoderPin,ACCFormerDecoderPin
from .gtn_pin import GTNEncoderPin,GTNDecoderPin
from .zerosim_pin import ZeroSimEncoderPin,make_decoder_pin

TORCH_BASELINES_PIN=('mlp_pin','gat_pin','deepgen_supervised_pin','deepgen_op_pretrained_pin',
                     'gtn_pin','zerosim_supervised_pin','accformer_pin')


def require_pin_preprocessing(prep):
    if prep.get('granularity')!='pin' or prep.get('pin_vocab')!=prep.get('device_vocab'):
        raise ValueError('Pin model factories require saved pin preprocessing and pin vocabulary')


class PinReadoutModel(ReadoutModel):
    """The architecture's own readout consumes graph+pin memory."""
    def memory(self,batch):
        inputs=design_input_pin(batch)
        if self.frozen:
            with torch.no_grad():
                return self.encoder(inputs)
        return self.encoder(inputs)


def build_encoder_pin(name,config,prep):
    require_pin_preprocessing(prep)
    factories={'gat_pin':GATEncoderPin,'deepgen_supervised_pin':DeepGENEncoderPin,
        'deepgen_op_pretrained_pin':DeepGENEncoderPin,'gtn_pin':GTNEncoderPin,
        'zerosim_supervised_pin':ZeroSimEncoderPin,'accformer_pin':ACCFormerEncoderPin}
    if name not in factories:
        raise ValueError(f'{name!r} is not a pin graph encoder')
    return factories[name](config,prep)


def build_decoder_pin(name,config,prep):
    require_pin_preprocessing(prep)
    cfg=model_config(config)
    count=len(prep['feature_spec']['metric_names'])
    if name=='zerosim_supervised_pin':
        return make_decoder_pin(cfg,prep)
    factories={'gat_pin':MeanReadoutPin,'deepgen_supervised_pin':DeepGENReadoutPin,
        'deepgen_op_pretrained_pin':DeepGENReadoutPin,'gtn_pin':GTNDecoderPin,
        'accformer_pin':ACCFormerDecoderPin}
    if name not in factories:
        raise ValueError(f'{name!r} is not a pin graph decoder')
    return factories[name](cfg,count)


def build_model_pin(name,config,prep):
    require_pin_preprocessing(prep)
    if name=='mlp_pin':
        return MLPPin(config,prep)
    if name not in TORCH_BASELINES_PIN:
        raise ValueError(f'Unknown pin Torch baseline {name!r}')
    return PinReadoutModel(build_encoder_pin(name,config,prep),build_decoder_pin(name,config,prep))


def prepare_model_pin(model,ctx,options=None,*,splits=('train','val')):
    modules=[module for module in model.modules() if hasattr(module,'prepare_resident_pin')]
    for split in splits:
        if modules and ctx.rows[split]:
            store=ctx.cache(split,op=False)
            for module in modules:
                module.prepare_resident_pin(store)
    return model
