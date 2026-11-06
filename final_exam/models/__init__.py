"""Factory for reproducible device-level final-exam baselines."""
from experiments.op_transfer.models import ReadoutModel, design_encoder, decoder, design_input

from .accformer import ACCFormerDecoder, ACCFormerEncoder
from .common import DeviceMLP, model_config, tabular_features
from .graph import DeepGENEncoder, DeepGENReadout, GATEncoder, MeanReadout


TORCH_BASELINES = ('mlp', 'gat', 'deepgen_supervised', 'deepgen_op_pretrained',
                   'gtn', 'zerosim_supervised', 'accformer')


class DeviceReadoutModel(ReadoutModel):
    def memory(self, batch):
        inputs = design_input(batch)
        # A derived topology tensor, never an OP/label-derived quantity. Keep
        # the central deployment allowlist otherwise unchanged.
        if 'gtn_loop_membership_by_device' in batch:
            inputs['gtn_loop_membership_by_device'] = batch['gtn_loop_membership_by_device']
        if self.frozen:
            import torch
            with torch.no_grad():
                return self.encoder(inputs)
        return self.encoder(inputs)


def prepare_model(model, ctx, options=None, *, splits=('train', 'val')):
    """Prepare topology-only state once; all training batches stay on device."""
    modules = [module for module in model.modules() if hasattr(module, 'prepare_resident')]
    if modules:
        for split in splits:
            if ctx.rows[split]:
                store = ctx.cache(split, op=False)
                for module in modules:
                    module.prepare_resident(store)
    return model


def build_encoder(name, config, prep):
    """Return (graph token + device tokens, padding) with .cfg metadata."""
    if name == 'zerosim_supervised':
        return design_encoder(config, prep)
    if name == 'gat':
        return GATEncoder(config, prep)
    if name in ('deepgen_supervised', 'deepgen_op_pretrained'):
        return DeepGENEncoder(config, prep)
    if name == 'accformer':
        return ACCFormerEncoder(config, prep)
    if name == 'gtn':
        from .gtn import GTNEncoder
        return GTNEncoder(config, prep)
    raise ValueError(f'Model {name!r} has no graph encoder')


def build_decoder(name, config, prep):
    cfg = model_config(config)
    count = len(prep['feature_spec']['metric_names'])
    if name == 'zerosim_supervised':
        return decoder(config, prep)
    if name == 'gat':
        return MeanReadout(cfg, count)
    if name in ('deepgen_supervised', 'deepgen_op_pretrained'):
        return DeepGENReadout(cfg, count)
    if name == 'accformer':
        return ACCFormerDecoder(cfg, count)
    if name == 'gtn':
        from .gtn import GTNDecoder
        return GTNDecoder(cfg, count)
    raise ValueError(f'Model {name!r} has no graph decoder')


def build_model(name, config, prep):
    """Each baseline preserves its own readout; only the input contract is shared.

    ``deepgen_op_pretrained`` builds the architecture, not its training schedule.
    The runner must first pretrain build_encoder() with full masked OP targets,
    then initialize this model's encoder and supervise circuit performance.
    """
    if name == 'mlp':
        return DeviceMLP(config, prep)
    if name not in TORCH_BASELINES:
        raise ValueError(f'Unknown Torch baseline {name!r}; XGBoost uses its own runner')
    return DeviceReadoutModel(build_encoder(name, config, prep), build_decoder(name, config, prep))


__all__ = ['TORCH_BASELINES', 'build_encoder', 'build_decoder', 'build_model',
           'prepare_model', 'tabular_features']
