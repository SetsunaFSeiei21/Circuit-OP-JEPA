"""Models for isolated, inference-OP-free transfer experiments.

All OP predictor outputs use the saved preprocessor's normalized OP units. A
cascade passes them directly to the OP encoder: no inverse/forward transform is
performed between the two modules. True-OP reference models are explicitly
different from the deployable design-input models below.
"""
from copy import deepcopy

import torch
from torch import nn

from device_jepa.baselines import _make_model
from device_jepa.data import canonical_model_name, is_op_target
from device_jepa.model import ModelConfig, make_decoder
from device_jepa.training import encoder_from_metadata


_TOPOLOGY = frozenset({"device_types", "device_categories", "device_models",
                       "device_mask", "adjacency", "edge_roles", "port_roles",
                       "named_port_roles"})
_DESIGN = frozenset({"parameters", "parameter_mask", "parameter_applicability"})
_OP = frozenset({"op", "op_mask", "op_target_mask"})


def design_input(batch):
    """Allow only topology and design data across the deployment boundary."""
    return {key: value for key, value in batch.items() if key in _TOPOLOGY | _DESIGN}


def op_input(batch, *, use_design=False):
    """True-OP reference input; design geometry is excluded by default."""
    allowed = _TOPOLOGY | _OP | (_DESIGN if use_design else frozenset())
    return {key: value for key, value in batch.items() if key in allowed}


def _config(config):
    return config.to_dict() if isinstance(config, ModelConfig) else dict(config)


def design_encoder(config, prep):
    return encoder_from_metadata(_config(config), prep)


def op_encoder(config, prep, *, use_design=False):
    mode = "zerosim_device_op" if use_design else "zerosim_device_op_only"
    return _make_model(mode, _config(config), prep).encoder


def decoder(config, prep):
    return make_decoder(ModelConfig(**_config(config)), len(prep["feature_spec"]["metric_names"]))


def freeze(module):
    """Freeze the COMPLETE teacher/guided module, including all embeddings."""
    module.requires_grad_(False)
    module.eval()
    return module


def _scaled(memory, padding, scaler, batch):
    if scaler is None:
        return memory
    if "_group_keys" not in batch:
        raise ValueError("Latent scaling requires circuit/topology/device group keys")
    return scaler.transform(memory, padding, batch["_group_keys"])


class ReadoutModel(nn.Module):
    """Performance head with an explicit input contract and frozen readout mode."""
    def __init__(self, encoder, decoder, *, source="design", scaler=None, frozen=False):
        super().__init__()
        if source not in {"design", "op", "design_op"}:
            raise ValueError("source must be design, op or design_op")
        self.encoder, self.decoder = encoder, decoder
        self.source, self.scaler, self.frozen = source, scaler, bool(frozen)
        if self.frozen:
            freeze(self.encoder)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def memory(self, batch):
        inputs = (design_input(batch) if self.source == "design" else
                  op_input(batch, use_design=self.source == "design_op"))
        if self.frozen:
            with torch.no_grad():
                memory, padding = self.encoder(inputs)
        else:
            memory, padding = self.encoder(inputs)
        return _scaled(memory, padding, self.scaler, batch), padding

    def forward(self, batch):
        memory, padding = self.memory(batch)
        return self.decoder(memory, padding)


def fit_op_support(train_rows, prep):
    """Fit field support ONLY on ID train observations, then freeze it.

    At inference, a known type/model pair uses its training support; a new
    physical model uses the known device-type support. Unseen device types have
    no supported fields. Eligibility comes from the trained type table and
    topology padding, never a held-out OP observation or simulation-status bit.
    """
    train_rows = list(train_rows)
    fitted = set(prep["fitted_sample_ids"])
    if not train_rows or any(row["sample_id"] not in fitted for row in train_rows):
        raise ValueError("OP support may be fitted only on nonempty ID training rows")
    types, models = prep["device_vocab"], prep.get("model_vocab", {})
    names = prep["feature_spec"]["op_names"]
    if not names:
        raise ValueError("OP prediction requires selected OP fields")
    n_models = max(1, len(models))
    type_support = torch.zeros(len(types), len(names), dtype=torch.bool)
    pair_support = torch.zeros(len(types), n_models, len(names), dtype=torch.bool)
    pair_seen = torch.zeros(len(types), n_models, dtype=torch.bool)
    eligible_types = torch.zeros(len(types), dtype=torch.bool)
    for row in train_rows:
        for dev in row["devices"]:
            kind = types.get(dev["device_type"], types["<UNK>"])
            name = canonical_model_name(dev)
            model = (models["<NONE>"] if name is None else models.get(name, models["<UNK>"])) if models else 0
            if not is_op_target(dev):
                continue
            eligible_types[kind] = True
            pair_seen[kind, model] = True
            observed = torch.tensor([dev.get("op", {}).get(field) is not None for field in names])
            type_support[kind] |= observed
            pair_support[kind, model] |= observed
    if not bool(type_support.any()):
        raise ValueError("No observed training OP field is available for prediction")
    return {"version": 1, "field_names": names[:],
            "fit_sample_ids": sorted(row["sample_id"] for row in train_rows),
            "fallback": "unseen_model_to_known_type; unseen_type_unsupported",
            "type_support": type_support.tolist(), "pair_support": pair_support.tolist(),
            "pair_seen": pair_seen.tolist(), "eligible_types": eligible_types.tolist()}


class OPPredictor(nn.Module):
    """Design encoder and physical-field head in saved normalized OP units."""
    def __init__(self, encoder, prep, support):
        super().__init__()
        self.encoder = encoder
        self.support_metadata = deepcopy(support)
        self.support = self.support_metadata
        self.op_names = list(prep["feature_spec"]["op_names"])
        if self.op_names != support["field_names"]:
            raise ValueError("OP support and preprocessor field order disagree")
        for name in ("type_support", "pair_support", "pair_seen", "eligible_types"):
            self.register_buffer(name, torch.tensor(support[name], dtype=torch.bool))
        dim = encoder.cfg.dim
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, len(self.op_names)))

    def topology_mask(self, batch):
        kinds = batch["device_types"]
        models = batch.get("device_models", torch.zeros_like(kinds))
        if bool((kinds < 0).any() | (kinds >= self.type_support.shape[0]).any()):
            raise ValueError("Device type is outside the saved training vocabulary")
        if bool((models < 0).any() | (models >= self.pair_support.shape[1]).any()):
            raise ValueError("Physical model is outside the saved training vocabulary")
        observed_pair = self.pair_seen[kinds, models]
        supported = torch.where(observed_pair.unsqueeze(-1), self.pair_support[kinds, models],
                                self.type_support[kinds])
        eligible = self.eligible_types[kinds] & batch["device_mask"]
        return supported & eligible.unsqueeze(-1)

    def forward(self, batch):
        memory, padding = self.encoder(design_input(batch))
        predicted = self.head(memory[:, 1:])
        supported = self.topology_mask(batch) & ~padding[:, 1:].unsqueeze(-1)
        return torch.where(supported, predicted, 0.), supported


class PredictedOPCascade(nn.Module):
    """Downstream +OP model consumes predicted OP in ALL dataset partitions."""
    def __init__(self, predictor, downstream, *, joint=False):
        super().__init__()
        self.predictor, self.downstream = predictor, downstream
        self.joint = bool(joint)
        if not self.joint:
            freeze(self.predictor)
        else:
            self.predictor.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if not self.joint:
            self.predictor.eval()
        return self

    def predicted_batch(self, batch):
        if self.joint:
            predicted, supported = self.predictor(batch)
        else:
            with torch.no_grad():
                predicted, supported = self.predictor(batch)
        result = design_input(batch)
        result.update(op=predicted, op_mask=supported,
                      op_target_mask=supported.any(-1) & batch["device_mask"])
        # Scaling metadata is input identity, not measured OP information.
        if "_group_keys" in batch:
            result["_group_keys"] = batch["_group_keys"]
        return result

    def memory(self, batch):
        """Cacheable predicted-OP features when the downstream encoder is frozen."""
        return self.downstream.memory(self.predicted_batch(batch))

    def forward(self, batch):
        return self.downstream(self.predicted_batch(batch))


class DualEncoderModel(nn.Module):
    """Two design-input branches, aligned device tokens, learned feature fusion.

    The control uses this identical architecture with two independently trained
    design encoders and no OP pretraining. A guided branch can instead be frozen
    after distillation or explicit OP prediction; both remain design-only here.
    """
    def __init__(self, direct, guided, decoder, dim, *, guided_frozen=True,
                 direct_scaler=None, guided_scaler=None):
        super().__init__()
        if direct.cfg.dim != dim or guided.cfg.dim != dim:
            raise ValueError("Dual branches must have the same configured latent width")
        self.direct, self.guided, self.decoder = direct, guided, decoder
        self.fusion = nn.Linear(2 * dim, dim)
        self.guided_frozen = bool(guided_frozen)
        self.direct_scaler, self.guided_scaler = direct_scaler, guided_scaler
        if self.guided_frozen:
            freeze(self.guided)

    def train(self, mode=True):
        super().train(mode)
        if self.guided_frozen:
            self.guided.eval()
        return self

    def forward(self, batch):
        inputs = design_input(batch)
        left, left_padding = self.direct(inputs)
        if self.guided_frozen:
            with torch.no_grad():
                right, right_padding = self.guided(inputs)
        else:
            right, right_padding = self.guided(inputs)
        if left.shape != right.shape or not torch.equal(left_padding, right_padding):
            raise ValueError("Dual encoders must return matching graph/device token identities")
        left = _scaled(left, left_padding, self.direct_scaler, batch)
        right = _scaled(right, right_padding, self.guided_scaler, batch)
        fused = self.fusion(torch.cat((left, right), -1)).masked_fill(left_padding.unsqueeze(-1), 0.)
        return self.decoder(fused, left_padding)


def describe_model(model):
    """Architecture recipe; weights are stored separately in ``state_dict``."""
    scaling = lambda value: None if value is None else value.state_dict()
    if isinstance(model, ReadoutModel):
        return {"kind": "readout", "source": model.source, "frozen": model.frozen,
                "scaler": scaling(model.scaler)}
    if isinstance(model, OPPredictor):
        return {"kind": "op_predictor", "support": deepcopy(model.support)}
    if isinstance(model, PredictedOPCascade):
        return {"kind": "cascade", "joint": model.joint,
                "predictor": describe_model(model.predictor),
                "downstream": describe_model(model.downstream)}
    if isinstance(model, DualEncoderModel):
        return {"kind": "dual", "guided_frozen": model.guided_frozen,
                "direct_scaler": scaling(model.direct_scaler),
                "guided_scaler": scaling(model.guided_scaler)}
    raise TypeError(f"Unsupported transfer checkpoint model: {type(model).__name__}")


def rebuild_model(recipe, config, prep):
    """Restore an inference architecture without a teacher or training dataset."""
    def scaling(state):
        if state is None:
            return None
        from .losses import LatentScaler
        return LatentScaler.from_state_dict(state)

    kind = recipe["kind"]
    if kind == "readout":
        source = recipe["source"]
        encoder = (design_encoder(config, prep) if source == "design" else
                   op_encoder(config, prep, use_design=source == "design_op"))
        return ReadoutModel(encoder, decoder(config, prep), source=source,
                            scaler=scaling(recipe.get("scaler")), frozen=recipe["frozen"])
    if kind == "op_predictor":
        return OPPredictor(design_encoder(config, prep), prep, recipe["support"])
    if kind == "cascade":
        return PredictedOPCascade(rebuild_model(recipe["predictor"], config, prep),
                                  rebuild_model(recipe["downstream"], config, prep),
                                  joint=recipe["joint"])
    if kind == "dual":
        return DualEncoderModel(design_encoder(config, prep), design_encoder(config, prep),
                                decoder(config, prep), ModelConfig(**_config(config)).dim,
                                guided_frozen=recipe["guided_frozen"],
                                direct_scaler=scaling(recipe.get("direct_scaler")),
                                guided_scaler=scaling(recipe.get("guided_scaler")))
    raise ValueError(f"Unknown transfer checkpoint architecture: {kind}")
