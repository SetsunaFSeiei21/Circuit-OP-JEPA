"""Training-only teacher guidance for a deployable predicted-OP cascade.

The teacher sees measured OP; the student always sees its frozen predictor's
outputs. Both encoders are compared in ONE coordinate system fitted on teacher
training features. Export the ``student`` cascade, not this training wrapper.
"""
import math

import torch
from torch import nn

from device_jepa.model import masked_mse
from .losses import LatentScaler, _groups
from .models import PredictedOPCascade, ReadoutModel, freeze, op_input


def _signature(module):
    return type(module), {name: tuple(value.shape) for name, value in module.state_dict().items()}


def _storage(module):
    # Detect tied storage as well as an accidentally reused module instance.
    return {(p.device, p.untyped_storage().data_ptr()) for p in module.parameters() if p.numel()}


class PredictedOPDistillation(nn.Module):
    """Frozen full-input teacher + frozen OP predictor + trainable student.

    Training loss equally averages the device-token and graph-token alignment
    terms before adding supervised performance MSE. Graph targets are used only
    for samples with an eligible observed OP device. Validation selection uses
    performance MSE alone and never calls the teacher or reads measured OP.
    """

    def __init__(self, student, teacher, teacher_scaler):
        super().__init__()
        if not isinstance(student, PredictedOPCascade) or student.joint:
            raise ValueError('Distillation requires a cascade with a frozen OP predictor')
        if not isinstance(teacher, ReadoutModel) or not isinstance(student.downstream, ReadoutModel):
            raise ValueError('Teacher and student downstream must be ReadoutModel instances')
        downstream = student.downstream
        if teacher.source != 'design_op' or downstream.source != 'design_op':
            raise ValueError('Teacher and student must both use topology, design and OP')
        if teacher.scaler is not None or downstream.scaler is not None:
            raise ValueError('Performance decoders must consume raw encoder features')
        if downstream.frozen:
            raise ValueError('Student downstream encoder must be trainable')
        if (teacher.encoder.cfg.to_dict() != downstream.encoder.cfg.to_dict()
                or _signature(teacher.encoder) != _signature(downstream.encoder)
                or _signature(teacher.decoder) != _signature(downstream.decoder)):
            raise ValueError('Teacher and student downstream architectures must match')
        if _storage(student) & _storage(teacher):
            raise ValueError('Teacher and student must not share parameter storage')
        if not isinstance(teacher_scaler, LatentScaler) or not teacher_scaler.finalized:
            raise ValueError('Teacher latent scaler must be finalized from training data')
        if teacher_scaler.hidden_dim != teacher.encoder.cfg.dim:
            raise ValueError('Teacher latent scaler width does not match the encoders')
        self.student = student
        self.teacher = freeze(teacher)
        self.teacher_scaler = teacher_scaler
        freeze(self.student.predictor)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        self.student.predictor.eval()
        return self

    def forward(self, batch):
        """This path is design-only even when the wrapper still has a teacher."""
        return self.student(batch)

    def teacher_memory(self, batch):
        """Frozen, scaled target suitable for one-time training feature caching."""
        with torch.no_grad():
            memory, padding = self.teacher.encoder(op_input(batch, use_design=True))
            return self.teacher_scaler.transform(memory, padding, batch['_group_keys']), padding

    def _alignment(self, batch, prepared, memory, padding):
        """Direct encoder alignment; deliberately independent of performance labels."""
        valid = (batch['device_mask'] & ~padding[:, 1:] & batch['op_target_mask']
                 & (batch['op_mask'] & prepared['op_mask']).any(-1))
        graph_valid = valid.any(-1) & ~padding[:, 0]
        if not bool(valid.any()):
            return None
        if '_teacher_memory' in batch:
            target, target_padding = batch['_teacher_memory'], batch['_teacher_padding']
        else:
            target, target_padding = self.teacher_memory(batch)
        if memory.shape != target.shape or not torch.equal(padding, target_padding):
            raise ValueError('Teacher/student graph and device tokens must match')
        target = target.detach()
        scaled = self.teacher_scaler.transform(memory, padding, batch['_group_keys'])
        device_delta = scaled[:, 1:][valid] - target[:, 1:][valid]
        graph_delta = scaled[:, 0][graph_valid] - target[:, 0][graph_valid]
        if not bool(torch.isfinite(device_delta).all() & torch.isfinite(graph_delta).all()):
            raise ValueError('Nonfinite eligible teacher/student alignment features')
        device_loss = device_delta.square().mean()
        graph_loss = graph_delta.square().mean()
        alignment = (device_loss + graph_loss) / 2
        # The common teacher scale recovers raw differences without another cache.
        raw_sse = memory.new_zeros(())
        for key, index in _groups(batch['_group_keys'], valid):
            if not index.shape[0]:
                continue
            delta = (scaled[index[:, 0], index[:, 1]+1]
                     - target[index[:, 0], index[:, 1]+1])
            scale = self.teacher_scaler.scale_for_key(key, delta)
            raw_sse = raw_sse + (delta * scale).square().sum()
        raw_device_loss = raw_sse / (valid.sum() * memory.shape[-1])
        return alignment, int(valid.sum()), {
            'device_alignment_mse': device_loss.detach(),
            'graph_alignment_mse': graph_loss.detach(),
            'alignment_mse': alignment.detach(),
            'raw_device_alignment_mse': raw_device_loss.detach(),
        }

    def objective(self, batch, *, alignment_weight=1., performance_weight=1.):
        for name, value, allow_zero in (('alignment_weight', alignment_weight, True),
                                        ('performance_weight', performance_weight, False)):
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or value < 0 or (not allow_zero and value == 0)):
                raise ValueError(f'{name} must be finite and {"nonnegative" if allow_zero else "positive"}')
        mask = batch['performance_mask']
        if not bool(mask.any()):
            return None

        prepared = self.student.predicted_batch(batch)
        memory, padding = self.student.downstream.encoder(op_input(prepared, use_design=True))
        prediction = self.student.downstream.decoder(memory, padding)
        performance = masked_mse(prediction, batch['performance'].to(prediction.dtype), mask)
        weight = int(mask.sum())
        if not self.training:
            # Neither alignment nor its coefficient may affect checkpoint selection.
            return performance, weight, {'perf_mse': performance.detach()}

        zero = performance.new_zeros(())
        alignment = zero
        detail = dict.fromkeys(('device_alignment_mse', 'graph_alignment_mse',
                                'alignment_mse', 'raw_device_alignment_mse'), zero)
        if alignment_weight > 0:
            aligned = self._alignment(batch, prepared, memory, padding)
            if aligned is not None:
                alignment, _, detail = aligned

        weighted_performance = performance_weight * performance
        weighted_alignment = alignment_weight * alignment
        return weighted_performance + weighted_alignment, weight, {
            **detail,
            'perf_mse': performance.detach(),
            'weighted_performance': weighted_performance.detach(),
            'weighted_alignment': weighted_alignment.detach(),
        }


class PredictedOPEncoderDistillation(PredictedOPDistillation):
    """Encoder-only stage; train AND validation select on teacher alignment.

    The copied performance decoder is frozen and never evaluated here. Measured
    OP supplies train/validation teacher targets, never student forward inputs.
    No performance labels or joint-student loss weights affect this objective.
    """

    def __init__(self, student, teacher, teacher_scaler):
        super().__init__(student, teacher, teacher_scaler)
        freeze(self.student.downstream.decoder)

    def train(self, mode=True):
        super().train(mode)
        self.student.downstream.decoder.eval()
        return self

    def objective(self, batch):
        prepared = self.student.predicted_batch(batch)
        memory, padding = self.student.downstream.encoder(op_input(prepared, use_design=True))
        return self._alignment(batch, prepared, memory, padding)
