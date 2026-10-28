"""CPU checks for input boundaries, frozen modules and deployable replay."""
import copy
import unittest

import torch

from device_jepa.data import build_manifest, fit_preprocessor, samples_for_split, synthetic_dataset
from device_jepa.model import ModelConfig
from device_jepa.tensors import collate_samples
from device_jepa.training import seed_everything, state_digest
from experiments.op_transfer.models import (
    DualEncoderModel, OPPredictor, PredictedOPCascade, ReadoutModel, decoder,
    describe_model, design_encoder, design_input, fit_op_support, freeze,
    op_encoder, op_input, rebuild_model,
)


class TransferModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything(17)
        cls.data = synthetic_dataset(6)
        cls.manifest = build_manifest(cls.data, ood_circuit="synthetic_4")
        cls.prep = fit_preprocessor(cls.data, cls.manifest, include_performance=True)
        cls.rows = samples_for_split(cls.data, cls.manifest, "train")
        cls.batch = collate_samples(cls.rows[:4], cls.prep, include_performance=True)
        cls.cfg = ModelConfig(dim=8, heads=2, encoder_layers=2, cross_layers=1,
                              ffn_dim=16, decoder_layers=1, decoder_ffn_dim=16, dropout=.3)
        cls.support = fit_op_support(cls.rows, cls.prep)

    def make_predictor(self):
        return OPPredictor(design_encoder(self.cfg, self.prep), self.prep, self.support)

    def make_readout(self, source="design", frozen=False):
        enc = (design_encoder(self.cfg, self.prep) if source == "design" else
               op_encoder(self.cfg, self.prep, use_design=source == "design_op"))
        return ReadoutModel(enc, decoder(self.cfg, self.prep), source=source, frozen=frozen)

    def corrupted_op(self):
        result = copy.deepcopy(self.batch)
        result["op"] = torch.full_like(result["op"], float("nan"))
        result["op_mask"] = ~result["op_mask"]
        result["op_target_mask"] = ~result["op_target_mask"]
        result["op_state"] = torch.tensor(float("nan"))
        result["performance"] = torch.full_like(result["performance"], float("nan"))
        result["physical_performance"] = result["performance"]
        return result

    def test_input_allowlists(self):
        design = design_input(self.corrupted_op())
        self.assertFalse(any(key.startswith("op") or "performance" in key for key in design))
        op = op_input(self.batch)
        self.assertFalse(any(key.startswith("parameter") or "performance" in key for key in op))
        self.assertIn("parameters", op_input(self.batch, use_design=True))

    def test_frozen_teacher_includes_embeddings_and_stays_eval(self):
        model = self.make_readout("op", frozen=True).train()
        before = state_digest(model.encoder)
        self.assertFalse(model.encoder.training)
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
        model(self.batch).sum().backward()
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.decoder.parameters()))
        self.assertEqual(before, state_digest(model.encoder))

    def test_op_teacher_does_not_see_design_parameters(self):
        model = self.make_readout("op").eval()
        changed = copy.deepcopy(self.batch)
        changed["parameters"].fill_(1e10)
        changed["parameter_mask"] = ~changed["parameter_mask"]
        with torch.no_grad():
            torch.testing.assert_close(model(self.batch), model(changed), rtol=0, atol=0)
        self.assertFalse(hasattr(model.encoder, "parameter_value"))

    def test_prediction_mask_and_values_ignore_measured_op(self):
        model = self.make_predictor().eval()
        with torch.no_grad():
            a, mask = model(self.batch)
            b, changed_mask = model(self.corrupted_op())
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(torch.equal(mask, changed_mask))
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue((a[~mask] == 0).all())
        resistor = self.batch["device_types"] == self.prep["device_vocab"]["resistor"]
        self.assertFalse(bool(mask[resistor].any()))

    def test_training_support_rejects_heldout_and_unknown_types_have_no_mask(self):
        with self.assertRaisesRegex(ValueError, "ID training"):
            fit_op_support(samples_for_split(self.data, self.manifest, "test"), self.prep)
        model = self.make_predictor().eval()
        changed = copy.deepcopy(self.batch)
        changed["device_types"][0, 0] = self.prep["device_vocab"]["<UNK>"]
        self.assertFalse(bool(model.topology_mask(changed)[0, 0].any()))

    def test_cascade_uses_prediction_in_same_normalized_units(self):
        predictor = self.make_predictor()
        cascade = PredictedOPCascade(predictor, self.make_readout("design_op")).train()
        self.assertFalse(cascade.predictor.training)
        predicted, mask = predictor(self.batch)
        prepared = cascade.predicted_batch(self.batch)
        torch.testing.assert_close(prepared["op"], predicted, rtol=0, atol=0)
        self.assertTrue(torch.equal(prepared["op_mask"], mask))
        cascade(self.batch).sum().backward()
        self.assertTrue(all(p.grad is None for p in predictor.parameters()))
        self.assertTrue(any(p.grad is not None for p in cascade.downstream.parameters()))
        cascade.eval()
        with torch.no_grad():
            torch.testing.assert_close(cascade(self.batch), cascade(self.corrupted_op()), rtol=0, atol=0)

    def test_joint_cascade_has_gradient_to_predictor(self):
        predictor = freeze(self.make_predictor())
        cascade = PredictedOPCascade(predictor, self.make_readout("design_op"), joint=True).train()
        self.assertTrue(predictor.training)
        self.assertTrue(all(p.requires_grad for p in predictor.parameters()))
        cascade(self.batch).sum().backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in predictor.parameters()))

    def test_dual_control_same_capacity_and_design_only_inputs(self):
        def make(frozen):
            return DualEncoderModel(design_encoder(self.cfg, self.prep),
                                    design_encoder(self.cfg, self.prep), decoder(self.cfg, self.prep),
                                    self.cfg.dim, guided_frozen=frozen)
        guided, control = make(True), make(False)
        self.assertEqual(sum(p.numel() for p in guided.parameters()), sum(p.numel() for p in control.parameters()))
        guided.train()
        self.assertFalse(guided.guided.training)
        guided(self.batch).sum().backward()
        self.assertTrue(all(p.grad is None for p in guided.guided.parameters()))
        self.assertTrue(any(p.grad is not None for p in guided.direct.parameters()))
        guided.eval()
        with torch.no_grad():
            torch.testing.assert_close(guided(self.batch), guided(self.corrupted_op()), rtol=0, atol=0)

    def test_checkpoint_recipe_replays_without_teacher(self):
        models = [self.make_readout(), self.make_readout("op"), self.make_predictor(),
                  PredictedOPCascade(self.make_predictor(), self.make_readout("design_op")),
                  DualEncoderModel(design_encoder(self.cfg, self.prep), design_encoder(self.cfg, self.prep),
                                   decoder(self.cfg, self.prep), self.cfg.dim)]
        for model in models:
            with self.subTest(kind=type(model).__name__):
                model.eval()
                recipe = describe_model(model)
                restored = rebuild_model(recipe, self.cfg, self.prep)
                restored.load_state_dict(copy.deepcopy(model.state_dict()))
                restored.eval()
                with torch.no_grad():
                    a, b = model(self.batch), restored(self.batch)
                if isinstance(a, tuple):
                    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
                    self.assertTrue(torch.equal(a[1], b[1]))
                else:
                    torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
