"""EMA is a parameter-update boundary, not a stop-gradient OP-input boundary."""
import copy
import io
import unittest
from unittest.mock import patch

import torch

from device_jepa.data import build_manifest, fit_preprocessor, samples_for_split, synthetic_dataset
from device_jepa.ema_model import EMAJEPAModel, EMA_ARCHITECTURE, EMA_MODE, ema_protocol, validate_ema_decay
from device_jepa.model import DeviceEncoder, JEPAModel, ModelConfig, alignment_loss
from device_jepa.tensors import collate_samples


class EMAModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        data = synthetic_dataset(8)
        manifest = build_manifest(data, "synthetic_4")
        self.prep = fit_preprocessor(data, manifest)
        self.rows = samples_for_split(data, manifest, "train")[:3]
        self.cfg = ModelConfig(dim=16, heads=4, encoder_layers=3, cross_layers=2,
                               ffn_dim=32, decoder_layers=1, decoder_ffn_dim=32,
                               max_devices=32, dropout=.4)
        self.encoder = DeviceEncoder(self.cfg, len(self.prep["device_vocab"]), 3, 6,
            input_contract_version=self.prep["input_contract_version"],
            num_categories=len(self.prep["device_category_vocab"]),
            num_models=len(self.prep["model_vocab"]), num_port_roles=len(self.prep["port_role_vocab"]))
        self.model = EMAJEPAModel(self.encoder, 3, ema_decay=.8)
        self.batch = collate_samples(self.rows, self.prep, include_op=True, include_performance=False)

    def test_exact_initial_copy_without_shared_parameters_or_storage(self):
        model = self.model
        self.assertIsNot(model.encoder.structure_encoder, model.op_encoder.structure_encoder)
        self.assertIsNot(model.encoder.cfg, model.op_encoder.cfg)
        all_parameters = list(model.named_parameters(remove_duplicate=False))
        self.assertEqual(len(all_parameters), len({id(value) for _, value in all_parameters}))
        for name, online, target in model.ema_parameter_pairs():
            torch.testing.assert_close(online, target, rtol=0, atol=0, msg=name)
            self.assertNotEqual(online.data_ptr(), target.data_ptr())
            self.assertFalse(target.requires_grad)
        self.assertFalse(hasattr(model.op_encoder, "parameter_names"))
        self.assertFalse(hasattr(model.op_encoder, "parameter_value"))
        self.assertTrue(model.op_encoder.op_names.weight.requires_grad)
        self.assertTrue(model.op_encoder.op_value.weight.requires_grad)
        self.assertEqual(model.ema_updates, 0)

    def test_loss_trains_online_and_op_embeddings_but_not_target_trunk(self):
        model = self.model.train()
        source_before = {name: p.detach().clone() for name, p in model.encoder.named_parameters()}
        target_before = {name: target.detach().clone() for name, _, target in model.ema_parameter_pairs()}
        op_before = model.op_encoder.op_value.weight.detach().clone()
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.01)
        predicted, target, valid = model(self.batch)
        self.assertTrue(target.requires_grad)
        alignment_loss(predicted, target, valid)[0].backward()
        for component in (model.encoder.structure_encoder, model.encoder.fusion_blocks,
                          model.encoder.parameter_names, model.encoder.parameter_value,
                          model.op_encoder.op_names, model.op_encoder.op_value):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in component.parameters()
                                   if p.grad is not None), 0.)
        self.assertTrue(all(target.grad is None for _, _, target in model.ema_parameter_pairs()))
        optimizer.step()
        self.assertTrue(any(not torch.equal(p, source_before[name]) for name, p in model.encoder.named_parameters()))
        self.assertFalse(torch.equal(model.op_encoder.op_value.weight, op_before))
        for name, _, target in model.ema_parameter_pairs():
            torch.testing.assert_close(target, target_before[name], rtol=0, atol=0)
        model.verify_ema_state()

    def test_target_only_loss_reaches_only_op_embeddings(self):
        self.model.eval()
        _, target, valid = self.model(self.batch)
        target[valid].square().mean().backward()
        self.assertTrue(all(p.grad is None for p in self.model.encoder.parameters()))
        self.assertGreater(float(self.model.op_encoder.op_value.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for _, _, p in self.model.ema_parameter_pairs()))

    def test_ema_formula_includes_structure_fusion_and_fallback_not_field_embeddings(self):
        model = self.model.train()
        before = {name: target.detach().clone() for name, _, target in model.ema_parameter_pairs()}
        op_before = {name: p.detach().clone() for name, p in model.op_encoder.named_parameters()
                     if p.requires_grad}
        with torch.no_grad():
            for _, online, _ in model.ema_parameter_pairs():
                online.add_(.25)
        model.update_ema()
        for name, online, target in model.ema_parameter_pairs():
            expected = (before[name].double() * .8 + online.double() * .2).to(target.dtype)
            torch.testing.assert_close(target, expected, rtol=2e-7, atol=1e-7, msg=name)
        for name, p in model.op_encoder.named_parameters():
            if p.requires_grad:
                torch.testing.assert_close(p, op_before[name], rtol=0, atol=0)
        self.assertEqual(model.ema_updates, 1)
        names = set(before)
        self.assertIn("null_parameter", names)
        self.assertTrue(any(name.startswith("structure_encoder.type_embedding") for name in names))
        self.assertTrue(any(name.startswith("fusion_blocks") for name in names))
        self.assertTrue(any(name.startswith("output_projection") for name in names))
        self.assertFalse(any(name.startswith("parameter_") for name in names))

    def test_zero_decay_copies_trunk_and_protocol_records_exact_boundaries(self):
        model = EMAJEPAModel(self.encoder, 3, ema_decay=0)
        with torch.no_grad():
            for _, online, _ in model.ema_parameter_pairs():
                online.add_(.25)
        model.update_ema()
        for name, online, target in model.ema_parameter_pairs():
            torch.testing.assert_close(target, online, rtol=0, atol=0, msg=name)
        self.assertEqual(EMA_MODE, "jepa_zerosim_ema")
        self.assertEqual(EMA_ARCHITECTURE, "separate-structure-ema_device-op-cross-attention_v1")
        self.assertEqual(ema_protocol(.8), {
            "decay": .8,
            "update_scope": ["structure_encoder", "fusion_blocks", "output_norm", "output_projection", "null_parameter"],
            "update_timing": "after_optimizer_step",
            "gradient_policy": "independent_input_embeddings_frozen_op_trunk"})
        self.assertEqual(validate_ema_decay(0), 0.)

    def test_weight_round_trip_restores_both_branches_without_aliasing(self):
        model = self.model.train()
        with torch.no_grad():
            model.encoder.output_projection.weight.add_(.25)
            model.op_encoder.op_value.weight.add_(.5)
        model.update_ema()
        model.eval()
        before = model(self.batch)
        stream = io.BytesIO()
        # Counter belongs to training protocol metadata, not model weights.
        torch.save({"model": model.state_dict(), "ema_updates": model.ema_updates}, stream)
        stream.seek(0)
        restored = torch.load(stream, weights_only=True)
        clone = EMAJEPAModel(copy.deepcopy(self.encoder), 3, ema_decay=.8)
        clone.load_state_dict(restored["model"], strict=True)
        clone.ema_updates = restored["ema_updates"]
        clone.verify_ema_state()
        after = clone.eval()(self.batch)
        self.assertEqual(clone.ema_updates, 1)
        for expected, actual in zip(before, after):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(all(online.data_ptr() != target.data_ptr()
                            for _, online, target in clone.ema_parameter_pairs()))

    def test_forward_validation_and_train_mode_never_update_ema(self):
        model = self.model.train()
        self.assertTrue(model.encoder.training)
        self.assertFalse(model.op_encoder.structure_encoder.training)
        self.assertFalse(model.op_encoder.fusion_blocks.training)
        before = {k: v.clone() for k, v in model.op_encoder.state_dict().items()}
        self.assertTrue(torch.equal(model(self.batch)[1], model(self.batch)[1]))
        model.eval()
        model(self.batch)
        with self.assertRaisesRegex(RuntimeError, "only during training"):
            model.update_ema()
        self.assertEqual(model.ema_updates, 0)
        for name, value in model.op_encoder.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_branches_each_evaluate_their_own_structure_once(self):
        with patch.object(self.model.encoder.structure_encoder, "forward",
                          wraps=self.model.encoder.structure_encoder.forward) as online, \
             patch.object(self.model.op_encoder.structure_encoder, "forward",
                          wraps=self.model.op_encoder.structure_encoder.forward) as target:
            self.model(self.batch)
        self.assertEqual(online.call_count, 1)
        self.assertEqual(target.call_count, 1)

    def test_design_and_op_information_boundaries(self):
        model = self.model.eval()
        a, b, _ = model(self.batch)
        changed = copy.deepcopy(self.batch)
        changed["op"] += 3.
        a2, b2, _ = model(changed)
        torch.testing.assert_close(a2, a, rtol=0, atol=0)
        self.assertGreater(float((b2 - b).abs().max()), 1e-6)
        changed = copy.deepcopy(self.batch)
        changed["parameters"] += 3.
        a3, b3, _ = model(changed)
        self.assertGreater(float((a3 - a).abs().max()), 1e-6)
        torch.testing.assert_close(b3, b, rtol=0, atol=0)
        for name in ("parameters", "parameter_mask", "parameter_applicability"):
            changed.pop(name, None)
        changed["performance"] = torch.full((3, 2), float("nan"))
        direct, _ = model.op_encoder(changed)
        torch.testing.assert_close(direct[:, 1:], b, rtol=0, atol=0)

    def test_missing_ineligible_and_padding_op_cannot_poison_target(self):
        model = self.model.eval()
        batch = copy.deepcopy(self.batch)
        batch["op_mask"][:, 0] = False
        baseline = model(batch)
        inactive = ~batch["op_mask"] | ~batch["op_target_mask"].unsqueeze(-1) | ~batch["device_mask"].unsqueeze(-1)
        batch["op"][inactive] = float("nan")
        output = model(batch)
        torch.testing.assert_close(output[1], baseline[1], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(output[1]).all())
        self.assertFalse(output[2][:, 0].any())
        self.assertFalse((output[2] & ~batch["op_target_mask"]).any())
        self.assertTrue((output[1][~batch["device_mask"]] == 0).all())

    def test_eligible_missing_fields_remain_named_but_ineligible_have_no_live_tokens(self):
        batch = copy.deepcopy(self.batch)
        batch["op_mask"][:, 0] = False
        _, live = self.model.op_encoder.input_tokens(batch)
        expected = (batch["device_mask"] & batch["op_target_mask"]).unsqueeze(-1).expand_as(live)
        self.assertTrue(torch.equal(live, expected))
        batch["op_mask"].zero_()
        batch["op"].fill_(float("nan"))
        _, target, valid = self.model.eval()(batch)
        self.assertTrue(torch.isfinite(target).all())
        self.assertFalse(valid.any())

    def test_padding_invariance_and_all_ineligible_safe_null_attention(self):
        model = self.model.eval()
        short = copy.deepcopy(self.rows[0])
        short["devices"] = short["devices"][:2]
        single = collate_samples([short], self.prep, include_op=True, include_performance=False)
        combined = collate_samples([short, self.rows[0]], self.prep, include_op=True, include_performance=False)
        one, many = model(single)[1], model(combined)[1]
        torch.testing.assert_close(one[0], many[0, :2], rtol=1e-5, atol=1e-5)
        self.assertTrue((many[0, 2:] == 0).all())
        combined["op_target_mask"].zero_()
        combined["op"].fill_(float("nan"))
        self.assertTrue(torch.isfinite(model(combined)[1]).all())
        self.assertFalse(model(combined)[2].any())

    def test_state_tokens_have_gradient_and_ignore_unobserved_and_ineligible_values(self):
        model = EMAJEPAModel(self.encoder, 3, num_state_features=2).eval()
        batch = copy.deepcopy(self.batch)
        shape = (*batch["op"].shape[:2], 2)
        batch["op_state"] = torch.zeros(shape)
        batch["op_state_mask"] = torch.ones(shape, dtype=torch.bool)
        batch["op_state_mask"][..., 1] = False
        a, target, valid = model(batch)
        target[valid].square().mean().backward()
        self.assertGreater(float(model.op_encoder.state_names.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.op_encoder.state_value.weight.grad.abs().sum()), 0.)
        changed = copy.deepcopy(batch)
        inactive = ~changed["op_state_mask"] | ~changed["op_target_mask"].unsqueeze(-1)
        changed["op_state"][inactive] = float("nan")
        torch.testing.assert_close(model(changed)[1], target, rtol=0, atol=0)
        changed["op_state"] = batch["op_state"].clone()
        changed["op_state"][..., 0] = 1.
        a2, target2, _ = model(changed)
        torch.testing.assert_close(a, a2, rtol=0, atol=0)
        self.assertGreater(float((target - target2).abs().max()), 1e-6)
        batch.pop("op_state_mask")
        with self.assertRaisesRegex(ValueError, "state features"):
            model(batch)

    def test_invalid_decay_or_schema_and_boundary_tampering_rejected(self):
        for decay in (-.1, 1., float("inf"), float("nan"), True, "0.9"):
            with self.subTest(decay=decay), self.assertRaisesRegex(ValueError, "ema_decay"):
                EMAJEPAModel(self.encoder, 3, ema_decay=decay)
        batch = copy.deepcopy(self.batch)
        batch.pop("op_target_mask")
        with self.assertRaisesRegex(ValueError, "eligibility"):
            self.model(batch)
        self.model.op_encoder.output_projection.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "parameter gradients"):
            self.model.verify_ema_state()

    def test_original_shared_mlp_model_keeps_its_existing_protocol(self):
        original = JEPAModel(self.encoder, 3)
        self.assertIs(original.encoder.structure_encoder, original.op_encoder.structure_encoder)
        self.assertTrue(all(p.requires_grad for p in original.parameters()))
        self.assertFalse(hasattr(original, "update_ema"))


if __name__ == "__main__":
    unittest.main()
