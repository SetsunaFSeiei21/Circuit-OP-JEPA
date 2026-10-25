"""Boundary tests for a single registered, single-evaluation structure prefix."""
import copy
import unittest

import torch

from device_jepa.data import build_manifest, fit_preprocessor, samples_for_split, synthetic_dataset
from device_jepa.model import DeviceEncoder, JEPAModel, ModelConfig
from device_jepa.tensors import collate_samples
from device_jepa.training import seed_everything


class SharedStructureTests(unittest.TestCase):
    def setUp(self):
        seed_everything(23)
        data = synthetic_dataset(8)
        manifest = build_manifest(data, "synthetic_4")
        self.prep = fit_preprocessor(data, manifest)
        self.batch = collate_samples(samples_for_split(data, manifest, "train")[:3], self.prep)
        cfg = ModelConfig(dim=16, heads=4, encoder_layers=3, cross_layers=2,
                          ffn_dim=32, decoder_layers=1, decoder_ffn_dim=32, dropout=.4)
        self.model = JEPAModel(DeviceEncoder(cfg, len(self.prep["device_vocab"]), 3, 6), 3)

    def test_one_shared_module_registered_once_no_predictor(self):
        m = self.model
        self.assertIs(m.encoder.structure_encoder, m.op_encoder.structure_encoder)
        ids = [id(p) for p in m.parameters()]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertFalse(hasattr(m, "predictor"))
        self.assertFalse(any(k.startswith("op_encoder.structure") or "_shared_structure" in k for k in m.state_dict()))
        self.assertEqual(len(m.encoder.structure_encoder.blocks), 1)
        self.assertEqual(len(m.encoder.fusion_blocks), 2)

    def test_structure_once_in_training_and_no_inplace_branch_mutation(self):
        m = self.model.train()
        seen = []
        hook = m.structure_encoder.register_forward_hook(lambda module, inputs, result: seen.append((result, result["tokens"].clone())))
        m(self.batch)
        hook.remove()
        self.assertEqual(len(seen), 1)
        torch.testing.assert_close(seen[0][0]["tokens"], seen[0][1], rtol=0, atol=0)

    def test_structure_does_not_see_parameter_op_state_or_label_values(self):
        m = self.model.eval()
        clean = m.structure_encoder(self.batch)["tokens"]
        poisoned = copy.deepcopy(self.batch)
        poisoned["parameters"].fill_(1e12)
        poisoned["op"].fill_(float("nan"))
        poisoned["performance"] = torch.full((3, 9), float("nan"))
        poisoned["op_state"] = torch.ones(3, 6, 9)
        poisoned["op_state_mask"] = torch.ones(3, 6, 9, dtype=torch.bool)
        torch.testing.assert_close(clean, m.structure_encoder(poisoned)["tokens"], rtol=0, atol=0)
        a, _ = m.encoder(self.batch)
        b, _ = m.encoder(poisoned)
        self.assertGreater(float((a-b).detach().abs().max()), 1e-6)

    def test_each_branch_updates_shared_parameters_without_detach(self):
        m = self.model.eval()
        for branch in (0, 1):
            m.zero_grad(set_to_none=True)
            output = m(self.batch)[branch]
            output.square().mean().backward()
            gradient = sum(float(p.grad.abs().sum()) for p in m.structure_encoder.parameters() if p.grad is not None)
            self.assertGreater(gradient, 0., f"branch {branch} failed to reach shared structure")

    def test_op_changes_target_only_and_design_parameters_change_input_only(self):
        m = self.model.eval()
        a, b, _ = m(self.batch)
        changed = copy.deepcopy(self.batch)
        changed["op"] += 2
        a2, b2, _ = m(changed)
        torch.testing.assert_close(a, a2, rtol=0, atol=0)
        self.assertGreater(float((b-b2).detach().abs().max()), 1e-6)
        changed = copy.deepcopy(self.batch)
        changed["parameters"] += 2
        a3, b3, _ = m(changed)
        self.assertGreater(float((a-a3).detach().abs().max()), 1e-6)
        torch.testing.assert_close(b, b3, rtol=0, atol=0)

    def test_non_mos_kept_in_graph_but_not_alignment(self):
        _, _, valid = self.model(self.batch)
        self.assertTrue((self.batch["device_mask"] & ~self.batch["op_target_mask"]).any())
        self.assertFalse((valid & ~self.batch["op_target_mask"]).any())

    def test_categorical_state_is_op_only(self):
        model = JEPAModel(self.model.encoder, 3, num_state_features=3).eval()
        batch = copy.deepcopy(self.batch)
        shape = (*batch["op"].shape[:2], 3)
        batch["op_state"], batch["op_state_mask"] = torch.zeros(shape), torch.ones(shape, dtype=torch.bool)
        a, b, _ = model(batch)
        batch["op_state"][..., 1] = 1.
        a2, b2, _ = model(batch)
        torch.testing.assert_close(a, a2, rtol=0, atol=0)
        self.assertGreater(float((b-b2).detach().abs().max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
