import copy
import unittest

import torch

from device_jepa.data import build_manifest, fit_preprocessor, samples_for_split, synthetic_dataset
from device_jepa.model import (DeviceEncoder, FrozenPerformanceModel, JEPAModel,
                               ModelConfig, alignment_loss, make_decoder, masked_mse)
from device_jepa.tensors import collate_samples
from device_jepa.training import seed_everything, state_digest


class ModelTests(unittest.TestCase):
    def setUp(self):
        seed_everything(17)
        self.data = synthetic_dataset(8, 17)
        self.manifest = build_manifest(self.data, "synthetic_4")
        self.prep = fit_preprocessor(self.data, self.manifest, include_performance=True)
        self.rows = samples_for_split(self.data, self.manifest, "train")[:3]
        self.cfg = ModelConfig(dim=16, heads=4, encoder_layers=2, cross_layers=1,
                               ffn_dim=32, decoder_layers=1, decoder_ffn_dim=32,
                               max_devices=32, dropout=.2)
        self.encoder = DeviceEncoder(self.cfg, len(self.prep["device_vocab"]), 3, 6)

    def batch(self, rows=None):
        return collate_samples(rows or self.rows, self.prep, include_op=True, include_performance=True)

    def test_full_memory_not_pooled(self):
        b = self.batch()
        memory, padding = self.encoder(b)
        self.assertEqual(memory.shape, (len(self.rows), b["device_types"].shape[1] + 1, 16))
        self.assertEqual(padding.shape, memory.shape[:2])
        self.assertFalse(padding[:, 0].any())
        self.assertTrue(torch.isfinite(memory).all())

    def test_stage1_trains_both_encoders_without_performance(self):
        model = JEPAModel(self.encoder, 3)
        b = collate_samples(self.rows, self.prep, include_op=True, include_performance=False)
        self.assertNotIn("performance", b)
        a, z, valid = model(b)
        loss, _ = alignment_loss(a, z, valid)
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None), 0)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.op_encoder.parameters() if p.grad is not None), 0)
        self.assertGreater(float(model.encoder.graph_token.grad.abs().sum()), 0)

    def test_missing_values_cannot_poison_outputs(self):
        self.encoder.eval()
        b = self.batch()
        for key in ("parameter_mask", "op_mask"):
            b[key][:, 0] = False
        base = self.encoder(b)[0]
        poison = copy.deepcopy(b)
        poison["parameters"][~poison["parameter_mask"]] = float("nan")
        poison["op"][~poison["op_mask"]] = float("nan")
        modified = self.encoder(poison)[0]
        torch.testing.assert_close(base, modified)
        jepa = JEPAModel(self.encoder, 3).eval()
        a, z, valid = jepa(poison)
        self.assertTrue(torch.isfinite(a).all() and torch.isfinite(z).all())
        self.assertFalse(valid[:, 0].any())

    def test_all_missing_parameters_finite(self):
        b = self.batch()
        b["parameter_mask"].zero_()
        b["parameters"].fill_(float("nan"))
        self.assertTrue(torch.isfinite(self.encoder(b)[0]).all())

    def test_padding_invariance(self):
        self.encoder.eval()
        short = copy.deepcopy(self.rows[0])
        short["devices"] = short["devices"][:2]
        single = self.encoder(self.batch([short]))[0]
        combined = self.encoder(self.batch([short, self.rows[0]]))[0]
        torch.testing.assert_close(single[0], combined[0, :3], atol=1e-5, rtol=1e-5)
        self.assertTrue((combined[0, 3:] == 0).all())

    def test_input_order_is_canonicalized(self):
        self.encoder.eval()
        altered = copy.deepcopy(self.rows)
        for row in altered:
            row["devices"].reverse()
        torch.testing.assert_close(self.encoder(self.batch())[0], self.encoder(self.batch(altered))[0])

    def test_terminal_roles_are_not_collapsed_to_adjacency_only(self):
        self.encoder.eval()
        b = self.batch()
        changed = copy.deepcopy(b)
        changed["edge_roles"] = b["edge_roles"].roll(1, dims=-1)
        a, _ = self.encoder(b)
        z, _ = self.encoder(changed)
        self.assertGreater(float((a-z).detach().abs().max()), 1e-6)

    def test_overflow_device_count_raises_not_truncates(self):
        self.encoder.cfg.max_devices = 1
        with self.assertRaisesRegex(ValueError, "no truncation"):
            self.encoder(self.batch())

    def test_unknown_device_is_not_padding(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["devices"][0]["device_type"] = "unseen_type"
        b = self.batch(rows)
        self.assertTrue((b["device_types"][0] == self.prep["device_vocab"]["<UNK>"]).any())
        self.assertEqual(int(b["device_mask"][0].sum()), len(rows[0]["devices"]))
        self.assertTrue(b["device_mask"][0][b["device_types"][0] == self.prep["device_vocab"]["<UNK>"]].all())

    def test_variable_feature_and_metric_dimensions(self):
        rows = copy.deepcopy(self.rows)
        prep = copy.deepcopy(self.prep)
        prep["feature_spec"]["parameter_names"] += ["C", "M"]
        prep["feature_spec"]["op_names"] += ["gds", "vgs"]
        prep["feature_spec"]["metric_names"] = [f"metric_{i}" for i in range(9)]
        for key in ("parameters", "op"):
            prep[key]["mean"] += [0., 0.]
            prep[key]["std"] += [1., 1.]
            prep[key]["count"] += [0, 0]
        b = collate_samples(rows, prep, include_op=True, include_performance=False)
        encoder = DeviceEncoder(self.cfg, len(prep["device_vocab"]), 5, 6)
        jepa = JEPAModel(encoder, 5)
        a, z, valid = jepa(b)
        self.assertTrue(torch.isfinite(alignment_loss(a, z, valid)[0]))
        prediction = FrozenPerformanceModel(encoder, make_decoder(self.cfg, 9))(b)
        self.assertEqual(prediction.shape, (len(rows), 9))

    def test_decoder_consumes_device_memory(self):
        decoder = make_decoder(self.cfg, 2).eval()
        memory = torch.randn(1, 5, 16)
        padding = torch.zeros(1, 5, dtype=torch.bool)
        base = decoder(memory, padding)
        changed = memory.clone()
        changed[:, 1:] += torch.randn_like(changed[:, 1:])
        self.assertGreater(float((base-decoder(changed, padding)).detach().abs().max()), 1e-6)

    def test_frozen_encoder_stays_eval_and_bit_identical(self):
        model = FrozenPerformanceModel(self.encoder, make_decoder(self.cfg, 2))
        model.encoder.register_buffer("scalar_probe", torch.tensor(0.))
        original = state_digest(model.encoder)
        model.train()
        self.assertFalse(model.encoder.training)
        self.assertTrue(model.decoder.training)
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
        b = self.batch()
        with torch.no_grad():
            first = model.encoder(b)[0]
            second = model.encoder(b)[0]
        self.assertTrue(torch.equal(first, second))
        decoder_before = state_digest(model.decoder)
        optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=.001)
        loss = masked_mse(model(b), b["performance"], b["performance_mask"])
        loss.backward()
        optimizer.step()
        self.assertEqual(state_digest(model.encoder), original)
        self.assertNotEqual(state_digest(model.decoder), decoder_before)
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))

    def test_masked_metric_loss_ignores_missing(self):
        pred = torch.tensor([[2., 1000.]])
        labels = torch.tensor([[1., float("nan")]])
        self.assertEqual(float(masked_mse(pred, labels, torch.tensor([[True, False]]))), 1.)

    def test_collapse_is_penalized(self):
        x = torch.zeros(2, 3, 16)
        valid = torch.ones(2, 3, dtype=torch.bool)
        loss, detail = alignment_loss(x, x, valid)
        self.assertEqual(float(detail["alignment_mse"]), 0.)
        self.assertGreater(float(loss), 20.)


if __name__ == "__main__":
    unittest.main()
