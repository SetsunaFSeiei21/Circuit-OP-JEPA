"""Device adaptation contracts: no measured inputs, padding, gradients and reload."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from device_jepa.data import synthetic_dataset, build_manifest, fit_preprocessor
from device_jepa.resident import ResidentDataset
from device_jepa.tensors import collate_samples
from experiments.op_transfer.models import OPPredictor, fit_op_support
from final_exam.models import TORCH_BASELINES, build_encoder, build_model
from final_exam.models.gtn import build_loop_membership


def padded(batch, extra=3):
    output = {}
    n = batch['device_mask'].shape[1]
    graph_keys = {'adjacency', 'edge_roles'}
    device_keys = {'device_types', 'device_categories', 'device_models', 'device_mask',
                   'port_roles', 'named_port_roles', 'parameters', 'parameter_mask',
                   'parameter_applicability', 'op', 'op_mask', 'op_target_mask'}
    for key, value in batch.items():
        if key in graph_keys:
            shape = list(value.shape)
            shape[1] = shape[2] = n + extra
            item = value.new_zeros(shape)
            item[:, :n, :n] = value
        elif key in device_keys:
            shape = list(value.shape)
            shape[1] = n + extra
            item = value.new_zeros(shape)
            item[:, :n] = value
        else:
            item = value
        output[key] = item
    return output


class DeviceBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        cls.data = synthetic_dataset(10)
        cls.manifest = build_manifest(cls.data, 'synthetic_4')
        cls.prep = fit_preprocessor(cls.data, cls.manifest, True)
        cls.rows = cls.data['samples'][::9][:3]
        cls.batch = collate_samples(cls.rows, cls.prep, include_op=True, include_performance=True)
        cls.cfg = dict(dim=8, heads=2, encoder_layers=2, cross_layers=1,
                       ffn_dim=16, decoder_layers=1, decoder_ffn_dim=16,
                       dropout=0., max_devices=32, terminal_bias=True)

    def test_all_models_forward_backward_and_checkpoint_reload(self):
        for name in TORCH_BASELINES:
            with self.subTest(model=name):
                model = build_model(name, self.cfg, self.prep)
                pred = model(self.batch)
                self.assertEqual(pred.shape, (len(self.rows), len(self.prep['feature_spec']['metric_names'])))
                self.assertTrue(torch.isfinite(pred).all())
                pred.square().mean().backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                model.eval()
                rebuilt = build_model(name, self.cfg, self.prep).eval()
                rebuilt.load_state_dict(model.state_dict(), strict=True)
                with torch.no_grad():
                    torch.testing.assert_close(model(self.batch), rebuilt(self.batch), rtol=0., atol=0.)

    def test_measured_op_and_performance_are_not_model_inputs(self):
        poisoned = copy.deepcopy(self.batch)
        for key in ('op', 'performance'):
            poisoned[key] = torch.full_like(poisoned[key], float('nan'))
        for key in ('op_mask', 'op_target_mask', 'performance_mask'):
            poisoned[key] = ~poisoned[key]
        poisoned['simulation_status'] = torch.ones(1)
        for name in TORCH_BASELINES:
            with self.subTest(model=name):
                model = build_model(name, self.cfg, self.prep).eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(self.batch), model(poisoned), rtol=0., atol=0.)

    def test_design_parameters_have_a_trainable_prediction_path(self):
        for name in TORCH_BASELINES:
            with self.subTest(model=name):
                torch.manual_seed(29)
                model = build_model(name, self.cfg, self.prep).eval()
                batch = dict(self.batch)
                batch['parameters'] = batch['parameters'].detach().clone().requires_grad_(True)
                gradient = torch.autograd.grad(model(batch).square().sum(), batch['parameters'])[0]
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_padding_does_not_change_predictions(self):
        extended = padded(self.batch)
        for name in TORCH_BASELINES:
            with self.subTest(model=name):
                model = build_model(name, self.cfg, self.prep).eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(self.batch), model(extended), atol=2e-6, rtol=2e-6)

    def test_accformer_and_deepgen_padding_excluded_from_training_batchnorm(self):
        for name in ('deepgen_supervised', 'accformer'):
            with self.subTest(model=name):
                first = build_model(name, self.cfg, self.prep).train()
                second = copy.deepcopy(first).train()
                torch.testing.assert_close(first(self.batch), second(padded(self.batch)), atol=2e-5, rtol=2e-5)
                for key, value in first.state_dict().items():
                    if 'running_mean' in key or 'running_var' in key:
                        torch.testing.assert_close(value, second.state_dict()[key], atol=1e-7, rtol=1e-6)

    def test_single_device_single_sample_batchnorm_safe(self):
        row = copy.deepcopy(self.rows[0])
        row['devices'] = row['devices'][:1]
        batch = collate_samples([row], self.prep, include_op=True)
        for name in TORCH_BASELINES:
            with self.subTest(model=name):
                output = build_model(name, self.cfg, self.prep).train()(batch)
                self.assertTrue(torch.isfinite(output).all())

    def test_device_limit_is_enforced_without_truncation(self):
        cfg = {**self.cfg, 'max_devices': 1}
        for name in TORCH_BASELINES:
            with self.subTest(model=name):
                with self.assertRaises(ValueError):
                    build_model(name, cfg, self.prep)(self.batch)

    def test_deepgen_encoder_supports_full_op_pretraining_and_replacement(self):
        train = [r for r in self.data['samples'] if r['sample_id'] in self.manifest['train']]
        support = fit_op_support(train, self.prep)
        predictor = OPPredictor(build_encoder('deepgen_supervised', self.cfg, self.prep), self.prep, support)
        predicted, support_mask = predictor(self.batch)
        self.assertEqual(predicted.shape, self.batch['op'].shape)
        self.assertEqual(support_mask.shape, self.batch['op_mask'].shape)
        predicted.square().mean().backward()
        supervised = build_model('deepgen_op_pretrained', self.cfg, self.prep)
        supervised.encoder = predictor.encoder
        self.assertTrue(torch.isfinite(supervised(self.batch)).all())

    def test_gtn_resident_forward_does_not_discover_cycles_per_batch(self):
        store = ResidentDataset(self.rows, self.prep, 'cpu', verbose=False)
        model = build_model('gtn', self.cfg, self.prep)
        model.encoder.prepare_resident(store)
        with patch('final_exam.models.gtn.build_loop_membership', side_effect=AssertionError('unexpected CPU cycle discovery')):
            for _, batch in store.batches(2):
                self.assertIn('gtn_loop_membership_by_device', batch)
                self.assertTrue(torch.isfinite(model(batch)).all())

    def test_gtn_distinguishes_one_net_clique_from_a_real_loop(self):
        # One shared net among three devices makes a device clique, not a loop.
        clique = torch.ones(1, 3, 3, 1)
        live = torch.ones(1, 3, dtype=torch.bool)
        self.assertFalse(build_loop_membership(clique, live).any())
        # Three two-terminal devices on three distinct nets form a real loop.
        edges = torch.zeros(1, 3, 3, 4)
        terminals = ((0, 1), (1, 2), (2, 0))
        for i, first in enumerate(terminals):
            for j, second in enumerate(terminals):
                for a, net_a in enumerate(first):
                    for b, net_b in enumerate(second):
                        edges[0, i, j, a * 2 + b] = float(net_a == net_b)
        torch.testing.assert_close(build_loop_membership(edges, live), torch.ones(1, 1, 3, dtype=torch.bool))


if __name__ == '__main__':
    unittest.main()
