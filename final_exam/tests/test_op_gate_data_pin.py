"""Physical-device input identity and train-only OP gate control fixtures."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from device_jepa.data import build_manifest, synthetic_dataset
from device_jepa.resident import ResidentDataset
from final_exam.data_pin import PinDataContext
from final_exam.op_gate_data_pin import (
    ShuffleBankPin, build_shuffle_bank_pin, describe_shuffle_bank_pin,
    device_input_pin, prepare_device_predictions_pin, rebuild_shuffle_bank_pin,
)


class _DevicePredictor(nn.Module):
    def __init__(self, fields):
        super().__init__()
        self.scale = nn.Parameter(torch.arange(1., fields + 1.))
        self.calls = 0

    def predict_devices(self, batch):
        self.calls += 1
        inputs = device_input_pin(batch)
        base = inputs['parameters'].sum(-1, keepdim=True)
        values = base * self.scale
        supported = inputs['device_mask'].unsqueeze(-1).expand_as(values)
        return values, supported


class OPGateDataPinTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = synthetic_dataset(24)
        self.data['samples'] = [row for row in self.data['samples']
                                if row['circuit_id'] == 'synthetic_0']
        for row in self.data['samples']:
            row['devices'] = row['devices'][:4]
        self.config = dict(model=dict(max_devices=32), normalization={}, dataset=dict(
            train_circuits=['synthetic_0'], ood_circuits=[], split_ratios=[.6, .2, .2],
            split_seed=17, finetune_label_fraction=1.))
        self.manifest = build_manifest(self.data, config=self.config['dataset'])
        with contextlib.redirect_stdout(io.StringIO()):
            self.ctx = PinDataContext(self.data, self.manifest, self.config, 'cpu', {}, self.root)
        self.predictor = _DevicePredictor(len(self.ctx.prep['feature_spec']['op_names']))
        self.predictor.requires_grad_(False).eval()

    def batch(self, split='train', size=3, seed=None, op=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return next(self.ctx.batches(split, size, seed=seed, op=op))

    def test_device_inputs_match_canonical_graph_and_numeric_tables(self):
        batch = self.batch(op=True)
        device = device_input_pin(batch)
        by_id = {row['sample_id']: row for row in self.ctx.rows['train']}
        rows = [by_id[sid] for sid in batch['_sample_ids']]
        with contextlib.redirect_stdout(io.StringIO()):
            resident = ResidentDataset(rows, self.ctx.device_preprocessing, 'cpu')
            _, expected = next(resident.batches(len(rows)))
        for name, values in device.items():
            torch.testing.assert_close(values, expected[name], msg=name)
        self.assertEqual(device['device_mask'].shape[1], 4)
        self.assertEqual(batch['device_mask'].shape[1], 16)
        self.assertNotIn('op', device)
        self.assertNotIn('performance', device)
        # Corrupting the expanded pin input cannot alter the owner input.
        corrupted = copy.deepcopy(batch)
        corrupted['parameters'].fill_(float('nan'))
        corrupted['op'].fill_(float('nan'))
        corrupted['owner_op'].fill_(float('nan'))
        for name, values in device_input_pin(corrupted).items():
            torch.testing.assert_close(values, device[name], atol=0, rtol=0)

    def test_invalid_owner_axis_and_missing_original_fields_rejected(self):
        batch = self.batch()
        malformed = dict(batch, owner_adjacency=batch['adjacency'])
        with self.assertRaisesRegex(ValueError, 'device axis'):
            device_input_pin(malformed)
        malformed = dict(batch)
        malformed.pop('owner_parameters')
        with self.assertRaisesRegex(ValueError, 'missing'):
            device_input_pin(malformed)
        malformed = dict(batch, owner_parameter_mask=batch['owner_parameter_mask'].float())
        with self.assertRaisesRegex(ValueError, 'parameter_mask'):
            device_input_pin(malformed)

    def test_cache_tracks_sample_order_and_predictor_state(self):
        with contextlib.redirect_stdout(io.StringIO()):
            digest = prepare_device_predictions_pin(self.ctx, self.predictor, {'batch_size': 3})
        calls = self.predictor.calls
        with contextlib.redirect_stdout(io.StringIO()):
            again = prepare_device_predictions_pin(self.ctx, self.predictor, {'batch_size': 8})
        self.assertEqual(digest, again)
        self.assertEqual(calls, self.predictor.calls)
        for split in ('train', 'val', 'test'):
            batch = self.batch(split, size=2, seed=47)
            expected, mask = self.predictor.predict_devices(batch)
            torch.testing.assert_close(batch['_predicted_device_op'], expected)
            torch.testing.assert_close(batch['_predicted_device_op_mask'], mask)
            self.assertEqual(batch['_predicted_device_op_digest'], digest)
        with torch.no_grad():
            self.predictor.scale.add_(1.)
        with contextlib.redirect_stdout(io.StringIO()):
            updated = prepare_device_predictions_pin(self.ctx, self.predictor,
                                                     {'batch_size': 3}, splits=('train',))
        self.assertNotEqual(digest, updated)
        self.assertEqual(set(self.ctx.device_predictions_pin), {'train'})
        self.assertNotIn('_predicted_device_op', self.batch('val'))
        batch = self.batch()
        torch.testing.assert_close(batch['_predicted_device_op'], self.predictor.predict_devices(batch)[0])

    def test_only_frozen_eval_predictor_can_be_cached(self):
        self.predictor.train()
        with self.assertRaisesRegex(ValueError, 'frozen eval'):
            prepare_device_predictions_pin(self.ctx, self.predictor, {'batch_size': 3})
        self.predictor.eval().requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'frozen eval'):
            prepare_device_predictions_pin(self.ctx, self.predictor, {'batch_size': 3})

    def test_shuffle_bank_uses_train_only_and_excludes_training_self(self):
        with contextlib.redirect_stdout(io.StringIO()):
            bank = build_shuffle_bank_pin(self.ctx, self.predictor, 3, seed=17)
        self.assertEqual(bank.sample_ids, [row['sample_id'] for row in self.ctx.rows['train']])
        self.assertTrue(set(bank.sample_ids).isdisjoint(self.manifest['test']))
        self.assertTrue(set(bank.sample_ids).isdisjoint(self.manifest['val']))
        indices = bank.donor_indices(bank.sample_ids)
        self.assertTrue(all(donor != own for own, donor in enumerate(indices)))
        self.assertEqual(set(self.ctx.device_predictions_pin), {'train'})
        # Whole device/field arrays are swapped, never independently shuffled.
        for split in ('train', 'val', 'test'):
            batch = self.batch(split, size=3)
            selected = bank.donor_indices(batch['_sample_ids'])
            values, mask = bank(batch)
            torch.testing.assert_close(values, bank.values[selected])
            torch.testing.assert_close(mask, bank.support[selected])

    def test_shuffle_mapping_independent_of_batch_order_size_and_checkpoint(self):
        with contextlib.redirect_stdout(io.StringIO()):
            bank = build_shuffle_bank_pin(self.ctx, self.predictor, 3, seed=29)
        expected = {}
        with contextlib.redirect_stdout(io.StringIO()):
            for batch in self.ctx.batches('test', 2):
                values, support = bank(batch)
                expected.update({sid: (values[i], support[i])
                                 for i, sid in enumerate(batch['_sample_ids'])})
        restored = rebuild_shuffle_bank_pin(describe_shuffle_bank_pin(bank))
        restored.load_state_dict(bank.state_dict(), strict=True)
        with contextlib.redirect_stdout(io.StringIO()):
            for batch in self.ctx.batches('test', 1, seed=123):
                values, support = restored(batch)
                original = expected[batch['_sample_ids'][0]]
                torch.testing.assert_close(values[0], original[0], atol=0, rtol=0)
                torch.testing.assert_close(support[0], original[1])
        bad = self.batch('test')
        bad['_owner_topology_pin'] = ['another circuit'] * len(bad['_sample_ids'])
        with self.assertRaisesRegex(ValueError, 'topology'):
            restored(bad)

    def test_shuffle_invalid_bank_and_anonymous_query_fail(self):
        with self.assertRaisesRegex(ValueError, 'at least two'):
            ShuffleBankPin(torch.ones(1, 2, 3), torch.ones(1, 2, 3, dtype=torch.bool), ['one'], 17)
        values = torch.arange(24.).reshape(2, 4, 3)
        bank = ShuffleBankPin(values, torch.ones_like(values, dtype=torch.bool), ['a', 'b'], 17)
        self.assertEqual(bank.donor_indices(['a', 'b']), [1, 0])
        batch = dict(owner_mask=torch.ones(1, 4, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, 'stable sample IDs'):
            bank(batch)


if __name__ == '__main__':
    unittest.main()
