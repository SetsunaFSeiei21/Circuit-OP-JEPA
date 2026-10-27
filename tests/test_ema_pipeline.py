import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from device_jepa.data import build_manifest, synthetic_dataset
from device_jepa.ema_model import EMAJEPAModel, EMA_MODE, EMA_ARCHITECTURE
from device_jepa.states import apply_op_states
from device_jepa.model import alignment_loss
from device_jepa.training import encoder_from_metadata, evaluate, load_checkpoint, pretrain, train_decoder


class EMAPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.options = json.loads((Path(__file__).parents[1] / 'configs/smoke.json').read_text())
        cls.options['training'].update(pretrain_epochs=2, decoder_epochs=1, batch_size=64,
                                      ema_decay=0.9,
                                      early_stopping={'enabled': True, 'min_delta': 1e12,
                                                      'min_epochs': 1, 'patience': 10})
        cls.options['model'].update(dim=16, heads=4, ffn_dim=32, decoder_ffn_dim=32, dropout=0.0)
        cls.data = apply_op_states(synthetic_dataset(6, 17), cls.options['op_state'])
        cls.manifest = build_manifest(cls.data, config=cls.options['dataset'])
        cls.temp = tempfile.TemporaryDirectory()
        val_calls = 0

        def controlled_validation(*args, **kwargs):
            # Exercise a best checkpoint from BEFORE the last optimizer step.
            # Keep the real train objective/backprop; make the second validation worse.
            nonlocal val_calls
            loss, detail = alignment_loss(*args, **kwargs)
            if not torch.is_grad_enabled():
                loss = loss + val_calls * 1e6
                val_calls += 1
            return loss, detail

        with patch('device_jepa.training.alignment_loss', controlled_validation), \
                patch('device_jepa.training._effective_alignment_loss', return_value=dict(cls.options['loss'])):
            cls.p = pretrain(cls.data, cls.manifest, cls.options, cls.temp.name, mode=EMA_MODE)
        cls.decoder_options = copy.deepcopy(cls.options)
        # The source checkpoint, not this new wish, defines the executed EMA.
        cls.decoder_options['training']['ema_decay'] = 0.5
        cls.d = train_decoder(cls.data, cls.p, cls.decoder_options, cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_selected_snapshot_and_actual_cost_are_distinct(self):
        source, decoder = load_checkpoint(self.p), load_checkpoint(self.d)
        self.assertEqual(source['epoch'], 1)
        self.assertEqual(source['ema_updates'], 1)
        self.assertEqual(source['training_cost']['pretrain_optimizer_steps'], 2)
        self.assertEqual(source['training_cost']['pretrain_ema_updates'], 2)
        self.assertEqual(decoder['ema_updates'], 1)
        self.assertEqual(decoder['training_cost']['pretrain_ema_updates'], 2)
        self.assertEqual(source['mode'], EMA_MODE)
        self.assertEqual(decoder['backend'], EMA_MODE)
        self.assertEqual(decoder['architecture'], EMA_ARCHITECTURE)
        self.assertEqual(decoder['shared_structure_layers'], 0)
        self.assertEqual(source['encoder_digest'], decoder['encoder_digest'])
        self.assertTrue(decoder['freeze_verified'])
        for key in ('pretrain_total', 'pretrain_trainable', 'pretrain_ema'):
            self.assertEqual(source['parameter_count'][key], decoder['parameter_count'][key])

    def test_decoder_preserves_executed_ema_protocol(self):
        decoder = load_checkpoint(self.d)
        self.assertEqual(decoder['options']['training']['ema_decay'], 0.5)
        self.assertEqual(decoder['ema']['decay'], 0.9)
        source = decoder['pretrain_provenance']
        self.assertEqual(source['protocol']['mode'], EMA_MODE)
        self.assertEqual(source['protocol']['ema'], decoder['ema'])
        self.assertEqual(source['source_options']['training']['ema_decay'], 0.9)
        self.assertEqual(source['selected_ema_updates'], 1)

    def test_full_pretrain_state_reconstructs_both_branches(self):
        source = load_checkpoint(self.p)
        prep = source['preprocessing']
        model = EMAJEPAModel(encoder_from_metadata(source['model_config'], prep),
                              len(prep['feature_spec']['op_names']),
                              len(prep['op_state_features']), ema_decay=source['ema']['decay'])
        model.load_state_dict(source['model_state'], strict=True)
        model.ema_updates = source['ema_updates']
        model.verify_ema_state()
        self.assertTrue(any(not torch.equal(a, b) for _, a, b in model.ema_parameter_pairs()))
        self.assertGreater(source['parameter_count']['pretrain_ema'], 0)
        self.assertEqual(source['parameter_count']['pretrain_total'],
                         source['parameter_count']['pretrain_ema'] + source['parameter_count']['pretrain_trainable'])

    def test_test_and_ood_reports_are_distinct_and_op_free_at_inference(self):
        from device_jepa.comparison import compare_reports
        reports = []
        for split, size in (('test', 4), ('ood', 6)):
            report = evaluate(self.data, self.d, split=split)
            reports.append(report)
            self.assertEqual(report['mode'], EMA_MODE)
            self.assertEqual(report['num_samples'], size)
            self.assertFalse(report['requires_op_at_inference'])
            self.assertFalse(report['selection_used_ood'])
            self.assertEqual(report['op_information']['training_use'], 'pretraining_target')
            self.assertEqual(report['op_information']['inference_use'], 'none')
            self.assertEqual(report['provenance']['ema']['decay'], 0.9)
            self.assertGreater(report['provenance']['parameter_count']['pretrain_ema'], 0)
        # The real generated report must satisfy the strict comparison contract.
        comparison = compare_reports(reports, expected_modes=[EMA_MODE])
        self.assertTrue(comparison['complete'])
        self.assertGreater(comparison['run_costs'][0]['parameter_count']['pretrain_ema'], 0)

    def test_identity_state_protocol_and_cost_tampering_are_rejected(self):
        cases = (
            ('mode', lambda c: c.update(mode='jepa_zerosim')),
            ('architecture', lambda c: c.update(architecture='shared-structure-prefix_direct-device-op-alignment_v2')),
            ('decay', lambda c: c['ema'].update(decay=0.1)),
            ('count', lambda c: c.update(ema_updates=1000)),
            ('state', lambda c: c['model_state']['op_encoder.op_names.weight'].add_(1)),
            ('export', lambda c: c['encoder_state'].pop(next(iter(c['encoder_state'])))),
        )
        for name, mutate in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as folder:
                checkpoint = load_checkpoint(self.p)
                mutate(checkpoint)
                path = Path(folder, 'bad.pt')
                torch.save(checkpoint, path)
                with self.assertRaisesRegex(ValueError, 'EMA'):
                    load_checkpoint(path)

    def test_invalid_decay_and_mode_fail_before_output_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            for decay in (True, -0.1, 1.0, float('nan'), float('inf')):
                options = copy.deepcopy(self.options)
                options['training']['ema_decay'] = decay
                output = Path(folder, 'new')
                with self.assertRaisesRegex(ValueError, 'ema_decay'):
                    pretrain(self.data, self.manifest, options, output, mode=EMA_MODE)
                self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, 'JEPA model'):
                pretrain(self.data, self.manifest, self.options, Path(folder, 'new'), mode='zerosim_device')


if __name__ == '__main__':
    unittest.main()
