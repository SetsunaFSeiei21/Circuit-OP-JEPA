"""Independent regression cases for pretraining identity, selection and output safety."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from device_jepa.baselines import BASELINE_MODES, evaluate_baseline, train_baseline
from device_jepa.comparison import compare_reports
from device_jepa.data import build_manifest, dataset_fingerprint, synthetic_dataset
from device_jepa.states import apply_op_states
from device_jepa.training import evaluate, load_checkpoint, pretrain, train_decoder


def options():
    config = json.loads((Path(__file__).parents[1] / 'configs/smoke.json').read_text())
    config['model'].update(dim=8, heads=2, ffn_dim=16, decoder_ffn_dim=16,
                           decoder_layers=1, dropout=0.)
    config['training'].update(pretrain_epochs=1, decoder_epochs=1, torch_threads=1)
    return config


def snapshot(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file()}


class TrainingAuditFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.options = options()
        cls.data = apply_op_states(synthetic_dataset(6, 17), cls.options['op_state'])
        cls.manifest = build_manifest(cls.data, config=cls.options['dataset'])
        cls.first = pretrain(cls.data, cls.manifest, cls.options, cls.root / 'source')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_actual_pretraining_protocol_survives_stage_two_and_comparison_rejects_mixing(self):
        altered = copy.deepcopy(self.options)
        altered['training'].update(seed=29, pretrain_epochs=2, pretrain_lr=.0005)
        altered['loss']['invariance'] = 0.
        second = pretrain(self.data, self.manifest, altered, self.root / 'different_source')
        reports = []
        for seed, source in [(17, self.first), (29, second)]:
            decoder_config = copy.deepcopy(self.options)
            decoder_config['training']['seed'] = seed
            checkpoint = train_decoder(self.data, source, decoder_config, self.root / f'protocol_{seed}')
            cp = load_checkpoint(checkpoint)
            actual = load_checkpoint(source)
            self.assertEqual(cp['pretrain_provenance']['source_options'], actual['options'])
            self.assertEqual(cp['pretrain_seed'], actual['options']['training']['seed'])
            self.assertEqual(cp['decoder_seed'], seed)
            self.assertEqual(cp['pretrain_provenance']['encoder_digest'], actual['encoder_digest'])
            actual_pretrain_epochs = actual['options']['training']['pretrain_epochs']
            self.assertEqual(cp['training_cost']['pretrain_epochs'], actual_pretrain_epochs)
            self.assertEqual(cp['training_cost']['supervised_epochs'], 1)
            self.assertEqual(cp['training_cost']['total_epochs'], actual_pretrain_epochs + 1)
            reports.append(evaluate(self.data, checkpoint))
        self.assertNotEqual(reports[0]['provenance']['pretraining_protocol_digest'],
                            reports[1]['provenance']['pretraining_protocol_digest'])
        with self.assertRaisesRegex(ValueError, 'retrain.*protocol|retraining.*protocol'):
            compare_reports(reports, expected_modes=['jepa_zerosim'])

    def test_pretrain_and_decoder_seeds_are_separate_and_run_id_is_split_stable(self):
        config = copy.deepcopy(self.options)
        config['training']['seed'] = 29
        checkpoint = train_decoder(self.data, self.first, config, self.root / 'separate_seeds')
        test = evaluate(self.data, checkpoint, 'test')
        ood = evaluate(self.data, checkpoint, 'ood')
        self.assertEqual(test['provenance']['pretrain_seed'], 17)
        self.assertEqual(test['provenance']['decoder_seed'], 29)
        self.assertTrue(test['provenance']['run_identity'])
        self.assertEqual(test['provenance']['run_identity'], ood['provenance']['run_identity'])

    def test_omitted_optional_loss_fields_record_effective_defaults(self):
        implicit = copy.deepcopy(self.options)
        del implicit['loss']['invariance']
        del implicit['loss']['variance_floor']
        source = pretrain(self.data, self.manifest, implicit, self.root / 'implicit_loss_source')
        cp = load_checkpoint(source)
        self.assertEqual(cp['effective_loss_config'], self.options['loss'])
        self.assertEqual(cp['encoder_digest'], load_checkpoint(self.first)['encoder_digest'])
        self.assertEqual(cp['options']['loss'], implicit['loss'])
        reports = []
        for seed, path in ((17, self.first), (29, source)):
            config = copy.deepcopy(self.options)
            config['training']['seed'] = seed
            decoded = train_decoder(self.data, path, config, self.root / f'loss_defaults_{seed}')
            result = evaluate(self.data, decoded)
            actual_loss = result['provenance']['pretrain_provenance']['protocol']['loss']
            self.assertEqual(set(actual_loss), {'invariance', 'variance', 'covariance', 'variance_floor'})
            self.assertEqual(actual_loss, self.options['loss'])
            reports.append(result)
        self.assertEqual(reports[0]['provenance']['pretraining_protocol_digest'],
                         reports[1]['provenance']['pretraining_protocol_digest'])
        self.assertTrue(compare_reports(reports, expected_modes=['jepa_zerosim'])['complete'])

    def test_validation_only_metric_cannot_change_selection_in_any_mode(self):
        data = copy.deepcopy(self.data)
        for row in data['samples']:
            if row['sample_id'] in self.manifest['labeled_train']:
                row['performance']['gain'] = None
        manifest = build_manifest(data, config=self.options['dataset'])
        config = copy.deepcopy(self.options)
        config['training']['decoder_epochs'] = 3
        source = pretrain(data, manifest, config, self.root / 'unobserved_source')
        for mode in ('jepa_zerosim', *BASELINE_MODES):
            selected = []
            for value in (None, 60.):
                changed = copy.deepcopy(data)
                for row in changed['samples']:
                    if row['sample_id'] in manifest['val']:
                        row['performance']['gain'] = value
                dest = self.root / 'unobserved' / mode / str(value)
                if mode == 'jepa_zerosim':
                    path = train_decoder(changed, source, config, dest)
                    history_path = dest / 'decoder_history.json'
                else:
                    changed_manifest = copy.deepcopy(manifest)
                    changed_manifest['dataset_fingerprint'] = dataset_fingerprint(changed)
                    path = train_baseline(changed, changed_manifest, config, dest, mode)
                    history_path = dest / f'{mode}_history.json'
                cp = load_checkpoint(path)
                self.assertEqual(cp['selection_metric_names'], ['power'])
                self.assertEqual(cp['unobserved_training_metrics'], ['gain'])
                self.assertEqual(cp['preprocessing']['performance']['count'][0], 0)
                history = json.loads(history_path.read_text())
                selected.append((cp['epoch'], cp['decoder_digest'], history))
            with self.subTest(mode=mode):
                self.assertEqual(selected[0], selected[1])

    def test_no_trained_validation_targets_fails_without_best_checkpoint(self):
        config = copy.deepcopy(self.options)
        data = copy.deepcopy(self.data)
        for row in data['samples']:
            if row['sample_id'] in self.manifest['labeled_train']:
                row['performance']['gain'] = None
            if row['sample_id'] in self.manifest['val']:
                row['performance'].update(gain=60., power=None)
        manifest = build_manifest(data, config=config['dataset'])
        source = pretrain(data, manifest, config, self.root / 'no_val_source')
        for mode in ('jepa_zerosim', *BASELINE_MODES):
            out = self.root / 'no_val' / mode
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'trained-metric.*val'):
                if mode == 'jepa_zerosim':
                    train_decoder(data, source, config, out)
                else:
                    train_baseline(data, manifest, config, out, mode)
            self.assertFalse(list(out.glob('*best.pt')))

    def test_completed_stage_outputs_cannot_be_overwritten(self):
        before = snapshot(self.root / 'source')
        changed = copy.deepcopy(self.options)
        changed['training']['seed'] = 29
        with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
            pretrain(self.data, self.manifest, changed, self.root / 'source')
        self.assertEqual(before, snapshot(self.root / 'source'))
        for mode in BASELINE_MODES:
            out = self.root / 'baseline_overwrite' / mode
            train_baseline(self.data, self.manifest, self.options, out, mode)
            before = snapshot(out)
            with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
                train_baseline(self.data, self.manifest, changed, out, mode)
            self.assertEqual(before, snapshot(out))

    def test_decoder_rejects_unrelated_manifest_or_stale_report_without_writes(self):
        for filename, payload, message in [
            ('manifest.json', {'unrelated': True}, 'different experiment manifest'),
            ('test_report.json', {'stale': True}, 'stale artifacts'),
        ]:
            out = self.root / 'stale' / filename
            out.mkdir(parents=True)
            (out / filename).write_text(json.dumps(payload))
            before = snapshot(out)
            with self.assertRaisesRegex(ValueError, message):
                train_decoder(self.data, self.first, self.options, out)
            self.assertEqual(before, snapshot(out))

    def test_valid_same_directory_stage_two_is_allowed_once(self):
        out = self.root / 'coexist'
        source = pretrain(self.data, self.manifest, self.options, out)
        checkpoint = train_decoder(self.data, source, self.options, out)
        self.assertTrue(Path(checkpoint).exists())
        before = snapshot(out)
        with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
            train_decoder(self.data, source, self.options, out)
        self.assertEqual(before, snapshot(out))

    def test_checkpoint_identity_rejects_corrupted_source_and_decoder_weights(self):
        corrupted = load_checkpoint(self.first)
        first_tensor = next(value for value in corrupted['encoder_state'].values()
                            if value.is_floating_point())
        first_tensor.add_(1.)
        source_path = self.root / 'corrupt_source.pt'
        torch.save(corrupted, source_path)
        with self.assertRaisesRegex(AssertionError, 'Encoder checkpoint integrity'):
            train_decoder(self.data, source_path, self.options, self.root / 'corrupt_source_output')
        path = train_decoder(self.data, self.first, self.options, self.root / 'integrity_decoder')
        corrupted = load_checkpoint(path)
        first_tensor = next(value for value in corrupted['decoder_state'].values()
                            if value.is_floating_point())
        first_tensor.add_(1.)
        decoder_path = self.root / 'corrupt_decoder.pt'
        torch.save(corrupted, decoder_path)
        with self.assertRaisesRegex(AssertionError, 'Decoder integrity'):
            evaluate(self.data, decoder_path)
        corrupted = load_checkpoint(path)
        corrupted['pretrain_provenance']['protocol']['loss']['invariance'] += 1.
        protocol_path = self.root / 'corrupt_protocol.pt'
        torch.save(corrupted, protocol_path)
        with self.assertRaisesRegex(ValueError, 'pretraining provenance integrity'):
            load_checkpoint(protocol_path)


if __name__ == '__main__':
    unittest.main()
