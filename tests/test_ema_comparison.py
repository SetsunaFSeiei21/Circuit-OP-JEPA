"""Reporting contracts for the separate-structure EMA JEPA experiment."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from device_jepa.comparison import compare_reports, main
from device_jepa.methods import DEFAULT_MODELS
from test_comparison import report


EMA_MODE = 'jepa_zerosim_ema'
EMA_ARCHITECTURE = 'separate-structure-ema_device-op-cross-attention_v1'


def sign_protocol(item):
    p = item['provenance']
    source = p['pretrain_provenance']
    digest = hashlib.sha256(json.dumps(source['protocol'], sort_keys=True,
                                      separators=(',', ':')).encode()).hexdigest()
    source['protocol_digest'] = p['pretraining_protocol_digest'] = digest


def ema_report(seed=17, split='test'):
    item = report('jepa_zerosim', seed, split)
    p = item['provenance']
    p.update(mode=EMA_MODE, backend=EMA_MODE, run_identity=f'{EMA_MODE}/{seed}')
    source = p['pretrain_provenance']
    source['source_options']['training']['ema_decay'] = .99
    source['protocol'].update(mode=EMA_MODE, architecture=EMA_ARCHITECTURE,
                             ema=dict(decay=.99, update_scope=['structure_encoder', 'fusion_blocks',
                                      'output_norm', 'output_projection', 'null_parameter'],
                                      update_timing='after_optimizer_step',
                                      gradient_policy='independent_input_embeddings_frozen_op_trunk'))
    sign_protocol(item)
    return item


class EMAComparisonTests(unittest.TestCase):
    def test_selected_ema_cohort_records_pretraining_costs_and_training_only_op(self):
        modes = ['zerosim_device', 'jepa_zerosim', EMA_MODE]
        reports = [factory(seed, split) for seed in (17, 29) for split in ('test', 'ood')
                   for factory in (lambda seed, split: report('zerosim_device', seed, split),
                                   lambda seed, split: report('jepa_zerosim', seed, split), ema_report)]
        result = compare_reports(reports, expected_modes=modes)
        self.assertTrue(result['complete'])
        self.assertEqual(result['total_epoch_budget'], 2)
        costs = [row for row in result['run_costs'] if row['mode'] == EMA_MODE]
        self.assertEqual(len(costs), 2)  # test/OOD share the same training run.
        for row in costs:
            self.assertEqual((row['pretrain_epochs'], row['supervised_epochs']), (1, 1))
            self.assertEqual(row['op_training_use'], 'pretraining_target')
            self.assertEqual(row['op_inference_use'], 'none')
            self.assertFalse(row['requires_op_at_inference'])
            self.assertIn('during JEPA pretraining only', row['op_availability_requirement'])
            self.assertEqual(row['evaluation_splits'], ['ood', 'test'])
        summary = [row for row in result['summary'] if row['mode'] == EMA_MODE]
        self.assertTrue(all(row['num_seeds'] == 2 and row['mae_std'] == 0 for row in summary))

    def test_ema_decoder_uses_its_own_epoch_budget(self):
        item = ema_report()
        item['provenance']['training_config']['decoder_epochs'] = 2
        with self.assertRaisesRegex(ValueError, 'Incompatible total training budget'):
            compare_reports([report('zerosim_device'), item], expected_modes=['zerosim_device', EMA_MODE])

    def test_old_seven_model_cohort_does_not_require_new_ema_model(self):
        modes = [*DEFAULT_MODELS, 'zerosim_pin_op_only', 'zerosim_device_op_only']
        result = compare_reports([report(mode) for mode in modes])
        self.assertEqual(result['expected_modes'], modes)
        self.assertTrue(result['complete'])

    def test_new_mode_requires_executed_ema_provenance(self):
        for section in ('pretrain_provenance', 'ema', 'mode', 'architecture'):
            with self.subTest(section=section):
                item = ema_report()
                p = item['provenance']
                if section == 'pretrain_provenance':
                    del p[section]
                else:
                    del p['pretrain_provenance']['protocol'][section]
                    sign_protocol(item)
                with self.assertRaisesRegex(ValueError, 'EMA'):
                    compare_reports([item], expected_modes=[EMA_MODE])

    def test_reject_ema_trunk_or_gradient_policy_changes_even_with_updated_digest(self):
        changes = {'decay': .98, 'update_scope': ['fusion_blocks'],
                   'update_timing': 'before_optimizer_step', 'gradient_policy': 'stop_gradient'}
        for field, value in changes.items():
            with self.subTest(field=field):
                item = ema_report()
                item['provenance']['pretrain_provenance']['protocol']['ema'][field] = value
                sign_protocol(item)
                with self.assertRaisesRegex(ValueError, 'EMA protocol'):
                    compare_reports([item], expected_modes=[EMA_MODE])

    def test_reject_changed_ema_decay_across_seeds(self):
        reports = [ema_report(17), ema_report(29)]
        source = reports[1]['provenance']['pretrain_provenance']
        source['source_options']['training']['ema_decay'] = .95
        source['protocol']['ema']['decay'] = .95
        sign_protocol(reports[1])
        with self.assertRaisesRegex(ValueError, 'pretraining protocol differs across seeds'):
            compare_reports(reports, expected_modes=[EMA_MODE])

    def test_invalid_or_nonfinite_ema_decay_is_rejected(self):
        for decay in (True, -1, 1, float('inf'), '0.99'):
            with self.subTest(decay=decay):
                item = ema_report()
                item['provenance']['pretrain_provenance']['source_options']['training']['ema_decay'] = decay
                with self.assertRaisesRegex(ValueError, 'EMA decay'):
                    compare_reports([item], expected_modes=[EMA_MODE])

    def test_ema_report_cannot_be_relabeled_as_legacy_jepa(self):
        item = ema_report()
        p = item['provenance']
        p.update(mode='jepa_zerosim', backend='jepa_zerosim')
        with self.assertRaisesRegex(ValueError, 'mode differs'):
            compare_reports([item], expected_modes=['jepa_zerosim'])
        del p['pretrain_provenance']['protocol']['mode']
        sign_protocol(item)
        with self.assertRaisesRegex(ValueError, 'cannot be reported as another mode'):
            compare_reports([item], expected_modes=['jepa_zerosim'])

    def test_report_backend_and_optional_metadata_must_match_ema_protocol(self):
        for field, value in (('backend', 'jepa_zerosim'), ('architecture', 'shared_structure'),
                             ('ema', {'decay': .99})):
            with self.subTest(field=field):
                item = ema_report()
                item['provenance'][field] = value
                with self.assertRaisesRegex(ValueError, 'EMA|architecture'):
                    compare_reports([item], expected_modes=[EMA_MODE])

    def test_decoder_config_does_not_replace_executed_ema_decay(self):
        item = ema_report()
        item['provenance']['training_config']['ema_decay'] = .5
        result = compare_reports([item], expected_modes=[EMA_MODE])
        self.assertTrue(result['complete'])

    def test_omitted_source_decay_uses_recorded_default(self):
        item = ema_report()
        del item['provenance']['pretrain_provenance']['source_options']['training']['ema_decay']
        self.assertTrue(compare_reports([item], expected_modes=[EMA_MODE])['complete'])

    def test_ema_digest_is_checked_against_executed_protocol(self):
        item = ema_report()
        source = item['provenance']['pretrain_provenance']
        source['source_options']['training']['ema_decay'] = .95
        source['protocol']['ema']['decay'] = .95
        with self.assertRaisesRegex(ValueError, 'digest does not match'):
            compare_reports([item], expected_modes=[EMA_MODE])

    def test_excel_and_csv_include_selected_ema_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'ema_report.json'
            source.write_text(json.dumps(ema_report()))
            output = root / 'comparison'
            main(['--reports', str(source), '--models', EMA_MODE, '--output', str(output)])
            result = json.loads((output / 'comparison.json').read_text())
            self.assertEqual(result['compared_modes'], [EMA_MODE])
            self.assertTrue(result['complete'])
            self.assertIn(EMA_MODE, (output / 'summary.csv').read_text())
            with ZipFile(output / 'model_comparison.xlsx') as workbook:
                xml = '\n'.join(workbook.read(name).decode() for name in workbook.namelist()
                                if name.endswith('.xml'))
                self.assertIn(EMA_MODE, xml)


if __name__ == '__main__':
    unittest.main()
