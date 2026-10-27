"""Device pipeline must preserve the original masks, physics and graph inputs."""
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from device_jepa.baselines import _collate, fit_pin_vocabulary, train_baseline
from device_jepa.data import (_statistics, build_manifest, fit_preprocessor,
                              synthetic_dataset, transform_value)
from device_jepa.model import FrozenPerformanceModel, ModelConfig, make_decoder
from device_jepa.methods import JEPA_MODES
from device_jepa.resident import (BatchProgress, ResidentDataset, _reserve,
                                  device_statistics, normalize, require_device, _prefix_sum)
from device_jepa.states import apply_op_states
from device_jepa.tensors import collate_samples, denormalize, _device_denormalize
from device_jepa.training import encoder_from_metadata, seed_everything
from test_input_identity import identity_dataset


class ResidentParityTests(unittest.TestCase):
    def setUp(self):
        seed_everything(17)
        self.data = apply_op_states(synthetic_dataset(8))
        self.manifest = build_manifest(self.data, 'synthetic_4')
        self.prep = fit_preprocessor(self.data, self.manifest, True)
        self.rows = self.data['samples'][::3]
        self.pins = fit_pin_vocabulary(self.data, self.manifest)

    def assert_batch(self, actual, expected):
        for name, value in expected.items():
            if isinstance(value, torch.Tensor):
                self.assertEqual(actual[name].dtype, value.dtype, name)
                torch.testing.assert_close(actual[name].cpu(), value, rtol=1e-6, atol=1e-7,
                                           msg=lambda text: name + ': ' + text)
            else:
                self.assertEqual(actual[name], value, name)

    def check_modes(self, device):
        for mode in ('zerosim_device', 'zerosim_pin', 'zerosim_device_op', 'zerosim_pin_op'):
            with self.subTest(mode=mode, device=device):
                cache = ResidentDataset(self.rows, self.prep, device,
                    include_op=mode.endswith('_op'), include_performance=True,
                    pin_vocab=self.pins if 'pin' in mode else None, max_tokens=200, verbose=False)
                order = list(range(len(self.rows)))
                random.Random(29).shuffle(order)
                for step, (_, batch) in enumerate(cache.batches(3, seed=29, metadata=True)):
                    rows = [self.rows[i] for i in order[step*3:step*3+3]]
                    expected = _collate(rows, self.prep, mode, {'max_devices': 200},
                                        self.pins if 'pin' in mode else None)
                    self.assert_batch(batch, expected)
                    self.assertTrue(all(v.device.type == device for v in batch.values()
                                        if isinstance(v, torch.Tensor)))

    def test_all_modes_and_variable_sizes_match_reference(self):
        self.check_modes('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware required')
    def test_all_numeric_cache_and_batches_stay_on_cuda(self):
        self.check_modes('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware required')
    def test_all_registered_modes_cuda_training_evaluation_and_frozen_cache(self):
        from device_jepa.baselines import evaluate_baseline
        from device_jepa.training import pretrain, train_decoder, evaluate, load_checkpoint
        from device_jepa.comparison import compare_reports, MODES
        config=json.loads((Path(__file__).parents[1]/'configs/smoke.json').read_text())
        config['model'].update(dim=8,heads=2,ffn_dim=16,decoder_ffn_dim=16,dropout=0.)
        config['training'].update(pretrain_epochs=1,decoder_epochs=1,baseline_epochs=2)
        data=apply_op_states(synthetic_dataset(6),config['op_state'])
        manifest=build_manifest(data,config=config['dataset'])
        reports=[]
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            for mode in MODES:
                dest=root/mode
                if mode in JEPA_MODES:
                    first=pretrain(data,manifest,config,dest,device='cuda',mode=mode)
                    checkpoint=train_decoder(data,first,config,dest,device='cuda')
                    self.assertGreater(load_checkpoint(checkpoint)['training_cost']['frozen_encoder_cache_bytes'],0)
                    reports.append(evaluate(data,checkpoint,device='cuda'))
                else:
                    checkpoint=train_baseline(data,manifest,config,dest,mode,device='cuda')
                    reports.append(evaluate_baseline(data,checkpoint,device='cuda'))
                self.assertTrue(load_checkpoint(checkpoint)['training_cost']['resident_device'].startswith('cuda'))
        self.assertTrue(compare_reports(reports)['complete'])

    def test_pretraining_state_channels_and_all_input_contracts(self):
        self.prep['op_state_features'] = list(next(d['op_state'] for r in self.rows for d in r['devices']
                                                   if d.get('op_state')).keys())
        for contract in (1, 2, 3):
            prep = copy.deepcopy(self.prep)
            prep['input_contract_version'] = contract
            if contract < 3:
                prep.pop('model_vocab')
                prep.pop('port_role_vocab')
            cache = ResidentDataset(self.rows, prep, 'cpu', include_op=True,
                                     include_op_states=True, verbose=False)
            batch = next(cache.batches(4, metadata=True))[1]
            self.assert_batch(batch, collate_samples(self.rows[:4], prep, include_op=True,
                                                     include_op_states=True))
            pins = fit_pin_vocabulary(self.data, self.manifest, contract)
            cache = ResidentDataset(self.rows, prep, 'cpu', include_op=True, include_performance=True,
                                     pin_vocab=pins, verbose=False)
            self.assert_batch(next(cache.batches(4, metadata=True))[1],
                              _collate(self.rows[:4], prep, 'zerosim_pin_op', {'max_devices': 200}, pins))

    def test_unseen_model_and_named_ports_match_reference_in_both_graphs(self):
        data = identity_dataset()
        manifest = build_manifest(data, 'B')
        prep = fit_preprocessor(data, manifest, True)
        pins = fit_pin_vocabulary(data, manifest)
        rows = [data['samples'][0], data['samples'][-1]]
        for mode in ('zerosim_pin_op', 'zerosim_device_op'):
            cache = ResidentDataset(rows, prep, 'cpu', include_op=True, include_performance=True,
                pin_vocab=pins if 'pin' in mode else None, verbose=False)
            self.assert_batch(next(cache.batches(2, metadata=True))[1],
                              _collate(rows, prep, mode, {'max_devices': 200},
                                       pins if 'pin' in mode else None))

    def test_sizing_points_share_graph_without_losing_identity_changes(self):
        a = copy.deepcopy(self.rows[0])
        b = copy.deepcopy(a)
        b['sample_id'] += '_b'
        b['devices'][0]['parameters']['W'] = 9e-6
        b['devices'][0]['op']['gm'] = 5.
        c = copy.deepcopy(b)
        c['devices'][0]['model_name'] = 'different_physical_model'
        cache = ResidentDataset([a,b,c], self.prep, 'cpu', include_op=True, verbose=False)
        self.assertEqual(cache.host_graph_ids, [0,0,1])
        batch = next(cache.batches(3))[1]
        self.assertFalse(torch.equal(batch['parameters'][0], batch['parameters'][1]))
        self.assertFalse(torch.equal(batch['device_models'][1], batch['device_models'][2]))

    def test_inactive_extreme_label_skips_transform_and_conversion(self):
        row = copy.deepcopy(self.rows[0])
        row['performance']['gain'] = 1e100
        cache = ResidentDataset([row], self.prep, 'cpu', include_performance=True,
            performance_metric_mask=[False,True], verbose=False)
        batch = next(cache.batches(1))[1]
        self.assertFalse(batch['performance_mask'][0,0])
        self.assertEqual(float(batch['performance'][0,0]), 0.)
        with self.assertRaisesRegex(ValueError, 'standardized feature gain'):
            ResidentDataset([row], self.prep, 'cpu', include_performance=True, verbose=False)

    def test_batch_iteration_never_calls_scalar_collators(self):
        cache = ResidentDataset(self.rows, self.prep, 'cpu', include_op=True,
                                 pin_vocab=self.pins, verbose=False)
        with patch('device_jepa.tensors.collate_samples', side_effect=AssertionError('CPU collator')), \
             patch('device_jepa.baselines.collate_pin_samples', side_effect=AssertionError('CPU pin collator')):
            for _ in range(3):
                self.assertEqual(sum(batch['device_mask'].shape[0] for _,batch in cache.batches(3)), len(self.rows))

    def test_only_frozen_features_are_cached_and_reused_across_epochs(self):
        cfg = ModelConfig(dim=8, heads=2, encoder_layers=2, cross_layers=1, ffn_dim=16,
                          decoder_ffn_dim=16, decoder_layers=1, dropout=.2).to_dict()
        encoder = encoder_from_metadata(cfg, self.prep)
        cache = ResidentDataset(self.rows, self.prep, 'cpu', verbose=False)
        with self.assertRaisesRegex(ValueError, 'frozen eval encoder'):
            cache.cache_encoder(encoder, 3)
        model = FrozenPerformanceModel(encoder, make_decoder(encoder.cfg, 2)).eval()
        expected = []
        with torch.no_grad():
            for _, batch in cache.batches(3):
                expected.append(model(batch))
        cache.cache_encoder(encoder, 3)
        with patch.object(encoder, 'forward', side_effect=AssertionError('Repeated frozen encoder')):
            for _ in range(3):
                actual = [model.decoder(batch['encoder_memory'],batch['encoder_padding'])
                          for _,batch in cache.batches(3)]
                torch.testing.assert_close(torch.cat(actual),torch.cat(expected),atol=1e-6,rtol=1e-6)

    def test_vectorized_own_node_mask_matches_original_node_loop(self):
        cfg = ModelConfig(dim=8, heads=2, encoder_layers=2, cross_layers=1,
                          ffn_dim=16, decoder_ffn_dim=16, dropout=0.).to_dict()
        encoder = encoder_from_metadata(cfg, self.prep).eval()
        batch = collate_samples(self.rows[:3], self.prep, include_op=False)
        batch['parameter_mask'][0].zero_()
        batch['parameter_applicability'][0].zero_()
        _, live = encoder.input_tokens(batch)
        b,n,features=live.shape
        expected=torch.zeros(b,n+1,n*features+1,dtype=torch.bool)
        expected[:,0,:-1]=live.flatten(1)
        for i in range(n):
            expected[:,i+1,i*features:(i+1)*features]=live[:,i]
        expected[:,:,-1]=~expected[:,:,:-1].any(-1)
        with patch.object(encoder.structure_encoder,'attention_mask',
                          wraps=encoder.structure_encoder.attention_mask) as masks:
            encoder(batch)
        self.assertTrue(torch.equal(masks.call_args_list[-1].args[0],expected))


class DeviceNormalizationTests(unittest.TestCase):
    def test_device_inverse_preserves_tiny_scales_large_offsets_and_log_range(self):
        cases = [('asinh',1e-250,0.,1.,1000.), ('asinh',1e-20,0.,1.,100.),
                 ('asinh',1e-250,0.,1.,1.), ('linear',1.,1e9,1.,-.125)]
        for kind, scale, mean, std, value in cases:
            stats=dict(kind=[kind],asinh_scale=[scale],mean=[mean],std=[std])
            raw=torch.tensor([[value]],dtype=torch.float64)
            torch.testing.assert_close(_device_denormalize(raw,stats),denormalize(raw,stats),rtol=1e-14,atol=0.)
        with self.assertRaisesRegex(ValueError,'finite physical range'):
            _device_denormalize(torch.tensor([[1000.]]),dict(kind=['asinh'],asinh_scale=[1.],mean=[0.],std=[1.]))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware required')
    def test_cuda_inverse_and_train_only_statistics(self):
        seed_everything(17)
        stats=dict(kind=['asinh'],asinh_scale=[1e-250],mean=[0.],std=[1.])
        raw=torch.tensor([[1000.]],device='cuda')
        torch.testing.assert_close(denormalize(raw,stats).cpu(),denormalize(raw.cpu(),stats),rtol=1e-14,atol=0.)
        rows=[('A',{'x':v}) for v in [0.,-1e-20,1e-10,3e-10]]+[('B',{'x':7e-10})]*3
        rules={'x':{'kind':'asinh','scale':'fit'}}
        actual=device_statistics(rows,['x'],rules,'equal_circuit','cuda')
        expected=_statistics(rows,['x'],rules,'equal_circuit')
        for key in ('mean','std','asinh_scale'):
            self.assertTrue(math.isclose(actual[key][0],expected[key][0],rel_tol=1e-12,abs_tol=1e-30))

    def test_deterministic_prefix_tree_matches_cumulative_weights(self):
        for count in (1,2,3,7,32,101):
            weights=torch.arange(1,count+1,dtype=torch.float64)/103.
            torch.testing.assert_close(_prefix_sum(weights),weights.cumsum(0),atol=1e-13,rtol=1e-13)

    def test_comparison_rejects_mixing_scalar_and_resident_runs(self):
        from device_jepa.comparison import compare_reports, MODES
        from test_comparison import report
        from test_ema_comparison import EMA_MODE, ema_report
        reports=[ema_report() if mode == EMA_MODE else report(mode) for mode in MODES]
        reports[0]['provenance']['tensor_pipeline']='device_resident_v1'
        with self.assertRaisesRegex(ValueError,'Tensor pipeline differs'):
            compare_reports(reports)

    def test_float64_physics_and_overflow_safe_asinh(self):
        stats = {'kind':['linear','asinh'], 'mean':[1e9,0.], 'std':[1.,1.],
                 'asinh_scale':[1.,1e-300]}
        values = [[1e9+.125,1e300], [1e9-.125,-1e300]]
        raw = torch.tensor(values,dtype=torch.float64)
        actual = normalize(raw,torch.ones_like(raw,dtype=torch.bool),stats,['gain','current'],torch.float64)
        expected = torch.tensor([[transform_value(v,stats,j) for j,v in enumerate(row)]
                                 for row in values],dtype=torch.float64)
        torch.testing.assert_close(actual,expected,atol=0.,rtol=1e-14)

    def test_circuit_balanced_device_statistics_match_scalar_reference(self):
        rows = [('A',{'x':v,'y':-v*1e-10}) for v in [0.,1.,2.,None] if v is not None]
        rows += [('B',{'x':5.,'y':None})]*7
        rules = {'x':{'kind':'linear','scale':1.},'y':{'kind':'asinh','scale':'fit'}}
        for weighting in ('equal_circuit','observation'):
            expected = _statistics(rows,['x','y'],rules,weighting)
            actual = device_statistics(rows,['x','y'],rules,weighting,'cpu')
            for key in ('mean','std','asinh_scale'):
                for a,b in zip(actual[key],expected[key]):
                    self.assertTrue(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-15),(key,a,b))
            for key in ('count','kind','circuit_count','version','weighting'):
                self.assertEqual(actual[key],expected[key])

    def test_cuda_missing_or_memory_shortage_never_falls_back_to_cpu(self):
        with patch.object(torch.cuda,'is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'refusing CPU fallback'):
                require_device('cuda')
        with patch.object(torch.cuda,'mem_get_info',return_value=(100,200)):
            with self.assertRaisesRegex(RuntimeError,'Refusing CPU fallback'):
                _reserve(torch.device('cuda'),80)

    def test_progress_reports_first_interval_last_and_eta(self):
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):
            progress=BatchProgress('epoch 1/2 train',5,every=2)
            for step in range(1,6):
                progress.update(step,torch.tensor(1.))
        text=stream.getvalue()
        self.assertIn('0/5 batches',text)
        self.assertIn('1/5 (20.0%)',text)
        self.assertIn('2/5 (40.0%)',text)
        self.assertIn('4/5 (80.0%)',text)
        self.assertIn('5/5 (100.0%)',text)
        self.assertNotIn('3/5',text)
        self.assertIn('ETA=',text)
        self.assertIn('loss=',text)

    def test_full_dataset_provenance_is_computed_once_per_baseline_run(self):
        from device_jepa.training import _provenance
        config=json.loads((Path(__file__).parents[1]/'configs/smoke.json').read_text())
        config['model'].update(dim=8,heads=2,ffn_dim=16,decoder_ffn_dim=16,dropout=0.)
        config['training'].update(baseline_epochs=3,batch_size=8)
        data=apply_op_states(synthetic_dataset(6),config['op_state'])
        manifest=build_manifest(data,config=config['dataset'])
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
             patch('device_jepa.baselines._provenance',wraps=_provenance) as compute:
            train_baseline(data,manifest,config,tmp)
        self.assertEqual(compute.call_count,1)


if __name__=='__main__':
    unittest.main()
