"""Pin data preserve device-weighted statistics and physical ownership."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from device_jepa.baselines import _pin_tokens, collate_pin_samples
from device_jepa.data import build_manifest, fit_preprocessor, synthetic_dataset
from final_exam.data_pin import PinDataContext, PinResidentDataset, group_keys_pin


class PinDataTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = synthetic_dataset(24)
        self.data['samples'] = [row for row in self.data['samples'] if row['circuit_id'] == 'synthetic_0']
        for row in self.data['samples']:
            row['devices'] = row['devices'][:4]
            for device in row['devices']:
                device['model_name'] = device['device_type'] + '_physical_model'
        self.config = dict(model=dict(max_devices=32), normalization={}, dataset=dict(
            train_circuits=['synthetic_0'], ood_circuits=[], split_ratios=[.6,.2,.2],
            split_seed=17, finetune_label_fraction=1.))
        self.manifest = build_manifest(self.data,config=self.config['dataset'])

    def context(self, name='context'):
        directory=self.root/name
        directory.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            return PinDataContext(self.data,self.manifest,self.config,'cpu',{},directory)

    def test_train_statistics_are_device_weighted_and_pin_vocab_train_only(self):
        # Introduce a genuinely unseen named port only in held-out rows.
        for row in self.data['samples']:
            if row['sample_id'] in self.manifest['test']:
                row['ports']['held_out_port'] = 'n0'
        self.manifest=build_manifest(self.data,config=self.config['dataset'])
        ctx=self.context()
        expected=fit_preprocessor(self.data,self.manifest,include_performance=True,
                                  config=self.config['normalization'],statistics_device='cpu')
        for name in ('parameters','op','performance'):
            self.assertEqual(ctx.prep[name],expected[name])
        self.assertEqual(ctx.prep['device_preprocessing']['device_vocab'],expected['device_vocab'])
        self.assertNotIn('port:held_out_port',ctx.prep['device_vocab'])
        with contextlib.redirect_stdout(io.StringIO()):
            batch=next(ctx.batches('test',8))
        # Held-out port has an unknown type, while preserving topology/port token.
        unknown=(batch['owner_index']<0)&batch['device_mask']&(batch['device_types']==1)
        self.assertTrue(unknown.any())
        files={path.name for path in (self.root/'context').iterdir()}
        self.assertEqual(files,{'cohort_pin.json','preprocessing_pin.json'})
        self.assertEqual(ctx.prep['granularity'],'pin')

    def test_pin_batch_matches_reference_and_keeps_owner_targets_once(self):
        ctx=self.context()
        with contextlib.redirect_stdout(io.StringIO()):
            batch=next(ctx.batches('train',8,op=True))
        row_map={row['sample_id']:row for row in ctx.rows['train']}
        rows=[row_map[sid] for sid in batch['_sample_ids']]
        reference=collate_pin_samples(rows,ctx.device_preprocessing,ctx.pin_vocab,
            include_op=True,performance_metric_mask=ctx.metric_mask)
        for name in ('device_types','device_models','device_categories','device_mask',
                     'adjacency','edge_roles','named_port_roles','parameters',
                     'parameter_mask','parameter_applicability','op','op_mask','op_target_mask'):
            torch.testing.assert_close(batch[name],reference[name],msg=name)
        self.assertEqual(batch['device_mask'].shape[1],16)  # 3 MOS x4 + resistor x2 + 2 ports
        self.assertEqual(batch['owner_mask'].shape[1],4)
        self.assertEqual(batch['owner_op'].shape[1],4)
        self.assertEqual(batch['owner_index'][0,:2].tolist(),[-1,-1])
        self.assertFalse(batch['op_mask'][:,:2].any())
        self.assertFalse(batch['parameter_mask'][:,:2].any())
        self.assertTrue((batch['pin_terminal_index'][:,:2]==-1).all())
        self.assertTrue((batch['owner_index']==batch['pin_to_device']).all())
        for owner,count in ((0,4),(1,4),(2,4),(3,2)):
            positions=batch['owner_index'][0]==owner
            self.assertEqual(int(positions.sum()),count)
            torch.testing.assert_close(batch['op'][0,positions],
                batch['owner_op'][0,owner].expand(count,-1))
        # Shared owner targets are counted per device, not weighted by pin count.
        self.assertEqual(batch['owner_op_mask'].shape[1],4)
        self.assertEqual(batch['owner_edge_roles'].shape[1:3],(4,4))
        self.assertEqual(len(batch['_group_keys'][0]),16)
        self.assertEqual(len(set(batch['_group_keys'][0])),16)
        self.assertEqual(len({key.rsplit(':',1)[0] for key in batch['_group_keys'][0]}),1)

    def test_untrained_op_and_metric_fields_mask_before_float32(self):
        self.data['feature_spec']['op_names'].append('held_out_only')
        self.data['feature_spec']['metric_names'].append('untrained_metric')
        for row in self.data['samples']:
            heldout=row['sample_id'] in self.manifest['test'] or row['sample_id'] in self.manifest['val']
            row['performance']['untrained_metric']=1e100 if heldout else None
            for device in row['devices']:
                device['op']['held_out_only']=1e100 if heldout else None
        self.manifest=build_manifest(self.data,config=self.config['dataset'])
        ctx=self.context()
        with contextlib.redirect_stdout(io.StringIO()):
            for split in ('train','val','test'):
                batch=next(ctx.batches(split,8,op=True))
                self.assertFalse(batch['owner_op_mask'][...,-1].any())
                self.assertFalse(batch['op_mask'][...,-1].any())
                self.assertFalse(batch['performance_mask'][...,-1].any())
                self.assertTrue(torch.isfinite(batch['owner_op']).all())
                self.assertTrue(torch.isfinite(batch['performance']).all())

    def test_pin_identity_uses_topology_not_sizes_op_or_metrics(self):
        row=copy.deepcopy(self.data['samples'][0])
        keys=group_keys_pin(row,self.data['feature_spec']['terminal_names'])
        row['devices'][0]['parameters']['W']=123
        row['devices'][0]['op']['id']=456
        row['performance']['gain']=789
        self.assertEqual(keys,group_keys_pin(row,self.data['feature_spec']['terminal_names']))
        row['devices'][0]['terminals']['d']='new_net'
        self.assertNotEqual(keys,group_keys_pin(row,self.data['feature_spec']['terminal_names']))

    def test_owner_and_padding_mapping_and_max_pin_budget(self):
        ctx=self.context()
        rows=copy.deepcopy(ctx.rows['train'][:2])
        rows[0]['devices']=rows[0]['devices'][:2]
        with contextlib.redirect_stdout(io.StringIO()):
            cache=PinResidentDataset(rows,ctx.prep,'cpu',include_op=True,max_tokens=32)
            _,batch=next(cache.batches(2))
            _,short=next(cache.batches(1,indices=[0]))
        self.assertEqual(cache.max_nodes,16)
        self.assertEqual(batch['owner_mask'][0].tolist(),[True,True,False,False])
        self.assertTrue((batch['owner_index'][0,10:]==-1).all())
        self.assertTrue((batch['pin_terminal_index'][0,10:]==-1).all())
        self.assertEqual(short['owner_edge_roles'].shape[1:3],(4,4))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError,'max_tokens'):
            PinResidentDataset(rows,ctx.prep,'cpu',max_tokens=15)
        with contextlib.redirect_stdout(io.StringIO()):
            design=next(ctx.batches('train',8,op=False))
        self.assertNotIn('op',design)
        self.assertNotIn('owner_op',design)


if __name__=='__main__':
    unittest.main()
