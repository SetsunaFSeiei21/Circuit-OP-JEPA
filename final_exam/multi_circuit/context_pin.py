"""CPU resident design/OP tables; only a topology-homogeneous batch visits CUDA."""
import copy
import torch
from final_exam.data_pin import PinDataContext


class MultiCircuitContextPin(PinDataContext):
    storage_device = torch.device('cpu')
    cache_frozen_features = False
    evaluation_split = 'val'

    def __init__(self, *args, preprocessing=None, **kwargs):
        saved = copy.deepcopy(preprocessing)
        if saved is not None:
            saved.pop('multi_circuit_flat_pin', None)
        super().__init__(*args, preprocessing=saved, **kwargs)
        self.prep['multi_circuit_flat_pin'] = dict(version=1,
            ordering='canonical pin order within each circuit; zero padding to model.max_devices',
            explicit_validity=True, circuit_ids=sorted(self.config['dataset']['train_circuits']))
        from experiments.op_transfer.resume import atomic_json
        atomic_json(self.output / 'preprocessing_pin.json', self.prep)
        atomic_json(self.output / 'pin_schema_pin.json', {
            'version': 1, 'capacity': self.config['model']['max_devices'],
            'circuits': {self.rows['train'][indices[0]]['circuit_id']: self.keys['train'][indices[0]]
                         for indices in self.groups['train'].values()},
            'padding': 'zero plus explicit validity for flat baselines; attention padding mask otherwise'})

    def cache(self, split, op=False):
        if split in ('test', 'ood') and self.evaluation_split != 'test':
            raise RuntimeError('Search cannot materialize a test/ood feature cache')
        return super().cache(split, op)

    def batches(self, split, *args, **kwargs):
        for batch in super().batches(split, *args, **kwargs):
            ids = batch['_indices'].detach().cpu().tolist()
            batch['_circuit_ids'] = [self.rows[split][i]['circuit_id'] for i in ids]
            yield batch
