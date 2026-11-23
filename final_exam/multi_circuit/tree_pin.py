"""Joint fixed-order padded pin XGBoost; one native booster per metric."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from final_exam.tree_pin import (pin_features_pin, tree_settings_pin, feature_schema_pin,
    _topology_schema_pin, _check_topology_pin, _backend)
from experiments.op_transfer.engine import denormalize
from experiments.op_transfer.resume import atomic_json
from device_jepa.comparison import regression_metrics


def features_pin(batch, prep, capacity):
    # A batch contains one topology. Remove split-wide padding before flattening.
    valid = batch['device_mask']
    n = int(valid[0].sum())
    if n > capacity or not torch.equal(valid, valid[:1].expand_as(valid)):
        raise ValueError('Joint tabular batch must have one topology within its saved capacity')
    clean = dict(batch)
    for key in ('parameters', 'parameter_mask', 'parameter_applicability', 'device_types',
                'device_categories', 'device_models', 'device_mask', 'port_roles', 'named_port_roles'):
        if key in clean:
            clean[key] = clean[key][:, :n]
    for key in ('adjacency', 'edge_roles'):
        clean[key] = clean[key][:, :n, :n]
    values = pin_features_pin(clean, prep, n).reshape(valid.shape[0], n, -1)
    values = torch.cat((values, values.new_ones(*values.shape[:2], 1)), -1)
    return F.pad(values, (0, 0, 0, capacity-n)).flatten(1)


class JointTreePin(nn.Module):
    def __init__(self, boosters, prep, capacity, topologies, device):
        super().__init__()
        self.boosters, self.prep, self.capacity, self.topologies = boosters, prep, capacity, topologies
        self.execution_device = torch.device(device)

    def forward(self, batch):
        import xgboost as xgb
        key = batch['_group_keys'][0][0]
        if key not in self.topologies:
            raise ValueError('Joint XGBoost received an unseen circuit/topology')
        _check_topology_pin(batch, self.topologies[key])
        features = features_pin(batch, self.prep, self.capacity)
        matrix = xgb.DMatrix(features.detach().cpu().numpy())
        out = np.zeros((len(features), len(self.boosters)), dtype=np.float32)
        for i, booster in enumerate(self.boosters):
            if booster is not None:
                out[:, i] = booster.predict(matrix)
                _backend(booster, self.execution_device)
        return torch.from_numpy(out).to(features.device)


def collect_pin(ctx, split, directory, batch_size, topologies=None):
    schemas = {} if topologies is None else topologies
    arrays = None
    circuits, offset = [], 0
    for batch in ctx.batches(split, batch_size):
        key = batch['_group_keys'][0][0]
        if key not in schemas:
            if topologies is not None:
                raise ValueError('Held-out topology absent from training schema')
            schemas[key] = _topology_schema_pin(batch)
        _check_topology_pin(batch, schemas[key])
        features = features_pin(batch, ctx.prep, ctx.config['model']['max_devices']).cpu().numpy()
        if arrays is None:
            shapes = ((len(ctx.rows[split]), features.shape[1]),
                      (len(ctx.rows[split]), batch['performance'].shape[1]))
            arrays = [np.lib.format.open_memmap(directory/f'{split}_{name}_pin.npy', mode='w+',
                       dtype=dtype, shape=shape) for name, dtype, shape in
                       [('x', 'float32', shapes[0]), ('y', 'float32', shapes[1]), ('mask', 'bool', shapes[1])]]
        end = offset + len(features)
        for dest, src in zip(arrays, (features, batch['performance'].cpu().numpy(), batch['performance_mask'].cpu().numpy())):
            dest[offset:end] = src
        offset = end
        circuits.extend(batch['_circuit_ids'])
    if arrays is None:
        raise ValueError('Empty tabular split')
    for array in arrays:
        array.flush()
    return *arrays, np.asarray(circuits), schemas


def fit_xgboost_joint_pin(ctx, config, options, seed, directory):
    import xgboost as xgb
    if ctx.device.type == 'cuda' and (not torch.cuda.is_available() or not xgb.build_info().get('USE_CUDA', False)):
        raise RuntimeError('Joint CUDA XGBoost needs a CUDA-enabled build and visible GPU')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    settings = tree_settings_pin(config)
    train_x, train_y, train_mask, _, topologies = collect_pin(ctx, 'train', directory, options['batch_size'])
    val_x, val_y, val_mask, val_circuits, _ = collect_pin(ctx, 'val', directory, options['batch_size'], topologies)
    params = {k: v for k, v in settings.items() if k != 'num_boost_round'}
    params.update(objective='reg:squarederror', tree_method='hist', device=str(ctx.device),
                  seed=seed, nthread=options['torch_threads'], disable_default_eval_metric=True)
    boosters, selections, histories = [], [], {}
    names = ctx.prep['feature_spec']['metric_names']
    for j, name in enumerate(names):
        train_valid, val_valid = train_mask[:, j], val_mask[:, j]
        if not train_valid.any():
            boosters.append(None)
            selections.append({'metric': name, 'trained': False})
            continue
        circuit_ids = val_circuits[val_valid]
        def physical(values):
            tensor = torch.zeros((len(values), len(names)), dtype=torch.float64)
            tensor[:, j] = torch.as_tensor(values.astype(np.float64))
            return denormalize(tensor, ctx.prep['performance'])[:, j].numpy()
        y = physical(np.asarray(val_y[val_valid, j]))
        eligible = [c for c in sorted(set(circuit_ids)) if regression_metrics(y[circuit_ids == c], y[circuit_ids == c])['r2'] is not None]
        def metric(prediction, matrix):
            p = physical(prediction)
            values = [regression_metrics(y[circuit_ids == c], p[circuit_ids == c])['r2'] for c in eligible]
            return 'negative_circuit_macro_r2', -float(np.mean(values))
        train = xgb.DMatrix(train_x[train_valid], label=train_y[train_valid, j])
        validation = xgb.DMatrix(val_x[val_valid], label=val_y[val_valid, j]) if eligible else None
        history = {}
        # No eligible validation targets: fixed rounds; never fall back to test.
        booster = xgb.train(params, train, settings['num_boost_round'],
            evals=[(validation, 'val')] if eligible else [], custom_metric=metric if eligible else None,
            maximize=False, early_stopping_rounds=options['patience'] or None if eligible else None,
            evals_result=history, verbose_eval=False)
        _backend(booster, ctx.device)
        rounds = booster.num_boosted_rounds()
        selected = getattr(booster, 'best_iteration', rounds-1) + 1
        booster = booster[:selected]
        filename = f'metric_{j}_pin.ubj'
        booster.save_model(directory/filename)
        boosters.append(booster)
        selections.append(dict(metric=name, trained=True, file=filename, rounds_run=rounds,
            selected_rounds=selected, eligible_validation_circuits=eligible,
            selection='negative_circuit_macro_r2' if eligible else 'fixed_rounds_no_eligible_validation'))
        histories[name] = history
    metadata = dict(format='multi_circuit_xgboost_pin_v1', granularity='pin', seed=seed,
        preprocessing=ctx.prep, config=config, manifest=ctx.full_manifest, capacity=config['model']['max_devices'],
        metric_names=names, metrics=selections, topologies=topologies,
        feature_schema={**feature_schema_pin(ctx.prep), 'explicit_validity': True, 'padding': 'zero'},
        device=str(ctx.device), xgboost_version=xgb.__version__, inference_requires_op=False)
    atomic_json(directory/'checkpoint_pin.json', metadata)
    atomic_json(directory/'history_pin.json', histories)
    cost = dict(stage_id='xgboost_pin', epochs=None, planned_epochs=None, optimizer_steps=0,
        examples_seen=int(train_mask.sum()), seconds=time.perf_counter()-start, parameters=None,
        trainable_parameters=None, boosting_rounds=selections, selection='validation circuit macro R2 per metric')
    atomic_json(directory/'cost_pin.json', cost)
    return JointTreePin(boosters, ctx.prep, config['model']['max_devices'], topologies, ctx.device), cost


def load_xgboost_joint_pin(path, device='cpu'):
    import xgboost as xgb
    path = Path(path)
    data = json.loads(path.read_text())
    boosters = []
    for j, item in enumerate(data['metrics']):
        booster = None
        if item['trained']:
            if item['file'] != f'metric_{j}_pin.ubj':
                raise ValueError('Invalid booster filename')
            booster = xgb.Booster()
            booster.load_model(path.parent/item['file'])
            booster.set_param({'device': str(device)})
        boosters.append(booster)
    return JointTreePin(boosters, data['preprocessing'], data['capacity'], data['topologies'], device).eval(), data
