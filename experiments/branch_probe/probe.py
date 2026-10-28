"""Paired frozen readouts of a single JEPA checkpoint, isolated from main runs.

Both heads read topology-eligible device tokens and no graph token. Eligibility
never depends on whether an OP value is observed. Missing OP samples stay in the
cohort. These diagnostic heads are not the production JEPA decoder protocol.
"""
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch

from device_jepa.data import (dataset_fingerprint, input_op_fingerprint,
                             partition_fingerprint, verify_manifest, is_op_target,
                             _statistics)
from device_jepa.early_stopping import EarlyStopping, SUPERVISED_OBJECTIVE, early_stopping_config
from device_jepa.ema_model import EMAJEPAModel, EMA_MODE
from device_jepa.methods import JEPA_MODES
from device_jepa.model import JEPAModel, make_decoder, masked_mse
from device_jepa.resident import ResidentDataset, BatchProgress, require_device, _reserve
from device_jepa.tensors import denormalize
from device_jepa.training import (load_checkpoint, encoder_from_metadata, state_digest,
                                  _state_dict_digest, _json_digest, _metrics,
                                  prediction_records, _pretraining_provenance, seed_everything)


READOUT_POLICY = 'topology_eligible_device_tokens_no_graph_v1'
BRANCHES = ('design', 'op')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def reconstruct_model(source):
    """Restore BOTH encoders; never reconstruct an OP branch from encoder_state."""
    mode = source.get('mode', 'jepa_zerosim')
    if source.get('stage') != 'pretrain' or mode not in JEPA_MODES:
        raise ValueError('Branch probing requires a JEPA pretrain_best.pt checkpoint')
    if 'model_state' not in source:
        raise ValueError('Pretrain checkpoint lacks full model_state containing the OP branch')
    prep = source['preprocessing']
    encoder = encoder_from_metadata(source['model_config'], prep)
    args = (encoder, len(prep['feature_spec']['op_names']), len(prep.get('op_state_features', [])))
    if mode == EMA_MODE:
        model = EMAJEPAModel(*args, ema_decay=source['ema']['decay'])
        model.ema_updates = source['ema_updates']
    else:
        if source.get('architecture') != 'shared-structure-prefix_direct-device-op-alignment_v2':
            raise ValueError('Unsupported legacy JEPA checkpoint architecture')
        model = JEPAModel(*args)
    model.load_state_dict(source['model_state'], strict=True)
    if (state_digest(model.encoder) != source['encoder_digest']
            or _state_dict_digest(source['encoder_state']) != source['encoder_digest']):
        raise ValueError('Pretrain full model and exported encoder integrity mismatch')
    if mode == EMA_MODE:
        model.verify_ema_state()
    return model.eval()


def prepare_probe(dataset, checkpoint_path, device='cpu'):
    """Validate source identity and split policy without creating output files."""
    require_device(device)
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint_hash = file_sha256(checkpoint_path)
    source = load_checkpoint(checkpoint_path)
    if file_sha256(checkpoint_path) != checkpoint_hash:
        raise ValueError('Source checkpoint changed while loading; wait for pretraining to finish')
    model = reconstruct_model(source)
    if input_op_fingerprint(dataset) != source['input_op_fingerprint']:
        raise ValueError('Probe data changed immutable design/input/OP provenance')
    if partition_fingerprint(source['manifest']) != source['partition_fingerprint']:
        raise ValueError('Source partition fingerprint mismatch')
    prep = copy.deepcopy(source['preprocessing'])
    if prep['fitted_sample_ids'] != source['manifest']['train']:
        raise ValueError('Source preprocessing was not fitted on its training partition')
    manifest = copy.deepcopy(source['manifest'])
    manifest['dataset_fingerprint'] = dataset_fingerprint(dataset)
    verify_manifest(dataset, manifest)
    lookup = {row['sample_id']: row for row in dataset['samples']}
    split_rows = {split: [lookup[sid] for sid in manifest['labeled_train' if split == 'train' else split]]
                  for split in ('train', 'val', 'test', 'ood')}
    if not split_rows['train'] or not split_rows['val']:
        raise ValueError('Paired readout requires nonempty labeled_train and validation splits')
    for split, rows in split_rows.items():
        for row in rows:
            if not any(is_op_target(d) for d in row['devices']):
                raise ValueError(f'No topology-eligible device tokens in {split} sample {row["sample_id"]}')
    return dict(dataset=dataset, source=source, source_checkpoint=str(checkpoint_path),
                source_sha256=checkpoint_hash, manifest=manifest, prep=prep,
                split_rows=split_rows, source_model_digest=state_digest(model))


def probe_training(context, seed, overrides=None):
    if type(seed) is not int or seed < 0:
        raise ValueError('Decoder seed must be a nonnegative integer')
    allowed = {'decoder_epochs', 'batch_size', 'decoder_lr', 'weight_decay', 'grad_clip',
               'early_stopping', 'torch_threads', 'progress_every'}
    overrides = overrides or {}
    if set(overrides) - allowed:
        raise ValueError('Unsupported probe training overrides: ' + ', '.join(sorted(set(overrides)-allowed)))
    source = context['source']['options']['training']
    tr = {key: copy.deepcopy(source[key]) for key in allowed if key in source}
    tr.update(copy.deepcopy(overrides))
    tr.setdefault('torch_threads', 1)
    tr.setdefault('progress_every', 100)
    tr['seed'] = seed
    for key in ('decoder_epochs', 'batch_size', 'torch_threads', 'progress_every'):
        if type(tr[key]) is not int or tr[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('decoder_lr', 'grad_clip', 'weight_decay'):
        value = tr[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key != 'weight_decay' and value == 0):
            raise ValueError(f'Invalid {key}')
    tr['early_stopping'] = early_stopping_config(tr)
    return tr


def fit_probe_preprocessor(context, device='cpu'):
    device = require_device(device)
    key = str(device)
    if key in context.get('_fitted_preprocessing', {}):
        return copy.deepcopy(context['_fitted_preprocessing'][key])
    prep = copy.deepcopy(context['prep'])
    rows = sorted(context['split_rows']['train'], key=lambda r: r['sample_id'])
    arguments = ([(row['circuit_id'], row['performance']) for row in rows],
                 prep['feature_spec']['metric_names'], prep['transform_config']['performance'],
                 prep['transform_config']['weighting'])
    if device.type == 'cuda':
        from device_jepa.resident import device_statistics
        prep['performance'] = device_statistics(*arguments, device)
    else:
        prep['performance'] = _statistics(*arguments)
    prep['performance_fitted_sample_ids'] = sorted(context['manifest']['labeled_train'])
    prep['fit_ids']['performance'] = list(prep['performance_fitted_sample_ids'])
    if not any(count > 0 for count in prep['performance']['count']):
        raise ValueError('No performance labels are observed in labeled_train')
    context.setdefault('_fitted_preprocessing', {})[key] = copy.deepcopy(prep)
    return prep


def branch_memory(model, batch, branch, eligible_mask):
    """Return matched device memory without crossing the input modality boundary."""
    if branch not in BRANCHES:
        raise ValueError('Unknown probe branch')
    clean = {key: value for key, value in batch.items()
             if key not in ('performance', 'performance_mask', 'physical_performance', 'probe_eligible_mask')}
    if branch == 'design':
        clean = {key: value for key, value in clean.items() if not key.startswith('op')}
        memory = model.encoder(clean)[0][:, 1:]
    else:
        clean = {key: value for key, value in clean.items() if not key.startswith('parameter')}
        if isinstance(model, EMAJEPAModel):
            memory = model.op_encoder(clean)[0][:, 1:]
        else:
            structure = model.structure_encoder(clean)
            memory = model.op_encoder(structure['tokens'][:, 1:], clean['op'], clean['op_mask'],
                                      clean.get('op_state'), clean.get('op_state_mask'))
    live = eligible_mask & batch['device_mask']
    if live.shape != memory.shape[:2] or not bool(live.any(dim=1).all()):
        raise ValueError('Every probe example requires matching eligible device tokens')
    return memory.masked_fill(~live.unsqueeze(-1), 0), ~live


class FeatureCache:
    """Branch-specific frozen features, targets and exact masks on one device."""
    def __init__(self, rows, prep, model, branch, device, training, evaluation=False):
        self.rows, self.device = rows, require_device(device)
        selected = None if evaluation else [count > 0 for count in prep['performance']['count']]
        resident = ResidentDataset(rows, prep, self.device, include_op=branch == 'op',
                                   include_op_states=branch == 'op', include_parameters=branch == 'design',
                                   include_performance=True, performance_metric_mask=selected,
                                   performance_dtype=torch.float64 if evaluation else torch.float32,
                                   max_tokens=model.encoder.cfg.max_devices, label=f'{branch} probe data')
        # One host packing pass and one transfer, not one tiny CUDA transfer per sample.
        eligibility = torch.tensor([
            [is_op_target(d) for d in sorted(row['devices'], key=lambda d: d['device_id'])]
            + [False] * (resident.max_nodes-len(row['devices'])) for row in rows],
            dtype=torch.bool).to(self.device)
        resident.tables['probe_eligible_mask'] = eligibility
        needed = len(rows) * resident.max_nodes * (model.encoder.cfg.dim * 4 + 1)
        _reserve(self.device, needed)
        self.memory = torch.zeros(len(rows), resident.max_nodes, model.encoder.cfg.dim, device=self.device)
        self.padding = torch.ones(len(rows), resident.max_nodes, dtype=torch.bool, device=self.device)
        self.target = resident.tables['performance']
        self.mask = resident.tables['performance_mask']
        self.physical = resident.tables.get('physical_performance')
        progress = BatchProgress(f'{branch} frozen feature cache', math.ceil(len(rows)/training['batch_size']),
                                 training['progress_every'], self.device)
        with torch.no_grad():
            for step, (idx, batch) in enumerate(resident.batches(training['batch_size']), 1):
                memory, padding = branch_memory(model, batch, branch, batch['probe_eligible_mask'])
                self.memory[idx, :memory.shape[1]] = memory
                self.padding[idx, :padding.shape[1]] = padding
                progress.update(step)
        self.nbytes = sum(t.numel()*t.element_size() for t in
                          (self.memory, self.padding, self.target, self.mask) if t is not None)

    def batches(self, batch_size, indices=None, seed=None):
        order = list(range(len(self.rows))) if indices is None else list(indices)
        if seed is not None:
            random.Random(seed).shuffle(order)
        indices = torch.tensor(order, dtype=torch.long, device=self.device)
        for start in range(0, len(order), batch_size):
            idx = indices[start:start+batch_size]
            yield idx, self.memory[idx], self.padding[idx], self.target[idx], self.mask[idx]


def _snapshot(model):
    return torch.cat([tensor.detach().contiguous().reshape(-1).view(torch.uint8)
                      for tensor in model.state_dict().values()]).clone()


def _verify_frozen(model, before):
    if (any(p.requires_grad or p.grad is not None for p in model.parameters())
            or any(module.training for module in model.modules())):
        raise AssertionError('Frozen probe encoder received gradients or left eval mode')
    if not torch.equal(_snapshot(model), before):
        raise AssertionError('Frozen source encoder parameters or buffers changed')


def _source_metadata(context, prep, tr, initial_digest):
    source = context['source']
    return dict(experiment_kind='frozen_branch_probe', probe_version=1,
                backend='frozen_branch_probe_v1', readout_policy=READOUT_POLICY,
                source_mode=source.get('mode', 'jepa_zerosim'),
                source_checkpoint_sha256=context['source_sha256'],
                source_checkpoint=context['source_checkpoint'],
                source_model_digest=context['source_model_digest'],
                source_encoder_digest=source['encoder_digest'],
                pretrain_seed=source['options']['training']['seed'],
                decoder_seed=tr['seed'], seed=tr['seed'], training_config=copy.deepcopy(tr),
                decoder_initial_digest=initial_digest,
                target_scaler_digest=_json_digest({'names':prep['feature_spec']['metric_names'], 'performance':prep['performance']}),
                input_transform_digest=_json_digest(context['prep']),
                partition_fingerprint=source['partition_fingerprint'], input_op_fingerprint=source['input_op_fingerprint'],
                data_fingerprint=context['manifest']['dataset_fingerprint'],
                train_ids=context['manifest']['train'], labeled_train_ids=context['manifest']['labeled_train'],
                metric_names=prep['feature_spec']['metric_names'],
                pretrain_provenance=_pretraining_provenance(source),
                shared_pretraining_cost=copy.deepcopy(source.get('training_cost', {})),
                synthetic=bool(source.get('synthetic', False)),
                unobserved_training_metrics=[name for name, count in zip(prep['feature_spec']['metric_names'],
                                                                        prep['performance']['count']) if not count])


def _evaluate(context, checkpoint, model, prep, split, device):
    rows = context['split_rows'][split]
    tr, branch = checkpoint['training_config'], checkpoint['branch']
    cache = FeatureCache(rows, prep, model, branch, device, tr, evaluation=True)
    decoder = make_decoder(model.encoder.cfg, len(prep['feature_spec']['metric_names'])).to(device)
    decoder.load_state_dict(checkpoint['decoder_state'], strict=True)
    if state_digest(decoder) != checkpoint['decoder_digest']:
        raise ValueError('Probe decoder checkpoint integrity mismatch')
    decoder.eval()
    predictions = []
    progress = BatchProgress(f'{branch} probe evaluate {split}', math.ceil(len(rows)/tr['batch_size']), tr['progress_every'], device)
    with torch.no_grad():
        for step, (_, memory, padding, _, _) in enumerate(cache.batches(tr['batch_size']), 1):
            predictions.append(decoder(memory, padding))
            progress.update(step)
    normalized = torch.cat(predictions)
    raw = denormalize(normalized, prep['performance']).cpu()
    normalized, target, mask, physical = normalized.cpu(), cache.target.cpu(), cache.mask.cpu(), cache.physical.cpu()
    report = {key: copy.deepcopy(value) for key, value in checkpoint.items()
              if key not in ('decoder_state', 'preprocessing', 'manifest')}
    report.update(split=split, num_samples=len(rows), evaluation_ids=[row['sample_id'] for row in rows],
                  selection_used_ood=False, has_ood_holdout=bool(context['manifest']['ood']),
                  metrics=_metrics(raw, physical, mask, prep, normalized, target), by_circuit={})
    for circuit in sorted({row['circuit_id'] for row in rows}):
        selected = torch.tensor([row['circuit_id'] == circuit for row in rows])
        report['by_circuit'][circuit] = _metrics(raw[selected], physical[selected], mask[selected], prep,
                                               normalized[selected], target[selected])
    report['predictions'] = prediction_records(raw, physical, mask, normalized, target, rows, prep)
    names = prep['feature_spec']['op_names']
    devices = [dev for row in rows for dev in row['devices'] if is_op_target(dev)]
    report['coverage'] = dict(samples=len(rows), excluded_samples=0, eligible_devices=len(devices),
                              observed_devices=sum(any(dev.get('op', {}).get(name) is not None for name in names) for dev in devices),
                              all_op_missing_samples=sum(not any(dev.get('op', {}).get(name) is not None
                                  for dev in row['devices'] if is_op_target(dev) for name in names) for row in rows))
    return report


def evaluate_probe(context, probe_checkpoint, split='val', device='cpu'):
    device = require_device(device)
    checkpoint = torch.load(probe_checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('experiment_kind') != 'frozen_branch_probe' or checkpoint.get('probe_version') != 1:
        raise ValueError('Not a supported frozen branch probe checkpoint')
    if checkpoint.get('source_checkpoint_sha256') != context['source_sha256']:
        raise ValueError('Probe checkpoint belongs to a different source checkpoint')
    if checkpoint.get('readout_policy') != READOUT_POLICY or checkpoint.get('branch') not in BRANCHES:
        raise ValueError('Probe readout policy or branch mismatch')
    branch = checkpoint['branch']
    checks = {
        'source_mode':context['source'].get('mode', 'jepa_zerosim'),
        'source_model_digest':context['source_model_digest'],
        'frozen_model_digest':context['source_model_digest'],
        'source_encoder_digest':context['source']['encoder_digest'],
        'partition_fingerprint':context['source']['partition_fingerprint'],
        'input_op_fingerprint':context['source']['input_op_fingerprint'],
        'data_fingerprint':context['manifest']['dataset_fingerprint'],
        'train_ids':context['manifest']['train'],
        'labeled_train_ids':context['manifest']['labeled_train'],
        'requires_op_at_inference':branch == 'op',
        'uses_design_parameters':branch == 'design',
        'encoder_frozen_verified':True,
    }
    if any(checkpoint.get(key) != value for key, value in checks.items()):
        raise ValueError('Probe checkpoint source or input-contract provenance mismatch')
    if (checkpoint.get('seed') != checkpoint.get('decoder_seed')
            or checkpoint['training_config'].get('seed') != checkpoint.get('decoder_seed')):
        raise ValueError('Probe checkpoint decoder seed mismatch')
    probe_training(context, checkpoint['decoder_seed'],
                   {key:value for key,value in checkpoint['training_config'].items() if key != 'seed'})
    if split not in context['split_rows'] or not context['split_rows'][split]:
        raise ValueError(f'Empty or unknown evaluation split: {split}')
    if checkpoint['manifest'] != context['manifest']:
        raise ValueError('Probe checkpoint manifest mismatch')
    expected = checkpoint['preprocessing']
    frozen = copy.deepcopy(expected)
    frozen.pop('performance', None)
    frozen.pop('performance_fitted_sample_ids', None)
    frozen['fit_ids'].pop('performance', None)
    source_prep = copy.deepcopy(context['prep'])
    source_prep.pop('performance', None)
    source_prep.pop('performance_fitted_sample_ids', None)
    source_prep['fit_ids'].pop('performance', None)
    labeled_ids = sorted(context['manifest']['labeled_train'])
    if (frozen != source_prep or expected.get('performance_fitted_sample_ids') != labeled_ids
            or expected.get('fit_ids', {}).get('performance') != labeled_ids
            or checkpoint.get('target_scaler_digest') != _json_digest({
                'names':expected['feature_spec']['metric_names'], 'performance':expected['performance']})):
        raise ValueError('Probe preprocessing differs from source and labeled training data')
    stats = expected['performance']
    width = len(expected['feature_spec']['metric_names'])
    for key in ('mean', 'std', 'count', 'kind', 'asinh_scale'):
        if not isinstance(stats.get(key), list) or len(stats[key]) != width:
            raise ValueError('Probe performance scaler schema mismatch')
    if (any(type(value) not in (int,float) or not math.isfinite(value) for value in stats['mean'])
            or any(type(value) not in (int,float) or not math.isfinite(value) or value <= 0 for value in stats['std'])
            or any(type(value) is not int or value < 0 for value in stats['count'])):
        raise ValueError('Probe performance scaler contains invalid statistics')
    model = reconstruct_model(context['source']).to(device).requires_grad_(False).eval()
    return _evaluate(context, checkpoint, model, expected, split, device)


def run_pair(context, output, seed, training_overrides=None, device='cuda', eval_splits=('val',)):
    """Train two identically initialized heads; output is exclusively a new pair."""
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Probe output must be new or empty; refusing to overwrite results')
    device = require_device(device)
    tr = probe_training(context, seed, training_overrides)
    if not eval_splits or len(set(eval_splits)) != len(eval_splits):
        raise ValueError('Choose distinct nonempty evaluation splits')
    for split in eval_splits:
        if split not in ('val', 'test', 'ood') or not context['split_rows'][split]:
            raise ValueError(f'Empty or unsupported evaluation split: {split}')
    prep = fit_probe_preprocessor(context, device)
    selected = [count > 0 for count in prep['performance']['count']]
    if not any(row['performance'].get(name) is not None for row in context['split_rows']['val']
               for name, active in zip(prep['feature_spec']['metric_names'], selected) if active):
        raise ValueError('Validation has no observed targets for trained metrics')
    seed_everything(seed, tr['torch_threads'])
    model = reconstruct_model(context['source']).to(device).requires_grad_(False).eval()
    before = _snapshot(model)
    decoder_template = make_decoder(model.encoder.cfg, len(prep['feature_spec']['metric_names']))
    initial_state = copy.deepcopy(decoder_template.state_dict())
    initial_digest = state_digest(decoder_template)
    metadata = _source_metadata(context, prep, tr, initial_digest)
    output.mkdir(parents=True, exist_ok=True)
    _json(output/'manifest.json', context['manifest'])
    _json(output/'preprocessing.json', prep)
    _json(output/'pair.json', {**metadata, 'branches':list(BRANCHES), 'complete':False})
    reports = []
    for branch in BRANCHES:
        started = time.perf_counter()
        destination = output/branch
        destination.mkdir()
        train_rows, val_rows = context['split_rows']['train'], context['split_rows']['val']
        cache = FeatureCache(train_rows+val_rows, prep, model, branch, device, tr)
        decoder = make_decoder(model.encoder.cfg, len(prep['feature_spec']['metric_names'])).to(device)
        decoder.load_state_dict(initial_state, strict=True)
        if state_digest(decoder) != initial_digest:
            raise AssertionError('Paired decoders did not start from identical parameters')
        optimizer = torch.optim.AdamW(decoder.parameters(), lr=tr['decoder_lr'], weight_decay=tr['weight_decay'])
        # Feature extraction/reconstruction must not change head dropout RNG.
        seed_everything(seed, tr['torch_threads'])
        stopping = EarlyStopping(tr, tr['decoder_epochs'], SUPERVISED_OBJECTIVE)
        history, steps = [], 0
        for epoch in range(tr['decoder_epochs']):
            record = {'epoch':epoch+1}
            for phase, indices in (('train',range(len(train_rows))), ('val',range(len(train_rows),len(cache.rows)))):
                decoder.train(phase == 'train')
                total = torch.zeros((), dtype=torch.float64, device=device)
                count = torch.zeros((), dtype=torch.long, device=device)
                progress = BatchProgress(f'{branch} probe epoch {epoch+1}/{tr["decoder_epochs"]} {phase}',
                                         math.ceil(len(indices)/tr['batch_size']),tr['progress_every'],device)
                for step, (_, memory, padding, target, mask) in enumerate(cache.batches(tr['batch_size'], indices,
                        seed+epoch if phase == 'train' else None), 1):
                    if not bool(mask.any()):
                        progress.update(step)
                        continue
                    with torch.set_grad_enabled(phase == 'train'):
                        prediction = decoder(memory, padding)
                        loss = masked_mse(prediction, target, mask)
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Non-finite branch probe objective')
                        if phase == 'train':
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(decoder.parameters(), tr['grad_clip'], error_if_nonfinite=True)
                            optimizer.step()
                            steps += 1
                    count += mask.sum()
                    total += loss.detach().double() * mask.sum()
                    progress.update(step, loss.detach())
                if not int(count):
                    raise ValueError(f'No observed trained-metric labels in {phase}')
                record[phase] = dict(normalized_mse=float(total/count),observed_targets=int(count))
            _verify_frozen(model,before)
            history.append(record)
            value=record['val']['normalized_mse']
            print(f'{branch} probe {epoch+1}/{tr["decoder_epochs"]}: val={value:.6g}',flush=True)
            if stopping.update(value,epoch+1):
                checkpoint = {**metadata, 'branch':branch, 'mode':branch+'_probe',
                              'requires_op_at_inference':branch=='op', 'uses_design_parameters':branch=='design',
                              'epoch':epoch+1,'selection':dict(split='val',objective=SUPERVISED_OBJECTIVE,value=value),
                              'decoder_state':copy.deepcopy(decoder.state_dict()),'decoder_digest':state_digest(decoder),
                              'encoder_frozen_verified':True,'frozen_model_digest':context['source_model_digest'],
                              'preprocessing':prep,'manifest':context['manifest']}
                torch.save(checkpoint,destination/'decoder_best.pt')
            if stopping.should_stop:
                stopping.announce(f'{branch} probe')
                break
        checkpoint = torch.load(destination/'decoder_best.pt',map_location='cpu',weights_only=True)
        checkpoint.update(early_stopping=stopping.summary(),training_cost=dict(
            decoder_optimizer_steps=steps,decoder_epochs=len(history),planned_decoder_epochs=tr['decoder_epochs'],
            labeled_train_examples=len(train_rows),seconds=time.perf_counter()-started,
            device=str(device),frozen_cache_bytes=cache.nbytes),
            parameter_count=dict(decoder=sum(p.numel() for p in decoder.parameters()),
                                 frozen_source_model=sum(p.numel() for p in model.parameters())))
        torch.save(checkpoint,destination/'decoder_best.pt')
        _json(destination/'history.json',history)
        del cache, optimizer, decoder
        for split in eval_splits:
            report = _evaluate(context,checkpoint,model,prep,split,device)
            _json(destination/f'{split}_report.json',report)
            reports.append(report)
        _verify_frozen(model,before)
    _json(output/'pair.json',{**metadata,'branches':list(BRANCHES),'complete':True,
                            'evaluation_splits':list(eval_splits),'source_encoder_frozen_verified':True})
    return reports
