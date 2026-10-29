"""Train-only transformations, explicit stage dependencies, and replayable models.

This module deliberately does not change the historical production model IDs.
All final design-input checkpoints are deployable without measured OP values.
"""
import copy
import csv
import hashlib
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

from device_jepa.data import (build_manifest, canonical_model_name, dataset_fingerprint,
    fit_preprocessor, load_dataset, samples_for_split, verify_manifest)
from device_jepa.early_stopping import EarlyStopping
from device_jepa.ema_model import EMAJEPAModel
from device_jepa.model import ModelConfig, alignment_loss, masked_mse
from device_jepa.resident import BatchProgress, ResidentDataset, require_device
from device_jepa.tensors import denormalize
from device_jepa.training import _metrics, seed_everything, state_digest
from .losses import (LatentScaler, grouped_alignment_loss, masked_field_mse,
                     residual_alignment_loss)
from .distillation import PredictedOPDistillation, PredictedOPEncoderDistillation
from .models import (DualEncoderModel, OPPredictor, PredictedOPCascade, ReadoutModel,
    decoder, describe_model, design_encoder, design_input, fit_op_support, freeze,
    op_encoder, op_input, rebuild_model)

MODELS = ('design_supervised', 'op_oracle', 'jepa_standardized', 'jepa_grouped',
          'jepa_residual', 'jepa_op_aux', 'op_distill', 'op_distill_finetune',
          'predicted_op', 'predicted_op_joint', 'dual_distill',
          'dual_predicted_op', 'dual_design_control', 'design_op_oracle',
          'student_predicted_op', 'student_predicted_op_distill')
MEASURED_OP_MODELS = frozenset({'op_oracle', 'design_op_oracle'})
DEFAULT_EPOCHS = dict(teacher=30, student=30, op_predictor=30, readout=50, dual=50,
                     jepa=30, student_predicted_op=50,
                     predicted_op_distill_encoder=60, predicted_op_distill_decoder=60)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def settings(config):
    tr, raw = config.get('training', {}), config.get('op_transfer', {})
    allowed = {'epochs', 'batch_size', 'lr', 'weight_decay', 'grad_clip', 'latent_floor',
               'patience', 'progress_every', 'op_aux_weight', 'ema_decay', 'variance_floor', 'torch_threads',
               'student_alignment_weight', 'student_performance_weight'}
    if set(raw) - allowed:
        raise ValueError(f'Unknown op_transfer settings: {sorted(set(raw)-allowed)}')
    if set(raw.get('epochs', {})) - set(DEFAULT_EPOCHS):
        raise ValueError('Unknown op_transfer stage epoch setting')
    result = dict(epochs={**DEFAULT_EPOCHS, **raw.get('epochs', {})},
        batch_size=tr.get('batch_size', 32), lr=tr.get('decoder_lr', 1e-4),
        weight_decay=tr.get('weight_decay', 1e-4), grad_clip=tr.get('grad_clip', 1.),
        latent_floor=1e-4, patience=10, progress_every=100, op_aux_weight=1.,
        ema_decay=.99, variance_floor=config.get('loss', {}).get('variance_floor', 1.),
        torch_threads=tr.get('torch_threads', 1), student_alignment_weight=1.,
        student_performance_weight=1.)
    result.update({k: v for k, v in raw.items() if k != 'epochs'})
    for key, value in {**result['epochs'], **{k: result[k] for k in
            ('batch_size', 'progress_every', 'torch_threads')}}.items():
        if type(value) is not int or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(result['patience']) is not int or result['patience'] < 0:
        raise ValueError('patience must be a nonnegative integer; 0 disables early stopping')
    for key in ('lr', 'grad_clip', 'latent_floor', 'op_aux_weight', 'variance_floor', 'weight_decay', 'ema_decay',
                'student_alignment_weight', 'student_performance_weight'):
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
        if key not in ('weight_decay', 'ema_decay', 'student_alignment_weight') and value == 0:
            raise ValueError(f'{key} must be positive')
    if not result['ema_decay'] < 1:
        raise ValueError('ema_decay must be <1')
    return result


def group_keys(sample):
    """Topology identity excludes design values, OP, labels, and split names."""
    devices = sorted(sample['devices'], key=lambda d: d['device_id'])
    topology = [sample.get('ports', {}), [(d['device_id'], d['device_type'],
                 canonical_model_name(d), d['terminals']) for d in devices]]
    digest = hashlib.sha256(json.dumps(topology, sort_keys=True).encode()).hexdigest()
    prefix = json.dumps([sample['circuit_id'], digest])
    return [prefix + ':' + d['device_id'] for d in devices]


def attach_identity(batch, rows):
    batch['_group_keys'] = [group_keys(row) for row in rows]
    return batch


class DataContext:
    def __init__(self, dataset, manifest, config, device, limits, output):
        self.device, self.config, self.output = device, config, Path(output)
        self.full_manifest = manifest
        # Subset selection is independent of performance/OP and of model seed.
        self.rows, selected = {}, copy.deepcopy(manifest)
        for split in ('train', 'val', 'test', 'ood'):
            rows = samples_for_split(dataset, manifest, split)
            limit = limits.get('test' if split == 'ood' else split)
            ranked = sorted(rows, key=lambda r: hashlib.sha256(
                f"{manifest['split_seed']}:{r['sample_id']}".encode()).hexdigest())
            self.rows[split] = ranked if limit is None else ranked[:limit]
            selected[split] = sorted(r['sample_id'] for r in self.rows[split])
        selected['labeled_train'] = sorted(set(selected['train']) & set(manifest['labeled_train']))
        if not self.rows['train'] or not self.rows['val'] or not selected['labeled_train']:
            raise ValueError('Nonempty train, validation and labeled train cohorts required')
        self.selection = selected
        print('Fitting input/OP/label transforms on selected ID train only', flush=True)
        self.prep = fit_preprocessor(dataset, selected, include_performance=True,
            config=config.get('normalization'), statistics_device=device)
        self.prep['op_state_features'] = []  # continuous OP; state bits are not inference inputs
        self.metric_mask = [n > 0 for n in self.prep['performance']['count']]
        if not any(self.metric_mask):
            raise ValueError('No training performance metric is observed')
        self.labeled = set(selected['labeled_train'])
        self.caches = {}
        self.keys = {s: [group_keys(r) for r in rows] for s, rows in self.rows.items()}
        self.groups = {}
        for split, keys in self.keys.items():
            groups = defaultdict(list)
            for i, row_keys in enumerate(keys):
                groups[row_keys[0].rsplit(':', 1)[0]].append(i)
            self.groups[split] = dict(groups)
        write_json(output / 'cohort.json', {k: selected[k] for k in
                   ('train', 'val', 'test', 'ood', 'labeled_train')})
        write_json(output / 'preprocessing.json', self.prep)

    def cache(self, split, op=False):
        key = (split, op)
        if key not in self.caches:
            rows = self.rows[split]
            if not rows:
                return None
            if op:
                supported = {name for name,count in zip(self.prep['feature_spec']['op_names'],
                             self.prep['op']['count']) if count > 0}
                if len(supported) != len(self.prep['feature_spec']['op_names']):
                    # Mask before normalization/float32, including finite 1e100
                    # held-out values in fields never observed during training.
                    rows = [{**row, 'devices': [{**dev, 'op': {name:value for name,value in
                        dev.get('op',{}).items() if name in supported}} for dev in row['devices']]}
                        for row in rows]
            self.caches[key] = ResidentDataset(rows, self.prep, self.device, include_op=op,
                include_performance=True, performance_metric_mask=self.metric_mask,
                performance_dtype=torch.float64, max_tokens=self.config['model']['max_devices'],
                label=f'op-transfer {split} ({"OP supervision" if op else "design only"})')
        return self.caches[key]

    def batches(self, split, batch_size, seed=None, op=False, repeated=False):
        cache = self.cache(split, op)
        if cache is None:
            return
        chunks = []
        rng = random.Random(seed)
        for source_indices in self.groups[split].values():
            indices = source_indices[:]
            if seed is not None:
                rng.shuffle(indices)
            parts = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]
            if repeated and len(parts) > 1 and len(parts[-1]) == 1:
                parts[-2].extend(parts.pop())
            chunks.extend(parts)
        if seed is not None:
            rng.shuffle(chunks)
        for indices in chunks:
            # No silent dropping of singleton or all-missing records: objective
            # can explicitly mark them unusable, and the stage logs that count.
            batch_indices, batch = next(cache.batches(len(indices), indices=indices))
            rows = [cache.rows[i] for i in indices]
            batch['_group_keys'] = [self.keys[split][i] for i in indices]
            batch['_indices'] = batch_indices
            batch['_sample_ids'] = [r['sample_id'] for r in rows]
            if split == 'train':
                labeled = torch.tensor([r['sample_id'] in self.labeled for r in rows], device=self.device)
                batch['performance_mask'] = batch['performance_mask'] & labeled[:, None]
            yield batch


def fit_scaler(ctx, encoder, options, source='design'):
    if source not in {'design', 'op', 'design_op'}:
        raise ValueError('Latent scale source must be design, op or design_op')
    scaler = LatentScaler(options['latent_floor'])
    was_training = encoder.training
    encoder.eval()
    progress = BatchProgress(f'fit train-only {source} latent scales',
        sum(math.ceil(len(v)/options['batch_size']) for v in ctx.groups['train'].values()),
        options['progress_every'])
    with torch.no_grad():
        for step,batch in enumerate(ctx.batches('train', options['batch_size'], op=source != 'design'),1):
            inputs = design_input(batch) if source == 'design' else op_input(batch, use_design=source == 'design_op')
            memory, padding = encoder(inputs)
            scaler.update(memory, padding, batch['_group_keys'])
            progress.update(step)
    encoder.train(was_training)
    return scaler.finalize()


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def fit_stage(ctx, model, stage_id, epochs, options, seed, objective, *, op=False,
              repeated=False, after_step=None, before_epoch=None, checkpoint_extra=None):
    """Exact val-best selection, actual costs, no test/OOD access in fitting."""
    seed_everything(seed, options['torch_threads'])
    start = time.perf_counter()
    model.to(ctx.device)
    frozen_features = {}
    cache_cascade_readout = (isinstance(model, PredictedOPCascade) and not model.joint
                             and model.downstream.frozen)
    cache_readout = (isinstance(model, ReadoutModel) and model.frozen) or cache_cascade_readout
    cache_cascade_teacher = isinstance(model, PredictedOPDistillation)
    encoder_only = isinstance(model, PredictedOPEncoderDistillation)
    cache_teacher = isinstance(model, DistillationPair) or cache_cascade_teacher
    if cache_readout or cache_teacher:
        # Encode each frozen branch once per stage, not once per epoch.
        model.eval()
        with torch.no_grad():
            # The joint student selects performance only. Encoder-only distillation
            # instead selects validation alignment, using fixed train-fit scales.
            for split in (('train',) if cache_cascade_teacher and not encoder_only else ('train', 'val')):
                cache = ctx.cache(split, op)
                enc = (model.downstream.encoder if cache_cascade_readout else
                       model.encoder if cache_readout else
                       model.teacher.encoder if cache_cascade_teacher else model.teacher)
                table = torch.zeros(len(cache.rows), cache.max_nodes+1, enc.cfg.dim,
                                    device=ctx.device)
                padding_table = torch.ones(table.shape[:2], dtype=torch.bool, device=ctx.device)
                print(f'{stage_id}: caching frozen {split} features on {ctx.device}', flush=True)
                progress = BatchProgress(f'{stage_id}: cache {split}',
                    sum(math.ceil(len(v)/options['batch_size']) for v in ctx.groups[split].values()),options['progress_every'])
                for step,batch in enumerate(ctx.batches(split, options['batch_size'], op=op),1):
                    if cache_readout:
                        memory, padding = model.memory(batch)
                    elif cache_cascade_teacher:
                        memory, padding = model.teacher_memory(batch)
                    else:
                        memory,padding=model.teacher(op_input(batch))
                        memory=model.teacher_scaler.transform(memory,padding,batch['_group_keys'])
                    n = memory.shape[1]
                    table[batch['_indices'], :n] = memory
                    padding_table[batch['_indices'], :n] = padding
                    progress.update(step)
                frozen_features[split] = table, padding_table
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=options['lr'], weight_decay=options['weight_decay'])
    stopping = EarlyStopping({'early_stopping': dict(enabled=options['patience'] > 0,
        patience=max(1, options['patience']), min_delta=0., min_epochs=1)}, epochs, stage_id)
    directory = ctx.output / 'stages' / stage_id
    directory.mkdir(parents=True)
    steps, examples, history = 0, 0, []
    best_state, best_extra = None, {}
    for epoch in range(1, epochs+1):
        if before_epoch:
            before_epoch(model)
        record = {'epoch': epoch}
        for phase in ('train', 'val'):
            model.train(phase == 'train')
            # Total is an upper bound because topology grouping splits batches.
            total_batches = sum(math.ceil(len(v)/options['batch_size']) -
                int(repeated and len(v)>options['batch_size'] and len(v)%options['batch_size']==1)
                for v in ctx.groups[phase].values())
            progress = BatchProgress(f'{stage_id} {epoch}/{epochs} {phase}', total_batches,
                                     options['progress_every'])
            total = torch.zeros((), dtype=torch.float64, device=ctx.device)
            count, skipped = 0, 0
            field_sse = field_counts = None
            detail_totals = {}
            for step, batch in enumerate(ctx.batches(phase, options['batch_size'],
                    seed=seed+epoch if phase == 'train' else None, op=op, repeated=repeated), 1):
                with torch.set_grad_enabled(phase == 'train'):
                    if phase in frozen_features:
                        table, pads = frozen_features[phase]
                        n = batch['device_mask'].shape[1]+1
                        prefix = '_frozen' if cache_readout else '_teacher'
                        batch[prefix+'_memory'] = table.index_select(0,batch['_indices'])[:,:n]
                        batch[prefix+'_padding'] = pads.index_select(0,batch['_indices'])[:,:n]
                    value = objective(model, batch)
                    if value is None:
                        skipped += batch['device_mask'].shape[0]
                        progress.update(step)
                        continue
                    loss, weight = value[:2]
                    if len(value) == 3 and isinstance(value[2], dict):
                        for name, detail in value[2].items():
                            if torch.is_tensor(detail):
                                detail_totals[name] = detail_totals.get(name,0.) + detail.detach().double()*weight
                    elif len(value) == 3:
                        sse, counts = value[2]
                        field_sse = sse.detach().double() if field_sse is None else field_sse+sse.detach().double()
                        field_counts = counts.detach() if field_counts is None else field_counts+counts.detach()
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f'{stage_id} non-finite {phase} loss')
                    if phase == 'train':
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(parameters, options['grad_clip'], error_if_nonfinite=True)
                        optimizer.step()
                        steps += 1
                        examples += batch['device_mask'].shape[0]
                        if after_step:
                            after_step(model)
                total += loss.detach().double() * weight
                count += weight
                progress.update(step, loss.detach())
            if not count:
                raise ValueError(f'{stage_id}: no usable {phase} targets/repeated device groups')
            epoch_loss = total/count
            if field_sse is not None:
                observed = field_counts > 0
                epoch_loss = (field_sse[observed]/field_counts[observed]).mean()
            record[phase] = dict(loss=float(epoch_loss), objective_weight=count, skipped_samples=skipped)
            if detail_totals:
                record[phase]['loss_components'] = {name:float(value/count) for name,value in detail_totals.items()}
        history.append(record)
        print(f"{stage_id}: epoch {epoch} train={record['train']['loss']:.6g} val={record['val']['loss']:.6g}", flush=True)
        if stopping.update(record['val']['loss'], epoch):
            best_state = _cpu_state(model)
            best_extra = copy.deepcopy(checkpoint_extra(model) if checkpoint_extra else {})
            # Preserve useful progress if a long HPC stage is interrupted.
            # This is a selected-weight snapshot, not optimizer-resume support.
            torch.save({'state_dict':best_state,'config':ctx.config,'preprocessing':ctx.prep,
                'stage_metadata':best_extra,'complete':False,
                'selection':{'epoch':epoch,'value':record['val']['loss'],'split':'val'}},
                directory/'best.pt')
        write_json(directory/'history.json',history)
        if stopping.should_stop:
            stopping.announce(stage_id)
            break
    if ctx.device.type == 'cuda':
        torch.cuda.synchronize(ctx.device)
    model.load_state_dict(best_state)
    model.eval()
    cost = dict(stage_id=stage_id, optimizer_steps=steps, examples_seen=examples,
        epochs=len(history), planned_epochs=epochs, seconds=time.perf_counter()-start,
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in parameters), device=str(ctx.device),
        selection=stopping.summary(), selected_state_digest=state_digest(model))
    torch.save({'state_dict': best_state, 'cost': cost, 'config': ctx.config,
                'preprocessing': ctx.prep, 'stage_metadata': best_extra, 'complete':True}, directory / 'best.pt')
    write_json(directory / 'history.json', history)
    write_json(directory / 'cost.json', cost)
    return model, cost


def performance_objective(model, batch):
    mask = batch['performance_mask']
    if not bool(mask.any()):
        return None
    if '_frozen_memory' in batch:
        readout = model.downstream.decoder if isinstance(model, PredictedOPCascade) else model.decoder
        prediction = readout(batch['_frozen_memory'], batch['_frozen_padding'])
    else:
        prediction = model(batch)
    return masked_mse(prediction, batch['performance'].to(prediction.dtype), mask), int(mask.sum())


def op_objective(model, batch):
    prediction, supported = model(batch)
    mask = batch['op_mask'] & supported
    if not bool(mask.any()):
        return None
    sse = torch.where(mask, (prediction.double()-batch['op'].double()).square(),0.).sum((0,1))
    counts = mask.sum((0,1))
    return masked_field_mse(prediction, batch['op'], mask), int(mask.sum()), (sse,counts)


class DistillationPair(nn.Module):
    def __init__(self, student, teacher, student_scaler, teacher_scaler):
        super().__init__()
        self.student, self.teacher = student, freeze(teacher)
        self.student_scaler, self.teacher_scaler = student_scaler, teacher_scaler

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def objective(self, batch):
        if '_teacher_memory' in batch:
            target,padding=batch['_teacher_memory'],batch['_teacher_padding']
        else:
            with torch.no_grad():
                target, padding = self.teacher(op_input(batch))
                target = self.teacher_scaler.transform(target, padding, batch['_group_keys'])
        predicted, student_padding = self.student(design_input(batch))
        if not torch.equal(padding, student_padding):
            raise ValueError('Teacher/student token identity or padding mismatch')
        predicted = self.student_scaler.transform(predicted, padding, batch['_group_keys'])
        valid = batch['device_mask'] & batch['op_target_mask'] & batch['op_mask'].any(-1)
        if not bool(valid.any()):
            return None
        # Deliberately no learned predictor/projection between encoders.
        return (predicted[:, 1:][valid] - target[:, 1:][valid]).square().mean(), int(valid.sum())


def evaluate_model(ctx, model, split, options, *, measured_op=False):
    if not ctx.rows[split]:
        return None
    model.eval()
    predictions, targets, masks, physical = [], [], [], []
    fallback_tokens = 0
    scalers = [getattr(model, name, None) for name in ('scaler','direct_scaler','guided_scaler')]
    with torch.no_grad():
        for batch in ctx.batches(split, options['batch_size'], op=measured_op):
            prediction = model(batch)
            fallback_tokens += sum(s.last_fallback_tokens for s in scalers if s is not None)
            if not bool(torch.isfinite(prediction).all()):
                raise FloatingPointError('Non-finite model prediction')
            predictions.append(prediction.double())
            targets.append(batch['performance'])
            physical.append(batch['physical_performance'])
            masks.append(batch['performance_mask'])
    pred, target, mask, raw = map(torch.cat, (predictions, targets, masks, physical))
    decoded = denormalize(pred, ctx.prep['performance'])
    return dict(split=split, count=len(ctx.rows[split]),
        requires_measured_op=measured_op,
        latent_fallback_tokens=fallback_tokens,
        excluded_untrained_metrics=[n for n, active in zip(ctx.prep['feature_spec']['metric_names'], ctx.metric_mask) if not active],
        normalized_mse=float(masked_mse(pred, target, mask)) if bool(mask.any()) else None,
        metrics=_metrics(decoded, raw, mask, ctx.prep, pred, target))


def evaluate_op(ctx, predictor, split, options):
    """Physical OP diagnostics; truth is used only for scoring after selection."""
    if not ctx.rows[split]:
        return None
    predictor.eval()
    names = ctx.prep['feature_spec']['op_names']
    lookup = {row['sample_id']: row for row in ctx.rows[split]}
    predictions, targets, raw, masks = [], [], [], []
    observed_counts = torch.zeros(len(names), dtype=torch.long, device=ctx.device)
    with torch.no_grad():
        for batch in ctx.batches(split, options['batch_size'], op=True):
            pred, supported = predictor(batch)
            observed_counts += batch['op_mask'].sum((0,1))
            shape = pred.shape
            physical = torch.zeros(shape, dtype=torch.float64, device=ctx.device)
            values = []
            for sample_id in batch['_sample_ids']:
                row=[]
                for dev in sorted(lookup[sample_id]['devices'],key=lambda d:d['device_id']):
                    row.append([dev.get('op',{}).get(name) or 0. for name in names])
                values.append(row)
            physical[:,:len(values[0])] = torch.tensor(values,dtype=torch.float64,device=ctx.device)
            predictions.append(pred.reshape(-1,len(names)).double())
            targets.append(batch['op'].reshape(-1,len(names)).double())
            raw.append(physical.reshape(-1,len(names)))
            masks.append((batch['op_mask'] & supported).reshape(-1,len(names)))
    pred,target,physical,mask = map(torch.cat,(predictions,targets,raw,masks))
    decoded=denormalize(pred,ctx.prep['op'])
    metric_prep={'feature_spec':{'metric_names':names},'performance':ctx.prep['op']}
    scores=_metrics(decoded,physical,mask,metric_prep,pred,target)
    for j,name in enumerate(names):
        scores[name]['supported_scored_count']=int(mask[:,j].sum())
        scores[name]['observed_in_train_supported_columns']=int(observed_counts[j])
        scores[name]['trained_field']=ctx.prep['op']['count'][j]>0
    return dict(split=split,physical_units='canonical OP field units from saved preprocessing',
        inference_uses_true_op=False,truth_used_for='scoring_only',metrics=scores)


def save_final(path, model, ctx, mode, seed, stage_ids, options):
    path.mkdir(parents=True, exist_ok=True)
    payload = dict(format='op_transfer_v1', mode=mode, seed=seed, recipe=describe_model(model),
        state_dict=_cpu_state(model), model_config=ctx.config['model'], preprocessing=ctx.prep,
        stage_ids=stage_ids, selected_sample_ids={k: ctx.selection[k] for k in ('train','val','test','ood','labeled_train')},
        manifest=ctx.full_manifest, options=options,
        inference_requires_op=mode in MEASURED_OP_MODELS)
    torch.save(payload, path / 'checkpoint.pt')


def load_model(path, device='cpu'):
    """Restore a final checkpoint from primitives/tensors, without a teacher."""
    device = require_device(device)
    value = torch.load(path, map_location='cpu', weights_only=True)
    if value.get('format') != 'op_transfer_v1':
        raise ValueError('Not an OP transfer checkpoint')
    model = rebuild_model(value['recipe'], value['model_config'], value['preprocessing']).to(device)
    model.load_state_dict(value['state_dict'])
    model.eval()
    return model, value


def export_summary(output, results, stages):
    groups = defaultdict(list)
    for result in results:
        for metric, values in result['report']['metrics'].items():
            for measure in ('mae', 'rmse', 'r2', 'normalized_rmse'):
                value = values.get(measure)
                if value is not None:
                    groups[(result['mode'], result['report']['split'], metric, measure)].append(value)
    rows = []
    for (mode, split, metric, measure), values in sorted(groups.items()):
        rows.append(dict(model=mode, split=split, metric=metric, measure=measure,
            seeds=len(values), mean=statistics.mean(values),
            std=statistics.stdev(values) if len(values)>1 else None))
    write_json(output / 'summary.json', rows)
    with (output / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model','split','metric','measure','seeds','mean','std'])
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'stages.json', {'unique_stages': stages,
        'optimizer_steps_deduplicated': sum(s['optimizer_steps'] for s in stages),
        'seconds_deduplicated': sum(s['seconds'] for s in stages),
        'note': 'Teacher and OP predictor costs counted once per seed; no equal-compute claim.'})


def run_experiment(data_path, config_path, output, models, seeds, device, manifest_path=None,
                   max_train=None, max_val=None, max_test=None):
    experiment_started = time.perf_counter()
    output = Path(output).resolve()
    # Must precede any write, including config/manifest creation.
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Refusing a nonempty experiment output directory')
    if not models or len(set(models)) != len(models) or set(models)-set(MODELS):
        raise ValueError('Select distinct supported model IDs')
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Seeds must be distinct nonnegative integers')
    for value in (max_train, max_val, max_test):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError('Subset caps must be positive integers')
    config = json.loads(Path(config_path).read_text())
    options = settings(config)
    config['model'] = ModelConfig(**config['model']).to_dict()
    device = require_device(device)
    seed_everything(seeds[0], options['torch_threads'])
    print(f'Loading canonical dataset: {data_path}', flush=True)
    dataset = load_dataset(data_path)
    manifest = (json.loads(Path(manifest_path).read_text()) if manifest_path else
                build_manifest(dataset, config=config['dataset']))
    verify_manifest(dataset, manifest)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'config.json', config)
    write_json(output / 'manifest.json', manifest)
    run = dict(status='running', models=models, seeds=seeds, device=str(device), options=options,
        dataset_fingerprint=dataset_fingerprint(dataset),
        measured_op_at_inference=[mode for mode in models if mode in MEASURED_OP_MODELS])
    write_json(output / 'run.json', run)
    ctx = DataContext(dataset, manifest, config, device,
                      dict(train=max_train,val=max_val,test=max_test), output)
    write_json(output / 'protocol.json', dict(
        teacher='supervised OP-only, exact ID-val best, entire encoder frozen, no EMA',
        student='direct encoder output alignment, no predictor; fixed train-only branch scales',
        cascade='train/val/test/OOD all predicted OP; normalized units shared with downstream encoder',
        student_predicted_op='design+true-OP performance teacher; copy encoder and decoder; frozen teacher/predictor; '
            'shared train-teacher latent scales in alignment loss only; graph/device alignment + performance; '
            'validation selects only unweighted performance MSE; deployed cascade has no teacher',
        student_predicted_op_distill='same teacher/predictor and copied initialization as joint student; '
            'encoder-only alignment followed by frozen-encoder decoder-only performance training; '
            'encoder selects validation alignment (measured OP targets), decoder selects validation performance; '
            'scales fit ID train only; deployed cascade has no teacher or measured OP input',
        op_mask='training field support + topology only; no measured test OP missingness',
        dual='feature concatenation plus linear fusion; matched architecture design-only control',
        scaler='same circuit/topology/device train stats; unseen device uses global train-device stats',
        cohorts='one fixed manifest/subset shared across models/seeds; no test/OOD fitting or selection',
        loss_ablation='grouped replaces pooled regularizers; residual builds on grouped; OP auxiliary builds on pooled',
        compute='separate stage epoch budgets and actual optimizer steps; no equal-compute claim',
        op_training_predictions='in-sample predictions; no cross-fitted/OOF claim',
        inference_diagnostics='real OP allowed only for op_oracle/design_op_oracle references or separate OP scoring'))
    stages, results = [], []
    original_output = ctx.output
    for seed in seeds:
        ctx.output = original_output / f'seed_{seed}'
        ctx.output.mkdir()
        shared, dependencies = {}, {}

        def new_design():
            seed_everything(seed, options['torch_threads'])
            return design_encoder(config['model'], ctx.prep).to(device)

        def head():
            # Selection/order of other models must not change initialization.
            seed_everything(seed+31, options['torch_threads'])
            return decoder(config['model'], ctx.prep).to(device)

        def stage(model, name, budget, objective=performance_objective, **kwargs):
            model, cost = fit_stage(ctx, model, name, options['epochs'][budget], options, seed,
                                    objective, **kwargs)
            cost['stage_id'] = f'seed_{seed}/{name}'
            stages.append(cost)
            return model, cost['stage_id']

        def teacher():
            if 'teacher' not in shared:
                seed_everything(seed, options['torch_threads'])
                model = ReadoutModel(op_encoder(config['model'],ctx.prep),head(),source='op').to(device)
                shared['teacher'], dep = stage(model,'teacher','teacher',op=True)
                dependencies['teacher'] = [dep]
                freeze(shared['teacher'])
            return shared['teacher']

        def op_predictor():
            if 'predictor' not in shared:
                model = OPPredictor(new_design(),ctx.prep,fit_op_support(ctx.rows['train'],ctx.prep)).to(device)
                shared['predictor'], dep = stage(model,'op_predictor','op_predictor',op_objective,op=True)
                dependencies['predictor'] = [dep]
                freeze(shared['predictor'])
                save_final(ctx.output/'op_predictor',shared['predictor'],ctx,'op_predictor',seed,[dep],options)
                for split in ('test','ood'):
                    report=evaluate_op(ctx,shared['predictor'],split,options)
                    if report is not None:
                        write_json(ctx.output/'op_predictor'/f'{split}_op.json',report)
            return shared['predictor']

        def design_op_teacher():
            if 'design_op_teacher' not in shared:
                seed_everything(seed, options['torch_threads'])
                model = ReadoutModel(op_encoder(config['model'], ctx.prep, use_design=True),
                                     head(), source='design_op').to(device)
                shared['design_op_teacher'], dep = stage(model, 'design_op_teacher', 'teacher', op=True)
                dependencies['design_op_teacher'] = [dep]
                freeze(shared['design_op_teacher'])
            return shared['design_op_teacher']

        def student():
            if 'student' not in shared:
                source = teacher().encoder
                before = state_digest(source)
                enc = new_design()
                pair = DistillationPair(enc,copy.deepcopy(source),
                    fit_scaler(ctx,enc,options),fit_scaler(ctx,source,options,'op')).to(device)
                pair, dep = stage(pair,'distillation','student',lambda m,b:m.objective(b),op=True)
                if state_digest(source) != before or state_digest(pair.teacher) != before:
                    raise AssertionError('Frozen teacher changed during distillation')
                shared['student'] = freeze(pair.student)
                dependencies['student'] = dependencies['teacher']+[dep]
                torch.save(dict(student_scaler=pair.student_scaler.state_dict(),
                    teacher_scaler=pair.teacher_scaler.state_dict(),teacher_digest=before,
                    direct_alignment=True,predictor=False),ctx.output/'stages/distillation/scales.pt')
            return shared['student']

        for mode in models:
            print(f'=== seed {seed}: {mode} ===',flush=True)
            deps = []
            if mode == 'op_oracle':
                model = teacher()
                deps = dependencies['teacher'][:]
            elif mode == 'design_op_oracle':
                model = design_op_teacher()
                deps = dependencies['design_op_teacher'][:]
            elif mode == 'design_supervised':
                model, dep = stage(ReadoutModel(new_design(),head()),mode,'readout')
                deps=[dep]
            elif mode.startswith('jepa_'):
                model, deps = _train_jepa(ctx, mode, config, options, seed, stage, new_design, head)
            elif mode in ('op_distill','op_distill_finetune'):
                enc = copy.deepcopy(student())
                scaled = fit_scaler(ctx,enc,options)
                finetune = mode.endswith('finetune')
                enc.requires_grad_(finetune)
                model = ReadoutModel(enc,head(),scaler=scaled,frozen=not finetune)
                model, dep = stage(model,mode,'readout')
                deps=dependencies['student']+[dep]
            elif mode in ('student_predicted_op', 'student_predicted_op_distill'):
                sequential = mode == 'student_predicted_op_distill'
                source = design_op_teacher()
                predictor = copy.deepcopy(op_predictor())
                before_teacher, before_predictor = state_digest(source), state_digest(predictor)
                downstream = copy.deepcopy(source).requires_grad_(True)
                initial_encoder = state_digest(downstream.encoder)
                initial_decoder = state_digest(downstream.decoder)
                scaler = fit_scaler(ctx, source.encoder, options, 'design_op')
                cascade = PredictedOPCascade(predictor, downstream).to(device)
                wrapper = PredictedOPEncoderDistillation if sequential else PredictedOPDistillation
                pair = wrapper(cascade, source, scaler).to(device)
                scales = dict(teacher_scaler=scaler.state_dict(),
                              fit_sample_ids=sorted(ctx.selection['train']),
                              shared_teacher_student_scale=True, loss_only=True)
                def distillation_objective(m, b):
                    return m.objective(b, alignment_weight=options['student_alignment_weight'],
                                       performance_weight=options['student_performance_weight'])
                phase_audit = {}
                if sequential:
                    encoder_stage = 'predicted_op_distill_encoder'
                    decoder_stage = 'predicted_op_distill_decoder'
                    pair, encoder_dep = stage(pair, encoder_stage, encoder_stage, lambda m,b: m.objective(b),
                                              op=True, checkpoint_extra=lambda m: scales)
                    copied_decoder = state_digest(pair.student.downstream.decoder)
                    if copied_decoder != initial_decoder:
                        raise AssertionError('Copied decoder changed during encoder-only distillation')
                    model = pair.student
                    model.downstream.frozen = True
                    freeze(model.downstream.encoder)
                    model.downstream.decoder.requires_grad_(True)
                    model.zero_grad(set_to_none=True)
                    encoder_before_readout = state_digest(model.downstream.encoder)
                    model, decoder_dep = stage(model, decoder_stage, decoder_stage)
                    encoder_after_readout = state_digest(model.downstream.encoder)
                    if encoder_before_readout != encoder_after_readout:
                        raise AssertionError('Distilled encoder changed during decoder-only training')
                    student_deps = [encoder_dep, decoder_dep]
                    phase_audit = dict(
                        encoder_epoch_key=encoder_stage, decoder_epoch_key=decoder_stage,
                        encoder_objective='unweighted mean of device and graph alignment MSE; no performance labels',
                        encoder_selection='validation alignment MSE; measured OP teacher targets; train-fit scales fixed',
                        decoder_objective='masked normalized performance MSE; no teacher or measured OP inputs',
                        decoder_selection='validation performance MSE',
                        decoder_frozen_in_encoder_stage=True, encoder_frozen_in_decoder_stage=True,
                        decoder_digest_after_encoder=copied_decoder,
                        encoder_digest_before_decoder=encoder_before_readout,
                        encoder_digest_after_decoder=encoder_after_readout)
                else:
                    pair, dep = stage(pair, mode, 'student_predicted_op', distillation_objective,
                                      op=True, checkpoint_extra=lambda m: scales)
                    model = pair.student
                    student_deps = [dep]
                after_teacher, after_predictor = state_digest(source), state_digest(pair.student.predictor)
                if before_teacher != after_teacher or before_predictor != after_predictor:
                    raise AssertionError('Frozen design+OP teacher or OP predictor changed during student training')
                deps = dependencies['design_op_teacher'] + dependencies['predictor'] + student_deps
                torch.save(scales, ctx.output / 'stages' / (encoder_stage if sequential else mode) / 'scales.pt')
                audit = dict(teacher_mode='design_op_oracle', teacher_stage_ids=dependencies['design_op_teacher'],
                    predictor_stage_ids=dependencies['predictor'],
                    training_scheme='encoder_then_decoder' if sequential else 'joint_encoder_decoder',
                    student_stage_ids=student_deps,
                    teacher_digest_before=before_teacher, teacher_digest_after=after_teacher,
                    predictor_digest_before=before_predictor, predictor_digest_after=after_predictor,
                    initial_student_encoder_digest=initial_encoder,
                    teacher_encoder_digest=state_digest(source.encoder),
                    initial_student_decoder_digest=initial_decoder,
                    teacher_decoder_digest=state_digest(source.decoder),
                    initialization='copy supervised teacher encoder and decoder; optimizer starts fresh',
                    teacher_frozen=True, predictor_frozen=True, ema=False,
                    alignment_weight=1. if sequential else options['student_alignment_weight'],
                    performance_weight=1. if sequential else options['student_performance_weight'],
                    joint_student_weights_used=not sequential,
                    alignment='equal mean of graph and eligible-device scaled MSE; no trainable projection',
                    eligible_devices='nonpadding and observed/predicted OP field intersection; graph if any eligible device',
                    validation_selection=('encoder: alignment MSE; decoder: performance MSE' if sequential else
                                          'unweighted masked performance MSE; no teacher alignment'),
                    scaler_fit_samples=scaler.fitted_samples,
                    deployment='predicted OP cascade only; no teacher or alignment scaler', **phase_audit)
                (ctx.output / mode).mkdir(parents=True)
                write_json(ctx.output / mode / 'distillation.json', audit)
                del pair
            elif mode in ('predicted_op','predicted_op_joint'):
                predictor=copy.deepcopy(op_predictor())
                seed_everything(seed,options['torch_threads'])
                downstream=ReadoutModel(op_encoder(config['model'],ctx.prep,use_design=True),head(),source='design_op')
                model=PredictedOPCascade(predictor,downstream,joint=mode.endswith('joint')).to(device)
                def joint_objective(m,b):
                    perf=performance_objective(m,b)
                    if perf is None:
                        return None
                    if not m.training:
                        return perf  # performance val-best; OP stays a training anchor
                    opvalue=op_objective(m.predictor,b)
                    return (perf[0]+options['op_aux_weight']*opvalue[0],perf[1]) if opvalue else perf
                model,dep=stage(model,mode,'readout',joint_objective if model.joint else performance_objective,op=model.joint)
                deps=dependencies['predictor']+[dep]
                if model.joint:
                    directory=ctx.output/mode
                    directory.mkdir()
                    for split in ('test','ood'):
                        op_report=evaluate_op(ctx,model.predictor,split,options)
                        if op_report is not None: write_json(directory/f'{split}_op.json',op_report)
            else:
                if mode == 'dual_distill':
                    guided=copy.deepcopy(student()); deps=dependencies['student'][:]
                elif mode == 'dual_predicted_op':
                    guided=copy.deepcopy(op_predictor().encoder); deps=dependencies['predictor'][:]
                else:
                    guided=new_design()
                control=mode=='dual_design_control'
                scaler=None if control else fit_scaler(ctx,guided,options)
                direct=new_design()
                if control:
                    # Equal architecture, independent initialization (not duplicated weights).
                    seed_everything(seed+104729,options['torch_threads'])
                    guided=design_encoder(config['model'],ctx.prep).to(device)
                model=DualEncoderModel(direct,guided,head(),config['model']['dim'],
                    guided_frozen=not control,guided_scaler=scaler).to(device)
                model,dep=stage(model,mode,'dual'); deps.append(dep)
            directory=ctx.output/mode
            save_final(directory,model,ctx,mode,seed,deps,options)
            for split in ('test','ood'):
                report=evaluate_model(ctx,model,split,options,measured_op=mode in MEASURED_OP_MODELS)
                if report is not None:
                    write_json(directory/f'{split}.json',report)
                    results.append(dict(mode=mode,seed=seed,stage_ids=deps,report=report))
            export_summary(output,results,stages)
            # Shared dependencies stay on device; completed nonshared models can be released.
            del model
    ctx.output=original_output
    run['status']='complete'
    run['elapsed_seconds_including_setup_and_evaluation']=time.perf_counter()-experiment_started
    write_json(output/'run.json',run)
    write_json(output/'results.json',results)
    from .summarize import export_results
    try:
        export_results(output/'results.json',output/'summary.xlsx')
    except Exception:
        run['status']='failed_export'
        write_json(output/'run.json',run)
        raise
    return {'output':str(output),'results':results,'stages':stages}


def _train_jepa(ctx, mode, config, options, seed, stage, new_design, head):
    jepa=EMAJEPAModel(new_design(),len(ctx.prep['feature_spec']['op_names']),0,
                       ema_decay=options['ema_decay']).to(ctx.device)
    model=nn.ModuleDict({'jepa':jepa})
    if mode=='jepa_op_aux':
        model['op_head']=nn.Sequential(nn.Linear(config['model']['dim'],config['model']['dim']),
            nn.GELU(),nn.Linear(config['model']['dim'],len(ctx.prep['feature_spec']['op_names']))).to(ctx.device)
    scale_holder={}
    def before_epoch(m):
        if mode=='jepa_residual':
            scale_holder['scaler']=fit_scaler(ctx,m['jepa'].op_encoder,options,'op')
    def objective(m,batch):
        a,b,valid=m['jepa'](batch)
        if int(valid.sum())<2:
            return None
        weights={**config.get('loss',{})}
        weights['variance_floor']=options['variance_floor']
        if mode in ('jepa_grouped','jepa_residual'):
            # The batch sampler guarantees matching circuit/topology/device order.
            if not bool((valid.sum(0)>=2).any()): return None
            if mode=='jepa_residual': weights['invariance']=0.
            loss,detail=grouped_alignment_loss(a,b,valid,batch['_group_keys'],**weights)
            if mode=='jepa_residual':
                # Scaler is frozen for this epoch, fitted exclusively on train.
                pad=torch.cat((torch.zeros_like(valid[:,:1]),~batch['device_mask']),1)
                aa=torch.cat((a.new_zeros(a.shape[0],1,a.shape[-1]),a),1)
                bb=torch.cat((b.new_zeros(b.shape[0],1,b.shape[-1]),b),1)
                residual,_=residual_alignment_loss(aa,bb,pad,batch['_group_keys'],scale_holder['scaler'],valid=valid)
                loss=loss+config.get('loss',{}).get('invariance',25.)*residual
                detail['residual_mse']=residual.detach()
        else:
            loss,detail=alignment_loss(a,b,valid,**weights)
        detail['weighted_alignment']=weights.get('invariance',25.)*detail['alignment_mse']
        detail['weighted_variance']=weights.get('variance',25.)*detail['variance']
        detail['weighted_covariance']=weights.get('covariance',1.)*detail['covariance']
        if 'residual_mse' in detail:
            detail['weighted_residual']=config.get('loss',{}).get('invariance',25.)*detail['residual_mse']
        if mode=='jepa_op_aux':
            aux=masked_field_mse(m['op_head'](a),batch['op'],batch['op_mask'] & valid.unsqueeze(-1))
            loss=loss+options['op_aux_weight']*aux
            detail['op_field_mse']=aux.detach()
            detail['weighted_op_aux']=options['op_aux_weight']*aux.detach()
        return loss,int(valid.sum()),detail
    model,predep=stage(model,mode+'_pretrain','jepa',objective,op=True,
        repeated=mode in ('jepa_grouped','jepa_residual'),
        before_epoch=before_epoch,after_step=lambda m:m['jepa'].update_ema(),
        checkpoint_extra=lambda m: dict(ema_updates_at_selection=m['jepa'].ema_updates,
            loss_variant=mode, variance_floor=options['variance_floor'],
            residual_scaler=scale_holder['scaler'].state_dict() if 'scaler' in scale_holder else None))
    enc=freeze(model['jepa'].encoder)
    scaler=fit_scaler(ctx,enc,options)
    readout=ReadoutModel(enc,head(),scaler=scaler,frozen=True)
    readout,dep=stage(readout,mode+'_readout','readout')
    return readout,[predep,dep]
