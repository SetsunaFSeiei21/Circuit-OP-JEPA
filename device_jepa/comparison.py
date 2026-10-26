"""Strict, dependency-free comparison of matched evaluation reports.

Never pool physical units across metrics or pool validation, test and OOD rows.
Seed summaries are unweighted means/sample SDs of per-seed metrics, not a new
pooled R². Missing labels are excluded; nonfinite predictions are errors.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from .early_stopping import (PRETRAIN_OBJECTIVE, SUPERVISED_OBJECTIVE,
                             early_stopping_config, executed_epochs)
from .methods import (DEFAULT_MODELS, LEGACY_MODES, MODES, OP_MODES, OP_PAIRS, MATCHED_BACKBONES,
                      add_model_arguments, method_spec)
from .data import OP_GEOMETRY_FIELDS


def _information_contract(report):
    """Make extra information and unmeasured acquisition costs explicit.

    Old three-mode reports have an unambiguous no-OP-at-inference contract.
    New OP-input reports must declare their requirement rather than silently
    masquerading as zero-simulation inference.
    """
    p = report['provenance']
    mode = p['mode']
    method = method_spec(mode)
    required = method.requires_op_at_inference
    declared = p.get('requires_op_at_inference', False)
    if type(declared) is not bool or declared != required:
        raise ValueError(f'Incorrect or missing requires_op_at_inference for {mode}')
    uses_design = p.get('uses_design_parameters', True)
    if type(uses_design) is not bool or uses_design != method.uses_design_parameters:
        raise ValueError(f'Incorrect or missing uses_design_parameters for {mode}')
    training_use = 'input' if required else ('pretraining_target' if method.workflow == 'jepa' else 'none')
    inference_use = 'input' if required else 'none'
    info = p.get('op_information', {})
    if not isinstance(info, dict):
        raise ValueError('op_information must be a dictionary')
    for key, value in (('training_use', training_use), ('inference_use', inference_use)):
        if key in info and info[key] != value:
            raise ValueError(f'Incorrect OP {key} for {mode}')
    if required and info.get('state_features'):
        raise ValueError('Continuous OP-input baselines must not consume discrete OP states')
    if not uses_design:
        fields = info.get('feature_names')
        if not isinstance(fields, list) or not fields or any(not isinstance(n, str) for n in fields):
            raise ValueError('OP-only reports must declare their OP input fields')
        if any(n.lower() in OP_GEOMETRY_FIELDS for n in fields):
            raise ValueError('OP-only reports contain design geometry fields in OP inputs')
    if training_use != 'none' and 'scaler_fitted_sample_ids' in info:
        if sorted(info['scaler_fitted_sample_ids']) != sorted(p['train_ids']):
            raise ValueError('OP scaler must be fitted on the shared ID training samples only')
    seconds = info.get('acquisition_cost_seconds')
    status = info.get('acquisition_cost_status', 'not_measured')
    if status not in ('not_measured', 'measured'):
        raise ValueError('OP acquisition cost status must be not_measured or measured')
    if seconds is not None:
        if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0
                or status != 'measured' or not info.get('acquisition_cost_scope')):
            raise ValueError('Measured OP acquisition cost requires finite seconds, measured status and scope')
    elif status == 'measured':
        raise ValueError('Measured OP acquisition cost requires seconds')
    return dict(requires_op_at_inference=required, op_training_use=training_use,
                uses_design_parameters=uses_design,
                op_inference_use=inference_use,
                op_source=info.get('source'), op_feature_names=info.get('feature_names'),
                op_state_features=info.get('state_features'), op_state_config=info.get('state_config'),
                op_eligibility=info.get('eligibility'), op_evaluation_coverage=info.get('evaluation_coverage'),
                op_availability_requirement=('continuous DC OP for every inference sample (missing fields remain masked)'
                                             if required else ('DC OP during JEPA pretraining only; no OP at inference'
                                                               if mode == 'jepa_zerosim' else 'no OP model input')),
                op_acquisition_cost_seconds=seconds, op_acquisition_cost_status=status,
                op_acquisition_cost_scope=info.get('acquisition_cost_scope'))


def regression_metrics(target, prediction):
    if len(target) != len(prediction):
        raise ValueError('Target/prediction length mismatch')
    pairs = []
    for y, p in zip(target, prediction):
        if y is None or not math.isfinite(y):
            continue
        if p is None or not math.isfinite(p):
            raise ValueError('Nonfinite prediction for an observed target')
        pairs.append((float(y), float(p)))
    if not pairs:
        return dict(count=0, mae=None, rmse=None, r2=None)
    ys, ps = zip(*pairs)
    errors = [p-y for y, p in pairs]
    if any(not math.isfinite(e) for e in errors):
        raise ValueError('Physical prediction error exceeds finite reporting range')

    def rms(values):
        scale = max(abs(v) for v in values)
        if scale == 0:
            return 0.
        return scale * math.sqrt(math.fsum((v / scale) ** 2 for v in values) / len(values))

    error_rms = rms(errors)
    r2 = None
    if len(pairs) > 1 and any(y != ys[0] for y in ys[1:]):
        # Power-of-two scaling is exact for adjacent normal floats. Center
        # differences from an anchor, not the large absolute observations:
        # mean([1e16, 1e16+2]) rounds to an endpoint in float64. Also keep the
        # error/variation ratio in scaled units so subnormal physical RMS
        # values need not be representable before computing R².
        exponent = math.frexp(max(abs(y) for y in ys))[1]
        scaled = [math.ldexp(y, -exponent) for y in ys]
        offsets = [y - scaled[0] for y in scaled]
        center = mean(offsets)
        target_rms = rms([y - center for y in offsets])
        try:
            scaled_error_rms = rms([math.ldexp(e, -exponent) for e in errors])
            ratio = scaled_error_rms / target_rms
        except (OverflowError, ZeroDivisionError) as exc:
            raise ValueError('R2 exceeds finite reporting range') from exc
        r2 = 1 - ratio * ratio
        if not math.isfinite(r2):
            raise ValueError('R2 exceeds finite reporting range')
    return dict(count=len(pairs), mae=mean(abs(e) for e in errors),
                rmse=error_rms, r2=r2)



def _identity(report):
    p = report['provenance']
    required = ('partition_fingerprint', 'input_op_fingerprint', 'target_scaler_digest', 'input_transform_digest',
                'labeled_train_ids', 'train_ids')
    for key in required:
        if key not in p:
            raise ValueError(f'Missing comparison provenance: {key}')
    data_id = p.get('dataset_fingerprint', p.get('performance_fingerprint'))
    if not data_id:
        raise ValueError('Missing full dataset/performance fingerprint')
    return (data_id, report.get('data_fingerprint'), *(json.dumps(p[k], sort_keys=True) for k in required),
            tuple(report['metric_names']), bool(report['synthetic']))


def _records(report):
    records = report['predictions']
    if not records:
        raise ValueError('Cannot compare an empty evaluation split')
    ids = [(r['circuit_id'], r['sample_id']) for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate evaluation sample IDs')
    if len(ids) != report['num_samples']:
        raise ValueError('Report sample count does not match predictions')
    metrics = report['metric_names']
    for r in records:
        for key in ('target', 'prediction'):
            if set(r[key]) != set(metrics):
                raise ValueError(f'{key} fields do not match metric_names')
        for metric in metrics:
            y = r['target'][metric]
            ny = r.get('normalized_target', {}).get(metric)
            if y is not None and math.isfinite(y) and (ny is None or not math.isfinite(ny)):
                raise ValueError('Normalized prediction/target missing for observed label')
            if (y is None or not math.isfinite(y)) and ny is not None and math.isfinite(ny):
                raise ValueError('Physical and normalized missing-label masks differ')
    return sorted(records, key=lambda r: (r['circuit_id'], r['sample_id']))


def _executed_protocol_gaps(source):
    """Distinguish incomplete legacy facts from contradictory executed facts."""
    protocol = source['protocol']
    gaps = []
    required = {'protocol_version', 'optimizer', 'training', 'model_config',
                'architecture', 'input_contract_version', 'loss', 'op_state',
                'feature_spec', 'normalization'}
    if required - set(protocol):
        gaps.append('incomplete executed pretraining protocol')
    for name in ('training', 'loss', 'model_config', 'op_state', 'feature_spec', 'normalization'):
        if name in protocol and not isinstance(protocol[name], dict):
            raise ValueError(f'Executed pretraining protocol {name} must be a dictionary')
    for name in ('optimizer', 'architecture'):
        if name in protocol and (not isinstance(protocol[name], str) or not protocol[name].strip()):
            raise ValueError(f'Executed pretraining protocol {name} must be a nonempty string')
    for name in ('protocol_version', 'input_contract_version'):
        if name in protocol and (type(protocol[name]) is not int or protocol[name] < 1):
            raise ValueError(f'Executed pretraining protocol {name} must be a positive integer')
    if protocol.get('protocol_version', 1) != 1:
        gaps.append('unsupported executed pretraining protocol version')
    training_rules = {'pretrain_epochs': 'positive_int', 'batch_size': 'positive_int',
                      'torch_threads': 'positive_int', 'pretrain_lr': 'positive',
                      'weight_decay': 'nonnegative', 'grad_clip': 'positive'}
    loss_rules = {'invariance': 'finite', 'variance': 'positive',
                  'covariance': 'nonnegative', 'variance_floor': 'finite'}
    for section, rules in (('training', training_rules), ('loss', loss_rules)):
        if section not in protocol:
            continue  # Already marked as a missing top-level contract field.
        values = protocol[section]
        if not isinstance(values, dict):
            raise ValueError(f'Executed pretraining protocol {section} must be a dictionary')
        if set(rules) - set(values):
            gaps.append(f'incomplete executed pretraining protocol {section} fields')
        for name, rule in rules.items():
            if name not in values:
                continue
            value = values[name]
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or (rule == 'positive_int' and (type(value) is not int or value < 1))
                    or (rule == 'positive' and value <= 0)
                    or (rule == 'nonnegative' and value < 0)):
                raise ValueError(f'Invalid executed pretraining protocol {section}.{name}')
    # These defaults belong to recorded protocol version 1 / alignment_loss.
    # The producer records effective values; raw source options may legitimately
    # omit optional kwargs, so compare the effective, not merely raw, loss.
    options = source.get('source_options')
    if options is None:
        gaps.append('missing executed pretraining source options')
        return gaps
    if not isinstance(options, dict):
        raise ValueError('Executed pretraining source_options must be a dictionary')
    for name in ('training', 'loss', 'model'):
        if name not in options:
            gaps.append(f'missing executed pretraining source options {name}')
        elif not isinstance(options[name], dict):
            raise ValueError(f'Executed pretraining source options {name} must be a dictionary')
    if isinstance(options.get('training'), dict):
        original = options['training']
        if 'early_stopping' in protocol:
            policy = early_stopping_config({'early_stopping': protocol['early_stopping']})
            if policy != early_stopping_config(original):
                raise ValueError('Executed pretraining stopping protocol differs from source options')
        elif early_stopping_config(original)['enabled']:
            gaps.append('missing executed pretraining stopping protocol')
        actual_training = protocol.get('training', {})
        for name in training_rules:
            if name not in original and name != 'torch_threads':
                gaps.append(f'missing executed pretraining source options training.{name}')
            elif name in actual_training and actual_training[name] != original.get(name, 1):
                raise ValueError('Executed pretraining protocol training differs from source options')
        if 'seed' not in original:
            gaps.append('missing executed pretraining source options training.seed')
        elif original['seed'] != source.get('seed'):
            raise ValueError('Executed pretraining source options seed differs from source provenance')
    if isinstance(options.get('loss'), dict) and 'loss' in protocol:
        effective_loss = dict(invariance=25., variance=25., covariance=1., variance_floor=1.)
        effective_loss.update(options['loss'])
        if protocol['loss'] and effective_loss != protocol['loss']:
            raise ValueError('Executed pretraining protocol loss differs from source options')
    if isinstance(options.get('model'), dict) and 'model_config' in protocol:
        if options['model'] != protocol['model_config']:
            raise ValueError('Executed pretraining protocol architecture differs from source options')
    if 'op_state' in protocol and protocol['op_state'] != options.get('op_state', {'features': []}):
        raise ValueError('Executed pretraining protocol OP state differs from source options')
    return gaps


def _epoch_budget(provenance, gaps):
    """Separate planned maxima from executed epochs and selected weights."""
    mode, config = provenance['mode'], provenance['training_config']
    if mode == 'jepa_zerosim':
        supervised = config.get('decoder_epochs')
        source = provenance.get('pretrain_provenance') or {}
        pretrain = source.get('protocol', {}).get('training', {}).get('pretrain_epochs')
        if pretrain is None:
            gaps.add('jepa_zerosim: missing executed pretraining epoch budget')
    else:
        pretrain = 0
        supervised = config.get('baseline_epochs')
        if supervised is None:
            # Preserve inspection of old reports, without certifying a budget
            # that was silently coupled to the JEPA decoder setting.
            gaps.add(f'{mode}: missing explicit baseline_epochs; legacy epoch budget unverified')
    for key, value in (('pretrain_epochs', pretrain), ('supervised_epochs', supervised)):
        minimum = 0 if key == 'pretrain_epochs' and mode != 'jepa_zerosim' else 1
        if value is not None and (type(value) is not int or value < minimum):
            raise ValueError(f'Invalid {mode} training budget: {key}')
    if mode == 'jepa_zerosim' and supervised is None:
        raise ValueError('Incomplete JEPA decoder training budget')
    planned_pretrain, planned_supervised = pretrain, supervised
    planned_total = pretrain + supervised if pretrain is not None and supervised is not None else None
    if mode == 'jepa_zerosim' and pretrain is not None:
        original = source.get('source_options', {}).get('training')
        if original is None:
            original = dict(source.get('protocol', {}).get('training', {}))
            if 'early_stopping' in source.get('protocol', {}):
                original['early_stopping'] = source['protocol']['early_stopping']
        pretrain = executed_epochs(source.get('execution'), original, pretrain, PRETRAIN_OBJECTIVE)
    if supervised is not None:
        supervised = executed_epochs(provenance.get('early_stopping'), config, supervised, SUPERVISED_OBJECTIVE)
    total = pretrain + supervised if pretrain is not None and supervised is not None else None
    budget = dict(pretrain_epochs=pretrain, supervised_epochs=supervised, total_epochs=total,
                  planned_pretrain_epochs=planned_pretrain, planned_supervised_epochs=planned_supervised,
                  planned_total_epochs=planned_total)
    cost = provenance.get('training_cost', {})
    if not isinstance(cost, dict):
        raise ValueError('training_cost must be a dictionary')
    for key, value in budget.items():
        if key in cost and (type(cost[key]) is not int or cost[key] != value):
            raise ValueError(f'Recorded {key} differs from executed training budget')
    return budget


def compare_reports(reports, require_all_modes=True, expected_modes=None):
    if not reports:
        raise ValueError('No reports supplied')
    supplied_modes = {r['provenance']['mode'] for r in reports}
    # Preserve strict comparison of pre-existing three-mode result sets. Once
    # an original OP-input report is supplied, expect the original five.
    # New cohorts should explicitly declare their selected modes.
    if expected_modes is None:
        expected_modes = (LEGACY_MODES if supplied_modes <= set(LEGACY_MODES) else
                          DEFAULT_MODELS if supplied_modes <= set(DEFAULT_MODELS) else MODES)
    expected_modes = tuple(expected_modes)
    if not expected_modes or len(expected_modes) != len(set(expected_modes)) or any(m not in MODES for m in expected_modes):
        raise ValueError('Expected modes must be a nonempty unique selection of supported modes')
    if not supplied_modes <= set(expected_modes):
        raise ValueError('Reports contain modes outside the declared comparison cohort')
    identity = _identity(reports[0])
    cohorts, seen, per_seed = {}, set(), []
    run_costs, evaluation_costs, acquisition_costs = {}, [], {}
    mode_seed_runs, pretraining_protocols, provenance_gaps = {}, {}, set()
    mode_seeds = defaultdict(set)
    backends, mode_configs, architectures, mode_op_protocols = {}, {}, {}, {}
    tensor_pipeline = None
    stopping_policy, has_early_stopping = None, False
    optimizer_budget, total_epoch_budget, op_transform, selection_metrics = None, None, None, None
    for report in reports:
        if _identity(report) != identity:
            raise ValueError('Incompatible data, split, labels, metric names or target preprocessing')
        p = report['provenance']
        pipeline = p.get('tensor_pipeline', 'legacy_scalar')
        if not isinstance(pipeline, str) or not pipeline:
            raise ValueError('Invalid tensor pipeline provenance')
        if tensor_pipeline is not None and pipeline != tensor_pipeline:
            raise ValueError('Tensor pipeline differs across reports; rerun the complete experiment with one implementation')
        tensor_pipeline = pipeline
        mode, seed, split = p['mode'], p['seed'], report['split']
        if split not in ('test', 'ood', 'val', 'train'):
            raise ValueError('Unknown evaluation split')
        if mode not in MODES:
            raise ValueError(f'Unknown mode: {mode}')
        run_identity = p.get('run_identity')
        if run_identity is not None and (not isinstance(run_identity, str) or not run_identity.strip()):
            raise ValueError('run_identity must be a nonempty string')
        if run_identity is None:
            provenance_gaps.add(f'{mode}: missing run_identity; legacy costs cannot be deduplicated reliably')
        else:
            run_key = mode, seed
            if run_key in mode_seed_runs and mode_seed_runs[run_key] != run_identity:
                raise ValueError('Different checkpoint run identities for the same mode/seed across splits')
            mode_seed_runs[run_key] = run_identity
        if mode == 'jepa_zerosim':
            protocol = p.get('pretraining_protocol_digest')
            source = p.get('pretrain_provenance')
            if source is None:
                provenance_gaps.add('jepa_zerosim: missing executed pretraining provenance')
            elif not isinstance(source, dict) or not isinstance(source.get('protocol'), dict):
                raise ValueError('Invalid executed pretraining provenance')
            else:
                provenance_gaps.update('jepa_zerosim: ' + gap for gap in _executed_protocol_gaps(source))
                actual_digest = hashlib.sha256(json.dumps(source['protocol'], sort_keys=True,
                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()
                if protocol != actual_digest or source.get('protocol_digest') != actual_digest:
                    raise ValueError('Pretraining protocol digest does not match executed provenance')
                if source.get('seed') != p.get('pretrain_seed'):
                    raise ValueError('Pretraining seed differs from executed provenance')
                for field in ('input_op_fingerprint', 'partition_fingerprint'):
                    if source.get(field) != p[field]:
                        raise ValueError('Pretraining data identity differs from evaluation provenance')
            if protocol is None:
                provenance_gaps.add('jepa_zerosim: missing verified pretraining protocol')
            elif not isinstance(protocol, str) or not protocol.strip():
                raise ValueError('pretraining_protocol_digest must be a nonempty string')
            else:
                if mode in pretraining_protocols and pretraining_protocols[mode] != protocol:
                    raise ValueError('Actual pretraining protocol differs across seeds or splits within a mode')
                pretraining_protocols[mode] = protocol
            for field in ('pretrain_seed', 'decoder_seed'):
                if p.get(field) is None:
                    provenance_gaps.add(f'jepa_zerosim: missing {field}')
                elif type(p[field]) is not int:
                    raise ValueError(f'{field} must be an integer')
            if p.get('decoder_seed', seed) != seed:
                raise ValueError('Report seed differs from the recorded decoder seed')
        selected_metrics = p.get('selection_metric_names')
        if selected_metrics is None:
            provenance_gaps.add(f'{mode}: missing checkpoint-selection metric names')
        else:
            if (not isinstance(selected_metrics, list) or not selected_metrics
                    or any(not isinstance(name, str) for name in selected_metrics)
                    or len(selected_metrics) != len(set(selected_metrics))
                    or not set(selected_metrics) <= set(report['metric_names'])):
                raise ValueError('Invalid checkpoint-selection metric names')
            if selection_metrics is not None and selection_metrics != selected_metrics:
                raise ValueError('Checkpoint-selection metrics differ across reports')
            selection_metrics = selected_metrics
        information = _information_contract(report)
        # Continuous OP scaling alone does not identify the JEPA target branch:
        # categorical state channels/configuration may change its input width
        # while leaving the structure encoder and raw dataset unchanged.
        op_protocol = {k: information[k] for k in ('op_source', 'op_feature_names',
                       'op_state_features', 'op_state_config', 'op_eligibility')}
        if mode in mode_op_protocols and mode_op_protocols[mode] != op_protocol:
            raise ValueError('OP information protocol differs across seeds or splits within a mode')
        mode_op_protocols[mode] = op_protocol
        if mode in OP_MODES and not p.get('op_input_transform_digest'):
            raise ValueError('Missing continuous OP input scaler provenance')
        if information['op_training_use'] != 'none' and p.get('op_input_transform_digest'):
            if op_transform is not None and op_transform != p['op_input_transform_digest']:
                raise ValueError('Incompatible continuous OP input scalers')
            op_transform = p['op_input_transform_digest']
        backend = p.get('backend')
        if not backend:
            raise ValueError('Missing backend provenance')
        if mode in backends and backends[mode] != backend:
            raise ValueError('Cannot mix backend implementations within a mode')
        backends[mode] = backend
        config = p.get('training_config')
        if not isinstance(config, dict):
            raise ValueError('Missing training_config provenance')
        policy = early_stopping_config(config)
        if stopping_policy is not None and stopping_policy != policy:
            raise ValueError('Incompatible supervised early-stopping policies across models')
        stopping_policy = policy
        has_early_stopping |= policy['enabled']
        keys = ('decoder_lr', 'batch_size', 'weight_decay', 'grad_clip')
        if any(k not in config for k in keys):
            raise ValueError('Incomplete supervised training budget')
        budget = tuple(config[k] for k in keys)
        if optimizer_budget is not None and optimizer_budget != budget:
            raise ValueError('Incompatible supervised training budget')
        optimizer_budget = budget
        epochs = _epoch_budget(p, provenance_gaps)
        execution = (p.get('pretrain_provenance') or {}).get('execution')
        if execution is not None:
            has_early_stopping |= execution['config']['enabled']
        if epochs['planned_total_epochs'] is not None:
            if total_epoch_budget is not None and total_epoch_budget != epochs['planned_total_epochs']:
                raise ValueError('Incompatible total training budget: JEPA pretraining + decoder epochs '
                                 'must equal baseline_epochs')
            total_epoch_budget = epochs['planned_total_epochs']
        mode_config = {k: v for k, v in config.items() if k != 'seed'}
        if mode in mode_configs and mode_configs[mode] != mode_config:
            raise ValueError('Training configuration differs across seeds within a mode')
        mode_configs[mode] = mode_config
        architecture = p.get('model_config')
        if not isinstance(architecture, dict):
            raise ValueError('Missing model_config provenance')
        if mode in architectures and architectures[mode] != architecture:
            raise ValueError('Model configuration differs across seeds within a mode')
        architectures[mode] = architecture
        key = mode, seed, split
        if key in seen:
            raise ValueError(f'Duplicate mode/seed/split: {key}')
        seen.add(key)
        mode_seeds[split, mode].add(seed)
        records = _records(report)
        targets = [(r['circuit_id'], r['sample_id'], r['target'], r.get('normalized_target')) for r in records]
        if split in cohorts and cohorts[split] != targets:
            raise ValueError('Evaluation IDs or targets differ within the same split')
        cohorts[split] = targets
        scopes = [('pooled', '__all__')] + [('circuit', c) for c in sorted({r['circuit_id'] for r in records})]
        for scope, circuit in scopes:
            subset = [r for r in records if scope == 'pooled' or r['circuit_id'] == circuit]
            for metric in report['metric_names']:
                for scale, ykey, pkey in [('physical', 'target', 'prediction'),
                                          ('normalized', 'normalized_target', 'normalized_prediction')]:
                    if scale == 'normalized' and not all(ykey in r and pkey in r for r in subset):
                        raise ValueError('Normalized prediction/target records are required for comparable metrics')
                    scores = regression_metrics([r[ykey][metric] for r in subset], [r[pkey][metric] for r in subset])
                    per_seed.append(dict(mode=mode, backend=p.get('backend', 'unspecified'), seed=seed,
                                         split=split, scope=scope, circuit=circuit, metric=metric, scale=scale,
                                         requires_op_at_inference=information['requires_op_at_inference'],
                                         uses_design_parameters=information['uses_design_parameters'],
                                         op_training_use=information['op_training_use'], **scores))
        # Training belongs to a checkpoint run, whereas coverage and OP
        # acquisition belong to the evaluation or an explicitly stated scope.
        # A legacy report has no trustworthy run identity: keep its costs
        # separate and mark the comparison non-strict instead of guessing.
        cost_key = ('verified', run_identity) if run_identity is not None else ('legacy', mode, seed, split)
        fixed_information = {k: v for k, v in information.items()
                             if k not in ('op_evaluation_coverage', 'op_acquisition_cost_seconds',
                                          'op_acquisition_cost_status', 'op_acquisition_cost_scope')}
        training_cost = p.get('training_cost', {})
        if not isinstance(training_cost, dict):
            raise ValueError('training_cost must be a dictionary')
        cost = dict(mode=mode, seed=seed, run_identity=run_identity,
                    identity_status='verified' if run_identity is not None else 'legacy_unverified',
                    pretrain_seed=p.get('pretrain_seed'), decoder_seed=p.get('decoder_seed', seed),
                    pretraining_protocol_digest=p.get('pretraining_protocol_digest'),
                    timing=p.get('timing', {}), parameter_count=p.get('parameter_count'),
                    training_cost=training_cost,
                    early_stopping=p.get('early_stopping'), pretraining_execution=execution,
                    pretrain_optimizer_steps=training_cost.get('pretrain_optimizer_steps'),
                    supervised_optimizer_steps=training_cost.get('supervised_optimizer_steps'),
                    **epochs,
                    **fixed_information)
        if cost_key in run_costs:
            previous = {k: v for k, v in run_costs[cost_key].items() if k != 'evaluation_splits'}
            if previous != cost:
                raise ValueError('Conflicting training costs or provenance for the same run identity')
        else:
            run_costs[cost_key] = {**cost, 'evaluation_splits': []}
        run_costs[cost_key]['evaluation_splits'].append(split)
        evaluation_costs.append(dict(mode=mode, seed=seed, split=split, run_identity=run_identity,
                                     **information))
        if information['op_acquisition_cost_status'] == 'measured':
            acquisition = dict(mode=mode, seed=seed, run_identity=run_identity,
                               scope=information['op_acquisition_cost_scope'],
                               seconds=information['op_acquisition_cost_seconds'], status='measured')
            acquisition_key = cost_key, json.dumps(acquisition['scope'], sort_keys=True)
            if acquisition_key in acquisition_costs:
                previous = {k: v for k, v in acquisition_costs[acquisition_key].items() if k != 'evaluation_splits'}
                if previous != acquisition:
                    raise ValueError('Conflicting OP acquisition cost for the same run and acquisition scope')
            else:
                acquisition_costs[acquisition_key] = {**acquisition, 'evaluation_splits': []}
            acquisition_costs[acquisition_key]['evaluation_splits'].append(split)
    # Device supervised and JEPA must differ in optimization, not architecture.
    # The pin reproduction may differ only in node-budget and terminal-role bias.
    if 'zerosim_device' in architectures and 'jepa_zerosim' in architectures:
        if architectures['zerosim_device'] != architectures['jepa_zerosim']:
            raise ValueError('Model configuration differs between device and JEPA modes')
    for _, baseline_mode, op_mode in OP_PAIRS:
        if baseline_mode in architectures and op_mode in architectures and architectures[baseline_mode] != architectures[op_mode]:
            raise ValueError(f'Model configuration differs within OP/no-OP pair {baseline_mode}/{op_mode}')
    for reference, ablation in MATCHED_BACKBONES:
        if reference in architectures and ablation in architectures and architectures[reference] != architectures[ablation]:
            raise ValueError(f'Model configuration differs within design-input ablation {reference}/{ablation}')
    if 'zerosim_device_op' in architectures and 'jepa_zerosim' in architectures:
        if architectures['zerosim_device_op'] != architectures['jepa_zerosim']:
            raise ValueError('Model configuration differs between device OP and JEPA modes')
    common_architectures = [{k: v for k, v in cfg.items() if k not in ('max_devices', 'terminal_bias')}
                            for cfg in architectures.values()]
    if any(cfg != common_architectures[0] for cfg in common_architectures[1:]):
        raise ValueError('Model configuration differs across comparison modes')
    if require_all_modes:
        for split in cohorts:
            sets = [mode_seeds[split, mode] for mode in expected_modes]
            if not sets[0] or any(s != sets[0] for s in sets):
                raise ValueError(f'All declared modes must have matching seed sets for {split}: {dict(zip(expected_modes, sets))}')
    groups = defaultdict(list)
    for row in per_seed:
        groups[tuple(row[k] for k in ('mode', 'backend', 'split', 'scope', 'circuit', 'metric', 'scale'))].append(row)
    summary = []
    for key, rows in sorted(groups.items()):
        item = dict(zip(('mode', 'backend', 'split', 'scope', 'circuit', 'metric', 'scale'), key))
        item.update(num_seeds=len(rows), count_per_seed=rows[0]['count'],
                    requires_op_at_inference=rows[0]['requires_op_at_inference'],
                    uses_design_parameters=rows[0]['uses_design_parameters'],
                    op_training_use=rows[0]['op_training_use'])
        for metric in ('mae', 'rmse', 'r2'):
            values = [r[metric] for r in rows if r[metric] is not None]
            item[metric+'_defined_seeds'] = len(values)
            item[metric+'_mean'] = mean(values) if values else None
            item[metric+'_std'] = stdev(values) if len(values)>1 else None
        summary.append(item)
    paired_per_seed, paired_summary = _paired_deltas(per_seed)
    cohort_complete = all(mode_seeds[split, expected_modes[0]] and
                   all(mode_seeds[split, mode] == mode_seeds[split, expected_modes[0]] for mode in expected_modes)
                   for split in cohorts)
    for cost in (*run_costs.values(), *acquisition_costs.values()):
        cost['evaluation_splits'].sort()
    return dict(schema_version=5, synthetic=reports[0]['synthetic'],
                tensor_pipeline=tensor_pipeline,
                epoch_budget_policy='equal_maximum_with_early_stopping' if has_early_stopping else 'equal_total',
                total_epoch_budget=total_epoch_budget, max_total_epochs=total_epoch_budget,
                supervised_early_stopping=stopping_policy,
                expected_modes=list(expected_modes), compared_modes=[m for m in MODES if m in supplied_modes],
                complete=cohort_complete and not provenance_gaps, cohort_complete=cohort_complete,
                strict_provenance=not provenance_gaps, provenance_gaps=sorted(provenance_gaps),
                per_seed=per_seed, summary=summary, run_costs=list(run_costs.values()),
                evaluation_costs=evaluation_costs, op_acquisition_costs=list(acquisition_costs.values()),
                paired_per_seed=paired_per_seed, paired_summary=paired_summary,
                warning='Smoke/toy results validate mechanics only, not generalization. R2 undefined for fewer than two observations or constant targets. Physical units are dataset units. Normalized metrics use training-only target transforms. scope distinguishes pooled results from individual circuits, even a circuit named __all__. Training costs are deduplicated by verified run identity; legacy unverified rows remain separate and prevent strict completeness. Evaluation coverage is split-specific. Measured OP acquisition costs are deduplicated only within the same run and explicit acquisition scope; scopes may overlap and are not assumed additive. OP-input baselines require extra DC OP information at inference, unlike no-OP baselines and JEPA (training-only OP). OP acquisition/simulation cost is unknown unless explicitly measured, never assumed zero or included in model phase timing. Maximum total-epoch parity counts JEPA pretraining plus decoder maxima against each baseline. Independent validation early stopping may produce different actual totals, recorded separately with best epochs and stopping reasons; it does not imply equal sample exposure, information, parameters or total compute. Paired deltas are OP minus no-OP: negative MAE/RMSE and positive R2 favor OP; these are descriptive, not significance tests.')


def _paired_deltas(per_seed):
    """Within-seed contrasts; never subtract unpaired across-seed means."""
    index = {(r['mode'], *(r[k] for k in ('seed', 'split', 'scope', 'circuit', 'metric', 'scale'))): r for r in per_seed}
    paired = []
    for granularity, baseline_mode, op_mode in OP_PAIRS:
        for row in per_seed:
            if row['mode'] != baseline_mode:
                continue
            key = tuple(row[k] for k in ('seed', 'split', 'scope', 'circuit', 'metric', 'scale'))
            op = index.get((op_mode, *key))
            if op is None:
                continue
            item = dict(node_granularity=granularity, baseline_mode=baseline_mode, op_mode=op_mode,
                        **dict(zip(('seed', 'split', 'scope', 'circuit', 'metric', 'scale'), key)),
                        count=row['count'], baseline_requires_op_at_inference=False, op_requires_op_at_inference=True)
            for metric in ('mae', 'rmse', 'r2'):
                baseline_value, op_value = row[metric], op[metric]
                item[f'no_op_{metric}'], item[f'op_{metric}'] = baseline_value, op_value
                delta = op_value - baseline_value if baseline_value is not None and op_value is not None else None
                if delta is not None and not math.isfinite(delta):
                    raise ValueError('Paired delta exceeds finite reporting range')
                item[f'{metric}_delta'] = delta
            paired.append(item)
    groups = defaultdict(list)
    fields = ('node_granularity', 'baseline_mode', 'op_mode', 'split', 'scope', 'circuit', 'metric', 'scale')
    for row in paired:
        groups[tuple(row[k] for k in fields)].append(row)
    summary = []
    for key, rows in sorted(groups.items()):
        item = dict(zip(fields, key))
        item.update(num_paired_seeds=len(rows), count_per_seed=rows[0]['count'])
        for metric in ('mae', 'rmse', 'r2'):
            values = [r[f'{metric}_delta'] for r in rows if r[f'{metric}_delta'] is not None]
            for prefix in ('no_op', 'op'):
                scores = [r[f'{prefix}_{metric}'] for r in rows if r[f'{prefix}_{metric}'] is not None]
                item[f'{prefix}_{metric}_mean'] = mean(scores) if scores else None
            item[f'{metric}_delta_defined_seeds'] = len(values)
            item[f'{metric}_delta_mean'] = mean(values) if values else None
            item[f'{metric}_delta_std'] = stdev(values) if len(values) > 1 else None
        summary.append(item)
    return paired, summary


def _csv(path, rows):
    if not rows:
        # Reprocessing a legacy cohort into an existing comparison directory
        # must not leave stale OP-pair files from an earlier five-mode cohort.
        path.unlink(missing_ok=True)
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-partial', action='store_true', help='Allow missing modes/seeds; still enforce provenance')
    add_model_arguments(parser, comparison=True)
    args = parser.parse_args(argv)
    result = compare_reports([json.loads(Path(p).read_text()) for p in args.reports], not args.allow_partial, args.models)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    _csv(out/'per_seed.csv', result['per_seed'])
    _csv(out/'summary.csv', result['summary'])
    _csv(out/'paired_per_seed.csv', result['paired_per_seed'])
    _csv(out/'paired_summary.csv', result['paired_summary'])
    for filename, rows in (('costs.csv', result['run_costs']), ('evaluation_costs.csv', result['evaluation_costs']),
                           ('op_acquisition_costs.csv', result['op_acquisition_costs'])):
        _csv(out/filename, [{k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                            for k, v in r.items()} for r in rows])
    (out/'README.txt').write_text(result['warning']+'\nsummary.csv: one row per mode/backend/split/scope/circuit/metric/scale. Across-seed SD uses ddof=1; blank with one seed. Never compare different units or different splits as one score.\npaired_per_seed.csv and paired_summary.csv: only matched seeds within pin/device OP/no-OP pairs. Deltas are OP minus no-OP. Missing/undefined values are blank, not zero. These files are omitted if no pairs exist.\ncosts.csv: training costs per verified run, with optimizer steps. evaluation_costs.csv: split coverage and declared OP acquisition metadata. op_acquisition_costs.csv: measured OP costs by run and explicit scope; absent when unmeasured. complete requires both cohort_complete and strict_provenance. Legacy reports remain readable but provenance_gaps identify why they cannot pass strict completeness.\n')
    print(f'Wrote matched comparison to {out}')


if __name__ == '__main__':
    main()
