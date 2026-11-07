#!/usr/bin/env python3
"""Matched pin-level single-circuit experiments with independently saved stages."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_NAMES_PIN = ('mlp_pin', 'xgboost_pin', 'gat_pin', 'deepgen_supervised_pin', 'deepgen_op_pretrained_pin',
               'gtn_pin', 'zerosim_supervised_pin', 'accformer_pin', 'predicted_op_pin',
               'student_predicted_op_pin', 'student_predicted_op_distill_pin',
               'op_oracle_pin', 'design_op_oracle_pin')
GATE_MODELS_PIN = ('device_op_cascade_pin', 'device_op_gate_pin',
                  'device_op_gate_oracle_pin', 'device_op_gate_shuffle_pin',
                  'device_op_gate_only_base_pin', 'device_op_gate_only_pin',
                  'device_op_gate_only_oracle_pin', 'device_op_gate_only_shuffle_pin')
MODEL_NAMES_PIN += GATE_MODELS_PIN
CASCADE_MODELS_PIN = frozenset({'predicted_op_pin', 'student_predicted_op_pin', 'student_predicted_op_distill_pin'})
REFERENCES_PIN = frozenset({'op_oracle_pin', 'design_op_oracle_pin',
                           'device_op_gate_oracle_pin', 'device_op_gate_only_oracle_pin'})


def validate_request_pin(config, models, seeds, device, smoke=False):
    from device_jepa.model import ModelConfig
    from device_jepa.resident import require_device
    from experiments.op_transfer.engine import settings
    config = copy.deepcopy(config)
    if 'final_exam' in config:
        raise ValueError('Use a pin configuration with final_exam_pin; device configuration is not interchangeable')
    extra = config.setdefault('final_exam_pin', {})
    if set(extra) - {'models', 'model_overrides', 'xgboost'}:
        raise ValueError('Unknown final_exam configuration keys')
    selected = list(extra.get('models', MODEL_NAMES_PIN) if models is None else models)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(MODEL_NAMES_PIN):
        raise ValueError('Select distinct model names from --list-models')
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError('Seeds must be unique integers in [0, 2**32)')
    data = config['dataset']
    if len(data.get('train_circuits', [])) != 1 or data.get('ood_circuits'):
        raise ValueError('Phase one requires exactly one train circuit and no OOD circuit')
    overrides = extra.get('model_overrides', {})
    if set(overrides) - set(MODEL_NAMES_PIN[:8]):
        raise ValueError('Architecture overrides use baseline IDs; cascades share zerosim_supervised settings')
    if smoke:
        config['op_transfer'] = settings(config)
        config['op_transfer']['epochs'] = dict.fromkeys(config['op_transfer']['epochs'], 1)
        config['op_transfer']['epochs'].update(readout=2, student_predicted_op=2)
        config['op_transfer']['patience'] = 0
        extra.setdefault('xgboost', {})['num_boost_round'] = 2
    config['model'] = ModelConfig(**config['model']).to_dict()
    for override in overrides.values():
        ModelConfig(**{**config['model'], **override})
    options = settings(config)
    if set(selected) & set(GATE_MODELS_PIN) or 'op_gate_pin' in config:
        from final_exam.models.op_gate_pin import validate_gate_config_pin
        config['op_gate_pin'] = validate_gate_config_pin(config.get('op_gate_pin'))
    epochs = options['epochs']
    if not epochs['readout'] == epochs['student_predicted_op'] == (
            epochs['predicted_op_distill_encoder'] + epochs['predicted_op_distill_decoder']):
        raise ValueError('Matched downstream budgets require readout == student_predicted_op '
                         '== predicted_op_distill_encoder + predicted_op_distill_decoder')
    require_device(device)
    if 'xgboost_pin' in selected:
        if importlib.util.find_spec('xgboost') is None:
            raise ValueError('Selected xgboost requires pip install -r final_exam/requirements.txt')
        from final_exam.tree_pin import validate_tree_config_pin
        validate_tree_config_pin(config)
    return config, selected


def effective_config_pin(config, mode, seed):
    value = copy.deepcopy(config)
    key = 'zerosim_supervised_pin' if mode in (CASCADE_MODELS_PIN | REFERENCES_PIN | set(GATE_MODELS_PIN)
        | {'op_predictor', 'op_predictor_pin', 'device_op_predictor_pin'}) else mode
    override = value.get('final_exam_pin', {}).get('model_overrides', {}).get(key, {})
    value['model'].update(override)
    value.setdefault('training', {})['seed'] = seed
    return value


def gate_contract_pin(mode):
    """Explicit inference inputs for the two paired gate studies."""
    if mode not in GATE_MODELS_PIN:
        raise ValueError('Not a device-OP gate study model')
    features = '_only_' not in mode and mode != 'device_op_gate_only_pin'
    source = ('none' if mode in ('device_op_cascade_pin', 'device_op_gate_only_base_pin') else
              'true_op' if mode in REFERENCES_PIN else
              'shuffled_predicted_op' if '_shuffle_' in mode else 'predicted_op')
    return dict(group='A_features_and_gate' if features else 'B_gate_only',
                uses_predicted_op_features=features, gate_source=source,
                predictor_granularity='none' if mode == 'device_op_gate_only_base_pin' else 'device',
                requires_measured_op=mode in REFERENCES_PIN,
                attention_scope='all_encoder_self_attention' if source != 'none' else 'none',
                diagnostic_control=source in ('true_op', 'shuffled_predicted_op'))


def oracle_gate_coverage_pin(ctx, predictor, split, batch_size):
    """Count true OP used only by the explicitly measured-OP gate references."""
    import torch
    from final_exam.op_gate_data_pin import device_input_pin
    observed = torch.zeros((), dtype=torch.long, device=ctx.device)
    supported_count = torch.zeros_like(observed)
    with torch.no_grad():
        for batch in ctx.batches(split, batch_size, op=True):
            support = predictor.topology_mask(device_input_pin(batch))
            actual = (support & batch['owner_op_mask'] & batch['owner_op_target_mask'].unsqueeze(-1)
                      & torch.isfinite(batch['owner_op']))
            observed += actual.sum()
            supported_count += support.sum()
    count, denominator = int(observed), int(supported_count)
    return dict(observed_supported_fields=count, supported_fields=denominator,
                fraction=count / denominator if denominator else None,
                missing_policy='fill from the same frozen predicted OP; no additional missingness feature')


def save_checkpoint_pin(directory, model, ctx, mode, seed, dependencies, options, config, kind='op_transfer'):
    import torch
    from experiments.op_transfer.engine import _cpu_state, write_json
    from final_exam.models.op_transfer_pin import describe_model_pin
    mode = mode if mode.endswith('_pin') else mode + '_pin'
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(format='final_exam_pin_v1', granularity='pin', kind=kind, mode=mode, seed=seed,
                   state_dict=_cpu_state(model), model_config=config['model'], config=config,
                   preprocessing=ctx.prep, manifest=ctx.full_manifest,
                   selected_sample_ids=ctx.selection, stage_ids=dependencies, options=options,
                   trainable_parameters=[name for name, parameter in model.named_parameters()
                                         if parameter.requires_grad],
                   requires_measured_op=mode in REFERENCES_PIN)
    if kind == 'op_transfer':
        payload['recipe'] = describe_model_pin(model)
    elif kind == 'deepgen_op_predictor':
        payload['support'] = model.support_metadata
    else:
        payload['factory_name'] = 'deepgen_supervised_pin' if mode == 'deepgen_op_pretrained_pin' else mode
    if mode in GATE_MODELS_PIN:
        payload['input_contract_pin'] = gate_contract_pin(mode)
    torch.save(payload, directory / 'checkpoint_pin.pt')
    write_json(directory / 'config_pin.json', config)
    write_json(directory / 'dependencies_pin.json', dict(stage_ids=dependencies))


def load_checkpoint_pin(path, device='cpu'):
    import torch
    from device_jepa.resident import require_device
    from final_exam.models.op_transfer_pin import OPPredictorPin, rebuild_model_pin
    from final_exam.models.models_pin import build_encoder_pin, build_model_pin
    path = Path(path)
    if path.suffix == '.json':
        from final_exam.tree_pin import load_xgboost_pin
        return load_xgboost_pin(path, device)
    value = torch.load(path, map_location='cpu', weights_only=True)
    if value.get('format') != 'final_exam_pin_v1':
        raise ValueError('Expected a final_exam checkpoint')
    cfg, prep = value['model_config'], value['preprocessing']
    if value['kind'] == 'op_transfer':
        model = rebuild_model_pin(value['recipe'], cfg, prep)
    elif value['kind'] == 'deepgen_op_predictor':
        model = OPPredictorPin(build_encoder_pin('deepgen_supervised_pin', cfg, prep), prep, value['support'], encoder_name='deepgen_supervised_pin')
    elif value['kind'] == 'baseline':
        model = build_model_pin(value['factory_name'], cfg, prep)
    else:
        raise ValueError('Unknown checkpoint kind')
    model.load_state_dict(value['state_dict'], strict=True)
    if 'trainable_parameters' in value:
        trainable = set(value['trainable_parameters'])
        if trainable - set(dict(model.named_parameters())):
            raise ValueError('Checkpoint gradient metadata does not match the architecture')
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in trainable)
    model.to(require_device(device)).eval()
    return model, value


def _revision():
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
                            capture_output=True, check=False)
    return result.stdout.strip() or 'unavailable'


def run_experiment_pin(data_path, config_path, output, models, seeds, device, *, log_dir=None,
                   manifest_path=None, smoke=False):
    import torch
    from device_jepa.data import load_dataset, build_manifest, verify_manifest, dataset_fingerprint
    from device_jepa.resident import require_device
    from device_jepa.training import _check_dataset_options, seed_everything, state_digest
    from experiments.op_transfer import engine
    from final_exam.data_pin import PinDataContext
    from final_exam.models.op_transfer_pin import op_objective_pin, op_input_pin
    from final_exam.models.op_transfer_pin import PredictedOPDistillationPin, PredictedOPEncoderDistillationPin
    from final_exam.models.op_transfer_pin import (OPPredictorPin, PredictedOPCascadePin, ReadoutModelPin,
        decoder_pin, design_encoder_pin, fit_op_support_pin, freeze, op_encoder_pin)
    from final_exam.logging_utils import log_to
    from final_exam.models.models_pin import build_encoder_pin, build_model_pin, prepare_model_pin
    from final_exam.models.op_gate_pin import DeviceOPPredictorPin, DeviceOPGateModelPin
    from final_exam.op_gate_data_pin import prepare_device_predictions_pin, build_shuffle_bank_pin

    output = Path(output).resolve()
    if not output.name.endswith('_pin'):
        raise ValueError('Pin output directory must end with _pin')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Choose a new/empty output directory')
    config, models = validate_request_pin(json.loads(Path(config_path).read_text()), models, seeds, device, smoke)
    device = require_device(device)
    options = engine.settings(config)
    seed_everything(seeds[0], options['torch_threads'])
    dataset = load_dataset(data_path)
    circuit = config['dataset']['train_circuits'][0]
    if {row['circuit_id'] for row in dataset['samples']} != {circuit}:
        raise ValueError('Canonical dataset must contain only the configured single circuit')
    # Fixed device ordering has semantic meaning for the tabular baselines.
    identities = {tuple(engine.group_keys(row)) for row in dataset['samples']}
    if len(identities) != 1:
        raise ValueError('Single-circuit task requires one fixed topology/device identity')
    representative = dataset['samples'][0]
    pin_count = len(representative.get('ports', {})) + sum(
        len(device['terminals']) for device in representative['devices'])
    capacities = {mode: effective_config_pin(config, mode, seeds[0])['model']['max_devices']
                  for mode in models}
    too_small = [mode for mode, capacity in capacities.items() if capacity < pin_count]
    if too_small:
        raise ValueError(f'{pin_count} pin/port tokens exceed max_devices capacity for {too_small}; no truncation')
    manifest = (json.loads(Path(manifest_path).read_text()) if manifest_path else
                build_manifest(dataset, config=config['dataset']))
    verify_manifest(dataset, manifest)
    _check_dataset_options(config, manifest)
    if manifest['ood'] or not all(manifest[k] for k in ('train', 'val', 'test')):
        raise ValueError('Single-circuit manifest requires train/val/test and an empty OOD split')
    if any('_shuffle_' in name for name in models) and len(manifest['train']) < 2:
        raise ValueError('Shuffled OP control requires at least two training samples')
    logs = Path(log_dir).resolve() if log_dir else output / 'logs_pin'
    if not logs.name.endswith('_pin'):
        raise ValueError('Pin log directory must end with _pin')
    if (logs / 'experiment_pin.log').exists():
        raise ValueError('Choose a fresh experiment log directory')
    output.mkdir(parents=True, exist_ok=True)
    # O_EXCL prevents concurrent runs claiming the same empty directory.
    with (output / '.run_pin.lock').open('x') as lock:
        lock.write(str(os.getpid()))
    run = dict(status='running', granularity='pin', models=models, seeds=seeds, circuit=circuit,
               pin_tokens=pin_count, physical_devices=len(representative['devices']),
               resident_token_capacity=max(capacities.values()),
               evaluation_splits=['test'], device=str(device), code_commit=_revision(),
               dataset_fingerprint=dataset_fingerprint(dataset), log_dir=str(logs),
               slurm_job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURM_NODELIST'),
               torch_version=str(torch.__version__), started_unix=time.time(), smoke=smoke,
               budget_policy='equal neural downstream maxima; prerequisite costs separate; XGBoost uses rounds')
    engine.write_json(output / 'run_pin.json', run)
    engine.write_json(output / 'config_pin.json', config)
    engine.write_json(output / 'manifest_pin.json', manifest)
    engine.write_json(output / 'model_sources_pin.json', json.loads(
        (ROOT / 'final_exam/models/SOURCES_pin.json').read_text()))
    results, stages = [], []
    try:
        with log_to(logs / 'experiment_pin.log'):
            try:
                context_config = copy.deepcopy(config)
                context_config['model']['max_devices'] = max(capacities.values())
                ctx = PinDataContext(dataset, manifest, context_config, device, {}, output)
                for seed in seeds:
                    ctx.output = output / f'seed_{seed}_pin'
                    ctx.output.mkdir()
                    shared, dependencies = {}, {}

                    def save(model, mode, deps, model_cfg, kind='op_transfer', shared_stage=False):
                        mode = mode if mode.endswith('_pin') else mode + '_pin'
                        path = ctx.output / 'shared_pin' / mode if shared_stage else ctx.output / mode
                        save_checkpoint_pin(path, model, ctx, mode, seed, deps, options, model_cfg, kind)

                    def stage(model, name, epoch_key, model_cfg, objective=engine.performance_objective,
                              *, log_name=None, **kwargs):
                        name = name if name.endswith('_pin') else name + '_pin'
                        if log_name:
                            log_name = '/'.join(p if p.endswith('_pin') else p+'_pin' for p in log_name.split('/'))
                        path = logs / f'seed_{seed}_pin' / (log_name or name) / 'train_pin.log'
                        original_config = ctx.config
                        ctx.config = model_cfg
                        try:
                            with log_to(path):
                                print(f'seed={seed} stage={name} budget={options["epochs"][epoch_key]}', flush=True)
                                prepare_model_pin(model, ctx, options)
                                model, cost = engine.fit_stage(ctx, model, name, options['epochs'][epoch_key],
                                                              options, seed, objective, artifact_suffix='_pin', **kwargs)
                        finally:
                            ctx.config = original_config
                        cost.update(stage_id=f'seed_{seed}_pin/{name}', seed=seed)
                        stages.append(cost)
                        engine.write_json(ctx.output / 'stages_pin' / name / 'config_pin.json', model_cfg)
                        engine.export_summary(output, results, stages, artifact_suffix='_pin')
                        return model, cost['stage_id']

                    def predictor():
                        if 'op_predictor' not in shared:
                            cfg = effective_config_pin(config, 'op_predictor', seed)
                            seed_everything(seed, options['torch_threads'])
                            model = OPPredictorPin(design_encoder_pin(cfg['model'], ctx.prep), ctx.prep,
                                                fit_op_support_pin(ctx.rows['train'], ctx.prep)).to(device)
                            model, dep = stage(model, 'op_predictor', 'op_predictor', cfg, op_objective_pin,
                                               op=True, log_name='shared_pin/op_predictor')
                            shared['op_predictor'] = freeze(model)
                            dependencies['op_predictor'] = [dep]
                            save(model, 'op_predictor', [dep], cfg, shared_stage=True)
                        return shared['op_predictor']

                    def teacher(use_design):
                        mode = 'design_op_oracle_pin' if use_design else 'op_oracle_pin'
                        if mode not in shared:
                            cfg = effective_config_pin(config, mode, seed)
                            seed_everything(seed, options['torch_threads'])
                            model = ReadoutModelPin(op_encoder_pin(cfg['model'], ctx.prep, use_design=use_design),
                                                 decoder_pin(cfg['model'], ctx.prep),
                                                 source='design_op' if use_design else 'op').to(device)
                            model, dep = stage(model, mode, 'teacher', cfg, op=True, log_name=f'shared_pin/{mode}')
                            shared[mode] = freeze(model)
                            dependencies[mode] = [dep]
                            save(model, mode, [dep], cfg, shared_stage=True)
                        return shared[mode]

                    def device_predictor():
                        name = 'device_op_predictor_pin'
                        if name not in shared:
                            cfg = effective_config_pin(config, name, seed)
                            seed_everything(seed, options['torch_threads'])
                            model = DeviceOPPredictorPin(cfg['model'], ctx.prep,
                                fit_op_support_pin(ctx.rows['train'], ctx.prep)).to(device)
                            model, dep = stage(model, name, 'op_predictor', cfg, op_objective_pin,
                                               op=True, log_name='shared_pin/' + name)
                            shared[name] = freeze(model)
                            dependencies[name] = [dep]
                            save(model, name, [dep], cfg, shared_stage=True)
                            started = time.perf_counter()
                            with log_to(logs / f'seed_{seed}_pin' / 'shared_pin' / name / 'cache_pin.log'):
                                # No measured OP or performance labels are used in this forward-only pass.
                                prepare_device_predictions_pin(ctx, model, options, splits=('train', 'val', 'test'))
                            if device.type == 'cuda':
                                torch.cuda.synchronize(device)
                            cache_id = f'seed_{seed}_pin/frozen_device_op_cache_pin'
                            cache_cost = dict(stage_id=cache_id, seed=seed, epochs=0, planned_epochs=0,
                                optimizer_steps=0, examples_seen=sum(len(ctx.rows[s]) for s in ('train','val','test')),
                                seconds=time.perf_counter()-started, parameters=sum(p.numel() for p in model.parameters()),
                                trainable_parameters=0, device=str(device),
                                selection={'kind': 'forward_only_preparation', 'uses_measured_op': False})
                            stages.append(cache_cost)
                            dependencies[name].append(cache_id)
                            engine.write_json(ctx.output / 'shared_pin' / name / 'prediction_cache_pin.json', cache_cost)
                        return shared[name]

                    def shuffle_bank():
                        if 'shuffle_bank_pin' not in shared:
                            predictor_model = device_predictor()
                            shared['shuffle_bank_pin'] = build_shuffle_bank_pin(ctx, predictor_model,
                                options['batch_size'], seed)
                        return shared['shuffle_bank_pin']

                    for mode in models:
                        print(f'=== {circuit} seed={seed} model={mode} ===', flush=True)
                        cfg = effective_config_pin(config, mode, seed)
                        seed_everything(seed, options['torch_threads'])
                        directory = ctx.output / mode
                        kind, deps = 'baseline', []
                        if mode == 'xgboost_pin':
                            from final_exam.tree_pin import fit_xgboost_pin
                            with log_to(logs / f'seed_{seed}_pin' / mode / 'train_pin.log'):
                                model, cost = fit_xgboost_pin(ctx, cfg, options, seed, directory)
                            cost.update(stage_id=f'seed_{seed}_pin/xgboost_pin', seed=seed)
                            stages.append(cost)
                            deps = [cost['stage_id']]
                            engine.write_json(directory / 'config_pin.json', cfg)
                            engine.write_json(directory / 'preprocessing_pin.json', ctx.prep)
                            engine.write_json(directory / 'manifest_pin.json', manifest)
                        elif mode in GATE_MODELS_PIN:
                            kind = 'op_transfer'
                            needs_predictor = mode != 'device_op_gate_only_base_pin'
                            op_model = device_predictor() if needs_predictor else None
                            bank = shuffle_bank() if '_shuffle_' in mode else None
                            predictor_digest = state_digest(op_model) if op_model is not None else None
                            deps = dependencies['device_op_predictor_pin'][:] if needs_predictor else []
                            # Shared dependencies must not consume the paired downstream initialization RNG.
                            seed_everything(seed, options['torch_threads'])
                            model = DeviceOPGateModelPin(mode, cfg['model'], ctx.prep,
                                predictor=op_model, gate_config=cfg.get('op_gate_pin'), shuffle_bank=bank).to(device)
                            initial_state = state_digest(model.encoder)
                            initial_decoder = state_digest(model.decoder)
                            model, dep = stage(model, mode, 'readout', cfg, op=mode in REFERENCES_PIN)
                            deps.append(dep)
                            if op_model is not None and state_digest(op_model) != predictor_digest:
                                raise AssertionError('Frozen device OP predictor changed during downstream training')
                            directory.mkdir(parents=True, exist_ok=True)
                            audit = gate_contract_pin(mode)
                            audit.update(predictor_frozen=needs_predictor, predictor_digest=predictor_digest,
                                initial_encoder_digest=initial_state, initial_decoder_digest=initial_decoder,
                                op_embedding_dim=cfg['model']['dim'], gate_heads=cfg['model']['heads'],
                                protected_edges='same net, self, external ports, non-MOS',
                                predictor_checkpoint='../shared_pin/device_op_predictor_pin/checkpoint_pin.pt'
                                    if needs_predictor else None)
                            engine.write_json(directory / 'op_gate_audit_pin.json', audit)
                        elif mode in REFERENCES_PIN:
                            model = teacher(mode == 'design_op_oracle_pin')
                            deps, kind = dependencies[mode], 'op_transfer'
                        elif mode in CASCADE_MODELS_PIN:
                            op_model = copy.deepcopy(predictor())
                            predictor_digest = state_digest(op_model)
                            kind = 'op_transfer'
                            deps = dependencies['op_predictor'][:]
                            if mode == 'predicted_op_pin':
                                seed_everything(seed, options['torch_threads'])
                                downstream = ReadoutModelPin(op_encoder_pin(cfg['model'], ctx.prep, use_design=True),
                                                          decoder_pin(cfg['model'], ctx.prep), source='design_op')
                                model = PredictedOPCascadePin(op_model, downstream).to(device)
                                model, dep = stage(model, mode, 'readout', cfg)
                                deps.append(dep)
                            else:
                                source = teacher(True)
                                teacher_digest = state_digest(source)
                                downstream = copy.deepcopy(source).requires_grad_(True)
                                cascade = PredictedOPCascadePin(op_model, downstream).to(device)
                                scale = engine.fit_scaler(ctx, source.encoder, options, 'design_op',
                                    input_filter=lambda batch: op_input_pin(batch, use_design=True))
                                sequential = mode == 'student_predicted_op_distill_pin'
                                pair_type = PredictedOPEncoderDistillationPin if sequential else PredictedOPDistillationPin
                                pair = pair_type(cascade, source, scale).to(device)
                                deps += dependencies['design_op_oracle_pin']
                                audit = dict(teacher_digest=teacher_digest, predictor_digest=predictor_digest,
                                             initialization='copy complete teacher encoder and decoder_pin',
                                             inference_requires_measured_op=False)
                                metadata = lambda m: dict(teacher_scaler=scale.state_dict(),
                                                           teacher_checkpoint='../../shared_pin/design_op_oracle_pin/checkpoint_pin.pt')
                                if sequential:
                                    before_decoder = state_digest(downstream.decoder)
                                    pair, dep = stage(pair, 'predicted_op_distill_encoder',
                                                      'predicted_op_distill_encoder', cfg, lambda m,b: m.objective(b),
                                                      op=True, log_name=mode+'/encoder', checkpoint_extra=metadata)
                                    deps.append(dep)
                                    if state_digest(downstream.decoder) != before_decoder:
                                        raise AssertionError('Decoder changed during encoder-only distillation')
                                    model = pair.student
                                    model.downstream.frozen = True
                                    freeze(model.downstream.encoder)
                                    before_encoder = state_digest(model.downstream.encoder)
                                    model.downstream.decoder.requires_grad_(True)
                                    model.zero_grad(set_to_none=True)
                                    model, dep = stage(model, 'predicted_op_distill_decoder',
                                                       'predicted_op_distill_decoder', cfg, log_name=mode+'/decoder_pin')
                                    deps.append(dep)
                                    if state_digest(model.downstream.encoder) != before_encoder:
                                        raise AssertionError('Encoder changed during decoder_pin-only training')
                                    audit.update(decoder_unchanged_during_encoder=True,
                                                 encoder_unchanged_during_decoder=True,
                                                 encoder_selection='validation alignment', decoder_selection='validation performance')
                                else:
                                    objective = lambda m,b: m.objective(b,
                                        alignment_weight=options['student_alignment_weight'],
                                        performance_weight=options['student_performance_weight'])
                                    pair, dep = stage(pair, mode, 'student_predicted_op', cfg, objective,
                                                       op=True, checkpoint_extra=metadata)
                                    model = pair.student
                                    deps.append(dep)
                                if state_digest(source) != teacher_digest:
                                    raise AssertionError('Frozen teacher changed')
                                directory.mkdir(parents=True, exist_ok=True)
                                audit.update(teacher_frozen=True, predictor_frozen=True)
                                engine.write_json(directory / 'distillation_pin.json', audit)
                                torch.save(scale.state_dict(), directory / 'alignment_scales_pin.pt')
                                del pair
                            if state_digest(model.predictor) != predictor_digest:
                                raise AssertionError('Frozen OP predictor changed')
                        else:
                            if mode == 'deepgen_op_pretrained_pin':
                                op_model = OPPredictorPin(build_encoder_pin('deepgen_supervised_pin', cfg['model'], ctx.prep),
                                    ctx.prep, fit_op_support_pin(ctx.rows['train'], ctx.prep), encoder_name='deepgen_supervised_pin').to(device)
                                op_model, dep = stage(op_model, 'deepgen_op_predictor', 'op_predictor', cfg,
                                    op_objective_pin, op=True, log_name='shared_pin/deepgen_op_predictor')
                                deps.append(dep)
                                save(op_model, 'deepgen_op_predictor', deps, cfg, 'deepgen_op_predictor', True)
                                seed_everything(seed, options['torch_threads'])
                                model = build_model_pin('deepgen_supervised_pin', cfg['model'], ctx.prep).to(device)
                                model.encoder = copy.deepcopy(op_model.encoder)
                                del op_model
                            else:
                                model = build_model_pin(mode, cfg['model'], ctx.prep).to(device)
                            model, dep = stage(model, mode, 'readout', cfg)
                            deps.append(dep)
                        if mode != 'xgboost_pin':
                            save(model, mode, deps, cfg, kind)
                        with log_to(logs / f'seed_{seed}_pin' / mode / 'evaluate_pin.log'):
                            prepare_model_pin(model, ctx, options, splits=('test',))
                            report = engine.evaluate_model(ctx, model, 'test', options, measured_op=mode in REFERENCES_PIN)
                            report.update(circuit=circuit, granularity='pin')
                            if mode in GATE_MODELS_PIN:
                                report['op_gate_pin'] = gate_contract_pin(mode)
                                if mode in REFERENCES_PIN:
                                    report['oracle_gate_coverage_pin'] = oracle_gate_coverage_pin(
                                        ctx, model.predictor, 'test', options['batch_size'])
                            engine.write_json(directory / 'test_pin.json', report)
                            print(json.dumps(report, allow_nan=False), flush=True)
                        results.append(dict(mode=mode, seed=seed, stage_ids=deps, report=report))
                        engine.write_json(output / 'results_pin.json', results)
                        engine.export_summary(output, results, stages, artifact_suffix='_pin')
                        del model
                    shared.clear()
                run['status'] = 'ready_to_export'
                engine.write_json(output / 'run_pin.json', run)
                from final_exam.summarize_pin import export_results_pin
                export_results_pin(output, finalizing=True)
                run.update(status='complete', elapsed_seconds=time.time()-run['started_unix'])
                engine.write_json(output / 'run_pin.json', run)
                print(f'Completed. Excel: {output / "summary_pin.xlsx"}', flush=True)
            except BaseException:
                traceback.print_exc()
                run['status'] = 'failed_export' if run['status'] == 'ready_to_export' else 'failed'
                run['elapsed_seconds'] = time.time()-run['started_unix']
                engine.write_json(output / 'run_pin.json', run)
                raise
    except BaseException:
        # Log acquisition itself can fail before entering the logged try block.
        if run['status'] in ('running', 'ready_to_export'):
            run['status'] = 'failed_export' if run['status'] == 'ready_to_export' else 'failed'
            run['elapsed_seconds'] = time.time()-run['started_unix']
            engine.write_json(output / 'run_pin.json', run)
        raise
    return dict(output=str(output), results=results, stages=stages)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list-models', action='store_true')
    parser.add_argument('--data')
    parser.add_argument('--config', default=str(ROOT / 'final_exam/configs/default_pin.json'))
    parser.add_argument('--output')
    parser.add_argument('--log-dir')
    parser.add_argument('--manifest')
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES_PIN)
    parser.add_argument('--seeds', nargs='+', type=int, default=[17, 29, 43])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    if args.list_models:
        print('\n'.join(MODEL_NAMES_PIN))
        return
    if not args.data or not args.output:
        parser.error('--data and --output are required')
    run_experiment_pin(args.data, args.config, args.output, args.models, args.seeds, args.device,
                   log_dir=args.log_dir, manifest_path=args.manifest, smoke=args.smoke)


if __name__ == '__main__':
    main()
