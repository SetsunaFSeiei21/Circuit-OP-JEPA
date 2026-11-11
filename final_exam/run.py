#!/usr/bin/env python3
"""Matched device-level single-circuit experiments with independently saved stages."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_NAMES = ('mlp', 'xgboost', 'gat', 'deepgen_supervised', 'deepgen_op_pretrained',
               'gtn', 'zerosim_supervised', 'accformer', 'predicted_op',
               'student_predicted_op', 'student_predicted_op_distill',
               'op_oracle', 'design_op_oracle')
CASCADE_MODELS = frozenset({'predicted_op', 'student_predicted_op', 'student_predicted_op_distill'})
REFERENCES = frozenset({'op_oracle', 'design_op_oracle'})


def validate_request(config, models, seeds, device, smoke=False):
    from device_jepa.model import ModelConfig
    from device_jepa.resident import require_device
    from experiments.op_transfer.engine import settings
    config = copy.deepcopy(config)
    extra = config.setdefault('final_exam', {})
    if set(extra) - {'models', 'model_overrides', 'xgboost'}:
        raise ValueError('Unknown final_exam configuration keys')
    selected = list(extra.get('models', MODEL_NAMES) if models is None else models)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(MODEL_NAMES):
        raise ValueError('Select distinct model names from --list-models')
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError('Seeds must be unique integers in [0, 2**32)')
    data = config['dataset']
    if len(data.get('train_circuits', [])) != 1 or data.get('ood_circuits'):
        raise ValueError('Phase one requires exactly one train circuit and no OOD circuit')
    overrides = extra.get('model_overrides', {})
    if set(overrides) - set(MODEL_NAMES[:8]):
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
    epochs = options['epochs']
    if not epochs['readout'] == epochs['student_predicted_op'] == (
            epochs['predicted_op_distill_encoder'] + epochs['predicted_op_distill_decoder']):
        raise ValueError('Matched downstream budgets require readout == student_predicted_op '
                         '== predicted_op_distill_encoder + predicted_op_distill_decoder')
    require_device(device)
    if 'xgboost' in selected:
        if importlib.util.find_spec('xgboost') is None:
            raise ValueError('Selected xgboost requires pip install -r final_exam/requirements.txt')
        from final_exam.tree import validate_tree_config
        validate_tree_config(config)
    return config, selected


def effective_config(config, mode, seed):
    value = copy.deepcopy(config)
    key = 'zerosim_supervised' if mode in CASCADE_MODELS | REFERENCES | {'op_predictor'} else mode
    override = value.get('final_exam', {}).get('model_overrides', {}).get(key, {})
    value['model'].update(override)
    value.setdefault('training', {})['seed'] = seed
    return value


def save_checkpoint(directory, model, ctx, mode, seed, dependencies, options, config, kind='op_transfer'):
    import torch
    from experiments.op_transfer.engine import _cpu_state, write_json
    from experiments.op_transfer.models import describe_model
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(format='final_exam_v1', kind=kind, mode=mode, seed=seed,
                   state_dict=_cpu_state(model), model_config=config['model'], config=config,
                   preprocessing=ctx.prep, manifest=ctx.full_manifest,
                   selected_sample_ids=ctx.selection, stage_ids=dependencies, options=options,
                   trainable_parameters=[name for name, parameter in model.named_parameters()
                                         if parameter.requires_grad],
                   requires_measured_op=mode in REFERENCES)
    if kind == 'op_transfer':
        payload['recipe'] = describe_model(model)
    elif kind == 'deepgen_op_predictor':
        payload['support'] = model.support_metadata
    else:
        payload['factory_name'] = 'deepgen_supervised' if mode == 'deepgen_op_pretrained' else mode
    torch.save(payload, directory / 'checkpoint.pt')
    write_json(directory / 'config.json', config)
    write_json(directory / 'dependencies.json', dict(stage_ids=dependencies))


def load_checkpoint(path, device='cpu'):
    import torch
    from device_jepa.resident import require_device
    from experiments.op_transfer.models import OPPredictor, rebuild_model
    from final_exam.models import build_encoder, build_model
    path = Path(path)
    if path.suffix == '.json':
        from final_exam.tree import load_xgboost
        return load_xgboost(path, device)
    value = torch.load(path, map_location='cpu', weights_only=True)
    if value.get('format') != 'final_exam_v1':
        raise ValueError('Expected a final_exam checkpoint')
    cfg, prep = value['model_config'], value['preprocessing']
    if value['kind'] == 'op_transfer':
        model = rebuild_model(value['recipe'], cfg, prep)
    elif value['kind'] == 'deepgen_op_predictor':
        model = OPPredictor(build_encoder('deepgen_supervised', cfg, prep), prep, value['support'])
    elif value['kind'] == 'baseline':
        model = build_model(value['factory_name'], cfg, prep)
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
    from final_exam.common.provenance import code_revision
    return code_revision(ROOT)


def run_experiment(data_path, config_path, output, models, seeds, device, *, log_dir=None,
                   manifest_path=None, smoke=False):
    import torch
    from device_jepa.data import load_dataset, build_manifest, verify_manifest, dataset_fingerprint
    from device_jepa.resident import require_device
    from device_jepa.training import _check_dataset_options, seed_everything, state_digest
    from experiments.op_transfer import engine
    from experiments.op_transfer.distillation import PredictedOPDistillation, PredictedOPEncoderDistillation
    from experiments.op_transfer.models import (OPPredictor, PredictedOPCascade, ReadoutModel,
        decoder, design_encoder, fit_op_support, freeze, op_encoder)
    from final_exam.logging_utils import log_to
    from final_exam.models import build_encoder, build_model, prepare_model

    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Choose a new/empty output directory')
    config, models = validate_request(json.loads(Path(config_path).read_text()), models, seeds, device, smoke)
    device = require_device(device)
    options = engine.settings(config)
    seed_everything(seeds[0], options['torch_threads'])
    manifest = json.loads(Path(manifest_path).read_text()) if manifest_path else None
    dataset = load_dataset(data_path, sampling_config=(manifest or {}).get('sampling', config.get('dataset')))
    circuit = config['dataset']['train_circuits'][0]
    if {row['circuit_id'] for row in dataset['samples']} != {circuit}:
        raise ValueError('Canonical dataset must contain only the configured single circuit')
    # Fixed device ordering has semantic meaning for the tabular baselines.
    identities = {tuple(engine.group_keys(row)) for row in dataset['samples']}
    if len(identities) != 1:
        raise ValueError('Single-circuit task requires one fixed topology/device identity')
    if manifest is None:
        manifest = build_manifest(dataset, config=config['dataset'])
    verify_manifest(dataset, manifest)
    _check_dataset_options(config, manifest)
    if manifest['ood'] or not all(manifest[k] for k in ('train', 'val', 'test')):
        raise ValueError('Single-circuit manifest requires train/val/test and an empty OOD split')
    logs = Path(log_dir).resolve() if log_dir else output / 'logs'
    if (logs / 'experiment.log').exists():
        raise ValueError('Choose a fresh experiment log directory')
    output.mkdir(parents=True, exist_ok=True)
    # O_EXCL prevents concurrent runs claiming the same empty directory.
    with (output / '.run.lock').open('x') as lock:
        lock.write(str(os.getpid()))
    run = dict(status='running', models=models, seeds=seeds, circuit=circuit,
               evaluation_splits=['test'], device=str(device), code_commit=_revision(),
               dataset_fingerprint=dataset_fingerprint(dataset), log_dir=str(logs),
               slurm_job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURM_NODELIST'),
               torch_version=str(torch.__version__), started_unix=time.time(), smoke=smoke,
               budget_policy='equal neural downstream maxima; prerequisite costs separate; XGBoost uses rounds')
    engine.write_json(output / 'run.json', run)
    engine.write_json(output / 'config.json', config)
    engine.write_json(output / 'manifest.json', manifest)
    engine.write_json(output / 'model_sources.json', json.loads(
        (ROOT / 'final_exam/models/SOURCES.json').read_text()))
    results, stages = [], []
    try:
        with log_to(logs / 'experiment.log'):
            try:
                ctx = engine.DataContext(dataset, manifest, config, device, {}, output)
                for seed in seeds:
                    ctx.output = output / f'seed_{seed}'
                    ctx.output.mkdir()
                    shared, dependencies = {}, {}

                    def save(model, mode, deps, model_cfg, kind='op_transfer', shared_stage=False):
                        path = ctx.output / 'shared' / mode if shared_stage else ctx.output / mode
                        save_checkpoint(path, model, ctx, mode, seed, deps, options, model_cfg, kind)

                    def stage(model, name, epoch_key, model_cfg, objective=engine.performance_objective,
                              *, log_name=None, **kwargs):
                        path = logs / f'seed_{seed}' / (log_name or name) / 'train.log'
                        original_config = ctx.config
                        ctx.config = model_cfg
                        try:
                            with log_to(path):
                                print(f'seed={seed} stage={name} budget={options["epochs"][epoch_key]}', flush=True)
                                prepare_model(model, ctx, options)
                                model, cost = engine.fit_stage(ctx, model, name, options['epochs'][epoch_key],
                                                              options, seed, objective, **kwargs)
                        finally:
                            ctx.config = original_config
                        cost.update(stage_id=f'seed_{seed}/{name}', seed=seed)
                        stages.append(cost)
                        engine.write_json(ctx.output / 'stages' / name / 'config.json', model_cfg)
                        engine.export_summary(output, results, stages)
                        return model, cost['stage_id']

                    def predictor():
                        if 'op_predictor' not in shared:
                            cfg = effective_config(config, 'op_predictor', seed)
                            seed_everything(seed, options['torch_threads'])
                            model = OPPredictor(design_encoder(cfg['model'], ctx.prep), ctx.prep,
                                                fit_op_support(ctx.rows['train'], ctx.prep)).to(device)
                            model, dep = stage(model, 'op_predictor', 'op_predictor', cfg, engine.op_objective,
                                               op=True, log_name='shared/op_predictor')
                            shared['op_predictor'] = freeze(model)
                            dependencies['op_predictor'] = [dep]
                            save(model, 'op_predictor', [dep], cfg, shared_stage=True)
                        return shared['op_predictor']

                    def teacher(use_design):
                        mode = 'design_op_oracle' if use_design else 'op_oracle'
                        if mode not in shared:
                            cfg = effective_config(config, mode, seed)
                            seed_everything(seed, options['torch_threads'])
                            model = ReadoutModel(op_encoder(cfg['model'], ctx.prep, use_design=use_design),
                                                 decoder(cfg['model'], ctx.prep),
                                                 source='design_op' if use_design else 'op').to(device)
                            model, dep = stage(model, mode, 'teacher', cfg, op=True, log_name=f'shared/{mode}')
                            shared[mode] = freeze(model)
                            dependencies[mode] = [dep]
                            save(model, mode, [dep], cfg, shared_stage=True)
                        return shared[mode]

                    for mode in models:
                        print(f'=== {circuit} seed={seed} model={mode} ===', flush=True)
                        cfg = effective_config(config, mode, seed)
                        seed_everything(seed, options['torch_threads'])
                        directory = ctx.output / mode
                        kind, deps = 'baseline', []
                        if mode == 'xgboost':
                            from final_exam.tree import fit_xgboost
                            with log_to(logs / f'seed_{seed}' / mode / 'train.log'):
                                model, cost = fit_xgboost(ctx, cfg, options, seed, directory)
                            cost.update(stage_id=f'seed_{seed}/xgboost', seed=seed)
                            stages.append(cost)
                            deps = [cost['stage_id']]
                            engine.write_json(directory / 'config.json', cfg)
                            engine.write_json(directory / 'preprocessing.json', ctx.prep)
                            engine.write_json(directory / 'manifest.json', manifest)
                        elif mode in REFERENCES:
                            model = teacher(mode == 'design_op_oracle')
                            deps, kind = dependencies[mode], 'op_transfer'
                        elif mode in CASCADE_MODELS:
                            op_model = copy.deepcopy(predictor())
                            predictor_digest = state_digest(op_model)
                            kind = 'op_transfer'
                            deps = dependencies['op_predictor'][:]
                            if mode == 'predicted_op':
                                seed_everything(seed, options['torch_threads'])
                                downstream = ReadoutModel(op_encoder(cfg['model'], ctx.prep, use_design=True),
                                                          decoder(cfg['model'], ctx.prep), source='design_op')
                                model = PredictedOPCascade(op_model, downstream).to(device)
                                model, dep = stage(model, mode, 'readout', cfg)
                                deps.append(dep)
                            else:
                                source = teacher(True)
                                teacher_digest = state_digest(source)
                                downstream = copy.deepcopy(source).requires_grad_(True)
                                cascade = PredictedOPCascade(op_model, downstream).to(device)
                                scale = engine.fit_scaler(ctx, source.encoder, options, 'design_op')
                                sequential = mode == 'student_predicted_op_distill'
                                pair_type = PredictedOPEncoderDistillation if sequential else PredictedOPDistillation
                                pair = pair_type(cascade, source, scale).to(device)
                                deps += dependencies['design_op_oracle']
                                audit = dict(teacher_digest=teacher_digest, predictor_digest=predictor_digest,
                                             initialization='copy complete teacher encoder and decoder',
                                             inference_requires_measured_op=False)
                                metadata = lambda m: dict(teacher_scaler=scale.state_dict(),
                                                           teacher_checkpoint='../../shared/design_op_oracle/checkpoint.pt')
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
                                                       'predicted_op_distill_decoder', cfg, log_name=mode+'/decoder')
                                    deps.append(dep)
                                    if state_digest(model.downstream.encoder) != before_encoder:
                                        raise AssertionError('Encoder changed during decoder-only training')
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
                                engine.write_json(directory / 'distillation.json', audit)
                                torch.save(scale.state_dict(), directory / 'alignment_scales.pt')
                                del pair
                            if state_digest(model.predictor) != predictor_digest:
                                raise AssertionError('Frozen OP predictor changed')
                        else:
                            if mode == 'deepgen_op_pretrained':
                                op_model = OPPredictor(build_encoder('deepgen_supervised', cfg['model'], ctx.prep),
                                    ctx.prep, fit_op_support(ctx.rows['train'], ctx.prep)).to(device)
                                op_model, dep = stage(op_model, 'deepgen_op_predictor', 'op_predictor', cfg,
                                    engine.op_objective, op=True, log_name='shared/deepgen_op_predictor')
                                deps.append(dep)
                                save(op_model, 'deepgen_op_predictor', deps, cfg, 'deepgen_op_predictor', True)
                                seed_everything(seed, options['torch_threads'])
                                model = build_model('deepgen_supervised', cfg['model'], ctx.prep).to(device)
                                model.encoder = copy.deepcopy(op_model.encoder)
                                del op_model
                            else:
                                model = build_model(mode, cfg['model'], ctx.prep).to(device)
                            model, dep = stage(model, mode, 'readout', cfg)
                            deps.append(dep)
                        if mode != 'xgboost':
                            save(model, mode, deps, cfg, kind)
                        with log_to(logs / f'seed_{seed}' / mode / 'evaluate.log'):
                            prepare_model(model, ctx, options, splits=('test',))
                            report = engine.evaluate_model(ctx, model, 'test', options, measured_op=mode in REFERENCES)
                            report['circuit'] = circuit
                            engine.write_json(directory / 'test.json', report)
                            print(json.dumps(report, allow_nan=False), flush=True)
                        results.append(dict(mode=mode, seed=seed, stage_ids=deps, report=report))
                        engine.write_json(output / 'results.json', results)
                        engine.export_summary(output, results, stages)
                        del model
                    shared.clear()
                run['status'] = 'ready_to_export'
                engine.write_json(output / 'run.json', run)
                from final_exam.summarize import export_results
                export_results(output, finalizing=True)
                run.update(status='complete', elapsed_seconds=time.time()-run['started_unix'])
                engine.write_json(output / 'run.json', run)
                print(f'Completed. Excel: {output / "summary.xlsx"}', flush=True)
            except BaseException:
                traceback.print_exc()
                run['status'] = 'failed_export' if run['status'] == 'ready_to_export' else 'failed'
                run['elapsed_seconds'] = time.time()-run['started_unix']
                engine.write_json(output / 'run.json', run)
                raise
    except BaseException:
        # Log acquisition itself can fail before entering the logged try block.
        if run['status'] in ('running', 'ready_to_export'):
            run['status'] = 'failed_export' if run['status'] == 'ready_to_export' else 'failed'
            run['elapsed_seconds'] = time.time()-run['started_unix']
            engine.write_json(output / 'run.json', run)
        raise
    return dict(output=str(output), results=results, stages=stages)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list-models', action='store_true')
    parser.add_argument('--data')
    parser.add_argument('--config', default=str(ROOT / 'final_exam/configs/default.json'))
    parser.add_argument('--output')
    parser.add_argument('--log-dir')
    parser.add_argument('--manifest')
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES)
    parser.add_argument('--seeds', nargs='+', type=int, default=[17, 29, 43])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    if args.list_models:
        print('\n'.join(MODEL_NAMES))
        return
    if not args.data or not args.output:
        parser.error('--data and --output are required')
    run_experiment(args.data, args.config, args.output, args.models, args.seeds, args.device,
                   log_dir=args.log_dir, manifest_path=args.manifest, smoke=args.smoke)


if __name__ == '__main__':
    main()
