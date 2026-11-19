"""LASD-MoE submits isolated circuit jobs with reproducible config snapshots."""
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import unittest

from final_exam.single_circuit_pin.test_staged_op_gate_launch_pin import StagedGateLauncherPinTests
from final_exam.single_circuit_pin import test_resume_staged_op_gate_launch_pin as resume_tests


class LASDMoELauncherPinTests(StagedGateLauncherPinTests):
    MODELS = ['device_op_cascade_pin', 'lasd_moe_pin', 'lasd_moe_detach_pin']
    SCRIPT_NAME = 'train_lasd_moe_pin.sh'
    CONFIG_NAME = 'lasd_moe_pin.json'
    RUN_TAG = '_lasd_moe_'

    def test_template_retains_single_decoder_and_explicit_experiment_scope(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs' / self.CONFIG_NAME).read_text())
        self.assertEqual(cfg['final_exam_pin']['models'], self.MODELS)
        self.assertEqual(cfg['model']['encoder_layers'], 6)
        self.assertEqual(cfg['model']['cross_layers'], 3)
        self.assertEqual(cfg['model']['decoder_layers'], 2)
        self.assertEqual(cfg['op_transfer']['epochs']['op_predictor'], 30)
        self.assertEqual(cfg['op_transfer']['epochs']['readout'], 60)
        self.assertEqual(cfg['lasd_moe_pin']['n_experts'], 4)
        self.assertEqual(cfg['lasd_moe_pin']['top_k'], 2)
        self.assertTrue(cfg['lasd_moe_pin']['checkpoint_scores'])
        self.assertNotIn('lasd_pin', cfg)
        self.assertNotIn('width_decoder_pin', cfg)
        from final_exam.lasd_moe_registry_pin import lasd_moe_spec_pin
        spec = lasd_moe_spec_pin('lasd_moe_pin', cfg['lasd_moe_pin'], cfg['model'])
        self.assertEqual(spec['attention_scope'], 'pe_self_attention_only')
        self.assertEqual(spec['base_layers'], cfg['model']['cross_layers'])
        self.assertEqual(spec['input_dim'], 96)
        self.assertEqual(spec['version'], 1)
        self.assertNotIn('layer_weight_axes', spec)
        per_head = lasd_moe_spec_pin('lasd_moe_headwise_pin', cfg['lasd_moe_pin'], cfg['model'])
        self.assertEqual((per_head['input_dim'], per_head['expert_output_dim'],
                          per_head['channels_per_head']), (96, 24, 16))

    def test_headwise_flag_selects_only_the_two_new_models(self):
        script, _, environment, _ = self.fixture('exit 99')
        run = subprocess.run(['bash', str(script), '--dry-run', '--headwise'], env=environment,
                             text=True, capture_output=True, check=True)
        args = shlex.split(run.stdout)
        self.assertEqual(args[args.index('--models') + 1:args.index('--seeds')],
                         ['lasd_moe_headwise_pin', 'lasd_moe_headwise_detach_pin'])

    def test_two_circuits_have_separate_jobs_and_run_roots(self):
        script, task, environment, _ = self.fixture('exit 99')
        script.write_text(script.read_text().replace(
            'CIRCUITS=(two_stage_folded_opamp)',
            'CIRCUITS=(two_stage_folded_opamp Opamp1)'))
        before = sorted(str(p) for p in self.root.rglob('*'))
        run = subprocess.run(['bash', str(script), '--dry-run'], env=environment,
                             text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        commands = [shlex.split(line) for line in run.stdout.splitlines()]
        self.assertEqual(len(commands), 2)
        self.assertEqual([cmd[cmd.index('--circuit') + 1] for cmd in commands],
                         ['two_stage_folded_opamp', 'Opamp1'])
        run_ids = [cmd[cmd.index('--run-id') + 1] for cmd in commands]
        self.assertEqual(len(set(run_ids)), 2)
        self.assertEqual(before, sorted(str(p) for p in self.root.rglob('*')))

    def test_hyper_connection_compares_the_detached_pair(self):
        script, _, environment, _ = self.fixture('exit 99')
        args = ['bash', str(script), '--hyper-connection', '--dry-run']
        cmd = shlex.split(subprocess.check_output(args, env=environment, text=True))
        self.assertEqual(cmd[cmd.index('--models') + 1:cmd.index('--seeds')],
                         ['lasd_moe_headwise_detach_pin', 'lasd_moe_headwise_detach_hc_pin'])
        self.assertEqual(subprocess.run([*args, '--headwise'], env=environment,
                                        capture_output=True).returncode, 2)

    def test_headwise_v1_keeps_circuits_and_seeds_and_rejects_mixed_presets(self):
        script, _, environment, _ = self.fixture('exit 99')
        script.write_text(script.read_text().replace(
            'CIRCUITS=(two_stage_folded_opamp)', 'CIRCUITS=(Opamp1 two_stage_folded_opamp)'
        ).replace('SEEDS=(17 29 43)', 'SEEDS=(29)'))
        args = ['bash', str(script), '--headwise-v1', '--dry-run']
        output = subprocess.check_output(args, env=environment, text=True)
        commands = [shlex.split(line) for line in output.splitlines()]
        self.assertEqual(len(commands), 2)
        self.assertEqual([cmd[cmd.index('--circuit') + 1] for cmd in commands],
                         ['Opamp1', 'two_stage_folded_opamp'])
        for cmd in commands:
            self.assertEqual(cmd[cmd.index('--models') + 1:cmd.index('--seeds')],
                             ['lasd_moe_headwise_pin_v1', 'lasd_moe_headwise_detach_pin_v1'])
            self.assertEqual(cmd[cmd.index('--seeds') + 1:cmd.index('--device')], ['29'])
        for other in ('--headwise', '--hyper-connection', '--hyper-connection-grad'):
            with self.subTest(preset=other):
                self.assertEqual(subprocess.run([*args, other], env=environment,
                                               capture_output=True).returncode, 2)

    def test_hyper_gradient_preset_preserves_circuits_seeds_and_old_hyper_preset(self):
        script, _, environment, _ = self.fixture('exit 99')
        script.write_text(script.read_text().replace(
            'CIRCUITS=(two_stage_folded_opamp)', 'CIRCUITS=(Opamp1 two_stage_folded_opamp)'
        ).replace('SEEDS=(17 29 43)', 'SEEDS=(29)'))
        args = ['bash', str(script), '--hyper-connection-grad', '--dry-run']
        commands = [shlex.split(line) for line in subprocess.check_output(
            args, env=environment, text=True).splitlines()]
        self.assertEqual([cmd[cmd.index('--circuit') + 1] for cmd in commands],
                         ['Opamp1', 'two_stage_folded_opamp'])
        for cmd in commands:
            self.assertEqual(cmd[cmd.index('--models') + 1:cmd.index('--seeds')],
                             ['lasd_moe_headwise_hc_pin', 'lasd_moe_headwise_detach_hc_pin'])
            self.assertEqual(cmd[cmd.index('--seeds') + 1:cmd.index('--device')], ['29'])
        for other in ('--headwise', '--headwise-v1', '--hyper-connection'):
            with self.subTest(preset=other):
                self.assertEqual(subprocess.run([*args, other], env=environment,
                                               capture_output=True).returncode, 2)

    def test_new_cluster_environment_is_forwarded_to_job(self):
        script, _, environment, _ = self.fixture(
            'test "$JEPA_CONDA_HOME" = /public/home/test/anaconda3 || exit 31\n'
            'test "$JEPA_CONDA_ENV" = newbase || exit 32\necho 98765')
        environment.update(JEPA_CONDA_HOME='/public/home/test/anaconda3',
                           JEPA_CONDA_ENV='newbase')
        run = subprocess.run(['bash', str(script)], env=environment,
                             text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('--account=b_phzhwu', run.stdout)
        self.assertIn('--partition=ex01A800', run.stdout)


class LASDMoEResumeLaunchPinTests(unittest.TestCase):
    setUp = resume_tests.ResumeStagedGateLauncherTests.setUp
    source_fixture = resume_tests.ResumeStagedGateLauncherTests.source_fixture
    inspect = resume_tests.ResumeStagedGateLauncherTests.inspect
    snapshot = resume_tests.ResumeStagedGateLauncherTests.snapshot
    bash_fixture = resume_tests.ResumeStagedGateLauncherTests.bash_fixture

    def test_existing_resume_launcher_accepts_lasd_models_without_raw_import_or_writes(self):
        source, training, _, environment = self.bash_fixture('exit 99')
        config = json.loads((resume_tests.FINAL_ROOT / 'configs/lasd_moe_pin.json').read_text())
        models = ['device_op_cascade_pin', 'lasd_moe_pin', 'lasd_moe_detach_pin',
                  'lasd_moe_headwise_pin', 'lasd_moe_headwise_detach_pin',
                  'lasd_moe_headwise_hc_pin', 'lasd_moe_headwise_detach_hc_pin',
                  'lasd_moe_headwise_pin_v1', 'lasd_moe_headwise_detach_pin_v1']
        record = json.loads((training / 'run_pin.json').read_text())
        record['models'] = models
        for path in (source / 'run_pin.json', training / 'run_pin.json'):
            path.write_text(json.dumps(record))
        for path in (source / 'config_pin.json', training / 'config_pin.json'):
            path.write_text(json.dumps(config))
        script = self.task / 'resume_width_decoder_pin.sh'
        shutil.copyfile(Path(__file__).with_name(script.name), script)
        before = self.snapshot(self.root)
        paths_before = sorted(str(path) for path in self.root.rglob('*'))
        metadata = self.inspect(source)
        self.assertEqual(metadata['models'], models)
        self.assertEqual(metadata['seeds'], resume_tests.SEEDS)
        self.assertEqual(metadata['data_path'], str(source / 'data_pin.json'))
        run = subprocess.run(['bash', str(script), str(source), '--dry-run'],
                             env=environment, text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        command = shlex.split(run.stdout)
        self.assertEqual(command[0], 'sbatch')
        self.assertIn(str(self.root / 'final_exam/common/resume_pin.py'), command)
        self.assertEqual(command[command.index('--resume-from') + 1], str(training))
        self.assertIn('_resume_width_decoder_', command[command.index('--run-id') + 1])
        self.assertNotIn('import_analog_agent', command)
        self.assertEqual(self.snapshot(self.root), before)
        self.assertEqual(sorted(str(path) for path in self.root.rglob('*')), paths_before)
