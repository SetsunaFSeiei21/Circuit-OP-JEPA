"""Raw circuit selection imports on the worker, then starts a fresh experiment."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments.op_transfer import run, run_raw


ROOT = Path(__file__).resolve().parents[2]


def snapshot(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob('*') if path.is_file()}


class RawTransferTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.experiment = self.root / 'experiment'
        self.raw = self.root / 'raw circuits'
        self.output = self.experiment / 'results' / 'single'
        self.config = self.root / 'config.json'
        self.config.write_text((ROOT / 'experiments/op_transfer/example_config.json').read_text())
        self.source('CircuitA')
        self.enterContext(patch.object(run, 'EXPERIMENT_ROOT', self.experiment))
        if hasattr(run_raw, 'EXPERIMENT_ROOT'):
            self.enterContext(patch.object(run_raw, 'EXPERIMENT_ROOT', self.experiment))
        self.capture = io.StringIO()
        self.enterContext(contextlib.redirect_stdout(self.capture))
        self.enterContext(contextlib.redirect_stderr(self.capture))

    def source(self, directory_name, circuit_name=None):
        circuit = circuit_name or directory_name
        directory = self.raw / directory_name
        directory.mkdir(parents=True)
        with sqlite3.connect(directory / 'sampling_history.sqlite3') as db:
            db.execute('CREATE TABLE dataset_schema(circuit_name TEXT)')
            db.execute('INSERT INTO dataset_schema VALUES (?)', (circuit,))
            db.execute('CREATE TABLE samples(sample_id INTEGER)')
            db.executemany('INSERT INTO samples VALUES (?)', [(i,) for i in range(8)])
        (directory / f'{circuit}.sp').write_text(f'.SUBCKT {circuit}\n.ENDS\n')
        return directory

    def args(self, *extra):
        return ['--circuit','CircuitA','--raw-root',str(self.raw),
                '--config',str(self.config),'--output',str(self.output),'--device','cpu',*extra]

    def invoke(self, args=None, *, complete=True):
        commands=[]
        def child(command, **kwargs):
            command=list(map(str,command))
            commands.append(command)
            self.assertEqual(Path(kwargs['cwd']), ROOT)
            self.assertTrue(kwargs['check'])
            output=Path(command[command.index('--output')+1])
            if '-m' in command:
                output.write_text('{"imported": true}')
            else:
                output.mkdir(parents=True)
                (output/'run.json').write_text(json.dumps({'status':'complete' if complete else 'running'}))
                (output/'results.json').write_text('[]')
                (output/'summary.xlsx').write_bytes(b'mocked Excel')
            return subprocess.CompletedProcess(command,0)
        with patch.object(run_raw.subprocess,'run',side_effect=child):
            run_raw.main(args or self.args())
        return commands

    def test_single_circuit_imports_fresh_data_then_only_selected_models(self):
        before=snapshot(self.raw)
        commands=self.invoke(self.args('--models','op_distill','predicted_op',
            '--seeds','17','29','--max-train','32','--max-val','8','--max-test','8'))
        self.assertEqual(len(commands),2)
        saved=json.loads((self.output/'config.json').read_text())
        self.assertEqual(saved['dataset']['train_circuits'],['CircuitA'])
        self.assertEqual(saved['dataset']['ood_circuits'],[])
        first,second=commands
        self.assertEqual(first[1:4],['-m','device_jepa','import_analog_agent'])
        self.assertEqual(first[first.index('--input')+1],str(self.raw/'CircuitA'))
        self.assertEqual(first[first.index('--max-table-rows')+1],'none')
        self.assertEqual(second[second.index('--data')+1],str(self.output/'data.json'))
        self.assertEqual(second[second.index('--output')+1],str(self.output/'results'))
        self.assertIn(str(self.experiment/'run.py'),second)
        index=second.index('--models')+1
        self.assertEqual(second[index:index+2],['op_distill','predicted_op'])
        self.assertNotIn('--manifest',second)
        self.assertEqual(second[second.index('--max-train')+1],'32')
        self.assertEqual(second[second.index('--max-val')+1],'8')
        self.assertEqual(second[second.index('--max-test')+1],'8')
        self.assertEqual(snapshot(self.raw),before)

    def test_joint_id_and_ood_roles_are_explicit_and_disjoint(self):
        self.source('CircuitB'); self.source('HeldOut')
        commands=self.invoke(['--train-circuits','CircuitA','CircuitB',
            '--ood-circuits','HeldOut','--raw-root',str(self.raw),'--config',str(self.config),
            '--output',str(self.output),'--device','cpu','--models','dual_distill'])
        saved=json.loads((self.output/'config.json').read_text())
        self.assertEqual(saved['dataset']['train_circuits'],['CircuitA','CircuitB'])
        self.assertEqual(saved['dataset']['ood_circuits'],['HeldOut'])
        first=commands[0]
        self.assertEqual(first[first.index('--input')+1:first.index('--config')],
            [str(self.raw/name) for name in ('CircuitA','CircuitB','HeldOut')])

    def test_multiple_id_circuits_can_run_without_ood(self):
        self.source('CircuitB')
        self.invoke(['--train-circuits','CircuitA','CircuitB','--raw-root',str(self.raw),
            '--config',str(self.config),'--output',str(self.output),'--device','cpu'])
        saved=json.loads((self.output/'config.json').read_text())
        self.assertEqual(saved['dataset']['train_circuits'],['CircuitA','CircuitB'])
        self.assertEqual(saved['dataset']['ood_circuits'],[])

    def test_existing_output_rejected_before_source_read_or_write(self):
        self.output.mkdir(parents=True)
        (self.output/'data.json').write_text('old dataset')
        (self.output/'old_report.json').write_text('old report')
        before=snapshot(self.experiment)
        with patch('scripts.run.run_raw_comparison.source_identity',side_effect=AssertionError('source read too early')), \
             patch.object(run_raw.subprocess,'run') as child, \
             self.assertRaises((SystemExit,ValueError)):
            run_raw.main(self.args())
        child.assert_not_called()
        self.assertEqual(snapshot(self.experiment),before)

    def test_bad_selection_never_imports_or_allocates_output(self):
        for extra in (['--models','not_a_model'],['--models','op_distill','op_distill'],
                      ['--seeds','17','17'],['--seeds','-1'],['--seeds',str(2**32)],
                      ['--max-train','0'],['--max-table-rows','0']):
            with self.subTest(extra=extra),patch.object(run_raw.subprocess,'run') as child, \
                 self.assertRaises((SystemExit,ValueError)):
                run_raw.main(self.args(*extra))
            child.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_sqlite_identity_mismatch_is_not_silently_renamed(self):
        self.source('WrongDirectory','ActualCircuit')
        args=self.args(); args[args.index('--circuit')+1]='WrongDirectory'
        with patch.object(run_raw.subprocess,'run') as child, self.assertRaises((SystemExit,ValueError)):
            run_raw.main(args)
        child.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_overlapping_or_duplicate_roles_rejected(self):
        for roles in (['--train-circuits','CircuitA','--ood-circuits','CircuitA'],
                      ['--train-circuits','CircuitA','CircuitA'],
                      ['--circuit','../outside']):
            args=[*roles,'--raw-root',str(self.raw),'--config',str(self.config),
                  '--output',str(self.output),'--device','cpu']
            with self.subTest(roles=roles),patch.object(run_raw.subprocess,'run') as child, \
                 self.assertRaises((SystemExit,ValueError)):
                run_raw.main(args)
            child.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_import_failure_never_starts_training(self):
        def fail(command,**kwargs):
            raise subprocess.CalledProcessError(1,command)
        with patch.object(run_raw.subprocess,'run',side_effect=fail) as child, \
             self.assertRaises(subprocess.CalledProcessError):
            run_raw.main(self.args())
        self.assertEqual(child.call_count,1)
        self.assertFalse((self.output/'results').exists())
        record=json.loads((self.output/'raw_run.json').read_text())
        self.assertEqual(record['status'],'failed')
        self.assertEqual(record['failure_type'],'CalledProcessError')

    def test_incomplete_training_is_not_reported_complete(self):
        with self.assertRaises((SystemExit,ValueError)):
            self.invoke(complete=False)
        self.assertEqual(json.loads((self.output/'raw_run.json').read_text())['status'],'failed')

    def test_default_output_name_includes_selected_circuit(self):
        args=self.args()
        index=args.index('--output'); del args[index:index+2]
        commands=self.invoke(args)
        path=Path(commands[0][commands[0].index('--output')+1]).parent
        self.assertEqual(path.parent,self.experiment/'results')
        self.assertTrue(path.name.startswith('CircuitA_'),path)

    def test_raw_slurm_preview_dispatches_without_reading_or_writing(self):
        source=ROOT/'experiments/op_transfer/submit.sh'
        script=self.experiment/'submit.sh'
        script.parent.mkdir(parents=True,exist_ok=True)
        script.write_text(source.read_text())
        for roles in (['--circuit','CircuitA'],['--circuit=CircuitA'],
                      ['--train-circuits','CircuitA','--ood-circuits','HeldOut']):
            result=subprocess.run(['bash',str(script),'--dry-run',*roles,
                '--raw-root','/not present on login node','--models','op_distill'],
                env={**os.environ,'DRY_RUN':'1'},text=True,capture_output=True,check=True)
            self.assertIn('run_raw.py',result.stdout)
            self.assertNotIn('UNEXPECTED',result.stderr)
            self.assertFalse((self.experiment/'logs').exists())
            self.assertFalse((self.experiment/'results').exists())


if __name__=='__main__':
    unittest.main()
