"""Small deterministic checks for immutable-source, epoch-boundary recovery."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from device_jepa.training import state_digest
from experiments.op_transfer import engine


class TinyContext:
    def __init__(self, output):
        self.output = Path(output)
        self.device = torch.device('cpu')
        self.config = {'model': {'width': 5}, 'dataset': {'circuit': 'fixture'}}
        self.prep = {'normalization': {'mean': [0., 0.], 'scale': [1., 1.]}}
        self.groups = {'train': {'one': list(range(12))}, 'val': {'one': list(range(4))}}
        self.epoch = 0

    def batches(self, split, batch_size, seed=None, **kwargs):
        if seed is not None:
            self.epoch = seed - 17
        indices = list(self.groups[split]['one'])
        if seed is not None:
            random.Random(seed).shuffle(indices)
        for start in range(0, len(indices), batch_size):
            ids = torch.tensor(indices[start:start + batch_size], dtype=torch.float32)
            x = torch.stack((ids / 11., (ids.remainder(3) - 1.) / 2.), -1)
            y = x[:, :1] * .7 - x[:, 1:] * .2
            yield {'x': x, 'y': y, 'device_mask': torch.ones(len(ids), 1, dtype=torch.bool),
                   'epoch': self.epoch}


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 5), nn.Tanh(), nn.Dropout(.35), nn.Linear(5, 1))

    def forward(self, batch):
        return self.layers(batch['x'])


def objective(model, batch):
    prediction = model(batch)
    # All three RNGs affect training, so restoring just torch RNG is insufficient.
    scale = (1. + random.random() / 8. + float(np.random.random()) / 8.) if model.training else 1.
    return (prediction - batch['y']).square().mean() * scale, len(prediction)


class StageResumeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.identity = {'dataset_fingerprint': 'fixture-v1', 'manifest': {'train': [1, 2], 'val': [3]}}
        self.options = engine.settings({'op_transfer': {'batch_size': 4, 'lr': .01,
            'patience': 0, 'progress_every': 1000, 'torch_threads': 1}})

    def model(self):
        torch.manual_seed(91)
        random.seed(92)
        np.random.seed(93)
        return TinyModel()

    def stage(self, name):
        return self.root / name / 'stages_pin' / 'tiny_pin'

    def run_stage(self, name, *, resume_from=None, identity=None, context=None,
                  model=None, objective_fn=objective, epochs=5, **kwargs):
        ctx = context or TinyContext(self.root / name)
        model = self.model() if model is None else model
        with contextlib.redirect_stdout(io.StringIO()):
            return engine.fit_stage(ctx, model, 'tiny_pin', epochs, self.options, 17,
                objective_fn, artifact_suffix='_pin', resume_from=resume_from,
                resume_identity=self.identity if identity is None else identity,
                checkpoint_extra=lambda _: {'selection_note': 'validation-only'}, **kwargs)

    def interrupt_before(self, epoch):
        seen = 0

        def interrupt(_):
            nonlocal seen
            seen += 1
            if seen == epoch:
                raise RuntimeError('simulated power loss')
        return interrupt

    def snapshot(self, directory):
        return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in directory.rglob('*') if p.is_file()}

    def test_epoch_resume_matches_uninterrupted_optimizer_rng_and_selection(self):
        reference, reference_cost = self.run_stage('reference')
        with self.assertRaisesRegex(RuntimeError, 'simulated power loss'):
            self.run_stage('interrupted', before_epoch=self.interrupt_before(3))
        source = self.stage('interrupted')
        self.assertTrue((source / 'last_pin.pt').is_file())
        source_before = self.snapshot(source)
        restored, cost = self.run_stage('resumed', resume_from=source)

        self.assertEqual(state_digest(reference), state_digest(restored))
        for key in ('optimizer_steps', 'examples_seen', 'epochs', 'planned_epochs', 'selection',
                    'selected_state_digest'):
            self.assertEqual(reference_cost[key], cost[key], key)
        self.assertEqual(json.loads((self.stage('reference') / 'history_pin.json').read_text()),
                         json.loads((self.stage('resumed') / 'history_pin.json').read_text()))
        self.assertEqual(source_before, self.snapshot(source))
        selected = torch.load(self.stage('resumed') / 'best_pin.pt', weights_only=True)
        self.assertTrue(selected['complete'])
        self.assertEqual(selected['stage_metadata'], {'selection_note': 'validation-only'})

    def test_completed_stage_is_reused_without_objective_or_optimizer_steps(self):
        reference, reference_cost = self.run_stage('complete')
        source = self.stage('complete')
        source_before = self.snapshot(source)

        def forbidden(*_):
            raise AssertionError('A completed stage must not train or evaluate again')

        restored, cost = self.run_stage('reused', resume_from=source, objective_fn=forbidden)
        self.assertEqual(state_digest(restored), state_digest(reference))
        for key in ('optimizer_steps', 'examples_seen', 'epochs', 'planned_epochs', 'selection'):
            self.assertEqual(cost[key], reference_cost[key], key)
        self.assertTrue((self.stage('reused') / 'best_pin.pt').is_file())
        self.assertEqual(source_before, self.snapshot(source))

    def test_changed_identity_or_configuration_is_rejected_before_output(self):
        self.run_stage('original')
        source = self.stage('original')
        source_before = self.snapshot(source)
        with self.assertRaises((ValueError, RuntimeError)):
            self.run_stage('wrong_identity', resume_from=source,
                           identity={**self.identity, 'dataset_fingerprint': 'different'})
        self.assertFalse(self.stage('wrong_identity').exists())
        ctx = TinyContext(self.root / 'wrong_config')
        ctx.config['model']['width'] = 99
        with self.assertRaises((ValueError, RuntimeError)):
            self.run_stage('wrong_config', resume_from=source, context=ctx)
        self.assertFalse(self.stage('wrong_config').exists())
        self.assertEqual(source_before, self.snapshot(source))

    def test_changed_initial_dependency_or_budget_is_rejected_before_output(self):
        self.run_stage('source')
        source = self.stage('source')
        changed = self.model()
        with torch.no_grad():
            next(changed.parameters()).add_(.25)
        with self.assertRaises((ValueError, RuntimeError)):
            self.run_stage('wrong_dependency', resume_from=source, model=changed)
        self.assertFalse(self.stage('wrong_dependency').exists())
        with self.assertRaises((ValueError, RuntimeError)):
            self.run_stage('wrong_budget', resume_from=source, epochs=6)
        self.assertFalse(self.stage('wrong_budget').exists())

    def test_resume_restores_early_stopping_patience(self):
        self.options['patience'] = 2

        def scheduled(model, batch):
            if model.training:
                return objective(model, batch)
            losses = [None, 1., .9, .91, .92, .93, .94, .95]
            return model(batch).sum() * 0 + losses[batch['epoch']], len(batch['x'])

        expected, expected_cost = self.run_stage('early_reference', objective_fn=scheduled, epochs=7)
        with self.assertRaisesRegex(RuntimeError, 'simulated power loss'):
            self.run_stage('early_partial', objective_fn=scheduled, epochs=7,
                           before_epoch=self.interrupt_before(4))
        actual, cost = self.run_stage('early_resumed', resume_from=self.stage('early_partial'),
                                      objective_fn=scheduled, epochs=7)
        self.assertEqual(cost['epochs'], 4)
        self.assertTrue(cost['selection']['stopped_early'])
        self.assertEqual(cost['selection'], expected_cost['selection'])
        self.assertEqual(state_digest(actual), state_digest(expected))

    def test_legacy_complete_stage_remains_reusable(self):
        expected, expected_cost = self.run_stage('old_complete')
        source = self.stage('old_complete')
        value = torch.load(source / 'best_pin.pt', weights_only=True)
        legacy = {key: value[key] for key in ('state_dict', 'config', 'preprocessing',
                  'stage_metadata', 'cost', 'complete')}
        torch.save(legacy, source / 'best_pin.pt')
        (source / 'last_pin.pt').unlink()
        before = self.snapshot(source)

        def forbidden(*_):
            raise AssertionError('legacy complete stage was retrained')

        actual, cost = self.run_stage('old_reused', resume_from=source, objective_fn=forbidden)
        self.assertEqual(state_digest(actual), state_digest(expected))
        self.assertEqual(cost['optimizer_steps'], expected_cost['optimizer_steps'])
        self.assertEqual(cost['resume']['new_optimizer_steps'], 0)
        self.assertEqual(before, self.snapshot(source))

    def test_legacy_incomplete_best_restarts_stage_budget_and_reports_limit(self):
        with self.assertRaisesRegex(RuntimeError, 'simulated power loss'):
            self.run_stage('old_partial', before_epoch=self.interrupt_before(3))
        source = self.stage('old_partial')
        value = torch.load(source / 'best_pin.pt', weights_only=True)
        legacy = {key: value[key] for key in ('state_dict', 'config', 'preprocessing',
                  'stage_metadata', 'complete', 'selection')}
        self.assertFalse(legacy['complete'])
        torch.save(legacy, source / 'best_pin.pt')
        (source / 'last_pin.pt').unlink()
        before = self.snapshot(source)
        expected_start = self.model()
        expected_start.load_state_dict(legacy['state_dict'])
        observed_start = []

        def observe(model, batch):
            if model.training and not observed_start:
                observed_start.append(state_digest(model))
            return objective(model, batch)

        _, cost = self.run_stage('old_restarted', resume_from=source, objective_fn=observe)
        self.assertEqual(observed_start, [state_digest(expected_start)])
        self.assertEqual(cost['epochs'], 5)
        self.assertEqual(cost['optimizer_steps'], 15)
        self.assertEqual(cost['resume']['policy'], 'legacy_weights_restart')
        self.assertFalse(cost['resume']['optimizer_state_restored'])
        self.assertTrue(cost['resume']['partial_prior_cost_unknown'])
        self.assertEqual(before, self.snapshot(source))

    def test_corrupt_last_checkpoint_is_not_silently_replaced_by_best(self):
        with self.assertRaisesRegex(RuntimeError, 'simulated power loss'):
            self.run_stage('corrupt_source', before_epoch=self.interrupt_before(3))
        source = self.stage('corrupt_source')
        (source / 'last_pin.pt').write_bytes(b'incomplete checkpoint')
        before = self.snapshot(source)
        with self.assertRaises(Exception):
            self.run_stage('corrupt_target', resume_from=source)
        self.assertFalse(self.stage('corrupt_target').exists())
        self.assertEqual(before, self.snapshot(source))

    def test_existing_empty_stage_records_unknown_discarded_work(self):
        source = self.stage('empty_partial')
        source.mkdir(parents=True)
        _, cost = self.run_stage('empty_restart', resume_from=source)
        self.assertEqual(cost['resume']['policy'], 'missing_checkpoint_restart')
        self.assertTrue(cost['resume']['partial_prior_cost_unknown'])
        self.assertFalse(cost['resume']['optimizer_state_restored'])
        self.assertEqual(cost['epochs'], 5)
        self.assertEqual(list(source.iterdir()), [])
        _, fresh = self.run_stage('unstarted', resume_from=self.stage('nonexistent'))
        self.assertEqual(fresh['resume']['policy'], 'fresh')
        self.assertNotIn('partial_prior_cost_unknown', fresh['resume'])

    def test_last_checkpoint_is_self_contained_without_best_or_history_files(self):
        expected, _ = self.run_stage('last_reference')
        with self.assertRaisesRegex(RuntimeError, 'simulated power loss'):
            self.run_stage('last_partial', before_epoch=self.interrupt_before(3))
        source = self.stage('last_partial')
        (source / 'best_pin.pt').unlink()
        (source / 'history_pin.json').unlink()
        before = self.snapshot(source)
        actual, cost = self.run_stage('last_resumed', resume_from=source)
        self.assertEqual(state_digest(actual), state_digest(expected))
        self.assertEqual(cost['resume']['policy'], 'epoch_resume')
        self.assertTrue(cost['resume']['optimizer_state_restored'])
        self.assertEqual(before, self.snapshot(source))

    def test_second_failure_before_next_epoch_retains_recoverable_boundary(self):
        expected, expected_cost = self.run_stage('retry_reference')
        with self.assertRaisesRegex(RuntimeError, 'simulated power loss'):
            self.run_stage('first_failure', before_epoch=self.interrupt_before(3))
        first_source = self.stage('first_failure')
        first_before = self.snapshot(first_source)
        with self.assertRaisesRegex(RuntimeError, 'simulated power loss'):
            self.run_stage('second_failure', resume_from=first_source,
                           before_epoch=self.interrupt_before(1))
        second_source = self.stage('second_failure')
        self.assertTrue((second_source / 'last_pin.pt').is_file())
        self.assertEqual(torch.load(second_source / 'last_pin.pt', weights_only=True)['epoch'], 2)
        self.assertEqual(first_before, self.snapshot(first_source))
        second_before = self.snapshot(second_source)
        actual, cost = self.run_stage('third_attempt', resume_from=second_source)
        self.assertEqual(state_digest(actual), state_digest(expected))
        self.assertEqual(cost['optimizer_steps'], expected_cost['optimizer_steps'])
        self.assertEqual(cost['selection'], expected_cost['selection'])
        self.assertEqual(json.loads((self.stage('third_attempt') / 'history_pin.json').read_text()),
                         json.loads((self.stage('retry_reference') / 'history_pin.json').read_text()))
        self.assertEqual(second_before, self.snapshot(second_source))

    def test_terminal_last_finalizes_without_reentering_training(self):
        def scheduled(model, batch):
            if model.training:
                return objective(model, batch)
            losses = [None, 1., .9, .91, .92, .93, .94, .95]
            return model(batch).sum() * 0 + losses[batch['epoch']], len(batch['x'])

        def forbidden(*_):
            raise AssertionError('A terminal last checkpoint must not train or evaluate again')

        for name, epochs, patience, objective_fn in (
                ('maximum', 5, 0, objective), ('patience', 7, 2, scheduled)):
            with self.subTest(termination=name):
                self.options['patience'] = patience
                expected, expected_cost = self.run_stage(name + '_source', epochs=epochs,
                    objective_fn=objective_fn)
                source = self.stage(name + '_source')
                # Simulate loss of power after the epoch checkpoint but before
                # a finalized selected-weight checkpoint has been written.
                (source / 'best_pin.pt').unlink()
                before = self.snapshot(source)
                actual, cost = self.run_stage(name + '_target', resume_from=source,
                    epochs=epochs, objective_fn=forbidden)
                self.assertEqual(state_digest(actual), state_digest(expected))
                self.assertEqual(cost['selection'], expected_cost['selection'])
                self.assertEqual(cost['epochs'], expected_cost['epochs'])
                self.assertEqual(cost['optimizer_steps'], expected_cost['optimizer_steps'])
                self.assertEqual(cost['resume']['new_epochs'], 0)
                self.assertEqual(cost['resume']['new_optimizer_steps'], 0)
                self.assertEqual(before, self.snapshot(source))

    def test_new_incomplete_best_without_last_explicitly_restarts_stage(self):
        with self.assertRaisesRegex(RuntimeError, 'simulated power loss'):
            self.run_stage('new_best_partial', before_epoch=self.interrupt_before(3))
        source = self.stage('new_best_partial')
        best = torch.load(source / 'best_pin.pt', weights_only=True)
        self.assertIn('resume_identity', best)
        (source / 'last_pin.pt').unlink()
        before = self.snapshot(source)
        _, cost = self.run_stage('new_best_restart', resume_from=source)
        self.assertEqual(cost['resume']['policy'], 'legacy_weights_restart')
        self.assertFalse(cost['resume']['optimizer_state_restored'])
        self.assertTrue(cost['resume']['partial_prior_cost_unknown'])
        self.assertEqual(cost['epochs'], 5)
        self.assertEqual(cost['optimizer_steps'], 15)
        self.assertEqual(before, self.snapshot(source))

    def test_atomic_checkpoint_failure_preserves_previous_snapshot(self):
        from experiments.op_transfer.resume import atomic_torch_save
        path = self.root / 'last_pin.pt'
        atomic_torch_save(path, {'epoch': 1, 'tensor': torch.tensor([1.])})
        before = path.read_bytes()

        def incomplete_save(value, target, *args, **kwargs):
            if hasattr(target, 'write'):
                target.write(b'incomplete')
            else:
                Path(target).write_bytes(b'incomplete')
            raise OSError('simulated interrupted write')

        with patch('experiments.op_transfer.resume.torch.save', side_effect=incomplete_save):
            with self.assertRaisesRegex(OSError, 'interrupted write'):
                atomic_torch_save(path, {'epoch': 2})
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(torch.load(path, weights_only=True)['epoch'], 1)


if __name__ == '__main__':
    unittest.main()
