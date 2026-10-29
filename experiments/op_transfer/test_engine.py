"""Small complete CPU experiments: selection, replay and held-out boundaries."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from device_jepa.data import build_manifest, samples_for_split, synthetic_dataset
from device_jepa.tensors import collate_samples
from experiments.op_transfer import engine


ROOT = Path(__file__).resolve().parents[2]


def small_config():
    config = json.loads((ROOT / "configs/smoke.json").read_text())
    config["model"].update(dim=8, heads=2, encoder_layers=2, cross_layers=1,
                           ffn_dim=16, decoder_ffn_dim=16, dropout=0.)
    config["dataset"]["split_ratios"] = [.6, .2, .2]
    config["op_transfer"] = {"epochs": dict.fromkeys(engine.DEFAULT_EPOCHS, 1),
                             "batch_size": 8, "patience": 0,
                             "progress_every": 1000, "lr": .001}
    return config


def fixed_topology_data():
    data = synthetic_dataset(12)
    # Preserve varying design values while making enough repeats of each device
    # identity in both train and val for conditional-variance ablations.
    for sample in data["samples"]:
        sample["devices"] = sample["devices"][:4]
    return data


def quiet_run(*args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return engine.run_experiment(*args, **kwargs)


class TransferEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.data, cls.config = fixed_topology_data(), small_config()
        cls.data_path, cls.config_path = cls.root / "data.json", cls.root / "config.json"
        cls.data_path.write_text(json.dumps(cls.data))
        cls.config_path.write_text(json.dumps(cls.config))
        cls.output = cls.root / "all_models"
        cls.result = quiet_run(cls.data_path, cls.config_path, cls.output,
                               list(engine.MODELS), [17], "cpu")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def checkpoint(self, mode):
        return self.output / "seed_17" / mode / "checkpoint.pt"

    def test_all_models_complete_and_report_test_and_ood(self):
        run = json.loads((self.output / "run.json").read_text())
        self.assertEqual(run["status"], "complete")
        results = self.result["results"]
        self.assertEqual({row["mode"] for row in results}, set(engine.MODELS))
        self.assertEqual(len(results), len(engine.MODELS) * 2)
        for result in results:
            self.assertEqual(result["report"]["requires_measured_op"], result["mode"] in engine.MEASURED_OP_MODELS)
            self.assertTrue(self.checkpoint(result["mode"]).is_file())
            self.assertGreater(result["report"]["count"], 0)
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertTrue(summary)
        self.assertTrue(all(row["seeds"] == 1 and row["std"] is None for row in summary))

    def test_teacher_and_predictor_costs_are_deduplicated(self):
        costs = json.loads((self.output / "stages.json").read_text())
        ids = [row["stage_id"] for row in costs["unique_stages"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids.count("seed_17/teacher"), 1)
        self.assertEqual(ids.count("seed_17/op_predictor"), 1)
        self.assertEqual(ids.count("seed_17/distillation"), 1)
        self.assertEqual(ids.count("seed_17/design_op_teacher"), 1)
        self.assertEqual(ids.count("seed_17/student_predicted_op"), 1)
        self.assertEqual(ids.count("seed_17/predicted_op_distill_encoder"), 1)
        self.assertEqual(ids.count("seed_17/predicted_op_distill_decoder"), 1)
        self.assertEqual(costs["optimizer_steps_deduplicated"],
                         sum(row["optimizer_steps"] for row in costs["unique_stages"]))
        for mode in ("op_distill", "op_distill_finetune", "dual_distill"):
            _, payload = engine.load_model(self.checkpoint(mode))
            self.assertIn("seed_17/teacher", payload["stage_ids"])
            self.assertIn("seed_17/distillation", payload["stage_ids"])

    def test_final_replay_needs_no_teacher_or_measured_op(self):
        # Reconstructed deployment modules have no teacher module, and produce
        # exactly the same result if arbitrary measured OP/labels are supplied.
        rows = self.data["samples"][-3:]
        for mode in engine.MODELS:
            if mode in engine.MEASURED_OP_MODELS:
                continue
            with self.subTest(mode=mode):
                model, payload = engine.load_model(self.checkpoint(mode))
                self.assertFalse(payload["inference_requires_op"])
                self.assertFalse(any("teacher" in name for name, _ in model.named_modules()))
                batch = collate_samples(rows, payload["preprocessing"], include_op=False,
                                         include_performance=False, include_op_states=False)
                engine.attach_identity(batch, rows)
                polluted = copy.deepcopy(batch)
                shape = (*batch["device_types"].shape, len(payload["preprocessing"]["feature_spec"]["op_names"]))
                polluted.update(op=torch.full(shape, float("nan")),
                                op_mask=torch.ones(shape, dtype=torch.bool),
                                op_target_mask=torch.zeros_like(batch["device_mask"]),
                                performance=torch.tensor(float("nan")))
                replay, _ = engine.load_model(self.checkpoint(mode))
                with torch.no_grad():
                    before, after, restored = model(batch), model(polluted), replay(batch)
                self.assertTrue(torch.isfinite(before).all())
                torch.testing.assert_close(before, after, rtol=0, atol=0)
                torch.testing.assert_close(before, restored, rtol=0, atol=0)

    def test_transforms_and_latent_scales_fit_selected_train_only(self):
        cohort = json.loads((self.output / "cohort.json").read_text())
        prep = json.loads((self.output / "preprocessing.json").read_text())
        self.assertEqual(set(prep["fitted_sample_ids"]), set(cohort["train"]))
        self.assertEqual(set(prep["performance_fitted_sample_ids"]), set(cohort["labeled_train"]))
        heldout = set(cohort["val"] + cohort["test"] + cohort["ood"])
        self.assertFalse(set(prep["fitted_sample_ids"]) & heldout)
        lookup = {row["sample_id"]: row for row in self.data["samples"]}
        train_keys = {key for sid in cohort["train"] for key in engine.group_keys(lookup[sid])}
        for mode in ("jepa_standardized", "jepa_grouped", "jepa_residual", "jepa_op_aux", "op_distill"):
            model, _ = engine.load_model(self.checkpoint(mode))
            self.assertEqual(model.scaler.fitted_samples, len(cohort["train"]))
            self.assertEqual(set(model.scaler.groups), train_keys)
        predicted, _ = engine.load_model(self.checkpoint("predicted_op"))
        self.assertEqual(set(predicted.predictor.support["fit_sample_ids"]), set(cohort["train"]))

    def test_model_selection_does_not_train_unused_dependencies(self):
        output = self.root / "design_only"
        result = quiet_run(self.data_path, self.config_path, output,
                           ["design_supervised"], [29], "cpu")
        self.assertEqual([row["stage_id"] for row in result["stages"]], ["seed_29/design_supervised"])
        self.assertEqual({p.name for p in (output / "seed_29/stages").iterdir()}, {"design_supervised"})
        self.assertFalse((output / "seed_29/op_oracle").exists())

    def test_untrained_metric_extreme_is_excluded_before_float32(self):
        data = copy.deepcopy(self.data)
        manifest = build_manifest(data, config=self.config["dataset"])
        training = set(manifest["train"])
        data["feature_spec"]["metric_names"].append("heldout_only")
        for row in data["samples"]:
            row["performance"]["heldout_only"] = None if row["sample_id"] in training else 1e100
        path, output = self.root / "extreme_data.json", self.root / "extreme_run"
        path.write_text(json.dumps(data))
        result = quiet_run(path, self.config_path, output, ["design_supervised"], [17], "cpu")
        for row in result["results"]:
            self.assertIn("heldout_only", row["report"]["excluded_untrained_metrics"])
            self.assertEqual(row["report"]["metrics"]["heldout_only"]["count"], 0)
        _, saved = engine.load_model(output / "seed_17/design_supervised/checkpoint.pt")
        self.assertEqual(saved["preprocessing"]["performance"]["count"][-1], 0)

    def test_refuse_existing_output_before_any_write(self):
        output = self.root / "preexisting"
        output.mkdir()
        original = {"data.json": b"old dataset", "report.json": b"old report"}
        for name, value in original.items():
            (output / name).write_bytes(value)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            quiet_run("missing_data.json", "missing_config.json", output,
                       ["design_supervised"], [17], "cpu")
        self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, original)

    def test_selected_model_order_does_not_change_shared_student_or_readout(self):
        outputs = [self.root / "distill_alone", self.root / "distill_reordered"]
        modes = [["op_distill"], ["dual_distill", "op_distill"]]
        for output, selected in zip(outputs, modes):
            quiet_run(self.data_path, self.config_path, output, selected, [17], "cpu")
        from device_jepa.training import state_digest
        states = [engine.load_model(output / "seed_17/op_distill/checkpoint.pt")[0] for output in outputs]
        self.assertEqual(state_digest(states[0]), state_digest(states[1]))
        a = json.loads((outputs[0] / "seed_17/op_distill/test.json").read_text())
        b = json.loads((outputs[1] / "seed_17/op_distill/test.json").read_text())
        self.assertEqual(a, b)

    def test_untrained_op_field_extreme_does_not_enter_teacher_or_cascade(self):
        data = copy.deepcopy(self.data)
        manifest = build_manifest(data, config=self.config["dataset"])
        training = set(manifest["train"])
        data["feature_spec"]["op_names"].append("never_trained_op")
        for row in data["samples"]:
            for device in row["devices"]:
                device["op"]["never_trained_op"] = None if row["sample_id"] in training else 1e100
        path, output = self.root / "op_extreme.json", self.root / "op_extreme_run"
        path.write_text(json.dumps(data))
        quiet_run(path, self.config_path, output, ["op_oracle", "predicted_op", "op_distill"], [17], "cpu")
        model, saved = engine.load_model(output / "seed_17/predicted_op/checkpoint.pt")
        self.assertEqual(saved["preprocessing"]["op"]["count"][-1], 0)
        self.assertFalse(bool(model.predictor.type_support[:, -1].any()))
        report = json.loads((output / "seed_17/op_predictor/test_op.json").read_text())
        self.assertEqual(report["metrics"]["never_trained_op"]["count"], 0)

    def test_test_and_ood_outcomes_do_not_change_training_or_selection(self):
        modified = copy.deepcopy(self.data)
        manifest = build_manifest(modified, config=self.config["dataset"])
        heldout = set(manifest["test"] + manifest["ood"])
        for sample in modified["samples"]:
            if sample["sample_id"] not in heldout:
                continue
            sample["performance"] = {name: None if value is None else value * 12. + 37.
                                     for name, value in sample["performance"].items()}
            for dev in sample["devices"]:
                dev["op"] = {name: None if value is None else value * 3. + .001
                             for name, value in dev["op"].items()}
        mutated_path = self.root / "heldout_mutated.json"
        mutated_path.write_text(json.dumps(modified))
        outputs = [self.root / "heldout_original", self.root / "heldout_mutated"]
        for path, output in zip((self.data_path, mutated_path), outputs):
            quiet_run(path, self.config_path, output,
                       ["design_supervised", "op_oracle", "op_distill", "student_predicted_op",
                        "student_predicted_op_distill"], [17, 29], "cpu")
        a, b = (json.loads((output / "stages.json").read_text())["unique_stages"] for output in outputs)
        self.assertEqual({stage["stage_id"]: stage["selected_state_digest"] for stage in a},
                         {stage["stage_id"]: stage["selected_state_digest"] for stage in b})
        self.assertEqual(json.loads((outputs[0] / "preprocessing.json").read_text()),
                         json.loads((outputs[1] / "preprocessing.json").read_text()))

    def test_standalone_prediction_accepts_no_outcomes_and_exports_physical_op(self):
        from experiments.op_transfer.predict import predict
        data = copy.deepcopy(self.data)
        data["samples"] = data["samples"][-3:]
        for sample in data["samples"]:
            sample.pop("performance")
            sample["op_status"] = "arbitrary ignored simulation metadata"
            for dev in sample["devices"]:
                dev.pop("op")
                dev["op_state"] = {"invalid": float("nan")}
        path = self.root / "design_only_inputs.json"
        path.write_text(json.dumps(data))
        original = path.read_bytes()
        for mode in ("op_distill", "predicted_op", "op_predictor", "student_predicted_op",
                     "student_predicted_op_distill"):
            output = self.root / f"{mode}_predictions.json"
            with contextlib.redirect_stdout(io.StringIO()):
                result = predict(self.output / f"seed_17/{mode}/checkpoint.pt", path,
                                 output, device="cpu", include_predicted_op=mode in {"predicted_op", "student_predicted_op"})
            self.assertFalse(result["requires_measured_op"])
            self.assertEqual(len(result["predictions"]), 3)
            self.assertEqual(path.read_bytes(), original)
            if mode in ("predicted_op", "op_predictor", "student_predicted_op"):
                self.assertIn("predicted_op", result["predictions"][0])
                device = result["predictions"][0]["predicted_op"][0]
                self.assertIn("physical", device)
                self.assertIn("normalized", device)
            if mode != "op_predictor":
                self.assertIn("performance", result["predictions"][0])
            snapshot = output.read_bytes()
            with self.assertRaisesRegex(ValueError, "new file"):
                predict("missing_checkpoint", "missing_data", output)
            self.assertEqual(output.read_bytes(), snapshot)

    def test_jepa_loss_components_reconstruct_reported_objective(self):
        for mode in ('jepa_standardized','jepa_grouped','jepa_residual','jepa_op_aux'):
            history = json.loads((self.output / f'seed_17/stages/{mode}_pretrain/history.json').read_text())
            for epoch in history:
                for phase in ('train','val'):
                    parts=epoch[phase]['loss_components']
                    weighted=sum(v for k,v in parts.items() if k.startswith('weighted_'))
                    self.assertAlmostEqual(weighted,epoch[phase]['loss'],delta=1e-5)

    def test_predicted_student_provenance_scaling_and_teacher_free_checkpoint(self):
        from device_jepa.training import state_digest
        audit = json.loads((self.output / 'seed_17/student_predicted_op/distillation.json').read_text())
        for subject in ('teacher', 'predictor'):
            self.assertEqual(audit[f'{subject}_digest_before'], audit[f'{subject}_digest_after'])
        for part in ('encoder', 'decoder'):
            self.assertEqual(audit[f'initial_student_{part}_digest'], audit[f'teacher_{part}_digest'])
        cohort = json.loads((self.output / 'cohort.json').read_text())
        stage = self.output / 'seed_17/stages/student_predicted_op'
        scales = torch.load(stage / 'scales.pt', weights_only=True)
        self.assertEqual(scales['fit_sample_ids'], sorted(cohort['train']))
        self.assertEqual(audit['scaler_fit_samples'], len(cohort['train']))
        self.assertTrue(scales['loss_only'])
        model, payload = engine.load_model(self.checkpoint('student_predicted_op'))
        predictor, _ = engine.load_model(self.checkpoint('op_predictor'))
        self.assertEqual(state_digest(model.predictor), state_digest(predictor))
        self.assertEqual(payload['recipe']['kind'], 'cascade')
        self.assertIsNone(payload['recipe']['downstream']['scaler'])
        self.assertEqual(payload['stage_ids'], ['seed_17/design_op_teacher', 'seed_17/op_predictor',
                                              'seed_17/student_predicted_op'])
        self.assertFalse(any('teacher' in key for key in payload['state_dict']))

    def test_predicted_student_selects_performance_only_and_logs_training_components(self):
        stage = self.output / 'seed_17/stages/student_predicted_op'
        history = json.loads((stage / 'history.json').read_text())
        for epoch in history:
            parts = epoch['train']['loss_components']
            self.assertAlmostEqual(parts['weighted_performance'] + parts['weighted_alignment'],
                                   epoch['train']['loss'], delta=2e-5)
        model, saved = engine.load_model(self.checkpoint('student_predicted_op'))
        rows = samples_for_split(self.data, saved['manifest'], 'val')
        batch = collate_samples(rows, saved['preprocessing'], include_op=False,
                                include_performance=True, include_op_states=False)
        with torch.no_grad():
            prediction = model(batch)
            expected = float(engine.masked_mse(prediction, batch['performance'].to(prediction.dtype),
                                               batch['performance_mask']))
        self.assertAlmostEqual(history[0]['val']['loss'], expected, delta=2e-5)

    def test_predicted_student_only_trains_required_stages_and_order_is_stable(self):
        from device_jepa.training import state_digest
        output = self.root / 'predicted_student_only'
        result = quiet_run(self.data_path, self.config_path, output,
                           ['student_predicted_op'], [17], 'cpu')
        self.assertEqual([s['stage_id'] for s in result['stages']],
                         ['seed_17/design_op_teacher', 'seed_17/op_predictor', 'seed_17/student_predicted_op'])
        self.assertFalse((output / 'seed_17/design_op_oracle').exists())
        alone, _ = engine.load_model(output / 'seed_17/student_predicted_op/checkpoint.pt')
        together, _ = engine.load_model(self.checkpoint('student_predicted_op'))
        self.assertEqual(state_digest(alone), state_digest(together))

    def test_sequential_student_freeze_audit_and_replay(self):
        mode = 'student_predicted_op_distill'
        audit = json.loads((self.output / f'seed_17/{mode}/distillation.json').read_text())
        self.assertEqual(audit['training_scheme'], 'encoder_then_decoder')
        self.assertEqual(audit['initial_student_decoder_digest'], audit['decoder_digest_after_encoder'])
        self.assertEqual(audit['encoder_digest_before_decoder'], audit['encoder_digest_after_decoder'])
        for subject in ('teacher', 'predictor'):
            self.assertEqual(audit[f'{subject}_digest_before'], audit[f'{subject}_digest_after'])
        self.assertFalse(audit['joint_student_weights_used'])
        model, payload = engine.load_model(self.checkpoint(mode))
        self.assertTrue(model.downstream.frozen)
        self.assertTrue(all(not p.requires_grad for p in model.downstream.encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.downstream.decoder.parameters()))
        self.assertEqual(payload['stage_ids'], ['seed_17/design_op_teacher', 'seed_17/op_predictor',
                                              'seed_17/predicted_op_distill_encoder',
                                              'seed_17/predicted_op_distill_decoder'])
        encoder = json.loads((self.output / 'seed_17/stages/predicted_op_distill_encoder/history.json').read_text())
        for epoch in encoder:
            for phase in ('train', 'val'):
                detail = epoch[phase]['loss_components']
                self.assertAlmostEqual(detail['alignment_mse'], epoch[phase]['loss'], delta=1e-5)
                self.assertNotIn('perf_mse', detail)
        stage = torch.load(self.output/'seed_17/stages/predicted_op_distill_encoder/best.pt', weights_only=True)
        self.assertEqual(stage['stage_metadata']['fit_sample_ids'],
                         sorted(json.loads((self.output/'cohort.json').read_text())['train']))

    def test_sequential_student_stage_budgets_are_independent_and_order_stable(self):
        from device_jepa.training import state_digest
        config = copy.deepcopy(self.config)
        config['op_transfer']['epochs'].update(predicted_op_distill_encoder=2,
                                              predicted_op_distill_decoder=3)
        path = self.root/'sequential_budget.json'
        path.write_text(json.dumps(config))
        output = self.root/'sequential_budget'
        result = quiet_run(self.data_path, path, output, ['student_predicted_op_distill'], [17], 'cpu')
        self.assertEqual({s['stage_id']:s['epochs'] for s in result['stages']}, {
            'seed_17/design_op_teacher':1, 'seed_17/op_predictor':1,
            'seed_17/predicted_op_distill_encoder':2, 'seed_17/predicted_op_distill_decoder':3})
        self.assertFalse((output/'seed_17/student_predicted_op').exists())
        for key in ('predicted_op_distill_encoder', 'predicted_op_distill_decoder'):
            for value in (0, -1, True, 1.5):
                bad = copy.deepcopy(config)
                bad['op_transfer']['epochs'][key] = value
                with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                    engine.settings(bad)
        output = self.root/'sequential_alone'
        quiet_run(self.data_path, self.config_path, output, ['student_predicted_op_distill'], [17], 'cpu')
        alone, _ = engine.load_model(output/'seed_17/student_predicted_op_distill/checkpoint.pt')
        together, _ = engine.load_model(self.checkpoint('student_predicted_op_distill'))
        self.assertEqual(state_digest(alone), state_digest(together))

    def test_student_loss_weights_validate_before_output(self):
        for key, values in {'student_alignment_weight': [-1, float('nan'), True],
                            'student_performance_weight': [0, -1, float('inf'), True]}.items():
            for value in values:
                config = copy.deepcopy(self.config)
                config['op_transfer'][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    engine.settings(config)
        config = copy.deepcopy(self.config)
        config['op_transfer']['student_alignment_weight'] = 0.
        self.assertEqual(engine.settings(config)['student_alignment_weight'], 0.)

    def test_op_validation_uses_global_equal_field_mse(self):
        from experiments.op_transfer.losses import masked_field_mse
        data=copy.deepcopy(self.data)
        field=data['feature_spec']['op_names'][0]
        for index,sample in enumerate(data['samples']):
            if index%3:
                for dev in sample['devices']: dev['op'][field]=None
        path=self.root/'uneven_op.json'; path.write_text(json.dumps(data))
        output=self.root/'uneven_op'
        result=quiet_run(path,self.config_path,output,['predicted_op'],[17],'cpu')
        model,saved=engine.load_model(output/'seed_17/op_predictor/checkpoint.pt')
        rows=samples_for_split(data,saved['manifest'],'val')
        batch=collate_samples(rows,saved['preprocessing'],include_op=True,
            include_performance=False,include_op_states=False)
        with torch.no_grad():
            pred,support=model(batch)
            expected=float(masked_field_mse(pred,batch['op'],support & batch['op_mask']))
        stage=next(s for s in result['stages'] if s['stage_id'].endswith('/op_predictor'))
        self.assertAlmostEqual(stage['selection']['best_value'],expected,delta=2e-6)


if __name__ == "__main__":
    unittest.main()
