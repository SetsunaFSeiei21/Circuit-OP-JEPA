"""Clean-room reference checks: graph geometry, budgets and no OP dependence."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from device_jepa.baselines import (SupervisedPerformanceModel, collate_pin_samples,
                                   evaluate_baseline, fit_pin_vocabulary,
                                   train_baseline)
from device_jepa.data import (build_manifest, dataset_fingerprint, fit_preprocessor,
                              samples_for_split, synthetic_dataset)
from device_jepa.model import DeviceEncoder, ModelConfig, make_decoder
from device_jepa.tensors import collate_samples
from device_jepa.training import load_checkpoint, seed_everything, state_digest


def tiny_options(label_fraction=1., no_ood=False):
    options = json.loads((Path(__file__).parents[1] / "configs" / "smoke.json").read_text())
    options["model"].update(dim=8, heads=2, ffn_dim=16, decoder_ffn_dim=16, dropout=0., max_devices=64)
    options["training"].update(pretrain_epochs=1, decoder_epochs=2, baseline_epochs=2, batch_size=8)
    options["dataset"]["finetune_label_fraction"] = label_fraction
    if no_ood:
        options["dataset"]["train_circuits"].append("synthetic_4")
        options["dataset"]["ood_circuits"] = []
    return options


class PinTensorizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything(17)
        cls.data = synthetic_dataset(6)
        cls.manifest = build_manifest(cls.data, "synthetic_4")
        cls.prep = fit_preprocessor(cls.data, cls.manifest, True)
        cls.vocab = fit_pin_vocabulary(cls.data, cls.manifest)

    def example(self):
        # Three devices, two pins on each resistor, four transistor pins. Two
        # named external ports share one net but remain independent port nodes.
        return {"sample_id": "geometry", "circuit_id": "geometry_circuit",
                "ports": {"z_out": "shared", "a_alias": "shared", "isolated": "unused"},
                "devices": [
                    {"device_id": "R2", "device_type": "resistor", "parameters": {"R": 7.},
                     "terminals": {"p": "private_r2", "n": "other"}},
                    {"device_id": "M1", "device_type": "nmos", "parameters": {"W": 2., "L": 3.},
                     "terminals": {"b": "body", "s": "source", "g": "gate", "d": "shared"}},
                    {"device_id": "R1", "device_type": "resistor", "parameters": {"R": 5.},
                     "terminals": {"n": "private_r1", "p": "shared"}}],
                "performance": {"gain": 2., "power": None}}

    def test_node_order_geometry_and_parameter_repetition(self):
        sample = self.example()
        original = copy.deepcopy(sample)
        batch = collate_pin_samples([sample], self.prep, self.vocab)
        ids = batch["device_ids"][0]
        self.assertEqual(ids, ["port:a_alias", "port:isolated", "port:z_out", "pin:M1:d", "pin:M1:g",
                               "pin:M1:s", "pin:M1:b", "pin:R1:p", "pin:R1:n", "pin:R2:p", "pin:R2:n"])
        self.assertEqual(int(batch["device_mask"].sum()), 11)
        adj = batch["adjacency"][0]
        self.assertTrue(torch.equal(adj, adj.T))
        # Complete intra-device clique, including pins on distinct nets.
        self.assertTrue(adj[3:7, 3:7].all())
        self.assertTrue(adj[7, 8])
        self.assertTrue(adj[9, 10])
        self.assertTrue(adj[3, 7])  # M1 drain to R1 positive, same net.
        self.assertFalse(adj[4, 7])  # Different devices, different nets.
        self.assertFalse(adj[7, 9])
        for port in (0, 2):
            self.assertTrue(adj[port, 3] and adj[port, 7])
            self.assertFalse(adj[port, 4] or adj[port, 8])
        self.assertFalse(adj[0, 2])  # No direct port-to-port edge.
        self.assertEqual(int(adj[1].sum()), 1)  # Isolated port's self loop.
        self.assertFalse(batch["parameter_mask"][0, :3].any())
        for pin in range(4, 7):
            torch.testing.assert_close(batch["parameters"][0, pin], batch["parameters"][0, 3])
            self.assertTrue(torch.equal(batch["parameter_mask"][0, pin], batch["parameter_mask"][0, 3]))
        device_batch = collate_samples([sample], self.prep, include_op=False, include_performance=True)
        torch.testing.assert_close(batch["parameters"][0, 3], device_batch["parameters"][0, 0])
        torch.testing.assert_close(batch["performance"], device_batch["performance"])
        self.assertEqual(sample, original)
        # Directional role channels are correct, not symmetric by role index.
        t = len(self.prep["feature_spec"]["terminal_names"])
        roles = self.prep["feature_spec"]["terminal_names"]
        self.assertEqual(float(batch["edge_roles"][0, 3, 7, roles.index("d") * t + roles.index("p")]), 1.)
        self.assertEqual(float(batch["edge_roles"][0, 7, 3, roles.index("p") * t + roles.index("d")]), 1.)
        self.assertFalse(batch["edge_roles"][0, 0].any())
        self.assertFalse(batch["port_roles"].any())

    def test_canonical_order_ignores_incoming_record_order(self):
        sample = self.example()
        reordered = copy.deepcopy(sample)
        reordered["devices"].reverse()
        reordered["ports"] = dict(reversed(list(reordered["ports"].items())))
        for device in reordered["devices"]:
            device["terminals"] = dict(reversed(list(device["terminals"].items())))
        first = collate_pin_samples([sample], self.prep, self.vocab)
        second = collate_pin_samples([reordered], self.prep, self.vocab)
        for key in first:
            if isinstance(first[key], torch.Tensor):
                self.assertTrue(torch.equal(first[key], second[key]), key)
            else:
                self.assertEqual(first[key], second[key])

    def test_no_op_or_state_fields_are_consulted(self):
        sample = self.example()
        before = collate_pin_samples([sample], self.prep, self.vocab)
        sample["op_status"] = "failed"
        for d in sample["devices"]:
            d.update(op={"id": float("nan")}, op_state={"anything": object()}, op_target=False)
        after = collate_pin_samples([sample], self.prep, self.vocab)
        self.assertFalse(any(key.startswith("op") for key in before))
        for key in before:
            if isinstance(before[key], torch.Tensor):
                self.assertTrue(torch.equal(before[key], after[key]), key)

    def test_train_only_vocabulary_unknown_and_padding_are_distinct(self):
        data = copy.deepcopy(self.data)
        target_id = self.manifest["ood"][0]
        sample = next(s for s in data["samples"] if s["sample_id"] == target_id)
        sample["devices"][0]["device_type"] = "never_seen"
        sample["ports"]["never_seen_port"] = "newnet"
        manifest = build_manifest(data, "synthetic_4")
        vocab = fit_pin_vocabulary(data, manifest)
        self.assertEqual(vocab, self.vocab)
        self.assertNotIn("pin:never_seen:d", vocab)
        self.assertNotIn("port:never_seen_port", vocab)
        short = copy.deepcopy(sample)
        short["devices"] = short["devices"][:1]
        batch = collate_pin_samples([sample, short], self.prep, vocab)
        self.assertTrue((batch["device_types"][batch["device_mask"]] > 0).all())
        self.assertTrue((batch["device_types"][~batch["device_mask"]] == 0).all())
        self.assertTrue((batch["device_types"][0] == vocab["<UNK>"]).any())
        self.assertNotEqual(vocab["<PAD>"], vocab["<UNK>"])

    def test_budget_raises_without_truncation(self):
        sample = self.example()
        with self.assertRaisesRegex(ValueError, "no truncation"):
            collate_pin_samples([sample], self.prep, self.vocab, max_tokens=10)
        batch = collate_pin_samples([sample], self.prep, self.vocab, max_tokens=11)
        self.assertEqual(batch["device_types"].shape[1], 11)

    def test_empty_batch_and_invalid_vocab_rejected(self):
        with self.assertRaises(ValueError):
            collate_pin_samples([], self.prep, self.vocab)
        with self.assertRaisesRegex(ValueError, "PAD=0 and UNK=1"):
            collate_pin_samples([self.example()], self.prep, {"<PAD>": 0, "<UNK>": 0})

    def test_padding_null_parameters_and_missing_labels_stay_finite(self):
        sample = self.example()
        short = copy.deepcopy(sample)
        short["sample_id"] = "short"
        short["devices"] = short["devices"][:1]
        short["ports"] = {}
        short["devices"][0]["parameters"] = {}
        short["performance"] = {}
        batch = collate_pin_samples([sample, short], self.prep, self.vocab)
        cfg = ModelConfig(dim=8, heads=2, encoder_layers=2, cross_layers=1, ffn_dim=16,
                          decoder_layers=1, decoder_ffn_dim=16, dropout=0., max_devices=64, terminal_bias=False)
        encoder = DeviceEncoder(cfg, len(self.vocab), len(self.prep["feature_spec"]["parameter_names"]),
                                len(self.prep["feature_spec"]["terminal_names"]))
        model = SupervisedPerformanceModel(encoder, make_decoder(cfg, 2))
        predictions = model(batch)
        self.assertTrue(torch.isfinite(predictions).all())
        predictions.square().sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertFalse(batch["parameter_mask"][1].any())
        self.assertFalse(batch["performance_mask"][1].any())
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.parameters()))


class BaselinePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = synthetic_dataset(6)
        cls.options = tiny_options(label_fraction=.5)
        cls.manifest = build_manifest(cls.data, config=cls.options["dataset"])
        cls.temp = tempfile.TemporaryDirectory()
        cls.initial_fingerprint = dataset_fingerprint(cls.data)
        cls.original_manifest = copy.deepcopy(cls.manifest)
        cls.original_options = copy.deepcopy(cls.options)
        cls.paths = {mode: train_baseline(cls.data, cls.manifest, cls.options, Path(cls.temp.name) / mode, mode)
                     for mode in ("zerosim_device", "zerosim_pin")}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_encoder_and_decoder_update_end_to_end(self):
        for mode, path in self.paths.items():
            source = load_checkpoint(path)
            with self.subTest(mode=mode):
                self.assertNotEqual(source["initial_encoder_digest"], source["encoder_digest"])
                self.assertNotEqual(source["initial_decoder_digest"], source["decoder_digest"])
                self.assertFalse(source["encoder_frozen"])
                self.assertFalse(source["op_inputs_used"])
                self.assertFalse(any("op_encoder" in key for key in source["encoder_state"]))
                self.assertEqual(source["selection"]["split"], "val")
                self.assertEqual(source["training_cost"]["pretrain_epochs"], 0)
                cost = source["training_cost"]
                self.assertEqual(cost["supervised_epochs"], 2)
                self.assertEqual(cost["total_epochs"], 2)
                self.assertEqual(cost["total_parameter_count"], cost["trainable_parameter_count"])
                self.assertEqual(cost["encoder_parameter_count"] + cost["decoder_parameter_count"], cost["total_parameter_count"])
                expected_steps = math.ceil(len(self.manifest["labeled_train"]) / self.options["training"]["batch_size"]) * 2
                self.assertEqual(cost["optimizer_steps"], expected_steps)
                self.assertGreater(cost["wall_time_seconds"], 0.)

    def test_inputs_dataset_manifest_options_and_scalers_are_unchanged(self):
        self.assertEqual(dataset_fingerprint(self.data), self.initial_fingerprint)
        self.assertEqual(self.manifest, self.original_manifest)
        self.assertEqual(self.options, self.original_options)
        expected = fit_preprocessor(self.data, self.manifest, True, self.options["normalization"])
        for path in self.paths.values():
            source = load_checkpoint(path)
            self.assertEqual(source["manifest"], self.manifest)
            self.assertEqual(source["preprocessing"], expected)
            self.assertEqual(source["preprocessing"]["fitted_sample_ids"], self.manifest["train"])
            self.assertEqual(source["preprocessing"]["performance_fitted_sample_ids"], self.manifest["labeled_train"])
        pin = load_checkpoint(self.paths["zerosim_pin"])
        self.assertEqual(pin["pin_vocab_fitted_sample_ids"], self.manifest["train"])
        self.assertFalse(pin["model_config"]["terminal_bias"])
        self.assertTrue(pin["requested_options"]["model"]["terminal_bias"])

    def test_checkpoint_reload_metrics_and_prediction_order(self):
        for mode, path in self.paths.items():
            for split in ("test", "ood"):
                with self.subTest(mode=mode, split=split):
                    report = evaluate_baseline(self.data, path, split=split, batch_size=2)
                    self.assertEqual(report["mode"], mode)
                    self.assertFalse(report["encoder_frozen_verified"])
                    self.assertFalse(report["selection_used_ood"])
                    self.assertTrue(report["synthetic"])
                    self.assertEqual(report["num_samples"], len(self.manifest[split]))
                    self.assertEqual([p["sample_id"] for p in report["predictions"]], self.manifest[split])
                    self.assertEqual(report, evaluate_baseline(self.data, path, split=split, batch_size=2))
                    for metrics in report["metrics"].values():
                        for key, value in metrics.items():
                            if value is not None:
                                self.assertTrue(math.isfinite(value), key)

    def test_report_contract_and_selected_validation_epoch(self):
        reports = {mode: evaluate_baseline(self.data, path) for mode, path in self.paths.items()}
        self.assertEqual(reports["zerosim_device"]["metric_preprocessing_fingerprint"],
                         reports["zerosim_pin"]["metric_preprocessing_fingerprint"])
        for mode, report in reports.items():
            self.assertEqual(report["backend"], f"cleanroom_{mode}")
            self.assertEqual(report["labeled_train_ids"], self.manifest["labeled_train"])
            self.assertEqual(report["evaluation_ids"], self.manifest["test"])
            self.assertEqual(report["data_fingerprint"], self.initial_fingerprint)
            self.assertEqual(report["timings"]["pretrain_seconds"], 0.)
            self.assertGreater(report["timings"]["supervised_seconds"], 0.)
            self.assertEqual(report["provenance"]["parameter_count"]["total"],
                             report["provenance"]["parameter_count"]["trainable"])
            history = json.loads((Path(self.temp.name) / mode / f"{mode}_history.json").read_text())
            selected = min(history, key=lambda row: row["val"]["normalized_mse"])
            self.assertEqual(report["selected_epoch"], selected["epoch"])
            self.assertEqual(report["selection"]["value"], selected["val"]["normalized_mse"])
            rows = {row["sample_id"]: row for row in self.data["samples"]}
            for prediction in report["predictions"]:
                for field in ("prediction", "target", "normalized_prediction", "normalized_target"):
                    self.assertEqual(set(prediction[field]), {"gain", "power"})
                for name in ("gain", "power"):
                    expected = rows[prediction["sample_id"]]["performance"].get(name)
                    self.assertEqual(prediction["target"][name], expected)
                    self.assertEqual(prediction["normalized_target"][name] is None, expected is None)

    def test_explicit_baseline_epoch_override(self):
        options = copy.deepcopy(self.options)
        options["training"]["baseline_epochs"] = 1
        with tempfile.TemporaryDirectory() as folder:
            path = train_baseline(self.data, self.manifest, options, folder)
            source = load_checkpoint(path)
            history = json.loads((Path(folder) / "zerosim_device_history.json").read_text())
            self.assertEqual(len(history), 1)
            self.assertEqual(source["training_cost"]["epochs"], 1)
            self.assertEqual(source["options"]["training"]["decoder_epochs"], 2)
            self.assertEqual(source["options"]["training"]["baseline_epochs"], 1)

    def test_missing_baseline_epoch_budget_cannot_fall_back_to_decoder(self):
        options = copy.deepcopy(self.options)
        del options["training"]["baseline_epochs"]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'baseline'
            with self.assertRaisesRegex(ValueError, 'baseline_epochs.*explicit positive integer'):
                train_baseline(self.data, self.manifest, options, output)
            self.assertFalse(output.exists())

    def test_signed_target_transform_metrics_and_json_round_trip(self):
        data = copy.deepcopy(self.data)
        for index, row in enumerate(data["samples"]):
            if row["performance"].get("gain") is not None:
                row["performance"]["gain"] *= (-1 if index % 2 else 1)
        options = copy.deepcopy(self.options)
        options["training"]["baseline_epochs"] = 1
        options["normalization"]["performance"] = {"gain": {"kind": "asinh", "scale": "fit"}}
        manifest = build_manifest(data, config=options["dataset"])
        with tempfile.TemporaryDirectory() as folder:
            path = train_baseline(data, manifest, options, folder, "zerosim_pin")
            report_path = Path(folder) / "report.json"
            report = evaluate_baseline(data, path, output=report_path)
            self.assertEqual(report, json.loads(report_path.read_text()))
            normalized_errors = [row["normalized_prediction"]["gain"] - row["normalized_target"]["gain"]
                                 for row in report["predictions"] if row["normalized_target"]["gain"] is not None]
            normalized_rmse = math.sqrt(sum(error ** 2 for error in normalized_errors) / len(normalized_errors))
            self.assertAlmostEqual(report["metrics"]["gain"]["normalized_rmse"], normalized_rmse, places=5)

    def test_op_and_state_changes_do_not_change_learned_weights(self):
        changed = copy.deepcopy(self.data)
        for sample in changed["samples"]:
            sample["op_status"] = "failed"
            for device in sample["devices"]:
                device["op"] = {name: -123. for name in changed["feature_spec"]["op_names"]}
                device["op_state"] = {"forward": 1, "saturated": 0}
                device["op_target"] = False
        manifest = build_manifest(changed, config=self.options["dataset"])
        self.assertEqual(manifest["labeled_train"], self.manifest["labeled_train"])
        with tempfile.TemporaryDirectory() as folder:
            for mode in self.paths:
                source = load_checkpoint(self.paths[mode])
                path = train_baseline(changed, manifest, self.options, Path(folder) / mode, mode)
                modified = load_checkpoint(path)
                self.assertEqual(source["encoder_digest"], modified["encoder_digest"])
                self.assertEqual(source["decoder_digest"], modified["decoder_digest"])
                self.assertEqual(source["selection"], modified["selection"])
                self.assertEqual(source["preprocessing"]["parameters"], modified["preprocessing"]["parameters"])
                self.assertEqual(source["preprocessing"]["performance"], modified["preprocessing"]["performance"])

    def test_optional_terminal_bias_is_explicit(self):
        options = copy.deepcopy(self.options)
        options["training"]["decoder_epochs"] = 1
        options["baseline"] = {"pin_terminal_bias": True}
        with tempfile.TemporaryDirectory() as folder:
            path = train_baseline(self.data, self.manifest, options, folder, "zerosim_pin")
            self.assertTrue(load_checkpoint(path)["model_config"]["terminal_bias"])

    def test_empty_ood_is_reported_without_substitution(self):
        options = tiny_options(no_ood=True)
        options["training"]["decoder_epochs"] = 1
        manifest = build_manifest(self.data, config=options["dataset"])
        with tempfile.TemporaryDirectory() as folder:
            path = train_baseline(self.data, manifest, options, folder)
            report = evaluate_baseline(self.data, path, split="ood")
        self.assertEqual(report["status"], "not_configured")
        self.assertFalse(report["has_ood_holdout"])
        self.assertEqual(report["num_samples"], 0)
        self.assertEqual(report["predictions"], [])
        self.assertEqual(report["by_circuit"], {})
        self.assertTrue(all(row["count"] == 0 for row in report["metrics"].values()))

    def test_invalid_mode_batch_size_and_provenance_are_rejected(self):
        with self.assertRaises(ValueError):
            train_baseline(self.data, self.manifest, self.options, self.temp.name, "not_a_mode")
        with self.assertRaises(ValueError):
            evaluate_baseline(self.data, self.paths["zerosim_pin"], batch_size=0)
        source = load_checkpoint(self.paths["zerosim_pin"])
        source["pin_vocab"]["port:leaked_ood"] = len(source["pin_vocab"])
        tampered = Path(self.temp.name) / "tampered.pt"
        torch.save(source, tampered)
        with self.assertRaisesRegex(ValueError, "train-only vocabulary"):
            evaluate_baseline(self.data, tampered)


if __name__ == "__main__":
    unittest.main()
