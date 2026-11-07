"""Tree regressors retain missing-label boundaries and native replay."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from device_jepa.data import build_manifest, synthetic_dataset
from final_exam.data_pin import PinDataContext
from final_exam.tree_pin import pin_features_pin, fit_xgboost_pin, load_xgboost_pin, tree_settings_pin


class TreePinTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.data = synthetic_dataset(24)
        self.data["samples"] = [row for row in self.data["samples"] if row["circuit_id"] == "synthetic_0"]
        self.data["feature_spec"]["metric_names"].append("untrained")
        for row in self.data["samples"]:
            row["devices"] = row["devices"][:4]
            row['ports'] = {'VIN': next(iter(row['devices'][0]['terminals'].values())),
                            'VOUT': next(iter(row['devices'][1]['terminals'].values()))}
            row["performance"]["untrained"] = None
        self.config = dict(model={"max_devices": 32},
            dataset=dict(train_circuits=["synthetic_0"], ood_circuits=[],
                         split_ratios=[0.6, 0.2, 0.2], split_seed=17, finetune_label_fraction=1.0),
            normalization={}, final_exam_pin={"xgboost": {"num_boost_round": 4, "max_depth": 2}})
        manifest = build_manifest(self.data, config=self.config["dataset"])
        for row in self.data["samples"]:
            if row["sample_id"] in manifest["val"]:
                row["performance"]["power"] = None
            if row["sample_id"] in manifest["test"]:
                row["performance"]["untrained"] = 1e100
        manifest = build_manifest(self.data, config=self.config["dataset"])
        with contextlib.redirect_stdout(io.StringIO()):
            self.ctx = PinDataContext(self.data, manifest, self.config, torch.device("cpu"),
                                   dict(train=None, val=None, test=None), self.directory)
        self.options = dict(batch_size=8, patience=2, progress_every=100, torch_threads=1)

    def fit(self, destination):
        with contextlib.redirect_stdout(io.StringIO()):
            return fit_xgboost_pin(self.ctx, self.config, self.options, 17, destination)

    def test_fit_reads_train_and_validation_only_and_native_replay(self):
        original_batches = self.ctx.batches
        seen = []

        def tracked(split, *args, **kwargs):
            seen.append(split)
            self.assertIn(split, ("train", "val"))
            return original_batches(split, *args, **kwargs)

        self.ctx.batches = tracked
        output = self.directory / "tree"
        model, cost = self.fit(output)
        self.assertEqual(set(seen), {"train", "val"})
        self.ctx.batches = original_batches
        restored, metadata = load_xgboost_pin(output)
        batch = next(self.ctx.batches("test", 8))
        torch.testing.assert_close(model(batch), restored(batch), atol=0, rtol=0)
        self.assertTrue((restored(batch)[:, 2] == 0).all())
        self.assertFalse(metadata["metrics"][2]["trained"])
        self.assertEqual(metadata["metrics"][1]["val_count"], 0)
        self.assertFalse(metadata["metrics"][1]["early_stopping_enabled"])
        self.assertEqual(metadata["metrics"][1]["actual_rounds"], 4)
        self.assertIsNone(cost["epochs"])
        self.assertEqual(cost["budget_unit"], "boosting_rounds_per_metric")
        self.assertEqual(metadata['granularity'],'pin')
        self.assertEqual(metadata['mode'],'xgboost_pin')
        self.assertGreater(metadata['num_pins'],4)
        self.assertTrue(all(path.stem.endswith('_pin') for path in output.iterdir()))
        self.assertTrue((output/'metric_0_pin.ubj').is_file())
        for index, booster in enumerate(restored.boosters):
            if booster is not None:
                self.assertEqual(booster.num_boosted_rounds(), metadata["metrics"][index]["selected_rounds"])

    def test_features_and_predictions_ignore_op_and_performance(self):
        model, _ = self.fit(self.directory / "tree")
        batch = next(self.ctx.batches("val", 8, op=True))
        poisoned = copy.deepcopy(batch)
        for name in ("op", "owner_op", "performance", "physical_performance"):
            poisoned[name].fill_(float("nan"))
        poisoned["op_mask"] = ~poisoned["op_mask"]
        poisoned["owner_op_mask"] = ~poisoned["owner_op_mask"]
        poisoned["performance_mask"] = ~poisoned["performance_mask"]
        torch.testing.assert_close(pin_features_pin(batch,self.ctx.prep), pin_features_pin(poisoned,self.ctx.prep), atol=0, rtol=0)
        torch.testing.assert_close(model(batch), model(poisoned), atol=0, rtol=0)

    def test_rejects_overwrite_schema_mismatch_and_unknown_settings(self):
        output = self.directory / "tree"
        model, _ = self.fit(output)
        original = (output / "checkpoint_pin.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.fit(output)
        self.assertEqual(original, (output / "checkpoint_pin.json").read_bytes())
        batch = next(self.ctx.batches("test", 8))
        batch["device_mask"][:, -1] = False
        with self.assertRaisesRegex(ValueError, "without padding"):
            model(batch)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            tree_settings_pin({"final_exam_pin": {"xgboost": {"typod": 1}}})
        with self.assertRaisesRegex(ValueError, "only one"):
            tree_settings_pin({"final_exam_pin": {"xgboost": {"eta": 0.1, "learning_rate": 0.2}}})
        with self.assertRaisesRegex(ValueError, "positive integer"):
            tree_settings_pin({"final_exam_pin": {"xgboost": {"num_boost_round": True}}})

    def test_fixed_round_budget_still_restores_validation_best(self):
        original_batches = self.ctx.batches

        def constant_labels(split, *args, **kwargs):
            for batch in original_batches(split, *args, **kwargs):
                batch["performance"] = batch["performance"].clone()
                batch["performance"][:, 0] = 0.0 if split == "train" else 10.0
                yield batch

        self.ctx.batches = constant_labels
        self.options["patience"] = 0
        output = self.directory / "fixed_budget"
        model, _ = self.fit(output)
        metadata = json.loads((output / "checkpoint_pin.json").read_text())
        gain = metadata["metrics"][0]
        self.assertEqual(gain["actual_rounds"], 4)
        self.assertEqual(gain["selected_rounds"], 1)
        self.assertEqual(gain["selection"], "val_rmse")
        self.assertEqual(model.boosters[0].num_boosted_rounds(), 1)

    def test_actual_pin_features_and_strict_topology_replay(self):
        model, _ = self.fit(self.directory/'topology_pin')
        batch = next(self.ctx.batches('test',8))
        features = pin_features_pin(batch,self.ctx.prep)
        owners = batch['owner_index']
        self.assertEqual(owners.shape[1],model.num_pins)
        self.assertTrue((owners == -1).any())
        self.assertGreater(features.shape[1],batch['parameters'].shape[1]*batch['parameters'].shape[2]*3)
        changed = copy.deepcopy(batch)
        changed['pin_net_index'][0,0] += 1
        with self.assertRaisesRegex(ValueError,'topology/order'):
            model(changed)
        changed = copy.deepcopy(batch)
        changed['_group_keys'][0][0] += ':wrong_pin'
        with self.assertRaisesRegex(ValueError,'identities/topology'):
            model(changed)
        changed = copy.deepcopy(batch)
        changed.pop('pin_to_device')
        with self.assertRaisesRegex(ValueError,'pin_to_device'):
            model(changed)


if __name__ == "__main__":
    unittest.main()
