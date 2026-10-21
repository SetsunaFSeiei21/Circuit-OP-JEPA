import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from device_jepa.data import build_manifest, dataset_fingerprint, fit_preprocessor, synthetic_dataset
from device_jepa.training import evaluate, load_checkpoint, pretrain, train_decoder


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.options = json.loads((Path(__file__).parents[1] / "configs" / "smoke.json").read_text())
        cls.options["training"]["pretrain_epochs"] = 1
        cls.options["training"]["decoder_epochs"] = 1
        cls.options["model"].update(dim=16, heads=4, ffn_dim=32, decoder_ffn_dim=32)
        cls.data = synthetic_dataset(6, 17)
        cls.manifest = build_manifest(cls.data, "synthetic_4", seed=17)
        cls.temp = tempfile.TemporaryDirectory()
        cls.p = pretrain(cls.data, cls.manifest, cls.options, cls.temp.name)
        cls.d = train_decoder(cls.data, cls.p, cls.options, cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_checkpoint_phase_boundaries_and_frozen_digest(self):
        p, d = load_checkpoint(self.p), load_checkpoint(self.d)
        self.assertNotIn("performance", p["preprocessing"])
        self.assertIn("performance", d["preprocessing"])
        self.assertEqual(p["encoder_digest"], d["encoder_digest"])
        self.assertTrue(d["freeze_verified"])
        self.assertTrue(p["synthetic"] and d["synthetic"])
        self.assertFalse(any("decoder" in k for k in p["model_state"]))
        self.assertEqual(p["selection"]["split"], "val")
        self.assertEqual(d["selection"]["split"], "val")

    def test_eval_separates_id_and_ood(self):
        for split, count in (("test", 4), ("ood", 6)):
            result = evaluate(self.data, self.d, split=split)
            self.assertEqual(result["num_samples"], count)
            self.assertTrue(result["synthetic"])
            self.assertFalse(result["selection_used_ood"])
            self.assertEqual(set(result["metrics"]), {"gain", "power"})
            self.assertEqual(len(result["predictions"]), count)
            if split == "ood":
                self.assertEqual(set(result["by_circuit"]), {"synthetic_4"})
            else:
                self.assertNotIn("synthetic_4", result["by_circuit"])

    def test_performance_values_do_not_change_stage1_preprocessing(self):
        changed = copy.deepcopy(self.data)
        for row in changed["samples"]:
            row["performance"] = {"gain": 1e9, "power": None}
        updated_manifest = build_manifest(changed, "synthetic_4", seed=17)
        self.assertEqual(self.manifest["train"], updated_manifest["train"])
        self.assertEqual(fit_preprocessor(self.data, self.manifest),
                         fit_preprocessor(changed, updated_manifest))

    def test_labels_can_be_attached_after_unlabeled_pretraining(self):
        unlabeled = copy.deepcopy(self.data)
        for row in unlabeled["samples"]:
            row["performance"] = {}
        manifest = build_manifest(unlabeled, "synthetic_4", seed=17)
        with tempfile.TemporaryDirectory() as folder:
            p = pretrain(unlabeled, manifest, self.options, folder)
            d = train_decoder(self.data, p, self.options, folder)
            before, after = load_checkpoint(p), load_checkpoint(d)
            self.assertEqual(before["input_op_fingerprint"], after["input_op_fingerprint"])
            self.assertEqual(before["partition_fingerprint"], after["partition_fingerprint"])
            self.assertEqual(before["encoder_digest"], after["encoder_digest"])
            self.assertNotEqual(before["manifest"]["dataset_fingerprint"], after["manifest"]["dataset_fingerprint"])
            self.assertEqual(after["manifest"]["dataset_fingerprint"], dataset_fingerprint(self.data))
            self.assertEqual(after["source_pretrain_dataset_fingerprint"], before["manifest"]["dataset_fingerprint"])
            self.assertEqual(after["preprocessing"]["performance"], fit_preprocessor(self.data, self.manifest, True)["performance"])
            self.assertTrue(evaluate(self.data, d, split="ood")["synthetic"])

    def test_stage2_rejects_design_op_and_record_changes(self):
        mutations = [
            lambda d: d["samples"][0]["devices"][0]["parameters"].update(W=99.),
            lambda d: d["samples"][0]["devices"][0]["op"].update(gm=99.),
            lambda d: d["samples"][0]["devices"][0]["terminals"].update(d="new_net"),
            lambda d: d["samples"][0].update(sample_id="replacement_id"),
            lambda d: d["samples"][0].update(design_id="replacement_design"),
            lambda d: d["samples"][0].update(circuit_id="synthetic_4"),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                changed = copy.deepcopy(self.data)
                mutate(changed)
                with self.assertRaisesRegex(ValueError, "immutable design/input/OP"):
                    train_decoder(changed, self.p, self.options, self.temp.name)

    def test_stage2_rejects_partition_and_scaler_identity_changes(self):
        for mutation in ("partition", "scaler"):
            checkpoint = load_checkpoint(self.p)
            if mutation == "partition":
                checkpoint["manifest"]["train"][0], checkpoint["manifest"]["val"][0] = checkpoint["manifest"]["val"][0], checkpoint["manifest"]["train"][0]
            else:
                checkpoint["preprocessing"]["fitted_sample_ids"] = []
            path = Path(self.temp.name) / f"tampered_{mutation}.pt"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "partition"):
                train_decoder(self.data, path, self.options, self.temp.name)

    def test_stage2_rejects_architecture_mismatch(self):
        options = copy.deepcopy(self.options)
        options["model"]["dim"] = 32
        with self.assertRaisesRegex(ValueError, "Architecture differs"):
            train_decoder(self.data, self.p, options, self.temp.name)


if __name__ == "__main__":
    unittest.main()
