"""Mechanism tests; these do not establish predictive gains on real circuits."""
import copy
import unittest

import torch

from experiments.op_transfer.losses import (
    LatentScaler, grouped_alignment_loss, masked_field_mse, residual_alignment_loss,
)


class LatentCalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.x = torch.randn(12, 3, 4, dtype=torch.float64) * 1e-4
        self.x[:, 1] += 1000.
        self.x[:, 2] -= 1000.
        self.padding = torch.zeros(12, 3, dtype=torch.bool)
        self.keys = [["circuit/topology/M1", "circuit/topology/M2"] for _ in range(12)]

    def test_streaming_welford_matches_direct_tiny_residuals(self):
        scaler = LatentScaler(floor=1e-7)
        scaler.update(self.x[:5], self.padding[:5], self.keys[:5])
        scaler.update(self.x[5:], self.padding[5:], self.keys[5:]).finalize()
        result = scaler.transform(self.x, self.padding, self.keys)
        torch.testing.assert_close(result.mean(0), torch.zeros(3, 4, dtype=torch.float64), atol=3e-9, rtol=0)
        torch.testing.assert_close(result.std(0, unbiased=False), torch.ones(3, 4, dtype=torch.float64), atol=3e-9, rtol=0)
        self.assertEqual(scaler.fitted_samples, 12)
        self.assertEqual(scaler.fitted_tokens, 24)

    def test_scaler_restore_and_frozen_train_statistics(self):
        scaler = LatentScaler().update(self.x[:8], self.padding[:8], self.keys[:8]).finalize()
        state = scaler.state_dict()
        restored = LatentScaler.from_state_dict(copy.deepcopy(state))
        shifted_val = self.x[8:] + 1000
        a = scaler.transform(shifted_val, self.padding[8:], self.keys[8:])
        b = restored.transform(shifted_val, self.padding[8:], self.keys[8:])
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.testing.assert_close(scaler.groups[self.keys[0][0]]["mean"], state["groups"][self.keys[0][0]]["mean"])
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            scaler.update(self.x, self.padding, self.keys)
        with self.assertRaisesRegex(ValueError, "training partition"):
            LatentScaler().update(self.x, self.padding, self.keys, partition="val")

    def test_repeated_device_move_preserves_moments_without_copying(self):
        scaler = LatentScaler().update(self.x, self.padding, self.keys).finalize()
        address = scaler.graph["mean"].data_ptr()
        self.assertIs(scaler.to("cpu"), scaler)
        self.assertEqual(scaler.graph["mean"].data_ptr(), address)
        scaler.transform(self.x, self.padding, self.keys)
        self.assertEqual(scaler.graph["mean"].data_ptr(), address)
        self.assertEqual(scaler.graph["mean"].dtype, torch.float64)

    def test_unseen_device_fallback_is_explicit_and_does_not_fit(self):
        scaler = LatentScaler().update(self.x, self.padding, self.keys).finalize()
        keys = [["ood/new/M1", "ood/new/M2"]]
        val = self.x[:1] * 2
        actual = scaler.transform(val, self.padding[:1], keys)
        expected = (val[:, 1:] - scaler.global_device["mean"]) / scaler.global_device["scale"]
        torch.testing.assert_close(actual[:, 1:], expected)
        self.assertEqual(scaler.last_fallback_tokens, 2)
        self.assertEqual(len(scaler.groups), 2)
        self.assertEqual(scaler.state_dict()["unknown_device_policy"], "global_train_device_moments")

    def test_padding_nan_is_ignored_and_transform_is_differentiable(self):
        x = self.x.clone()
        padding = self.padding.clone()
        padding[:, 2] = True
        x[:, 2] = float("nan")
        scaler = LatentScaler().update(x, padding, self.keys).finalize()
        x.requires_grad_()
        actual = scaler.transform(x, padding, self.keys)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue((actual[:, 2] == 0).all())
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue((x.grad[:, 2] == 0).all())

    def test_duplicate_keys_or_missing_keys_fail(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            LatentScaler().update(self.x, self.padding, [["same", "same"]] * 12)
        with self.assertRaisesRegex(ValueError, "Every non-padding"):
            LatentScaler().update(self.x, self.padding, [["one"]] * 12)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            LatentScaler().update(self.x * float("nan"), self.padding, self.keys)

    def test_variable_key_lengths_allow_only_padded_unkeyed_slots(self):
        x = self.x[:2].clone()
        padding = self.padding[:2].clone()
        padding[1, 2] = True
        x[1, 2] = float("nan")
        keys = [self.keys[0], self.keys[1][:1]]
        scaler = LatentScaler().update(x, padding, keys).finalize()
        transformed = scaler.transform(x, padding, keys)
        self.assertTrue(torch.isfinite(transformed).all())
        self.assertTrue((transformed[1, 2] == 0).all())
        self.assertEqual(scaler.fitted_tokens, 3)
        padding[1, 2] = False
        with self.assertRaisesRegex(ValueError, "Every non-padding"):
            scaler.transform(x, padding, keys)


class ConditionalLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        self.keys = [["c/M1", "c/M2"]] * 8
        self.valid = torch.ones(8, 2, dtype=torch.bool)

    def test_device_identity_variation_cannot_satisfy_grouped_variance(self):
        z = torch.zeros(8, 2, 4)
        z[:, 0] = -100
        z[:, 1] = 100
        loss, info = grouped_alignment_loss(z, z, self.valid, self.keys)
        self.assertAlmostEqual(float(info["variance"]), .99, places=6)
        self.assertAlmostEqual(float(info["covariance"]), 0.)
        self.assertAlmostEqual(float(loss), 24.75, places=5)
        z = z + torch.arange(8).float()[:, None, None] * 2
        _, varied = grouped_alignment_loss(z, z, self.valid, self.keys)
        self.assertLess(float(varied["variance"]), float(info["variance"]))

    def test_grouped_loss_has_gradients_on_both_learned_branches(self):
        a = torch.randn(8, 2, 4, requires_grad=True)
        b = torch.randn(8, 2, 4, requires_grad=True)
        loss, info = grouped_alignment_loss(a, b, self.valid, self.keys)
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all() and (a.grad != 0).any())
        self.assertTrue(torch.isfinite(b.grad).all() and (b.grad != 0).any())
        self.assertEqual(info["repeated_groups"], 2)

    def test_singletons_and_no_eligible_targets_do_not_silently_train(self):
        z = torch.randn(8, 2, 4)
        unique = [[f"c{i}/M1", f"c{i}/M2"] for i in range(8)]
        with self.assertRaisesRegex(ValueError, "grouped batches"):
            grouped_alignment_loss(z, z, self.valid, unique)
        with self.assertRaisesRegex(ValueError, "No eligible"):
            grouped_alignment_loss(z, z, torch.zeros_like(self.valid), self.keys)

    def test_residual_alignment_removes_identity_offsets_and_detaches_teacher(self):
        target = torch.randn(8, 3, 4)
        padding = torch.zeros(8, 3, dtype=torch.bool)
        scaler = LatentScaler().update(target, padding, self.keys).finalize()
        student = target.clone()
        student[:, 0] += 9000  # graph is excluded from residual loss
        student[:, 1] += 50
        student[:, 2] -= 100
        student.requires_grad_()
        target.requires_grad_()
        loss, info = residual_alignment_loss(student, target, padding, self.keys, scaler)
        self.assertLess(float(loss), 2e-10)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertIsNotNone(student.grad)
        self.assertTrue((student.grad[:, 0] == 0).all())
        self.assertEqual(info["regularized_tokens"], 16)

    def test_residual_alignment_detects_design_residual_mismatch(self):
        target = torch.randn(8, 3, 4)
        padding = torch.zeros(8, 3, dtype=torch.bool)
        scaler = LatentScaler().update(target, padding, self.keys).finalize()
        student = target.clone()
        student[:, 1:] *= -1
        loss, _ = residual_alignment_loss(student, target, padding, self.keys, scaler)
        self.assertAlmostEqual(float(loss), 4., places=5)

    def test_op_loss_equal_fields_and_missing_before_arithmetic(self):
        prediction = torch.tensor([[[1., 3.]], [[1., float("nan")]], [[1., float("inf")]]], requires_grad=True)
        target = torch.zeros_like(prediction)
        mask = torch.tensor([[[True, True]], [[True, False]], [[True, False]]])
        loss = masked_field_mse(prediction, target, mask)
        self.assertAlmostEqual(float(loss), 5.)  # mean(field0=1, field1=9), not element mean=3
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(float(prediction.grad[1, 0, 1]), 0.)
        with self.assertRaisesRegex(ValueError, "No observed"):
            masked_field_mse(prediction, target, torch.zeros_like(mask))


if __name__ == "__main__":
    unittest.main()
