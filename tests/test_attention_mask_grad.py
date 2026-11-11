"""Regression coverage for attention whose only input gradient is its bias.

PyTorch's CUDA efficient-attention implementation can omit the LSE needed by
backward when Q/K/V are frozen but the additive mask requires gradients. A
staged OP gate must retain its gradient without unfreezing the encoder.
"""
import unittest
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from device_jepa.model import EncoderBlock, ModelConfig


def backend_flags():
    return (
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.cudnn_sdp_enabled(),
    )


class AttentionMaskGradientTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        self.cfg = ModelConfig(dim=16, heads=2, encoder_layers=2, cross_layers=1,
                               ffn_dim=32, dropout=0.)

    def inputs(self, block, *, batch_size=2, length=7):
        parameter = next(block.parameters())
        x = torch.randn(batch_size, length, block.self_attn.embed_dim,
                        dtype=parameter.dtype, device=parameter.device)
        bias = torch.randn(batch_size * block.heads, length, length,
                           dtype=parameter.dtype, device=parameter.device).mul_(.1)
        bias.requires_grad_()
        allowed = torch.ones_like(bias, dtype=torch.bool)
        # Exercise topology exclusions while keeping at least one valid key.
        allowed[:, 0, -1] = False
        mask = bias.masked_fill(~allowed, float('-inf'))
        valid = torch.ones(batch_size, length, dtype=torch.bool, device=x.device)
        return x, bias, mask, valid, allowed

    def frozen_block(self):
        return EncoderBlock(self.cfg, cross=False).double().eval().requires_grad_(False)

    def test_mask_gradient_matches_explicit_attention_with_frozen_weights(self):
        block = self.frozen_block()
        x, bias, mask, valid, allowed = self.inputs(block)
        reference_bias = bias.detach().clone().requires_grad_()
        reference_mask = reference_bias.masked_fill(~allowed, float('-inf'))
        actual = block(x, mask, None, None, valid)

        # need_weights=True uses explicit bmm/softmax instead of SDPA. This
        # reference checks the gate gradient independently of backend routing.
        update = block.self_attn(x, x, x, attn_mask=reference_mask,
                                 need_weights=True)[0]
        expected = block.self_norm(x + block.drop(update))
        expected = block.ffn_norm(expected + block.drop(block.ffn(expected)))
        expected = expected * valid.unsqueeze(-1)
        probe = torch.randn_like(actual)
        (actual * probe).sum().backward()
        (expected * probe).sum().backward()

        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
        torch.testing.assert_close(bias.grad, reference_bias.grad, atol=1e-10, rtol=1e-8)
        self.assertTrue(bool(torch.isfinite(bias.grad).all()))
        self.assertGreater(float(bias.grad.abs().sum()), 0.)
        self.assertEqual(float(bias.grad[~allowed].abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in block.parameters()))
        self.assertFalse(x.requires_grad)

    def capture_backend(self, block, x, mask, valid):
        observed = []
        original_forward = block.self_attn.forward

        def capture(*args, **kwargs):
            observed.append(backend_flags())
            return original_forward(*args, **kwargs)

        with patch.object(block.self_attn, 'forward', side_effect=capture):
            output = block(x, mask, None, None, valid)
        self.assertEqual(len(observed), 1)
        return observed[0], output

    def test_mask_only_route_is_local_and_restores_backend_settings(self):
        block = self.frozen_block()
        x, _, mask, valid, _ = self.inputs(block)
        original_flags = backend_flags()
        with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
            outer_flags = backend_flags()
            self.assertTrue(outer_flags[2])
            observed, _ = self.capture_backend(block, x, mask, valid)
            self.assertEqual(observed, (True, False, False, False))
            self.assertEqual(backend_flags(), outer_flags)
            with patch.object(block.self_attn, 'forward', side_effect=RuntimeError('probe')):
                with self.assertRaisesRegex(RuntimeError, 'probe'):
                    block(x, mask, None, None, valid)
            self.assertEqual(backend_flags(), outer_flags)
        self.assertEqual(backend_flags(), original_flags)

    def test_trainable_qkv_and_no_grad_keep_normal_backend_selection(self):
        for trainable in ('input', 'projection_weight', 'projection_bias', 'no_grad', 'fixed_mask'):
            with self.subTest(trainable=trainable):
                block = self.frozen_block()
                x, _, mask, valid, _ = self.inputs(block)
                if trainable == 'input':
                    x.requires_grad_()
                elif trainable == 'projection_weight':
                    block.self_attn.in_proj_weight.requires_grad_()
                elif trainable == 'projection_bias':
                    block.self_attn.in_proj_bias.requires_grad_()
                elif trainable == 'fixed_mask':
                    mask = mask.detach()
                with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                    outer_flags = backend_flags()
                    with torch.set_grad_enabled(trainable != 'no_grad'):
                        observed, _ = self.capture_backend(block, x, mask, valid)
                    self.assertEqual(observed, outer_flags)
                    self.assertEqual(backend_flags(), outer_flags)

    def test_output_projection_gradient_and_absent_bias_keep_safe_route(self):
        for variation in ('output_projection_only', 'no_projection_bias'):
            with self.subTest(variation=variation):
                block = self.frozen_block()
                if variation == 'output_projection_only':
                    # Training W_O does not make any Q/K/V require gradients.
                    block.self_attn.out_proj.requires_grad_(True)
                else:
                    block.self_attn.in_proj_bias = None
                x, bias, mask, valid, _ = self.inputs(block)
                with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                    outer_flags = backend_flags()
                    observed, output = self.capture_backend(block, x, mask, valid)
                    self.assertEqual(observed, (True, False, False, False))
                    self.assertEqual(backend_flags(), outer_flags)
                (output * torch.randn_like(output)).sum().backward()
                self.assertTrue(bool(torch.isfinite(bias.grad).all()))
                self.assertGreater(float(bias.grad.abs().sum()), 0.)
                self.assertIsNone(block.self_attn.in_proj_weight.grad)
                if variation == 'output_projection_only':
                    self.assertIsNotNone(block.self_attn.out_proj.weight.grad)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for the fused-kernel regression')
    def test_cuda_mask_only_backward_overrides_efficient_backend(self):
        cfg = ModelConfig(dim=128, heads=8, encoder_layers=2, cross_layers=1,
                          ffn_dim=256, dropout=0.)
        block = EncoderBlock(cfg, cross=False).cuda().eval().requires_grad_(False)
        x, bias, mask, valid, _ = self.inputs(block, length=97)
        # Use an odd sequence length representative of graph-plus-pin tokens.
        # The failure is caused by missing LSE, not by odd lengths alone. The
        # local guard must override even an efficient-only caller.
        original_flags = backend_flags()
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            outer_flags = backend_flags()
            output = block(x, mask, None, None, valid)
            self.assertEqual(backend_flags(), outer_flags)
            (output * torch.randn_like(output)).sum().backward()
            torch.cuda.synchronize()
            self.assertTrue(output.is_cuda)
            self.assertTrue(bias.grad.is_cuda)
            self.assertTrue(bool(torch.isfinite(bias.grad).all()))
            self.assertGreater(float(bias.grad.abs().sum()), 0.)
            self.assertTrue(all(parameter.grad is None for parameter in block.parameters()))
        self.assertEqual(backend_flags(), original_flags)


if __name__ == '__main__':
    unittest.main()
