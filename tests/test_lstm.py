"""test_lstm.py — LSTM Autoencoder 기반 파이프라인 테스트.

검증 항목:
  1. LSTMEncoder: all_hidden shape, bottleneck z shape, nan 없음, output_dim property
  2. LSTMDecoder: reconstruction shape, nan 없음
  3. LSTMAutoencoder: 입출력 shape 동일, nan 없음, gradient 흐름
  4. LSTMDetectionHead: (B,) logit shape, nan 없음
  5. LSTMClassificationHead: (B, num_classes) shape, nan 없음
  6. encoder 파라미터 업데이트 확인 (학습 시 실제 변경)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.adt.models.lstm_ae import (
    LSTMAutoencoder,
    LSTMClassificationHead,
    LSTMDecoder,
    LSTMDetectionHead,
    LSTMEncoder,
)

B, T, C, H = 4, 96, 4, 64  # batch, seq_len, features, hidden_dim


class TestLSTMEncoder:
    def test_all_hidden_shape(self):
        enc = LSTMEncoder(C, H, num_layers=2)
        x = torch.randn(B, T, C)
        all_h, z = enc(x)
        assert all_h.shape == (B, T, 2 * H), \
            f"all_hidden: expected ({B}, {T}, {2*H}), got {all_h.shape}"

    def test_bottleneck_shape(self):
        enc = LSTMEncoder(C, H, num_layers=2)
        x = torch.randn(B, T, C)
        _, z = enc(x)
        assert z.shape == (B, 2 * H), \
            f"z: expected ({B}, {2*H}), got {z.shape}"

    def test_no_nan_in_outputs(self):
        enc = LSTMEncoder(C, H, num_layers=2)
        x = torch.randn(B, T, C)
        all_h, z = enc(x)
        assert not all_h.isnan().any(), "NaN in all_hidden"
        assert not z.isnan().any(),     "NaN in bottleneck z"

    def test_output_dim_property(self):
        enc = LSTMEncoder(C, H)
        assert enc.output_dim == 2 * H


class TestLSTMDecoder:
    def test_reconstruction_shape(self):
        dec = LSTMDecoder(bottleneck_dim=2 * H, n_features=C, num_layers=1)
        z = torch.randn(B, 2 * H)
        out = dec(z, T)
        assert out.shape == (B, T, C), \
            f"expected ({B}, {T}, {C}), got {out.shape}"

    def test_no_nan(self):
        dec = LSTMDecoder(2 * H, C)
        out = dec(torch.randn(B, 2 * H), T)
        assert not out.isnan().any()


class TestLSTMAutoencoder:
    def test_reconstruction_shape_equals_input(self):
        ae = LSTMAutoencoder(C, H, num_layers=2)
        x = torch.randn(B, T, C)
        x_recon = ae(x)
        assert x_recon.shape == x.shape, \
            f"recon shape {x_recon.shape} != input shape {x.shape}"

    def test_no_nan_in_reconstruction(self):
        ae = LSTMAutoencoder(C, H)
        x = torch.randn(B, T, C)
        assert not ae(x).isnan().any()

    def test_gradient_flows_through_all_params(self):
        ae = LSTMAutoencoder(C, H)
        x = torch.randn(B, T, C)
        loss = F.mse_loss(ae(x), x)
        loss.backward()
        for name, p in ae.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert not p.grad.isnan().any(), f"NaN gradient for {name}"


class TestLSTMDetectionHead:
    def test_output_shape(self):
        head = LSTMDetectionHead(input_dim=2 * H, hidden_dim=32)
        z = torch.randn(B, 2 * H)
        out = head(z)
        assert out.shape == (B,), f"expected ({B},), got {out.shape}"

    def test_no_nan(self):
        head = LSTMDetectionHead(2 * H, hidden_dim=32)
        assert not head(torch.randn(B, 2 * H)).isnan().any()


class TestLSTMClassificationHead:
    def test_output_shape_5class(self):
        head = LSTMClassificationHead(2 * H, num_classes=5, hidden_dim=32)
        out = head(torch.randn(B, 2 * H))
        assert out.shape == (B, 5), f"expected ({B}, 5), got {out.shape}"

    def test_output_shape_4class(self):
        head = LSTMClassificationHead(2 * H, num_classes=4, hidden_dim=32)
        out = head(torch.randn(B, 2 * H))
        assert out.shape == (B, 4)

    def test_no_nan(self):
        head = LSTMClassificationHead(2 * H, num_classes=5, hidden_dim=32)
        assert not head(torch.randn(B, 2 * H)).isnan().any()


class TestEncoderUnfrozen:
    def test_all_params_require_grad_by_default(self):
        """LSTMEncoder는 기본적으로 모든 파라미터가 requires_grad=True."""
        enc = LSTMEncoder(C, H)
        for name, p in enc.named_parameters():
            assert p.requires_grad, f"Param {name} should require grad"

    def test_encoder_params_updated_after_optimizer_step(self):
        """optimizer step 후 encoder 파라미터가 실제로 변경됨 (완전 unfreeze 확인)."""
        enc = LSTMEncoder(C, H)
        det = LSTMDetectionHead(2 * H, hidden_dim=32)
        params_before = {n: p.clone().detach() for n, p in enc.named_parameters()}

        optim = torch.optim.SGD(
            list(enc.parameters()) + list(det.parameters()), lr=1.0
        )
        x  = torch.randn(B, T, C)
        bl = torch.zeros(B)
        _, z   = enc(x)
        loss   = F.binary_cross_entropy_with_logits(det(z), bl)
        optim.zero_grad()
        loss.backward()
        optim.step()

        changed = sum(
            1 for n, p in enc.named_parameters()
            if not torch.equal(p, params_before[n])
        )
        assert changed > 0, "No encoder parameters changed after optimizer step"
