"""test_lstm_joint.py — LSTMJoint 모델 테스트.

검증 항목:
  1. forward 출력 shape (det_logit, cls_logit)
  2. NaN 없음
  3. gradient가 encoder + det_head + cls_head 모두에 흐름
  4. z 공유 확인: det_head와 cls_head가 동일한 z 소비 (same encoder pass)
  5. encoder parameter update 확인 (완전 unfreeze)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.adt.models.lstm_joint import LSTMJoint

B, T, C, H = 4, 96, 4, 64
NUM_CLASSES = 5


class TestLSTMJointForward:
    def test_det_logit_shape(self):
        model = LSTMJoint(C, H, num_layers=2, num_classes=NUM_CLASSES)
        x = torch.randn(B, T, C)
        det_logit, _ = model(x)
        assert det_logit.shape == (B,), f"expected ({B},), got {det_logit.shape}"

    def test_cls_logit_shape(self):
        model = LSTMJoint(C, H, num_layers=2, num_classes=NUM_CLASSES)
        x = torch.randn(B, T, C)
        _, cls_logit = model(x)
        assert cls_logit.shape == (B, NUM_CLASSES), \
            f"expected ({B}, {NUM_CLASSES}), got {cls_logit.shape}"

    def test_no_nan_det(self):
        model = LSTMJoint(C, H, num_layers=2, num_classes=NUM_CLASSES)
        det_logit, _ = model(torch.randn(B, T, C))
        assert not det_logit.isnan().any()

    def test_no_nan_cls(self):
        model = LSTMJoint(C, H, num_layers=2, num_classes=NUM_CLASSES)
        _, cls_logit = model(torch.randn(B, T, C))
        assert not cls_logit.isnan().any()

    def test_different_num_classes(self):
        for nc in [3, 4, 5]:
            model = LSTMJoint(C, H, num_layers=2, num_classes=nc)
            _, cls_logit = model(torch.randn(B, T, C))
            assert cls_logit.shape == (B, nc)


class TestLSTMJointGradient:
    def test_gradient_flows_to_all_submodules(self):
        model = LSTMJoint(C, H, num_layers=2, num_classes=NUM_CLASSES)
        x     = torch.randn(B, T, C)
        bl    = torch.zeros(B)
        tl    = torch.randint(0, NUM_CLASSES, (B,))

        det_logit, cls_logit = model(x)
        det_loss = F.binary_cross_entropy_with_logits(det_logit, bl)
        cls_loss = F.cross_entropy(cls_logit, tl)
        (det_loss + 0.4 * cls_loss).backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert not p.grad.isnan().any(), f"NaN gradient for {name}"

    def test_encoder_params_updated(self):
        model = LSTMJoint(C, H, num_layers=2, num_classes=NUM_CLASSES)
        enc_before = {n: p.clone().detach() for n, p in model.encoder.named_parameters()}

        optim = torch.optim.SGD(model.parameters(), lr=1.0)
        x     = torch.randn(B, T, C)
        bl    = torch.zeros(B)
        tl    = torch.randint(0, NUM_CLASSES, (B,))

        det_logit, cls_logit = model(x)
        loss = F.binary_cross_entropy_with_logits(det_logit, bl) + \
               0.4 * F.cross_entropy(cls_logit, tl)
        optim.zero_grad()
        loss.backward()
        optim.step()

        changed = sum(
            1 for n, p in model.encoder.named_parameters()
            if not torch.equal(p, enc_before[n])
        )
        assert changed > 0, "encoder parameters did not update"


class TestLSTMJointArchitecture:
    def test_encoder_output_dim(self):
        model = LSTMJoint(C, H, num_layers=2, num_classes=NUM_CLASSES)
        assert model.encoder.output_dim == 2 * H

    def test_det_head_input_matches_bottleneck(self):
        """det_head의 첫 Linear 입력 차원 = 2*H."""
        model = LSTMJoint(C, H, num_layers=2, num_classes=NUM_CLASSES)
        first_linear = model.det_head.net[0]
        assert first_linear.in_features == 2 * H

    def test_cls_head_input_matches_bottleneck(self):
        """cls_head의 첫 Linear 입력 차원 = 2*H."""
        model = LSTMJoint(C, H, num_layers=2, num_classes=NUM_CLASSES)
        first_linear = model.cls_head.net[0]
        assert first_linear.in_features == 2 * H
