"""Encoder / TemporalEncoding / Masking / Loss / ForecastingHead 스모크 테스트.

실행:
    pytest tests/test_models.py -v
"""
import pytest
import torch

from src.adt.models.encoder import TimeSeriesTransformerEncoder
from src.adt.models.positional_encoding import TemporalEncoding
from src.adt.models.heads.reconstruction_head import MaskedReconstructionHead
from src.adt.models.heads.forecasting_head import ForecastingHead
from src.adt.ssl.masking import generate_mask, segment_mask, channel_mask
from src.adt.ssl.losses import masked_reconstruction_loss, forecast_loss, pretrain_joint_loss


# ------------------------------------------------------------------
# 공용 픽스처
# ------------------------------------------------------------------

B, T, C, D = 4, 96, 4, 128  # batch, time, channels, d_model


@pytest.fixture()
def encoder() -> TimeSeriesTransformerEncoder:
    return TimeSeriesTransformerEncoder(
        n_features=C,
        d_model=D,
        n_heads=4,
        n_layers=4,
        d_ff=256,
        dropout=0.0,   # 테스트에서 dropout=0 → 결정론적 출력
    ).eval()


@pytest.fixture()
def sample_x() -> torch.Tensor:
    return torch.randn(B, T, C)


@pytest.fixture()
def sample_time_feat() -> torch.Tensor:
    """(B, T, 2): [hour_of_day(0-23), day_of_week(0-6)]"""
    hour = torch.randint(0, 24, (B, T)).float()
    dow = torch.randint(0, 7, (B, T)).float()
    return torch.stack([hour, dow], dim=-1)


# ------------------------------------------------------------------
# TemporalEncoding
# ------------------------------------------------------------------

class TestTemporalEncoding:
    def test_output_shape(self, sample_x, sample_time_feat):
        x_proj = torch.randn(B, T, D)   # 이미 d_model 차원으로 투영된 토큰
        enc = TemporalEncoding(d_model=D, max_len=512, dropout=0.0)
        out = enc(x_proj, sample_time_feat)
        assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"

    def test_different_positions_differ(self):
        """같은 채널값이라도 위치가 다르면 출력이 달라야 한다."""
        enc = TemporalEncoding(d_model=D, max_len=512, dropout=0.0).eval()
        x = torch.zeros(1, T, D)
        # 동일한 hour/dow (변수 없음)
        time_feat = torch.zeros(1, T, 2)
        out = enc(x, time_feat)
        # 첫 번째와 마지막 타임스텝의 PE가 달라야 함 (sinusoidal)
        assert not torch.allclose(out[0, 0], out[0, -1])

    def test_hour_embedding_differs(self):
        """hour가 다르면 출력이 달라야 한다."""
        enc = TemporalEncoding(d_model=D, max_len=512, dropout=0.0).eval()
        x = torch.zeros(1, 1, D)
        tf_morning = torch.tensor([[[8.0, 0.0]]])   # 8시 월요일
        tf_night = torch.tensor([[[23.0, 0.0]]])    # 23시 월요일
        out_m = enc(x, tf_morning)
        out_n = enc(x, tf_night)
        assert not torch.allclose(out_m, out_n)


# ------------------------------------------------------------------
# TimeSeriesTransformerEncoder
# ------------------------------------------------------------------

class TestEncoder:
    def test_output_shape_no_mask(self, encoder, sample_x, sample_time_feat):
        """마스킹 없이 순방향 통과 — shape 확인."""
        with torch.no_grad():
            out = encoder(sample_x, sample_time_feat)
        assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"

    def test_output_shape_mask_2d(self, encoder, sample_x, sample_time_feat):
        """(B, T) bool 마스크 — shape 보존."""
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, 30:50] = True   # 30~49 타임스텝 마스킹
        with torch.no_grad():
            out = encoder(sample_x, sample_time_feat, mask=mask)
        assert out.shape == (B, T, D)

    def test_output_shape_mask_3d(self, encoder, sample_x, sample_time_feat):
        """(B, T, C) bool 마스크 — shape 보존."""
        mask = torch.zeros(B, T, C, dtype=torch.bool)
        mask[:, :, 0] = True   # 채널 0 전체 마스킹
        with torch.no_grad():
            out = encoder(sample_x, sample_time_feat, mask=mask)
        assert out.shape == (B, T, D)

    def test_mask_token_applied(self, encoder, sample_x, sample_time_feat):
        """마스킹된 위치의 입력이 실제로 바뀌는지 확인.

        mask_token이 적용되면 마스킹 있는 경우와 없는 경우의 출력이 달라야 한다.
        """
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, 40:60] = True

        with torch.no_grad():
            out_no_mask = encoder(sample_x, sample_time_feat, mask=None)
            out_masked = encoder(sample_x, sample_time_feat, mask=mask)

        assert not torch.allclose(out_no_mask, out_masked), \
            "mask_token이 적용됐으면 출력이 달라야 합니다."

    def test_mask_token_is_parameter(self, encoder):
        """mask_token이 nn.Parameter로 등록되어 학습 가능한지 확인."""
        param_names = [name for name, _ in encoder.named_parameters()]
        assert "mask_token" in param_names

    def test_invalid_mask_dim_raises(self, encoder, sample_x, sample_time_feat):
        """1D mask는 ValueError를 발생시켜야 한다."""
        bad_mask = torch.zeros(T, dtype=torch.bool)
        with pytest.raises(ValueError):
            encoder(sample_x, sample_time_feat, mask=bad_mask)

    def test_no_nan_in_output(self, encoder, sample_x, sample_time_feat):
        """정상 입력에서 NaN 출력 없음."""
        with torch.no_grad():
            out = encoder(sample_x, sample_time_feat)
        assert not torch.isnan(out).any(), "출력에 NaN이 포함되어 있습니다."


# ------------------------------------------------------------------
# Masking
# ------------------------------------------------------------------

SSL_CFG = {
    "mask_mode": "mixed",
    "segment_prob": 0.7,
    "mask_ratio": 0.40,
}


class TestMasking:
    def test_segment_mask_shape(self, sample_x):
        mask = segment_mask(sample_x, mask_ratio=0.4)
        assert mask.shape == (B, T)
        assert mask.dtype == torch.bool

    def test_segment_mask_ratio_approx(self, sample_x):
        """실제로 ~40% 타임스텝이 마스킹되는지 확인 (1~2 segment이므로 근사)."""
        mask = segment_mask(sample_x, mask_ratio=0.4)
        actual_ratio = mask.float().mean().item()
        # 1~2 연속 구간이라 ±15% 허용
        assert 0.10 < actual_ratio < 0.65, f"mask_ratio 이상: {actual_ratio:.2f}"

    def test_channel_mask_shape(self, sample_x):
        mask = channel_mask(sample_x, mask_ratio=0.4)
        assert mask.shape == (B, T, C)
        assert mask.dtype == torch.bool

    def test_channel_mask_exactly_one_channel(self, sample_x):
        """각 배치 아이템마다 정확히 채널 1개만 마스킹."""
        mask = channel_mask(sample_x, mask_ratio=0.4)
        for b in range(B):
            masked_channels = mask[b, 0]   # (C,) — 타임스텝 0의 채널별 마스크
            assert masked_channels.sum().item() == 1, \
                f"배치 {b}: 마스킹된 채널이 1개가 아님"

    def test_generate_mask_mixed_shape(self, sample_x):
        mask = generate_mask(sample_x, ssl_cfg=SSL_CFG)
        assert mask.shape == (B, T, C)

    def test_generate_mask_segment_mode(self, sample_x):
        cfg = {**SSL_CFG, "mask_mode": "segment"}
        mask = generate_mask(sample_x, ssl_cfg=cfg)
        assert mask.shape == (B, T)

    def test_generate_mask_invalid_mode(self, sample_x):
        cfg = {**SSL_CFG, "mask_mode": "invalid"}
        with pytest.raises(ValueError):
            generate_mask(sample_x, ssl_cfg=cfg)


# ------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------

class TestLoss:
    def test_loss_scalar(self, sample_x):
        pred = torch.randn(B, T, C)
        target = torch.randn(B, T, C)
        mask = segment_mask(sample_x, mask_ratio=0.4)   # (B, T)
        loss = masked_reconstruction_loss(pred, target, mask)
        assert loss.shape == (), f"loss가 스칼라가 아님: {loss.shape}"

    def test_loss_not_nan(self, sample_x):
        pred = torch.randn(B, T, C)
        target = torch.randn(B, T, C)
        mask = segment_mask(sample_x, mask_ratio=0.4)
        loss = masked_reconstruction_loss(pred, target, mask)
        assert not torch.isnan(loss), "loss가 NaN"

    def test_loss_3d_mask(self, sample_x):
        pred = torch.randn(B, T, C)
        target = torch.randn(B, T, C)
        mask = channel_mask(sample_x, mask_ratio=0.4)   # (B, T, C)
        loss = masked_reconstruction_loss(pred, target, mask)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_loss_only_on_masked(self):
        """마스킹된 위치만 loss에 포함되는지 검증.

        마스킹된 위치의 pred == target이면 loss=0,
        비마스킹 위치를 아무리 틀려도 loss에 영향 없어야 함.
        """
        pred = torch.ones(B, T, C)
        target = torch.ones(B, T, C)     # 마스킹 위치: pred==target → 0

        # 비마스킹 위치(mask=False)에서만 pred를 크게 틀리게
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, :T // 2] = True          # 앞 절반 마스킹
        pred[:, T // 2 :] = 999.0        # 뒤 절반(비마스킹) 크게 틀림

        loss = masked_reconstruction_loss(pred, target, mask)
        assert loss.item() < 1e-6, f"비마스킹 위치가 loss에 포함됨: {loss.item()}"

    def test_loss_has_gradient(self):
        """loss에서 역전파가 흐르는지 확인."""
        pred = torch.randn(B, T, C, requires_grad=True)
        target = torch.randn(B, T, C)
        mask = torch.ones(B, T, dtype=torch.bool)
        loss = masked_reconstruction_loss(pred, target, mask)
        loss.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()


# ------------------------------------------------------------------
# 통합 스모크 테스트: encoder → head → loss 전체 forward pass
# ------------------------------------------------------------------

class TestPretrainPipeline:
    def test_full_forward_pass(self, encoder, sample_x, sample_time_feat):
        """encoder + pretrain_head + mask + loss 전체 파이프라인 스모크 테스트."""
        head = MaskedReconstructionHead(d_model=D, n_features=C)

        # 1. mask 생성 (epoch=10, mixed 모드)
        mask = generate_mask(sample_x, ssl_cfg=SSL_CFG)
        assert mask.shape in [(B, T), (B, T, C)]

        # 2. encoder forward (마스킹 적용)
        enc_out = encoder(sample_x, sample_time_feat, mask=mask)
        assert enc_out.shape == (B, T, D)

        # 3. head → 복원값
        pred = head(enc_out)
        assert pred.shape == (B, T, C)

        # 4. loss 계산
        loss = masked_reconstruction_loss(pred, sample_x, mask)

        assert loss.shape == (), "loss가 스칼라가 아님"
        assert not torch.isnan(loss), "loss가 NaN"
        assert not torch.isinf(loss), "loss가 Inf"
        assert loss.item() >= 0.0, "MSE loss는 음수일 수 없음"

    def test_gradient_flows_through_pipeline(self, sample_x, sample_time_feat):
        """역전파가 encoder 파라미터까지 흐르는지 확인."""
        enc = TimeSeriesTransformerEncoder(
            n_features=C, d_model=D, n_heads=4, n_layers=2, d_ff=128, dropout=0.0
        ).train()
        head = MaskedReconstructionHead(d_model=D, n_features=C)

        mask = generate_mask(sample_x, ssl_cfg=SSL_CFG)
        pred = head(enc(sample_x, sample_time_feat, mask=mask))
        loss = masked_reconstruction_loss(pred, sample_x, mask)
        loss.backward()

        # encoder의 mask_token gradient 확인
        assert enc.mask_token.grad is not None
        assert not torch.isnan(enc.mask_token.grad).any()


# ------------------------------------------------------------------
# ForecastingHead
# ------------------------------------------------------------------

H = 8   # forecast_horizon

class TestForecastingHead:
    def test_output_shape(self, encoder, sample_x, sample_time_feat):
        """출력 shape이 (B, h, C)인지 확인."""
        head = ForecastingHead(d_model=D, forecast_horizon=H, n_features=C)
        with torch.no_grad():
            enc_out = encoder(sample_x, sample_time_feat)
            pred = head(enc_out)
        assert pred.shape == (B, H, C), f"Expected {(B, H, C)}, got {pred.shape}"

    def test_no_nan(self, encoder, sample_x, sample_time_feat):
        head = ForecastingHead(d_model=D, forecast_horizon=H, n_features=C)
        with torch.no_grad():
            pred = head(encoder(sample_x, sample_time_feat))
        assert not torch.isnan(pred).any()

    def test_uses_last_timestep(self):
        """h_L (마지막 타임스텝)만 사용하므로 마지막 timestep을 바꾸면 출력이 달라진다."""
        head = ForecastingHead(d_model=D, forecast_horizon=H, n_features=C).eval()
        z1 = torch.randn(1, T, D)
        z2 = z1.clone()
        z2[0, -1, :] += 10.0   # 마지막 타임스텝만 변경
        with torch.no_grad():
            out1 = head(z1)
            out2 = head(z2)
        assert not torch.allclose(out1, out2)

    def test_gradient_flows(self, sample_x, sample_time_feat):
        """역전파가 ForecastingHead 파라미터까지 흐르는지."""
        enc = TimeSeriesTransformerEncoder(
            n_features=C, d_model=D, n_heads=4, n_layers=2, d_ff=128, dropout=0.0
        ).train()
        head = ForecastingHead(d_model=D, forecast_horizon=H, n_features=C).train()
        future_gt = torch.randn(B, H, C)

        pred = head(enc(sample_x, sample_time_feat))
        loss = forecast_loss(pred, future_gt)
        loss.backward()

        for name, p in head.named_parameters():
            assert p.grad is not None, f"{name} grad is None"
            assert not torch.isnan(p.grad).any(), f"{name} grad has NaN"



# ------------------------------------------------------------------
# Joint loss
# ------------------------------------------------------------------

class TestJointLoss:
    def test_forecast_loss_scalar(self):
        pred = torch.randn(B, H, C)
        true = torch.randn(B, H, C)
        loss = forecast_loss(pred, true)
        assert loss.shape == (), f"forecast_loss가 스칼라가 아님: {loss.shape}"
        assert not torch.isnan(loss)
        assert loss.item() >= 0.0

    def test_joint_loss_correct_sum(self):
        """pretrain_joint_loss = l_mask + w * l_forecast."""
        l_mask = torch.tensor(1.0)
        l_fc = torch.tensor(2.0)
        w = 0.15
        total = pretrain_joint_loss(l_mask, l_fc, w)
        expected = 1.0 + 0.15 * 2.0
        assert abs(total.item() - expected) < 1e-5, f"expected {expected}, got {total.item()}"

    def test_joint_loss_gradient_to_both_heads(self, sample_x, sample_time_feat):
        """joint loss에서 역전파 시 reconstruction_head와 forecasting_head 모두 gradient 수신."""
        enc = TimeSeriesTransformerEncoder(
            n_features=C, d_model=D, n_heads=4, n_layers=2, d_ff=128, dropout=0.0
        ).train()
        recon_head = MaskedReconstructionHead(d_model=D, n_features=C).train()
        fc_head = ForecastingHead(d_model=D, forecast_horizon=H, n_features=C).train()

        mask = generate_mask(sample_x, ssl_cfg=SSL_CFG)
        enc_out = enc(sample_x, sample_time_feat, mask=mask)

        pred_recon = recon_head(enc_out)
        pred_future = fc_head(enc_out)
        future_gt = torch.randn(B, H, C)

        l_mask = masked_reconstruction_loss(pred_recon, sample_x, mask)
        l_fc = forecast_loss(pred_future, future_gt)
        loss = pretrain_joint_loss(l_mask, l_fc, forecast_weight=0.15)
        loss.backward()

        # recon_head gradient 확인
        for name, p in recon_head.named_parameters():
            assert p.grad is not None, f"recon_head.{name} grad is None"

        # forecast_head gradient 확인
        for name, p in fc_head.named_parameters():
            assert p.grad is not None, f"forecast_head.{name} grad is None"

        # encoder gradient 확인 (두 head로부터 흘러야 함)
        assert enc.mask_token.grad is not None
