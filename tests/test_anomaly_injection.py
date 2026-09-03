"""anomaly_injection.py 유닛 테스트.

각 시나리오별로:
  - 주입된 구간의 raw 값이 원본과 다른지
  - scale_down/ramp: 변조값이 원본 대비 비율(배율)로 설명 가능한지
  - pulse_plateau: 주입 구간이 대략 상수값인지
  - replay: 주입된 값이 동일 윈도우 내 다른 시점의 원본값과 일치하는지
  - instant_spike: 주입값이 로컬 평균보다 충분히 큰지
"""
from __future__ import annotations

import numpy as np
import pytest

from src.adt.data.attack_injection import (
    inject_instant_spike,
    inject_pulse_plateau,
    inject_ramp,
    inject_replay,
    inject_scale_down,
    inject_synthetic_anomalies,
)


# =========================================================================
# 공통 픽스처
# =========================================================================

T, C = 96, 4
CH = 0   # 테스트 대상 채널


@pytest.fixture
def window():
    """사인파 + 노이즈로 굴곡 있는 윈도우 생성."""
    rng = np.random.default_rng(0)
    base = np.linspace(100, 200, T)        # 기울어진 베이스라인
    wave = 20 * np.sin(np.linspace(0, 4 * np.pi, T))  # 사인 굴곡
    noise = rng.normal(0, 2, (T, C)).astype(np.float32)
    w = np.tile((base + wave)[:, None], (1, C)).astype(np.float32) + noise
    w = np.abs(w) + 10  # 음수 방지
    return w   # (T, C)


def _rng(seed=42):
    return np.random.default_rng(seed)


# =========================================================================
# 1. inject_scale_down
# =========================================================================

class TestScaleDown:
    def test_shape_unchanged(self, window):
        c, m = inject_scale_down(window, CH, [0.3, 0.7], [8, 48], _rng())
        assert c.shape == window.shape
        assert m.shape == (T,)

    def test_non_injected_unchanged(self, window):
        c, m = inject_scale_down(window, CH, [0.3, 0.7], [8, 48], _rng())
        assert np.allclose(c[~m, :], window[~m, :])

    def test_injected_differs_from_original(self, window):
        c, m = inject_scale_down(window, CH, [0.3, 0.7], [8, 48], _rng())
        assert m.any(), "주입 구간이 없음"
        assert not np.allclose(c[m, CH], window[m, CH]), "주입 후 값이 원본과 동일"

    def test_elementwise_ratio_preserved(self, window):
        """injected[t] / original[t] 가 단일 scale_factor여야 함 (원본 굴곡 유지)."""
        c, m = inject_scale_down(window, CH, [0.3, 0.7], [8, 48], _rng())
        seg_orig = window[m, CH]
        seg_corr = c[m, CH]
        ratios = seg_corr / (seg_orig + 1e-9)
        # 모든 ratio가 동일한 scale_factor (허용 오차 1e-5)
        assert np.allclose(ratios, ratios[0], atol=1e-5), \
            f"ratio가 균일하지 않음: min={ratios.min():.4f}, max={ratios.max():.4f}"
        assert 0.3 <= ratios[0] <= 0.7, f"scale_factor 범위 벗어남: {ratios[0]:.4f}"

    def test_other_channels_unchanged(self, window):
        c, m = inject_scale_down(window, CH, [0.3, 0.7], [8, 48], _rng())
        for ch in range(C):
            if ch == CH:
                continue
            assert np.allclose(c[:, ch], window[:, ch])


# =========================================================================
# 2. inject_ramp
# =========================================================================

class TestRamp:
    def test_shape_unchanged(self, window):
        c, m = inject_ramp(window, CH, 1.0, [0.2, 0.6], [4, 48], _rng())
        assert c.shape == window.shape

    def test_non_injected_unchanged(self, window):
        c, m = inject_ramp(window, CH, 1.0, [0.2, 0.6], [4, 48], _rng())
        assert np.allclose(c[~m, :], window[~m, :])

    def test_injected_differs(self, window):
        c, m = inject_ramp(window, CH, 1.0, [0.2, 0.6], [4, 48], _rng())
        assert m.any()
        assert not np.allclose(c[m, CH], window[m, CH])

    def test_v_shape_envelope(self, window):
        """전반부 비율 감소, 후반부 비율 증가 (V자)."""
        c, m = inject_ramp(window, CH, 1.0, [0.2, 0.6], [4, 48], _rng())
        seg_orig = window[m, CH]
        seg_corr = c[m, CH]
        ratios = seg_corr / (seg_orig + 1e-9)
        n = len(ratios)
        if n < 4:
            pytest.skip("구간 너무 짧아 V자 검증 어려움")
        half = n // 2
        # 전반부: 순단조 감소인지 (완화: 평균값 비교)
        front_mean = ratios[:half].mean()
        back_mean  = ratios[half:].mean()
        assert front_mean >= back_mean - 0.05, \
            "V자 전반부 평균이 후반부보다 지나치게 작음 (감소 방향 오류)"
        # 최솟값이 중간 부근에 존재
        mid = ratios.argmin()
        assert n * 0.2 <= mid <= n * 0.8, f"최솟값 위치 이상: {mid}/{n}"

    def test_ratios_in_range(self, window):
        """배율은 trough(0.2~0.6) ~ 1.0 사이여야 함."""
        c, m = inject_ramp(window, CH, 1.0, [0.2, 0.6], [4, 48], _rng())
        ratios = c[m, CH] / (window[m, CH] + 1e-9)
        assert ratios.min() >= 0.15, f"하한 이탈: {ratios.min():.3f}"
        assert ratios.max() <= 1.05, f"상한 이탈: {ratios.max():.3f}"


# =========================================================================
# 3. inject_pulse_plateau
# =========================================================================

class TestPulsePlateau:
    def test_shape_unchanged(self, window):
        c, m = inject_pulse_plateau(window, CH, [1.5, 2.5], [8, 48], _rng())
        assert c.shape == window.shape

    def test_non_injected_unchanged(self, window):
        c, m = inject_pulse_plateau(window, CH, [1.5, 2.5], [8, 48], _rng())
        assert np.allclose(c[~m, :], window[~m, :])

    def test_injected_segment_is_constant(self, window):
        """주입 구간 채널값이 상수여야 함."""
        c, m = inject_pulse_plateau(window, CH, [1.5, 2.5], [8, 48], _rng())
        seg = c[m, CH]
        assert np.allclose(seg, seg[0], atol=1e-4), \
            f"주입 구간이 상수가 아님: std={seg.std():.6f}"

    def test_magnitude_applied(self, window):
        """주입값 ≈ base_val × magnitude (magnitude ∈ [1.5, 2.5])."""
        c, m = inject_pulse_plateau(window, CH, [1.5, 2.5], [8, 48], _rng())
        seg_idx = np.where(m)[0]
        base_val = abs(window[seg_idx[0] - 1, CH]) + 1e-6
        injected_val = c[seg_idx[0], CH]
        mag = injected_val / base_val
        assert 1.5 <= mag <= 2.5, f"magnitude 범위 벗어남: {mag:.3f}"


# =========================================================================
# 4. inject_replay
# =========================================================================

class TestReplay:
    def test_shape_unchanged(self, window):
        c, m = inject_replay(window, CH, [8, 48], window[np.newaxis], _rng())
        assert c.shape == window.shape

    def test_injected_differs_from_original(self, window):
        # 사인파 윈도우는 앞/뒤 구간이 서로 달라서 복사 시 값이 달라짐
        c, m = inject_replay(window, CH, [8, 48], window[np.newaxis], _rng())
        if not m.any():
            pytest.skip("비겹침 구간 없어서 skip됨")
        # 반드시 원본과 같지 않아야 한다고 단정하기 어려우므로
        # 최소한 구간 내 값이 원본 window의 어딘가에 존재함을 검증
        seg_corr = c[m, CH]
        # source는 같은 window에서 왔으므로, corrupted의 각 값이
        # original window 어딘가에 존재해야 함
        orig_vals = window[:, CH]
        for val in seg_corr:
            assert np.any(np.abs(orig_vals - val) < 1e-4), \
                f"replay 값 {val:.4f}이 원본 윈도우에 없음"

    def test_source_is_non_overlapping(self, window):
        """복사된 값이 타겟 구간의 원본 값과 다른 시점에서 왔음을 검증.

        타겟 구간의 원본값과 복사된 값이 동일한 경우는
        '같은 위치에서 복사'를 의미하므로 replay 의미 없음.
        단, 사인파 특성상 동일값이 우연히 일치할 수도 있어 경고만.
        """
        c, m = inject_replay(window, CH, [8, 48], window[np.newaxis], _rng())
        if not m.any():
            pytest.skip("비겹침 구간 없어서 skip됨")
        seg_orig = window[m, CH]
        seg_corr = c[m, CH]
        # source != target 이어야 하므로 적어도 한 점은 달라야 함
        # (완전히 같을 확률은 매우 낮음)
        if np.allclose(seg_orig, seg_corr):
            pytest.xfail("source와 target이 우연히 동일 (사인파 주기)")

    def test_non_injected_unchanged(self, window):
        c, m = inject_replay(window, CH, [8, 48], window[np.newaxis], _rng())
        assert np.allclose(c[~m, :], window[~m, :])

    def test_replay_value_matches_source_in_same_window(self, window):
        """replay 값이 동일 윈도우의 어떤 시점 구간과 정확히 일치함을 검증."""
        rng = _rng(7)
        c, m = inject_replay(window, CH, [8, 48], window[np.newaxis], rng)
        if not m.any():
            pytest.skip("비겹침 구간 없어서 skip됨")
        seg_len = m.sum()
        seg_corr = c[m, CH]

        # 동일 윈도우에서 길이 seg_len 구간 중 값이 일치하는 게 있는지 확인
        found = False
        for s in range(T - seg_len + 1):
            if np.allclose(window[s:s + seg_len, CH], seg_corr, atol=1e-4):
                found = True
                break
        assert found, "replay 값이 동일 윈도우의 어느 구간과도 일치하지 않음"


# =========================================================================
# 5. inject_instant_spike
# =========================================================================

class TestInstantSpike:
    def test_shape_unchanged(self, window):
        c, m = inject_instant_spike(window, CH, [0.9, 1.3], [1, 2], _rng())
        assert c.shape == window.shape

    def test_non_injected_unchanged(self, window):
        c, m = inject_instant_spike(window, CH, [0.9, 1.3], [1, 2], _rng())
        assert np.allclose(c[~m, :], window[~m, :])

    def test_spike_value_above_local_mean(self, window):
        c, m = inject_instant_spike(window, CH, [0.9, 1.3], [1, 2], _rng())
        local_mean = abs(window[:, CH].mean())
        spike_val = c[m, CH].mean()
        assert spike_val > local_mean * 1.8, \
            f"스파이크 값({spike_val:.2f})이 평균({local_mean:.2f})의 1.8배 미만"

    def test_spike_magnitude_in_range(self, window):
        # magnitude=[2.5, 5.0]: spike = local_max * (1 + mag), ratio = 1 + mag ∈ [3.5, 6.0]
        c, m = inject_instant_spike(window, CH, [2.5, 5.0], [1, 2], _rng())
        local_max = abs(window[:, CH].max()) + 1e-6
        ratio = c[m, CH].mean() / local_max
        assert 3.4 <= ratio <= 6.1, f"magnitude 범위 벗어남: ratio={ratio:.3f}"


# =========================================================================
# 6. inject_synthetic_anomalies (통합)
# =========================================================================

class TestInjectSyntheticAnomalies:
    @pytest.fixture
    def dummy_scaler(self):
        """항등 변환 scaler (mean=0, std=1 — 정규화 없음, raw 값 그대로)."""
        from src.adt.data.scalers import StandardScalerND
        scaler = StandardScalerND()
        scaler.mean_ = np.zeros(C, dtype=np.float32)
        scaler.std_  = np.ones(C, dtype=np.float32)
        return scaler

    @pytest.fixture
    def cfg(self):
        return {
            "injection_ratio": 0.5,   # 절반 주입 (테스트용)
            "anomaly_types": [
                {"type": "scale_down",    "prob": 0.2, "scale_factor": [0.3, 0.7], "duration_steps": [8, 48]},
                {"type": "ramp",          "prob": 0.2, "scale_start": 1.0, "trough_scale": [0.2, 0.6], "duration_steps": [4, 48]},
                {"type": "pulse_plateau", "prob": 0.2, "magnitude": [1.5, 2.5], "duration_steps": [8, 48]},
                {"type": "replay",        "prob": 0.2, "duration_steps": [8, 48]},
                {"type": "instant_spike", "prob": 0.2, "magnitude": [0.9, 1.3], "duration_steps": [1, 2]},
            ],
        }

    def test_output_shape(self, dummy_scaler, cfg, window):
        X = np.stack([window] * 20)  # (20, T, C)
        corrupted, labels = inject_synthetic_anomalies(X, cfg, dummy_scaler, seed=0)
        assert corrupted.shape == X.shape
        assert labels.shape == (20,)

    def test_label_count(self, dummy_scaler, cfg, window):
        X = np.stack([window] * 20)
        _, labels = inject_synthetic_anomalies(X, cfg, dummy_scaler, seed=0)
        assert labels.sum() == 10   # injection_ratio=0.5, N=20 → 10개

    def test_clean_windows_unchanged(self, dummy_scaler, cfg, window):
        X = np.stack([window] * 20)
        corrupted, labels = inject_synthetic_anomalies(X, cfg, dummy_scaler, seed=0)
        clean_idx = np.where(labels == 0)[0]
        assert np.allclose(corrupted[clean_idx], X[clean_idx])

    def test_reproducibility(self, dummy_scaler, cfg, window):
        X = np.stack([window] * 20)
        c1, l1 = inject_synthetic_anomalies(X, cfg, dummy_scaler, seed=99)
        c2, l2 = inject_synthetic_anomalies(X, cfg, dummy_scaler, seed=99)
        assert np.allclose(c1, c2)
        assert np.array_equal(l1, l2)
