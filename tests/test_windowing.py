"""make_windows forecast_horizon 유닛 테스트."""
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adt.data.windowing import make_windows, process_meter_segments

C = 4
WINDOW_SIZE = 10
STRIDE = 1
H = 3


# -------------------------------------------------------------------------
# 헬퍼
# -------------------------------------------------------------------------

def _vals(T: int) -> np.ndarray:
    """단조증가 값 — segment 경계 leakage 검증에 사용."""
    return np.arange(T * C, dtype=np.float32).reshape(T, C)


def _tfeats(T: int) -> np.ndarray:
    return np.zeros((T, 2), dtype=np.float32)


# -------------------------------------------------------------------------
# future_target shape
# -------------------------------------------------------------------------

class TestFutureTargetShape:
    def test_shape_with_horizon(self):
        T = 20
        X, tf, future = make_windows(_vals(T), _tfeats(T), WINDOW_SIZE, STRIDE, forecast_horizon=H)
        expected_n = T - WINDOW_SIZE - H + 1
        assert X.shape == (expected_n, WINDOW_SIZE, C)
        assert future.shape == (expected_n, H, C)

    def test_shape_no_horizon(self):
        """forecast_horizon=0 이면 future_target shape (N, 0, C)."""
        T = 20
        X, tf, future = make_windows(_vals(T), _tfeats(T), WINDOW_SIZE, STRIDE, forecast_horizon=0)
        expected_n = T - WINDOW_SIZE + 1
        assert X.shape == (expected_n, WINDOW_SIZE, C)
        assert future.shape == (expected_n, 0, C)

    def test_n_windows_reduced_by_horizon(self):
        """forecast_horizon=H 이면 H=0 대비 H개 window가 drop된다."""
        T = 30
        X0, _, _ = make_windows(_vals(T), _tfeats(T), WINDOW_SIZE, STRIDE, forecast_horizon=0)
        Xh, _, _ = make_windows(_vals(T), _tfeats(T), WINDOW_SIZE, STRIDE, forecast_horizon=H)
        assert len(Xh) == len(X0) - H


# -------------------------------------------------------------------------
# segment 끝부분 drop
# -------------------------------------------------------------------------

class TestTailDrop:
    def test_exactly_min_len_gives_one_window(self):
        """T == window_size + H 이면 정확히 1개 window만 생성."""
        T = WINDOW_SIZE + H
        X, _, future = make_windows(_vals(T), _tfeats(T), WINDOW_SIZE, STRIDE, forecast_horizon=H)
        assert len(X) == 1
        assert future.shape == (1, H, C)

    def test_one_short_gives_zero_windows(self):
        """T == window_size + H - 1 이면 window 0개 (미래 h step 부족)."""
        T = WINDOW_SIZE + H - 1
        X, _, future = make_windows(_vals(T), _tfeats(T), WINDOW_SIZE, STRIDE, forecast_horizon=H)
        assert len(X) == 0
        assert len(future) == 0


# -------------------------------------------------------------------------
# future_target 값 정확성 + segment 경계 leakage 없음
# -------------------------------------------------------------------------

class TestFutureTargetValues:
    def test_future_target_correct_values(self):
        """future_target[i] == values[i+window_size : i+window_size+H]."""
        T = 20
        values = _vals(T)
        X, _, future = make_windows(values, _tfeats(T), WINDOW_SIZE, STRIDE, forecast_horizon=H)
        for i in range(len(X)):
            expected = values[i + WINDOW_SIZE : i + WINDOW_SIZE + H]
            np.testing.assert_array_equal(future[i], expected)

    def test_no_cross_segment_leakage(self):
        """process_meter_segments 사용 시 segment 경계를 넘는 future_target 없음.

        두 segment를 이어서 처리하면 각 segment의 tail이 drop되고
        다음 segment 값이 future_target에 섞이지 않아야 한다.
        """
        import pandas as pd

        def _make_seg(start_val: int, T: int) -> pd.DataFrame:
            vals = np.arange(start_val, start_val + T * C, dtype=np.float32).reshape(T, C)
            timestamps = pd.date_range("2024-01-01", periods=T, freq="15min")
            df = pd.DataFrame(
                {
                    "일자시간": timestamps,
                    "ch0": vals[:, 0],
                    "ch1": vals[:, 1],
                    "ch2": vals[:, 2],
                    "ch3": vals[:, 3],
                }
            )
            return df

        feature_cols = ["ch0", "ch1", "ch2", "ch3"]
        seg1 = _make_seg(0, WINDOW_SIZE + H)       # 정확히 1개 window 가능
        seg2 = _make_seg(10000, WINDOW_SIZE + H)   # 구분 가능한 큰 값으로 시작

        splits = process_meter_segments(
            [seg1, seg2], feature_cols, WINDOW_SIZE, STRIDE,
            ratios=[1.0, 0.0, 0.0], forecast_horizon=H
        )
        future = splits["train"]["future_target"]  # (2, H, C) — seg1 1개 + seg2 1개
        assert future.shape[0] == 2

        # seg1의 future 값은 모두 seg1 범위(0 ~ WINDOW_SIZE+H-1)
        seg1_max = float((WINDOW_SIZE + H) * C - 1)
        assert float(future[0].max()) <= seg1_max, "seg1 future가 seg2 값을 포함함 (leakage)"

        # seg2의 future 값은 모두 seg2 범위(10000 ~)
        assert float(future[1].min()) >= 10000.0, "seg2 future가 seg1 값을 포함함 (leakage)"
