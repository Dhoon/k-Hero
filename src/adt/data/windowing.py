"""Sliding window → (N, T, C) 텐서 생성 + 시간 특징 추출.

Transformer Input 구성 (pretrain_methodology.md §2):
  - X        : (N, window_size, n_features)  — 채널별 정규화된 전력값
  - time_feat: (N, window_size, 2)           — hour_of_day (0-23), day_of_week (0-6)

윈도우 경계 규칙:
  - 계량기 경계를 넘는 윈도우 생성 금지
  - segment 경계를 넘는 윈도우 생성 금지 (gap 기반 분리 후 segment별 독립 처리)
train/val/test 분할은 시간 순서를 지켜서 수행 (셔플 금지 — leakage 방지).
"""
from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# 타입 정의
# -------------------------------------------------------------------------

class SplitArrays(TypedDict):
    X: np.ndarray           # (N, window_size, C)
    time_feat: np.ndarray   # (N, window_size, 2)


# -------------------------------------------------------------------------
# 시간 특징
# -------------------------------------------------------------------------

def extract_time_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """각 타임스텝의 [hour_of_day, day_of_week] 반환.

    Args:
        timestamps: DatetimeIndex, 길이 T

    Returns:
        ndarray shape (T, 2), dtype float32
          col 0 — hour_of_day  (0–23)
          col 1 — day_of_week  (0=월, 6=일)
    """
    hour = timestamps.hour.values.astype(np.float32)   # (T,)
    dow = timestamps.dayofweek.values.astype(np.float32)  # (T,)
    return np.stack([hour, dow], axis=1)                # (T, 2)


# -------------------------------------------------------------------------
# 슬라이딩 윈도우
# -------------------------------------------------------------------------

def make_windows(
    values: np.ndarray,
    time_feats: np.ndarray,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """슬라이딩 윈도우 생성.

    Args:
        values   : (T, C)  — 정규화된 전력 피처
        time_feats: (T, 2) — hour_of_day, day_of_week
        window_size: 창 크기 (timestep)
        stride   : 슬라이딩 보폭

    Returns:
        X        : (N, window_size, C)
        T_feat   : (N, window_size, 2)
    """
    T = len(values)
    if T < window_size:
        return np.empty((0, window_size, values.shape[1]), dtype=np.float32), \
               np.empty((0, window_size, 2), dtype=np.float32)

    starts = range(0, T - window_size + 1, stride)
    X = np.stack([values[i : i + window_size] for i in starts]).astype(np.float32)
    T_feat = np.stack([time_feats[i : i + window_size] for i in starts]).astype(np.float32)
    return X, T_feat


# -------------------------------------------------------------------------
# 시간 순서 분할
# -------------------------------------------------------------------------

def split_windows(
    X: np.ndarray,
    time_feat: np.ndarray,
    ratios: list[float],
) -> dict[str, SplitArrays]:
    """시간 순서를 지켜 train/val/test 분할 (셔플 금지).

    Args:
        X        : (N, window_size, C)
        time_feat: (N, window_size, 2)
        ratios   : [train_ratio, val_ratio, test_ratio], 합계 1.0

    Returns:
        {'train': SplitArrays, 'val': SplitArrays, 'test': SplitArrays}
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, f"ratios 합계가 1이 아님: {ratios}"
    N = len(X)
    n_train = int(N * ratios[0])
    n_val = int(N * ratios[1])

    return {
        "train": {
            "X": X[:n_train],
            "time_feat": time_feat[:n_train],
        },
        "val": {
            "X": X[n_train : n_train + n_val],
            "time_feat": time_feat[n_train : n_train + n_val],
        },
        "test": {
            "X": X[n_train + n_val :],
            "time_feat": time_feat[n_train + n_val :],
        },
    }


# -------------------------------------------------------------------------
# per-segment 처리 + 결합
# -------------------------------------------------------------------------

def _process_single_segment(
    seg_df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
    stride: int,
    ratios: list[float],
) -> dict[str, SplitArrays] | None:
    """단일 segment DataFrame → split별 윈도우. 윈도우가 0개면 None 반환."""
    timestamps = pd.DatetimeIndex(seg_df["일자시간"])
    values = seg_df[feature_cols].to_numpy(dtype=np.float32)
    time_feats = extract_time_features(timestamps)

    X, T_feat = make_windows(values, time_feats, window_size, stride)
    if len(X) == 0:
        return None
    return split_windows(X, T_feat, ratios)


def _empty_splits(window_size: int, n_features: int) -> dict[str, SplitArrays]:
    empty_X = np.empty((0, window_size, n_features), dtype=np.float32)
    empty_t = np.empty((0, window_size, 2), dtype=np.float32)
    return {s: {"X": empty_X, "time_feat": empty_t} for s in ("train", "val", "test")}


def process_meter_segments(
    segments: list[pd.DataFrame],
    feature_cols: list[str],
    window_size: int,
    stride: int,
    ratios: list[float],
) -> dict[str, SplitArrays]:
    """여러 segment(gap 분리 후) → 각각 windowing → split별 concat.

    segment 경계를 넘는 윈도우는 생성되지 않는다.
    각 segment 내에서 시간 순서 유지 split 후 전체 concat.

    Args:
        segments    : preprocessing.preprocess_meter()가 반환한 segment DataFrame 리스트
        feature_cols: 사용할 전력 채널 컬럼명
        window_size : 창 크기
        stride      : 슬라이딩 보폭
        ratios      : [train, val, test] 비율

    Returns:
        {'train': SplitArrays, 'val': SplitArrays, 'test': SplitArrays}
    """
    if not segments:
        return _empty_splits(window_size, len(feature_cols))

    all_splits = []
    for seg_df in segments:
        missing = [c for c in feature_cols if c not in seg_df.columns]
        if missing:
            continue
        result = _process_single_segment(seg_df, feature_cols, window_size, stride, ratios)
        if result is not None:
            all_splits.append(result)

    if not all_splits:
        return _empty_splits(window_size, len(feature_cols))

    return concat_meter_splits(all_splits)


def concat_meter_splits(
    meter_splits: list[dict[str, SplitArrays]],
) -> dict[str, SplitArrays]:
    """여러 계량기의 split 결과를 split별로 concat.

    각 계량기는 독립적인 시계열이므로 per-meter split 후 concat한다
    (계량기 경계를 넘는 윈도우 생성 방지).
    """
    result: dict[str, SplitArrays] = {}
    for split in ("train", "val", "test"):
        Xs = [m[split]["X"] for m in meter_splits if len(m[split]["X"]) > 0]
        Ts = [m[split]["time_feat"] for m in meter_splits if len(m[split]["time_feat"]) > 0]
        result[split] = {
            "X": np.concatenate(Xs, axis=0) if Xs else np.empty((0,), dtype=np.float32),
            "time_feat": np.concatenate(Ts, axis=0) if Ts else np.empty((0,), dtype=np.float32),
        }
    return result
