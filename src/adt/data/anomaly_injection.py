"""합성 이상치 주입 모듈.

고장/이상 레이블이 없는 데이터에서 파이프라인을 정량 검증하기 위해
정상 윈도우에 인위적인 이상 패턴을 주입하고 주입 위치를 정답 레이블로 사용한다.
configs/eval/synthetic.yaml 파라미터를 따른다.
"""
from __future__ import annotations

import numpy as np


# -------------------------------------------------------------------------
# 개별 주입 함수  (window: (T, C), channel: int → corrupted, timestep_mask)
# -------------------------------------------------------------------------

def inject_spike(
    window: np.ndarray,
    channel: int,
    magnitude_std: list[float],
    duration_steps: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """특정 채널 값을 짧은 구간 동안 크게 튀게 만든다 (양/음 랜덤)."""
    corrupted = window.copy()
    T = len(window)
    ch_std = window[:, channel].std() + 1e-8
    duration = int(rng.integers(duration_steps[0], duration_steps[1] + 1))
    start = int(rng.integers(0, max(1, T - duration + 1)))
    magnitude = float(rng.uniform(magnitude_std[0], magnitude_std[1]))
    sign = rng.choice([-1.0, 1.0])
    corrupted[start:start + duration, channel] += sign * magnitude * ch_std
    mask = np.zeros(T, dtype=bool)
    mask[start:start + duration] = True
    return corrupted, mask


def inject_drop(
    window: np.ndarray,
    channel: int,
    magnitude_std: list[float],
    duration_steps: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """spike의 반대 방향 — 급격한 하락."""
    corrupted = window.copy()
    T = len(window)
    ch_std = window[:, channel].std() + 1e-8
    duration = int(rng.integers(duration_steps[0], duration_steps[1] + 1))
    start = int(rng.integers(0, max(1, T - duration + 1)))
    magnitude = float(rng.uniform(magnitude_std[0], magnitude_std[1]))
    corrupted[start:start + duration, channel] -= magnitude * ch_std
    mask = np.zeros(T, dtype=bool)
    mask[start:start + duration] = True
    return corrupted, mask


def inject_flatline(
    window: np.ndarray,
    channel: int,
    duration_steps: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """센서 고착 — 특정 구간 동안 값이 변하지 않음."""
    corrupted = window.copy()
    T = len(window)
    duration = int(rng.integers(duration_steps[0], duration_steps[1] + 1))
    start = int(rng.integers(0, max(1, T - duration + 1)))
    flat_val = corrupted[start, channel]
    corrupted[start:start + duration, channel] = flat_val
    mask = np.zeros(T, dtype=bool)
    mask[start:start + duration] = True
    return corrupted, mask


def inject_drift(
    window: np.ndarray,
    channel: int,
    magnitude_std: list[float],
    duration_steps: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """완만한 선형 drift — 센서 캘리브레이션 이슈 흉내."""
    corrupted = window.copy()
    T = len(window)
    ch_std = window[:, channel].std() + 1e-8
    duration = int(rng.integers(duration_steps[0], duration_steps[1] + 1))
    start = int(rng.integers(0, max(1, T - duration + 1)))
    magnitude = float(rng.uniform(magnitude_std[0], magnitude_std[1]))
    sign = rng.choice([-1.0, 1.0])
    drift = np.linspace(0.0, sign * magnitude * ch_std, duration, dtype=np.float32)
    corrupted[start:start + duration, channel] += drift
    mask = np.zeros(T, dtype=bool)
    mask[start:start + duration] = True
    return corrupted, mask


def inject_trend_break(
    window: np.ndarray,
    channel: int,
    duration_steps: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """다른 구간 패턴을 붙여넣어 일간 주기를 깨뜨림."""
    corrupted = window.copy()
    T = len(window)
    duration = int(rng.integers(duration_steps[0], min(duration_steps[1], T) + 1))
    start = int(rng.integers(0, max(1, T - duration + 1)))

    candidates = [s for s in range(0, T - duration + 1) if abs(s - start) >= duration]
    if candidates:
        src = int(rng.choice(candidates))
        corrupted[start:start + duration, channel] = window[src:src + duration, channel]
    else:
        corrupted[start:start + duration, channel] = window[start:start + duration, channel][::-1]

    mask = np.zeros(T, dtype=bool)
    mask[start:start + duration] = True
    return corrupted, mask


# -------------------------------------------------------------------------
# 통합 주입 함수
# -------------------------------------------------------------------------

_INJECTORS = {
    "spike": inject_spike,
    "drop": inject_drop,
    "flatline": inject_flatline,
    "drift": inject_drift,
    "trend_break": inject_trend_break,
}


def inject_synthetic_anomalies(
    clean_windows: np.ndarray,
    cfg: dict,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """injection_ratio만큼 윈도우를 골라 anomaly_types prob 비율로 이상치 주입.

    Args:
        clean_windows: (N, T, C) float32
        cfg          : synthetic.yaml 내용 (injection_ratio, anomaly_types, ...)
        seed         : 재현성 시드

    Returns:
        corrupted_windows: (N, T, C) — 주입된 윈도우 (나머지는 원본 그대로)
        labels          : (N,) int32 — 1=주입됨, 0=정상
    """
    rng = np.random.default_rng(seed)
    N, T, C = clean_windows.shape

    n_inject = max(1, int(N * cfg["injection_ratio"]))
    inject_indices = rng.choice(N, size=n_inject, replace=False)

    types = cfg["anomaly_types"]
    probs = np.array([t["prob"] for t in types], dtype=float)
    probs /= probs.sum()

    corrupted = clean_windows.copy()
    labels = np.zeros(N, dtype=np.int32)

    for idx in inject_indices:
        ti = int(rng.choice(len(types), p=probs))
        atype = types[ti]
        name = atype["type"]
        channel = int(rng.integers(0, C))
        window = corrupted[idx]

        if name in ("spike", "drop", "drift"):
            corrupted[idx], _ = _INJECTORS[name](
                window, channel,
                atype["magnitude_std"], atype["duration_steps"], rng,
            )
        elif name == "flatline":
            corrupted[idx], _ = inject_flatline(window, channel, atype["duration_steps"], rng)
        elif name == "trend_break":
            corrupted[idx], _ = inject_trend_break(window, channel, atype["duration_steps"], rng)

        labels[idx] = 1

    return corrupted, labels
