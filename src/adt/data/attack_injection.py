"""합성 이상치 주입 모듈 (물리적 스케일 기반).

모든 변조는 scaler.inverse_transform → 변조 → scaler.transform 순으로
원본(kWh) 단위에서 적용한다.
z-score 위에서 배율을 곱하면 "사용량 30% 축소" 같은 물리적 의미가 틀어진다.

공격 5종:
  scale_down    : 구간 균일 축소 (계량기 조작/절취)
  ramp          : V자형 드리프트 — 1.0→trough→1.0 선형 envelope 적용
  pulse_plateau : 계단식 상승 후 유지 (부하 주입/역류)
  replay        : 동일 윈도우의 다른 시점 구간 복사 (재생 공격)
  instant_spike : 순간 급증 (이상 부하)
"""
from __future__ import annotations

import numpy as np


# -------------------------------------------------------------------------
# 개별 주입 함수  (window_raw: (T, C) kWh 단위, in-place 금지 → copy 반환)
# -------------------------------------------------------------------------

def inject_scale_down(
    window_raw: np.ndarray,
    channel: int,
    scale_factor: list[float],
    duration_steps: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """구간 raw 값에 동일한 scale_factor를 elementwise 곱 (원본 굴곡 유지).

    injected[t] = original_raw[t] * scale_factor  (구간 내 모든 t)
    값의 형태(기복)는 그대로, 크기만 scale_factor(0.3~0.7)배로 축소된다.
    """
    corrupted = window_raw.copy()
    T = len(window_raw)
    duration = int(rng.integers(duration_steps[0], duration_steps[1] + 1))
    start = int(rng.integers(0, max(1, T - duration + 1)))
    factor = float(rng.uniform(scale_factor[0], scale_factor[1]))
    corrupted[start:start + duration, channel] *= factor   # elementwise, 원본 굴곡 유지
    mask = np.zeros(T, dtype=bool)
    mask[start:start + duration] = True
    return corrupted, mask


def inject_ramp(
    window_raw: np.ndarray,
    channel: int,
    scale_start: float,
    trough_scale: list[float],
    duration_steps: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """V자형 드리프트: scale_start → trough → scale_start 선형 envelope를 원본에 곱함.

    전반부: scale_start(1.0) → trough(0.2~0.6) 선형 감소
    후반부: trough → scale_start(1.0) 선형 증가 (대칭 복귀)
    injected[t] = original_raw[t] * envelope(t)  — 원본 굴곡은 유지됨.

    Args:
        trough_scale: [min, max] — V자 최저점 배율 범위
    """
    corrupted = window_raw.copy()
    T = len(window_raw)
    duration = int(rng.integers(duration_steps[0], duration_steps[1] + 1))
    start = int(rng.integers(0, max(1, T - duration + 1)))
    trough = float(rng.uniform(trough_scale[0], trough_scale[1]))

    half = duration // 2
    # 전반부: 1.0 → trough
    front = np.linspace(scale_start, trough, half, dtype=np.float32) if half > 0 else np.empty(0, dtype=np.float32)
    # 후반부: trough → 1.0 (홀수 길이면 후반부가 1 step 더 가짐)
    back = np.linspace(trough, scale_start, duration - half, dtype=np.float32) if duration - half > 0 else np.empty(0, dtype=np.float32)
    envelope = np.concatenate([front, back])  # length == duration

    corrupted[start:start + duration, channel] *= envelope
    mask = np.zeros(T, dtype=bool)
    mask[start:start + duration] = True
    return corrupted, mask


def inject_pulse_plateau(
    window_raw: np.ndarray,
    channel: int,
    magnitude: list[float],
    duration_steps: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """구간 시작 직전 값의 magnitude배로 계단식 상승 후 유지."""
    corrupted = window_raw.copy()
    T = len(window_raw)
    duration = int(rng.integers(duration_steps[0], duration_steps[1] + 1))
    # start >= 1 이어야 '직전 값'을 참조할 수 있음
    start = int(rng.integers(1, max(2, T - duration + 1)))
    mag = float(rng.uniform(magnitude[0], magnitude[1]))
    base_val = abs(corrupted[start - 1, channel]) + 1e-6  # 직전 값 기준
    corrupted[start:start + duration, channel] = base_val * mag
    mask = np.zeros(T, dtype=bool)
    mask[start:start + duration] = True
    return corrupted, mask


def inject_replay(
    window_raw: np.ndarray,
    channel: int,
    duration_steps: list[int],
    source_pool: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """재생 공격: 동일 윈도우의 비겹침 시점 구간을 그대로 복사해 붙여넣기.

    injected[target] = original_raw[source]  — 배율/오프셋 조정 없이 원본 값 그대로.

    1. 먼저 동일 윈도우(window_raw) 내에서 target과 겹치지 않는 source 구간 탐색
    2. 없으면 source_pool(다른 정상 윈도우)에서 fallback
    3. 둘 다 없으면 skip하고 경고 출력 (원본 반환, mask는 모두 False)

    Args:
        source_pool: (N, T, C) kWh — fallback용 다른 정상 윈도우들
    """
    T = len(window_raw)
    duration = int(rng.integers(duration_steps[0], min(duration_steps[1], T - 1) + 1))
    target_start = int(rng.integers(0, max(1, T - duration + 1)))

    # 동일 윈도우 내 비겹침 후보 (최소 duration만큼 떨어진 구간)
    same_win_candidates = [
        s for s in range(0, T - duration + 1)
        if s + duration <= target_start or s >= target_start + duration
    ]

    corrupted = window_raw.copy()
    mask = np.zeros(T, dtype=bool)

    if same_win_candidates:
        src_start = int(rng.choice(same_win_candidates))
        corrupted[target_start:target_start + duration, channel] = (
            window_raw[src_start:src_start + duration, channel]
        )
        mask[target_start:target_start + duration] = True

    elif len(source_pool) > 0:
        pool_idx = int(rng.integers(0, len(source_pool)))
        src_start = int(rng.integers(0, max(1, T - duration + 1)))
        corrupted[target_start:target_start + duration, channel] = (
            source_pool[pool_idx, src_start:src_start + duration, channel]
        )
        mask[target_start:target_start + duration] = True
        print(f"[replay] 동일 윈도우 비겹침 구간 없음 → source_pool fallback (pool_idx={pool_idx})")

    else:
        print(f"[replay] 소스 구간 없음 — skip (원본 유지)")

    return corrupted, mask


def inject_instant_spike(
    window_raw: np.ndarray,
    channel: int,
    magnitude: list[float],
    duration_steps: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """윈도우 채널 로컬 최대값 × (1 + magnitude) 값으로 duration 동안 설정."""
    corrupted = window_raw.copy()
    T = len(window_raw)
    duration = int(rng.integers(duration_steps[0], duration_steps[1] + 1))
    start = int(rng.integers(0, max(1, T - duration + 1)))
    local_max = abs(window_raw[:, channel].max()) + 1e-6
    mag = float(rng.uniform(magnitude[0], magnitude[1]))
    corrupted[start:start + duration, channel] = local_max * (1.0 + mag)
    mask = np.zeros(T, dtype=bool)
    mask[start:start + duration] = True
    return corrupted, mask


# -------------------------------------------------------------------------
# 통합 주입 함수
# -------------------------------------------------------------------------

def inject_synthetic_anomalies(
    clean_windows: np.ndarray,
    cfg: dict,
    scaler,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """injection_ratio만큼 윈도우를 골라 5종 공격 중 하나를 원본 스케일로 주입.

    Args:
        clean_windows: (N, T, C) 정규화된 float32
        cfg          : synthetic.yaml 내용
        scaler       : StandardScalerND (inverse_transform / transform 사용)
        seed         : 재현성 시드

    Returns:
        corrupted_windows: (N, T, C) 정규화된 float32
        labels          : (N,) int32  1=주입됨, 0=정상
    """
    rng = np.random.default_rng(seed)
    N, T, C = clean_windows.shape

    # 1. 원본 스케일로 복원
    raw_windows = scaler.inverse_transform(clean_windows)  # (N, T, C) kWh

    n_inject = max(1, int(N * cfg["injection_ratio"]))
    inject_indices = rng.choice(N, size=n_inject, replace=False)

    types = cfg["anomaly_types"]
    probs = np.array([t["prob"] for t in types], dtype=float)
    probs /= probs.sum()

    corrupted_raw = raw_windows.copy()
    labels = np.zeros(N, dtype=np.int32)

    for idx in inject_indices:
        ti = int(rng.choice(len(types), p=probs))
        atype = types[ti]
        name = atype["type"]
        channel = int(rng.integers(0, C))
        window_raw = corrupted_raw[idx]  # (T, C)

        if name == "scale_down":
            corrupted_raw[idx], _ = inject_scale_down(
                window_raw, channel,
                atype["scale_factor"], atype["duration_steps"], rng,
            )
        elif name == "ramp":
            corrupted_raw[idx], _ = inject_ramp(
                window_raw, channel,
                atype["scale_start"], atype["trough_scale"], atype["duration_steps"], rng,
            )
        elif name == "pulse_plateau":
            corrupted_raw[idx], _ = inject_pulse_plateau(
                window_raw, channel,
                atype["magnitude"], atype["duration_steps"], rng,
            )
        elif name == "replay":
            corrupted_raw[idx], _ = inject_replay(
                window_raw, channel,
                atype["duration_steps"], raw_windows, rng,
            )
        elif name == "instant_spike":
            corrupted_raw[idx], _ = inject_instant_spike(
                window_raw, channel,
                atype["magnitude"], atype["duration_steps"], rng,
            )

        labels[idx] = 1

    # 3. 다시 정규화
    corrupted_norm = scaler.transform(corrupted_raw)
    return corrupted_norm, labels
