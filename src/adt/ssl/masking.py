"""SSL 마스킹 전략.

설계 (pretrain_methodology.md §1):
  - segment masking: 연속된 구간(1~2개 블록)을 통째로 마스킹 — trivial shortcut 방지
  - channel masking: 채널 1개 전체를 마스킹 — 채널 간 물리적 관계 학습
  - mixed: 배치 아이템별로 segment/channel 중 하나를 확률적으로 선택

반환 shape:
  mask_mode="segment" → (B, T)      bool  — encoder의 (B,T) 경로 사용
  mask_mode="channel" → (B, T, C)   bool  — encoder의 (B,T,C) 경로 사용
  mask_mode="mixed"   → (B, T, C)   bool  — 두 방식을 통합 표현 (항상 3D)

mask_guard_tail (ssl_cfg 키):
  segment 마스킹 시 window 마지막 guard_tail timestep을 마스킹 후보에서 제외.
  forecasting head가 마지막 hidden state h_L을 사용하므로 anchor 구간을 보호.
  channel 마스킹에는 적용하지 않는다 (채널 차원 마스킹이므로 시간 경계와 무관).
"""
from __future__ import annotations

import torch


# -------------------------------------------------------------------------
# 단일 시퀀스 마스킹 (1D)
# -------------------------------------------------------------------------

def _segment_mask_1d(
    T: int,
    mask_ratio: float,
    device: torch.device,
    guard_tail: int = 0,
) -> torch.Tensor:
    """길이 T 시퀀스에서 연속 구간 1~2개로 mask_ratio 비율 마스킹.

    guard_tail > 0 이면 마지막 guard_tail timestep은 마스킹 후보에서 제외.
    즉 모든 마스크 구간은 [0, T - guard_tail) 범위 안에만 배치된다.

    Returns:
        (T,) bool tensor
    """
    T_eff = T - guard_tail  # 마스킹 가능한 유효 구간 길이
    if T_eff <= 0:
        return torch.zeros(T, dtype=torch.bool, device=device)

    n_masked = max(1, int(T * mask_ratio))
    n_masked = min(n_masked, T_eff)           # 유효 구간을 넘을 수 없음
    mask = torch.zeros(T, dtype=torch.bool, device=device)

    n_segs = 1 if torch.rand(1).item() < 0.5 else 2

    remaining = n_masked
    for i in range(n_segs):
        if remaining <= 0:
            break
        seg_len = remaining if i == n_segs - 1 else max(1, remaining // 2)
        seg_len = min(seg_len, T_eff)         # 개별 segment도 유효 구간 내로 제한
        max_start = max(0, T_eff - seg_len)
        start = torch.randint(0, max_start + 1, (1,), device=device).item()
        mask[start : start + seg_len] = True
        remaining -= seg_len

    return mask


def _channel_mask_3d(T: int, C: int, device: torch.device) -> torch.Tensor:
    """채널 1개를 무작위 선택해 전체 타임스텝에 걸쳐 마스킹.

    Returns:
        (T, C) bool tensor
    """
    mask = torch.zeros(T, C, dtype=torch.bool, device=device)
    ch = torch.randint(0, C, (1,), device=device).item()
    mask[:, ch] = True
    return mask


# -------------------------------------------------------------------------
# 공개 API
# -------------------------------------------------------------------------

def segment_mask(
    x: torch.Tensor,
    mask_ratio: float,
    guard_tail: int = 0,
) -> torch.Tensor:
    """배치 전체에 segment 마스킹 적용.

    Args:
        x         : (B, T, C)  — shape/device 참조용
        mask_ratio: 마스킹 비율
        guard_tail: 마스킹 불가 tail timestep 수

    Returns:
        (B, T) bool  — True = 마스킹된 타임스텝 (전체 채널)
    """
    B, T, _ = x.shape
    masks = torch.stack(
        [_segment_mask_1d(T, mask_ratio, x.device, guard_tail) for _ in range(B)]
    )  # (B, T)
    return masks


def channel_mask(x: torch.Tensor, mask_ratio: float) -> torch.Tensor:  # noqa: ARG001
    """배치 전체에 channel 마스킹 적용.

    Args:
        x         : (B, T, C)
        mask_ratio: 사용하지 않음 (항상 채널 1개 마스킹), 인터페이스 통일용

    Returns:
        (B, T, C) bool  — True = 마스킹된 (타임스텝, 채널)
    """
    B, T, C = x.shape
    masks = torch.stack(
        [_channel_mask_3d(T, C, x.device) for _ in range(B)]
    )  # (B, T, C)
    return masks


def mixed_mask(
    x: torch.Tensor,
    mask_ratio: float,
    segment_prob: float = 0.7,
    guard_tail: int = 0,
) -> torch.Tensor:
    """배치 아이템별로 segment/channel 방식을 확률적으로 선택.

    두 방식의 mask를 통합하기 위해 항상 (B, T, C) 반환.
    segment 아이템: 마스킹된 타임스텝의 모든 채널을 True로 확장.
    channel 아이템: 선택된 채널 전체를 True.
    guard_tail은 segment 방식에만 적용 (channel 방식은 시간 경계와 무관).

    Args:
        x           : (B, T, C)
        mask_ratio  : 현재 curriculum mask ratio
        segment_prob: segment 방식 선택 확률
        guard_tail  : segment 마스킹 불가 tail timestep 수

    Returns:
        (B, T, C) bool
    """
    B, T, C = x.shape
    mask = torch.zeros(B, T, C, dtype=torch.bool, device=x.device)

    for b in range(B):
        if torch.rand(1, device=x.device).item() < segment_prob:
            seg = _segment_mask_1d(T, mask_ratio, x.device, guard_tail)  # (T,)
            mask[b] = seg.unsqueeze(-1).expand(T, C)                      # all channels
        else:
            mask[b] = _channel_mask_3d(T, C, x.device)                   # one channel

    return mask  # (B, T, C)


def generate_mask(x: torch.Tensor, ssl_cfg: dict) -> torch.Tensor:
    """설정에 따라 마스크를 생성하는 메인 인터페이스.

    forecast branch가 clean pass를 별도로 사용하므로 guard_tail 없이
    window 전체에 균등 마스킹.

    Args:
        x      : (B, T, C)  — 입력 텐서 (shape/device 참조만)
        ssl_cfg: pretrain yaml의 ssl 섹션

    Returns:
        mask_mode="segment" → (B, T)    bool
        mask_mode="channel" → (B, T, C) bool
        mask_mode="mixed"   → (B, T, C) bool
    """
    mask_ratio: float = ssl_cfg.get("mask_ratio", 0.40)
    mode: str = ssl_cfg.get("mask_mode", "mixed")

    if mode == "segment":
        return segment_mask(x, mask_ratio, guard_tail=0)

    if mode == "channel":
        return channel_mask(x, mask_ratio)

    if mode == "mixed":
        seg_prob: float = ssl_cfg.get("segment_prob", 0.7)
        return mixed_mask(x, mask_ratio, segment_prob=seg_prob, guard_tail=0)

    raise ValueError(f"지원하지 않는 mask_mode: {mode!r}. segment | channel | mixed 중 선택.")
