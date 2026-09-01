"""SSL 손실함수.

masked_reconstruction_loss:
  마스킹된 위치에서만 MSE를 계산한다.
  마스킹 안 된 위치는 loss에 포함하지 않는다 — 마스킹된 곳만 어렵게 학습.

mask shape 두 가지 모두 지원:
  (B, T)    — segment 마스크: 해당 타임스텝의 모든 채널에 loss 적용
  (B, T, C) — channel/mixed 마스크: True인 (타임스텝, 채널) 원소에만 loss 적용
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """마스킹된 위치에서만 MSE loss 계산.

    Args:
        pred  : (B, T, C) — MaskedReconstructionHead 출력 (예측값)
        target: (B, T, C) — 원본 정규화된 전력값
        mask  : (B, T)    bool — segment 마스크, 타임스텝 전체 채널에 적용
                (B, T, C) bool — element-wise 마스크

    Returns:
        scalar tensor — 마스킹된 위치 평균 MSE
    """
    if mask.dim() == 2:
        # (B, T) → (B, T, C): 해당 타임스텝의 모든 채널에 loss 적용
        mask = mask.unsqueeze(-1).expand_as(pred)

    elif mask.dim() != 3:
        raise ValueError(
            f"mask는 2D(B,T) 또는 3D(B,T,C)여야 합니다. 받은 shape: {mask.shape}"
        )

    n_masked = mask.sum()
    if n_masked == 0:
        # 마스킹된 위치가 없으면 gradient 흐름은 유지하되 loss=0
        return (pred * 0.0).sum()

    return F.mse_loss(pred[mask], target[mask], reduction="mean")
