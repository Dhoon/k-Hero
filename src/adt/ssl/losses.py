"""SSL 손실함수.

masked_reconstruction_loss:
  마스킹된 위치에서만 MSE를 계산한다.
  마스킹 안 된 위치는 loss에 포함하지 않는다 — 마스킹된 곳만 어렵게 학습.

mask shape 두 가지 모두 지원:
  (B, T)    — segment 마스크: 해당 타임스텝의 모든 채널에 loss 적용
  (B, T, C) — channel/mixed 마스크: True인 (타임스텝, 채널) 원소에만 loss 적용

forecast_loss:
  미래 h step × C채널 전체에 대해 MSE.
  future_target이 데이터 생성 단계에서 segment 내부 값만 보장하므로 전체에 적용.

pretrain_joint_loss:
  L_pretrain = L_mask + forecast_weight * L_forecast
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


def forecast_loss(
    pred_future: torch.Tensor,
    true_future: torch.Tensor,
) -> torch.Tensor:
    """미래 h step × C채널 전체 MSE.

    windowing 단계에서 segment 경계 밖 window는 이미 drop됐으므로
    future_target은 전부 유효한 값이다 — 마스킹 없이 전체에 loss 적용.

    Args:
        pred_future: (B, h, C) — ForecastingHead 출력
        true_future: (B, h, C) — 정규화된 실제 미래값

    Returns:
        scalar tensor — 평균 MSE
    """
    return F.mse_loss(pred_future, true_future, reduction="mean")


def pretrain_joint_loss(
    l_mask: torch.Tensor,
    l_forecast: torch.Tensor,
    forecast_weight: float,
) -> torch.Tensor:
    """L_pretrain = L_mask + forecast_weight * L_forecast.

    Args:
        l_mask          : masked_reconstruction_loss 결과
        l_forecast      : forecast_loss 결과
        forecast_weight : lambda (pretrain yaml의 forecast_loss_weight)

    Returns:
        scalar tensor
    """
    return l_mask + forecast_weight * l_forecast
