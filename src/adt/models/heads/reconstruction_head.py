"""이상탐지 head 모듈.

ReconstructionAnomalyHead:
  encoder 출력 (B, T, d_model) → Linear(d_model → n_features) → (B, T, n_features)
  masking 없이 전체 window를 복원하고, 복원 오차를 이상 점수로 사용한다.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ReconstructionAnomalyHead(nn.Module):
    """encoder 표현 → 전체 window 복원 (이상 점수 = 재구성 오차).

    Args:
        d_model   : encoder 출력 차원
        n_features: 복원할 채널 수
    """

    def __init__(self, d_model: int = 128, n_features: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, n_features)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, n_features)
        """
        return self.proj(x)

    @classmethod
    def from_pretrain_state(
        cls,
        d_model: int,
        n_features: int,
        head_state_dict: dict,
    ) -> "ReconstructionAnomalyHead":
        """pretrain head 가중치로 warm start.

        MaskedReconstructionHead와 구조(proj: Linear)가 동일하므로
        state_dict를 그대로 복사할 수 있다. shape 불일치 시 fresh init으로 fallback.
        """
        instance = cls(d_model, n_features)
        try:
            instance.load_state_dict(head_state_dict, strict=True)
        except RuntimeError:
            pass  # shape 불일치 → xavier init 유지
        return instance


# pretrain_head.py의 MaskedReconstructionHead와 구조가 동일하므로 alias로 통합
MaskedReconstructionHead = ReconstructionAnomalyHead
