"""Self-supervised pretraining용 head 모듈.

MaskedReconstructionHead:
  encoder 출력 (B, T, d_model) → Linear(d_model → n_features) → (B, T, n_features)
  마스킹된 위치의 원본 전력값을 복원하는 head.
  losses.masked_reconstruction_loss와 쌍으로 사용.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MaskedReconstructionHead(nn.Module):
    """encoder 표현 → 원본 채널값 복원.

    Args:
        d_model   : encoder 출력 차원
        n_features: 복원할 채널 수 (입력과 동일, 기본 4)
    """

    def __init__(self, d_model: int = 128, n_features: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, n_features)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) — encoder 출력

        Returns:
            (B, T, n_features) — 복원된 전력값 예측
        """
        return self.proj(x)
