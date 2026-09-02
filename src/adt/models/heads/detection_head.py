"""이진 탐지 헤드 — Normal vs Attack.

forward는 raw logit(B,)을 반환. BCEWithLogitsLoss와 함께 사용.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    """MeanPool+MaxPool concat → MLP → 1 raw logit.

    Args:
        d_model   : Encoder 출력 차원
        hidden_dim: MLP 중간 차원
        dropout   : Dropout 확률
    """

    def __init__(
        self,
        d_model: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (B, L, d_model) — frozen encoder 출력

        Returns:
            (B,) raw logit — BCEWithLogitsLoss 입력
        """
        z = torch.cat([H.mean(dim=1), H.max(dim=1).values], dim=-1)  # (B, 2*d_model)
        return self.mlp(z).squeeze(-1)  # (B,)
