"""분류 헤드 — Encoder 출력에서 window-level 이진 분류.

forward는 raw logit을 반환. sigmoid/threshold는 호출부에서 처리.
BCEWithLogitsLoss와 함께 사용.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Mean/Max pooling + MLP 이진 분류 헤드.

    z = concat[MeanPool(H), MaxPool(H)]  (pooling="mean_max")
    logit = MLP(z)  → BCEWithLogitsLoss

    Args:
        d_model   : Encoder 출력 차원
        hidden_dim: MLP 중간 차원 (config로 제어, 기본 64)
        dropout   : Dropout 확률
        pooling   : "mean_max" | "mean" | "max"
    """

    def __init__(
        self,
        d_model: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        pooling: str = "mean_max",
    ) -> None:
        super().__init__()
        self.pooling = pooling
        pool_dim = 2 * d_model if pooling == "mean_max" else d_model

        self.mlp = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
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
            H: (B, L, d_model) — Encoder 출력 (mask=None로 통과한 것)

        Returns:
            (B,) — raw attack logit
        """
        if self.pooling == "mean_max":
            z = torch.cat([H.mean(dim=1), H.max(dim=1).values], dim=-1)
        elif self.pooling == "mean":
            z = H.mean(dim=1)
        elif self.pooling == "max":
            z = H.max(dim=1).values
        else:
            raise ValueError(f"pooling={self.pooling!r} 지원 안 함 (mean_max | mean | max)")

        return self.mlp(z).squeeze(-1)  # (B,)
