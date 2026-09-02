"""다중 클래스 분류 헤드 — Attack type classification.

forward는 raw logits(B, num_classes)를 반환. CrossEntropyLoss와 함께 사용.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """MeanPool+MaxPool concat → MLP → num_classes logits.

    Args:
        d_model   : Encoder 출력 차원
        num_classes: 분류 클래스 수 (all_type=5, unseen_X=4)
        hidden_dim: MLP 중간 차원
        dropout   : Dropout 확률
    """

    def __init__(
        self,
        d_model: int = 128,
        num_classes: int = 5,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
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
            H: (B, L, d_model) — frozen encoder 출력 (Attack 샘플만)

        Returns:
            (B, num_classes) raw logits — CrossEntropyLoss 입력
        """
        z = torch.cat([H.mean(dim=1), H.max(dim=1).values], dim=-1)  # (B, 2*d_model)
        return self.mlp(z)  # (B, num_classes)
