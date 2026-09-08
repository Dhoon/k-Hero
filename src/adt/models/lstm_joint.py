"""LSTM Joint Multi-task model.

Shared encoder → det_head + cls_head.
z는 encoder에서 한 번만 계산되고 두 head가 공유한다.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.adt.models.lstm_ae import (
    LSTMClassificationHead,
    LSTMDetectionHead,
    LSTMEncoder,
)


class LSTMJoint(nn.Module):
    """Encoder + Detection head + Classification head.

    Args:
        n_features  : 입력 채널 수 (C)
        hidden_dim  : encoder 단방향 LSTM hidden size; bottleneck = 2*hidden_dim
        num_layers  : encoder LSTM 레이어 수
        num_classes : classification head 출력 클래스 수
        det_hidden  : det_head MLP hidden size
        cls_hidden  : cls_head MLP hidden size
        dropout     : det_head / cls_head dropout rate
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        det_hidden: int = 32,
        cls_hidden: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        bottleneck    = 2 * hidden_dim
        self.encoder  = LSTMEncoder(n_features, hidden_dim, num_layers)
        self.det_head = LSTMDetectionHead(bottleneck, det_hidden, dropout)
        self.cls_head = LSTMClassificationHead(bottleneck, num_classes, cls_hidden, dropout)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x : (B, T, C)

        Returns:
            det_logit : (B,)              binary detection logit
            cls_logit : (B, num_classes)  classification logits (전체 배치)
        """
        _, z      = self.encoder(x)
        det_logit = self.det_head(z)
        cls_logit = self.cls_head(z)
        return det_logit, cls_logit
