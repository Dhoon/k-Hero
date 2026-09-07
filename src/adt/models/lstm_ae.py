"""LSTM Autoencoder — 경량 비교 모델.

Architecture:
  Encoder : 2-layer BiLSTM → bottleneck z (2*H,)
  Decoder : z repeat → 1-layer LSTM → Linear projection → (T, C)

Downstream heads (z 직접 입력, 시간축 pooling 불필요):
  LSTMDetectionHead      : z → (1,)  raw logit
  LSTMClassificationHead : z → (num_classes,) raw logits
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """2-layer Bidirectional LSTM encoder.

    Args:
        n_features : 입력 채널 수 (C)
        hidden_dim : 단방향 LSTM hidden size; 출력 dim = 2*hidden_dim
        num_layers : LSTM 레이어 수 (기본 2)
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    @property
    def output_dim(self) -> int:
        return 2 * self.hidden_dim

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x : (B, T, C)

        Returns:
            all_hidden : (B, T, 2*H)  전체 timestep hidden states
            z          : (B, 2*H)     bottleneck — 마지막 레이어 fwd+bwd concat
        """
        all_hidden, (h_n, _) = self.lstm(x)
        # h_n: (num_layers*2, B, H)  마지막 레이어: [-2]=forward, [-1]=backward
        z = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, 2*H)
        return all_hidden, z


class LSTMDecoder(nn.Module):
    """z를 매 timestep 반복 입력하는 단방향 LSTM 디코더.

    Args:
        bottleneck_dim : 2 * encoder_hidden_dim
        n_features     : 출력 채널 수 (원본 x와 동일)
        num_layers     : LSTM 레이어 수 (기본 1)
    """

    def __init__(
        self,
        bottleneck_dim: int,
        n_features: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=bottleneck_dim,
            hidden_size=bottleneck_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.proj = nn.Linear(bottleneck_dim, n_features)

    def forward(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Args:
            z       : (B, bottleneck_dim)
            seq_len : 복원할 timestep 수 (원본 T)

        Returns:
            x_recon : (B, T, n_features)
        """
        inp = z.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T, 2*H)
        out, _ = self.lstm(inp)                        # (B, T, 2*H)
        return self.proj(out)                          # (B, T, C)


class LSTMAutoencoder(nn.Module):
    """Encoder + Decoder wrapper.  forward(x) -> x_reconstructed."""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = LSTMEncoder(n_features, hidden_dim, num_layers)
        self.decoder = LSTMDecoder(
            bottleneck_dim=self.encoder.output_dim,
            n_features=n_features,
            num_layers=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) → (B, T, C)"""
        _, z = self.encoder(x)
        return self.decoder(z, x.size(1))


# ── Downstream heads ──────────────────────────────────────────────────────────

class LSTMDetectionHead(nn.Module):
    """bottleneck z (B, input_dim) → (B,) raw logit."""

    def __init__(
        self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)  # (B,)


class LSTMClassificationHead(nn.Module):
    """bottleneck z (B, input_dim) → (B, num_classes) raw logits."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)  # (B, num_classes)
