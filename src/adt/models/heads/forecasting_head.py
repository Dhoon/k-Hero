"""SSL pretraining용 Forecasting head.

ForecastingHead:
  encoder 출력의 마지막 타임스텝 h_L (B, d_model) →
  MLP → (B, forecast_horizon * n_features) → reshape (B, forecast_horizon, n_features)

pretrain_methodology.md §2: masked reconstruction(주)과 함께 joint 학습하는 보조 objective.
downstream에서는 사용하지 않으며, pretrain 완료 후 이 head는 버린다.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ForecastingHead(nn.Module):
    """encoder 마지막 타임스텝 표현 → 미래 h step 예측.

    Args:
        d_model         : encoder 출력 차원
        forecast_horizon: 예측할 미래 스텝 수 h
        n_features      : 예측할 채널 수 (입력과 동일, 기본 4)
        hidden_dim      : MLP 중간 차원 (None이면 d_model 사용)
    """

    def __init__(
        self,
        d_model: int = 128,
        forecast_horizon: int = 8,
        n_features: int = 4,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.n_features = n_features
        hid = hidden_dim if hidden_dim is not None else d_model

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Linear(hid, forecast_horizon * n_features),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, T, d_model) — encoder 출력 (전체 시퀀스)

        Returns:
            (B, forecast_horizon, n_features)
        """
        h_L = z[:, -1, :]                                    # (B, d_model) — 마지막 타임스텝
        out = self.mlp(h_L)                                   # (B, forecast_horizon * n_features)
        return out.view(-1, self.forecast_horizon, self.n_features)
