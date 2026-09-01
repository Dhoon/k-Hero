"""시계열 위치 인코딩 + 시간 특징 임베딩.

설계 선택: Sinusoidal PE (fixed, not learned)
이유:
  - window_size=96 고정이라 learnable PE도 무방하지만,
    sinusoidal은 파라미터 없이 위치 순서 inductive bias를 제공하고
    window_size가 달라져도 재학습 없이 동작한다.
  - 시간 정보(몇 시, 무슨 요일)는 이미 별도 learnable Embedding으로 추가하므로
    PE 자체를 learnable로 만들면 역할이 중복된다.
  - BERT/ViT 등은 learnable PE를 쓰지만, 우리 시퀀스 길이(96)는 짧고
    주기성이 강한 시계열이라 sinusoidal의 정현파 구조가 잘 맞는다.

TemporalEncoding:
  input:  x         (B, T, d_model) — Linear 투영 후 토큰
          time_feat (B, T, 2)       — [hour_of_day(0-23), day_of_week(0-6)], float32
  output: (B, T, d_model)           — PE + hour_emb + dow_emb 더한 결과
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _build_sinusoidal_table(max_len: int, d_model: int) -> torch.Tensor:
    """Vaswani et al. 2017 sinusoidal PE 테이블 생성."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)   # (max_len, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (max_len, d_model)


class TemporalEncoding(nn.Module):
    """Sinusoidal positional encoding + learnable hour/day-of-week embedding.

    Args:
        d_model  : 토큰 임베딩 차원 (= encoder d_model)
        max_len  : 최대 시퀀스 길이 (window_size 이상이면 됨, 기본 512)
        dropout  : encoding 후 dropout 확률
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()

        # sinusoidal PE — buffer로 등록 (파라미터 아님, 학습 안 됨)
        pe = _build_sinusoidal_table(max_len, d_model)  # (max_len, d_model)
        self.register_buffer("pe", pe)

        # 절대 시간 임베딩 (learnable)
        self.hour_embedding = nn.Embedding(24, d_model)   # hour_of_day: 0-23
        self.dow_embedding = nn.Embedding(7, d_model)     # day_of_week: 0-6 (월=0)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, time_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : (B, T, d_model)
            time_feat: (B, T, 2)  float32  — col0=hour, col1=dow

        Returns:
            (B, T, d_model)
        """
        T = x.size(1)

        # sinusoidal PE: (T, d_model) → broadcast to (1, T, d_model)
        pos_enc = self.pe[:T].unsqueeze(0)  # (1, T, d_model)

        # time feature: float32 → long for embedding lookup
        tf = time_feat.long()               # (B, T, 2)
        hour_enc = self.hour_embedding(tf[..., 0])  # (B, T, d_model)
        dow_enc = self.dow_embedding(tf[..., 1])    # (B, T, d_model)

        return self.dropout(x + pos_enc + hour_enc + dow_enc)
