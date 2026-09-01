"""Transformer 인코더 백본.

pretrain / finetune / inference 단계에서 공유되는 핵심 모듈.
pretrain 단계에서 학습한 가중치를 finetune 단계에서 load_state_dict로 그대로 불러온다.

forward(x, time_feat, mask=None)
  x         : (B, T, C)          raw 전력값 (정규화된 float32)
  time_feat : (B, T, 2)          [hour_of_day, day_of_week] float32
  mask      : (B, T) or (B, T, C) bool  — True 위치를 [MASK] 벡터로 치환
              None이면 마스킹 없이 통과 (target task / inference)
  returns   : (B, T, d_model)    토큰별 컨텍스트 표현
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.adt.models.positional_encoding import TemporalEncoding


class TimeSeriesTransformerEncoder(nn.Module):
    """LP 전력 시계열 Transformer 인코더.

    Args:
        n_features: 입력 채널 수 (수전 4채널 기본)
        d_model   : Transformer 임베딩 차원
        n_heads   : Multi-head attention 헤드 수
        n_layers  : TransformerEncoder 레이어 수
        d_ff      : FeedForward 내부 차원
        dropout   : dropout 확률
        max_len   : TemporalEncoding 최대 시퀀스 길이
    """

    def __init__(
        self,
        n_features: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
    ) -> None:
        super().__init__()

        # 학습 가능한 [MASK] 벡터 — raw 채널 공간 (n_features,)
        # Linear 투영 전에 적용: 0으로 채우면 심야 실제 낮은 값과 구분 불가
        self.mask_token = nn.Parameter(torch.zeros(n_features))

        # 채널 → d_model 투영 (timestep-as-token: 매 타임스텝을 토큰 하나로)
        self.input_proj = nn.Linear(n_features, d_model)

        # Sinusoidal PE + hour/dow learnable embedding
        self.temporal_encoding = TemporalEncoding(d_model, max_len=max_len, dropout=dropout)

        # Transformer 인코더 (Pre-LN for training stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LayerNorm: Post-LN보다 깊은 네트워크에서 안정적
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,  # mask 없을 때도 동작 일관성 유지
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        time_feat: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x        : (B, T, C)              정규화된 전력값
            time_feat: (B, T, 2)              [hour_of_day, day_of_week] float32
            mask     : (B, T) bool            — True 타임스텝 전체를 [MASK]로 치환
                       (B, T, C) bool         — True 채널별로 [MASK] 해당 값만 치환
                       None                   — 마스킹 없음

        Returns:
            (B, T, d_model)
        """
        # ---- 1. [MASK] 벡터 치환 (Linear 투영 전, raw 채널 단계) ----
        if mask is not None:
            x = x.clone()
            mask_token = self.mask_token.to(x.dtype)

            if mask.dim() == 2:
                # (B, T) → True 타임스텝 전체 채널을 mask_token으로 치환
                # mask_token: (C,) → (1, 1, C) broadcast
                expanded = mask.unsqueeze(-1).expand_as(x)           # (B, T, C)
                x = torch.where(expanded, mask_token.expand_as(x), x)

            elif mask.dim() == 3:
                # (B, T, C) → True인 채널 개별 치환
                x = torch.where(mask, mask_token.expand_as(x), x)

            else:
                raise ValueError(f"mask는 2D(B,T) 또는 3D(B,T,C)여야 합니다. 받은 shape: {mask.shape}")

        # ---- 2. 채널 → d_model 투영 ----
        x = self.input_proj(x)   # (B, T, d_model)

        # ---- 3. Positional + Temporal Encoding ----
        x = self.temporal_encoding(x, time_feat)   # (B, T, d_model)

        # ---- 4. Transformer Encoder ----
        x = self.transformer_encoder(x)            # (B, T, d_model)

        return x

    # ------------------------------------------------------------------
    # 체크포인트 유틸
    # ------------------------------------------------------------------

    def load_pretrained(self, ckpt_path: str, strict: bool = True) -> None:
        """pretrain 단계에서 저장한 encoder 가중치를 로드."""
        state = torch.load(ckpt_path, map_location="cpu")
        # 체크포인트가 {'encoder': state_dict, ...} 형태일 수도 있음
        if "encoder" in state:
            state = state["encoder"]
        self.load_state_dict(state, strict=strict)
