"""Transformer 인코더 백본. pretrain / finetune 단계에서 공유되는 핵심 모듈.

TODO:
- TimeSeriesTransformerEncoder(nn.Module)
    입력: (batch, window_size, n_features) -> 채널 임베딩(Linear) -> +positional encoding
    -> nn.TransformerEncoder(d_model, n_heads, n_layers, d_ff, dropout)
    출력: (batch, window_size, d_model) 토큰별 표현
- pretrain 단계에서 학습된 가중치를 finetune 단계에서 그대로 load_state_dict 하는 것을 전제로 설계
"""
