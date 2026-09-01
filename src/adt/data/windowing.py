"""Sliding window로 (N, T, C) 텐서를 만드는 모듈.

Transformer 입력 형태: (batch, window_size, n_features)

TODO:
- make_windows(df, window_size, stride) -> np.ndarray
- train/val/test로 시간 순서를 지켜서 분할 (shuffle 금지 - 시계열 leakage 방지)
"""
