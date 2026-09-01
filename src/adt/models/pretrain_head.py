"""Self-supervised pretraining용 head.

TODO:
- MaskedReconstructionHead: encoder 출력 -> 원본 window 값 재구성 (Linear projection back to n_features)
- (선택) ContrastiveProjectionHead: SimCLR류 대조학습을 쓸 경우의 projection head
"""
