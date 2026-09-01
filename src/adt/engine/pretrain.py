"""Self-supervised pretraining 학습 루프.

TODO:
- run(cfg): DataLoader(레이블 없는 정상 위주 window) -> encoder + pretrain_head
  -> masking 적용 -> masked reconstruction loss -> optimizer step
- 매 epoch checkpoints/pretrain 에 저장 (best.pt = val loss 최저)
- utils.logger 로 tensorboard 로깅
"""
