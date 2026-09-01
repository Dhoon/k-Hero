"""이상탐지 파인튜닝 루프.

TODO:
- run(cfg): checkpoints/pretrain 의 encoder 가중치 로드 -> anomaly_head 부착
  -> (freeze_encoder 옵션에 따라) 전체 또는 head만 학습
- 재구성 오차 기반이면 정상 데이터의 오차 분포로 threshold 산출 (configs/finetune 의 head.threshold_method)
- checkpoints/finetune 에 저장
"""
