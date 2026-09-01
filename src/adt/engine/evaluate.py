"""평가 루프.

실측 고장 레이블이 없으므로, data/processed/synthetic_eval/ 의 합성 이상치 주입 결과
(scripts/inject_synthetic_anomalies.py 로 생성, configs/eval/synthetic.yaml 참고)를
정답 레이블처럼 사용해 AUROC/AUPRC/F1/Precision@K 를 계산한다.

TODO:
- run(cfg, ckpt): data/processed/synthetic_eval/ 로드 -> 이상 점수 계산 -> 지표 산출
- outputs/figures 에 재구성 오차 & 주입된 이상 구간 vs 예측 이상 구간 시각화 저장
- outputs/scores 에 결과 csv 저장
- (참고) 합성 주입은 실제 이상치의 완전한 대체가 아니므로, threshold를 이 결과에만
  과최적화하지 않도록 주의 (README '검증 방법' 절 참고)
"""
