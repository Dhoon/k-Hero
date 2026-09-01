"""학습된 모델로 새 LP 데이터에 대한 이상 점수를 산출.

TODO:
- load_model(ckpt_path) -> (encoder, anomaly_head)
- predict(model, raw_df, data_cfg) -> pd.DataFrame(시간, 이상점수, 이상여부)
- outputs/scores 에 저장, outputs/figures 에 타임라인 시각화
"""
