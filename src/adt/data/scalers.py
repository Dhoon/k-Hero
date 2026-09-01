"""채널별(feature-wise) 정규화.

TODO:
- StandardScalerND: (N, T, C) 텐서에서 채널(C) 축 기준으로 fit/transform
- 저장/로드 (finetune, inference 단계에서 동일 scaler 재사용 필수)
"""
