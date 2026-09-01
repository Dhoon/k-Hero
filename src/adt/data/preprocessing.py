"""결측치 처리, 리샘플링, 정렬 등 전처리.

TODO:
- clean(df, cfg) -> pd.DataFrame        : datetime 파싱, 정렬, 중복 제거
- resample(df, freq, method)             : configs/data/default.yaml 의 resample_freq 적용
- fit_scaler / apply_scaler               : 채널별 StandardScaler 등, train만으로 fit
"""
