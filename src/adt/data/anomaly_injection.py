"""레이블이 없는 데이터를 검증하기 위한 합성 이상치 주입 모듈.

고장/이상 레이블이 전혀 없는 데이터라 모델을 정량적으로 검증할 방법이 없다.
정상 위주 구간(보통 test split)에 아래와 같은 패턴을 인위적으로 주입하고,
주입한 위치를 정답 레이블로 삼아 auroc/auprc/f1/precision_at_k 등을 계산한다.
실제 이상치와 100% 같지는 않지만, "이 정도 편차는 최소한 잡아내는지"를
검증하는 용도로 널리 쓰이는 방법이다 (configs/eval/synthetic.yaml 참고).

TODO — 아래 함수들은 모두 (윈도우 텐서, 채널) 단위로 원본을 훼손하지 않고
        corrupted copy + binary label mask(시점 단위, 1=주입된 이상)를 반환해야 함:

- inject_spike(window, channel, magnitude_std, duration_steps)
    특정 채널의 값을 짧은 구간 동안 크게 튀게 만듦 (정상 구간 std 기준 배수)
- inject_drop(window, channel, magnitude_std, duration_steps)
    spike의 반대 방향 (급격한 하락)
- inject_flatline(window, channel, duration_steps)
    값을 특정 구간 동안 고정 (계량기 통신 오류/고착 흉내)
- inject_drift(window, channel, magnitude_std, duration_steps)
    완만한 선형/지수 drift를 더함 (센서 캘리브레이션 이슈 흉내)
- inject_trend_break(window, channel, duration_steps)
    다른 시간대(예: 다른 요일)의 구간을 붙여넣어 주기 패턴을 깨뜨림

- inject_synthetic_anomalies(dataset, cfg) -> (corrupted_dataset, labels)
    configs/eval/synthetic.yaml 의 anomaly_types/prob/injection_ratio 를 따라
    위 함수들을 랜덤하게 조합 적용, seed 고정으로 재현 가능해야 함
    출력은 configs/eval/synthetic.yaml 의 output_dir 에 저장
"""
