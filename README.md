# campus-power-ad — deploy

LSTM Joint 모델(이상 탐지 + 공격 유형 분류)의 배포용 weight와 메타데이터만 담은 브랜치.
학습·평가 코드는 `main` 브랜치에 있다.

- 출처: `main` 의 `checkpoints/lstm/joint/all_type/best.pt` (epoch 21)
- 성능 (test_50_50): Detection F1 0.825 / AUC-ROC 0.895, Classification macro-F1 0.668
- 파라미터 143,622개, 전체 약 0.6MB

## 파일

| 파일 | 내용 |
|---|---|
| `model/model.pt` | `LSTMJoint.state_dict()` (float32, optimizer 제외) |
| `model/config.json` | 입력 형식, 모델 하이퍼파라미터, 텐서별 shape |
| `model/scaler.json` | 채널별 mean / std (train split으로 fit) |
| `model/threshold.json` | detection threshold (sigmoid 확률 기준) |
| `model/class_names.json` | classification 출력 index → 공격 유형 |
| `reference/` | PyTorch 모델 정의 (C 구현 시 참조용) |

## 추론 절차

입력은 계량기 LP 데이터의 수전 4채널(`유효`, `지상무효`, `진상무효`, `피상` 전력량, 이 순서).

1. **적산값 → 구간 사용량**: 채널별 1-step diff, 음수(계량기 리셋)는 0으로 clip
2. **리샘플링**: 15분 간격, 결측은 선형 보간
3. **정규화**: `x = (x - mean[c]) / std[c]` (`scaler.json`)
4. **윈도우**: 최근 96 step (24시간) → 입력 `(T=96, C=4)`
5. **Encoder**: 2-layer BiLSTM (hidden 64). 마지막 레이어의 forward 최종 hidden과 backward 최종 hidden을 concat → `z (128,)`
6. **Detection**: `Linear(128,32) → ReLU → Linear(32,1)` → sigmoid → `p > threshold` 이면 공격
7. **Classification**: `Linear(128,32) → ReLU → Linear(32,5)` → argmax → `class_names.json` (공격 판정 시에만 의미 있음)

dropout은 추론 시 사용하지 않는다.

## LSTM weight 레이아웃 (PyTorch 규약)

`encoder.lstm.weight_ih_l{k}[_reverse]` 는 `(4*H, in)`, 게이트 순서는 **i, f, g, o**.
bias는 `bias_ih + bias_hh` 를 합쳐서 쓰면 된다.

```
i = σ(W_ii x + b_ii + W_hi h + b_hi)
f = σ(W_if x + b_if + W_hf h + b_hf)
g = tanh(W_ig x + b_ig + W_hg h + b_hg)
o = σ(W_io x + b_io + W_ho h + b_ho)
c' = f*c + i*g,   h' = o * tanh(c')
```

layer 1의 입력은 layer 0의 fwd/bwd 출력을 timestep마다 concat한 `(T, 128)`.
backward 방향은 시퀀스를 t=T-1 → 0 순서로 처리하고, 최종 hidden은 t=0 처리 후의 값.
