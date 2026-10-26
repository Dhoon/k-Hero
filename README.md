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
| `reference/` | PyTorch 모델 정의 (검증·참조용) |
| `c/` | 순수 C 추론 코드 (외부 의존성 없음, `model_weights.h`에 weight 내장) |
| `tools/` | `export_c.py` (weight → C 헤더), `verify_c.py` (C ↔ PyTorch 일치 검증) — PC 전용 |
| `examples/` | 정상 / 공격(scale_down) 샘플 입력 CSV |

## 보드에서 실행 (Python 불필요)

보드(AM43xx, armv7l)에 gcc·make가 있으므로 소스를 복사해서 보드에서 빌드한다.

```bash
# PC → 보드
scp -r c examples khero:/home/root/lstm_joint/

# 보드에서
cd /home/root/lstm_joint/c
make ARM=1                                   # 약 20초
./lstm_joint ../examples/sample_attack.csv
# end_row  p_attack  is_attack  type           score_scale_down  score_ramp  score_pulse_plateau  score_replay  score_instant_spike
#      95  0.999987          1  scale_down             3.591340    3.415424            -0.206889      0.602079            -7.296520
```

| 컬럼 | 의미 |
|---|---|
| `end_row` | 판정한 window(96행) 마지막 행 번호 (0부터, 헤더 제외) |
| `p_attack` | 공격 확률. `threshold.json` 값(0.391)보다 크면 공격 |
| `is_attack` | 1 = 공격, 0 = 정상 |
| `type` | 공격 유형 (정상이면 `normal`) |
| `score_*` | 유형별 점수 (logit). 공격일 때 가장 큰 값이 `type`. 정상이면 무시 |

| 옵션 | 설명 |
|---|---|
| (파일 생략) | stdin 에서 읽음 |
| `--cumulative` | 입력이 적산값이면 diff(음수→0)로 구간 사용량 변환 (97행 이상 필요) |
| `--all` | 모든 sliding window(stride 1) 결과 출력. 기본은 마지막 window 하나 |
| `--bench N` | 추론 N회 평균 시간 (stderr) |
| `--no-header` | 헤더 줄 생략 (다른 프로그램에서 파싱할 때) |

입력 CSV: 한 줄 = 한 시점(15분 간격, 오래된 것 먼저), 각 줄의 **마지막 4개 필드**를 수전 4채널로 사용
(앞에 시간 컬럼이 있어도 됨). 숫자가 아닌 줄(헤더)은 무시. 리샘플링·결측 보간은 하지 않으므로
입력은 15분 간격으로 빠짐없이 들어와야 한다.

성능 (보드, `make ARM=1`): window 당 약 140ms, 실행 파일 0.6MB, 추가 메모리 100KB 미만.

C 라이브러리로 쓸 때는 `lstm_joint.h` 의 `lj_infer(x[96][4], &result)` 하나만 호출하면 된다
(입력은 정규화 전 값, 내부에서 scaler 적용).

## weight 갱신 시 (PC)

```bash
# model/ 의 파일을 새 checkpoint 기준으로 교체한 뒤
python tools/export_c.py                          # c/model_weights.h 재생성
gcc -O2 -std=gnu99 -o c/lstm_joint.exe c/main.c c/lstm_joint.c -lm
python tools/verify_c.py --npy <main>/data/processed/downstream/all_type/test/X.npy -n 300
# → PASS 확인 후 보드에 c/ 다시 복사
```

검증 결과 (현재 weight, test window 300개): p_attack 최대 오차 6e-6, 공격 판정·유형 argmax 불일치 0.

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
