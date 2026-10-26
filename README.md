# 전력량계 데이터 공격 탐지 모델 — 에지 단말 배포판

전력량계 LP 데이터(15분 단위) 24시간 구간을 넣으면 **위·변조 공격 여부**와 **공격 유형**을 알려 주는 AI 모델입니다.
Python 없이 C로만 동작하도록 만들어서, 에지 단말(ARM 보드)에서 바로 빌드하고 실행할 수 있습니다.

> 모델 학습·평가 코드는 이 저장소의 `main` 브랜치에 있습니다. 이 `deploy` 브랜치는 **실행에 필요한 것만** 담았습니다.

| 항목 | 값 |
|---|---|
| 판정 단위 | 15분 간격 96개 시점 (= 24시간) |
| 입력 | 수전 4채널 (유효 / 지상무효 / 진상무효 / 피상 전력량) |
| 출력 | 공격 확률, 공격 여부(0/1), 공격 유형 5종 중 하나 |
| 성능 | 탐지 F1 0.82, AUC-ROC 0.90 / 유형 분류 macro-F1 0.67 |
| 크기 | 실행 파일 약 0.6MB, 실행 중 추가 메모리 100KB 미만 |
| 속도 | 1회 판정 약 0.14초 (ARM Cortex-A9 보드 기준) |

---

## 1. 빠른 시작 (5분)

보드에 gcc와 make만 있으면 됩니다. Python, 외부 라이브러리는 필요 없습니다.

**① PC에서 보드로 파일 복사**

```bash
# 이 폴더(deploy 브랜치)에서 실행
ssh -p 52022 root@<보드IP> "mkdir -p /home/root/lstm_joint"
scp -P 52022 -r c examples root@<보드IP>:/home/root/lstm_joint/
```

> 보드의 SSH가 오래된 방식만 지원하면 `ssh`/`scp` 에 `-o HostKeyAlgorithms=+ssh-rsa` 를 붙이세요.

**② 보드에서 빌드** (약 20초)

```bash
cd /home/root/lstm_joint/c
make ARM=1
```

`lstm_joint` 실행 파일이 생기면 성공입니다.

**③ 샘플로 실행해 보기**

```bash
./lstm_joint ../examples/sample_normal.csv
./lstm_joint ../examples/sample_attack.csv
```

아래처럼 나오면 정상 동작입니다.

```
end_row  p_attack  is_attack  type           score_scale_down  score_ramp  score_pulse_plateau  score_replay  score_instant_spike
     95  0.064271          0  normal                 0.515684    1.049638            -0.438651      0.091517            -1.281374

end_row  p_attack  is_attack  type           score_scale_down  score_ramp  score_pulse_plateau  score_replay  score_instant_spike
     95  0.999987          1  scale_down             3.591340    3.415424            -0.206889      0.602079            -7.296520
```

정상 샘플은 공격 확률 6% → **정상**, 공격 샘플은 99.99% → **공격(scale_down)** 으로 판정한 결과입니다.

---

## 2. 결과 읽는 법

**판정에 필요한 건 앞의 4칸뿐입니다.** 뒤의 `score_*` 는 참고용입니다.

| 컬럼 | 의미 | 예시 |
|---|---|---|
| `end_row` | 이 판정에 쓴 24시간 구간의 **마지막 행 번호** (0부터, 헤더 제외) | `95` → 0~95행을 보고 판정 |
| `p_attack` | 공격일 확률 (0~1) | `0.064` = 6.4% |
| `is_attack` | **최종 판정.** `p_attack` 이 0.391 보다 크면 `1`(공격), 아니면 `0`(정상) | `0` |
| `type` | 공격 유형. 정상이면 `normal` | `scale_down` |
| `score_*` | 유형별 점수. 공격일 때 가장 큰 점수가 `type` 이 됨. **정상일 때는 무시** | |

공격 유형 5종:

| 유형 | 어떤 공격인가 |
|---|---|
| `scale_down` | 일정 시간 동안 값을 일정 비율로 줄여서 보냄 (사용량 축소) |
| `ramp` | 값을 서서히 늘리거나 줄여서 정상에서 점점 벗어나게 함 |
| `pulse_plateau` | 일정 시간 동안 값을 정상보다 높게 유지 |
| `replay` | 과거의 정상 구간을 복사해서 다시 보냄 |
| `instant_spike` | 1~2 시점만 순간적으로 크게 튐 |

> 기준값 0.391 은 검증 데이터에서 F1 이 가장 높게 나오는 값으로 정했습니다 (`model/threshold.json`).

---

## 3. 내 데이터로 실행하기

### 입력 파일 형식 (CSV)

- **한 줄 = 한 시점**, 15분 간격, **오래된 시점이 위**
- 각 줄의 **마지막 4칸**을 `유효, 지상무효, 진상무효, 피상` 순서로 읽습니다
- 앞에 시간 컬럼이 있어도 되고, 헤더 줄처럼 숫자가 아닌 줄은 알아서 건너뜁니다
- 구분자는 쉼표 `,` · 탭 · 세미콜론 `;` 모두 가능
- **최소 96줄** 필요 (24시간)

```csv
일자시간,유효,지상무효,진상무효,피상
2025-03-01 00:15,0.0412,0.0058,0.0131,0.0566
2025-03-01 00:30,0.0398,0.0061,0.0127,0.0551
...
```

### 실행 예시

```bash
./lstm_joint data.csv                    # 가장 최근 24시간 구간 한 번 판정
./lstm_joint --all data.csv              # 15분씩 밀면서 모든 구간 판정
./lstm_joint --cumulative data.csv       # 입력이 누적(적산)값일 때
./lstm_joint --all data.csv > result.txt # 결과를 파일로 저장
cat data.csv | ./lstm_joint              # 파일 대신 표준입력으로
```

| 옵션 | 설명 |
|---|---|
| `--all` | 모든 24시간 구간을 15분씩 밀면서 판정. 1000줄이면 905개 결과 |
| `--cumulative` | 전력량계 원본처럼 **누적값**이 들어올 때. 직전 값과의 차이(구간 사용량)로 바꿔서 판정. 97줄 이상 필요 |
| `--no-header` | 맨 위 컬럼 이름 줄을 빼고 출력 (다른 프로그램이 결과를 읽을 때) |
| `--bench N` | 판정을 N번 반복해서 평균 시간 측정 |

### ⚠️ 입력 데이터 주의사항

모델은 학습 때와 **같은 형태의 데이터**를 받아야 정확합니다.

1. **15분 간격이 빠짐없이** 들어와야 합니다. 중간 시점이 빠지면 정상 데이터도 공격으로 오탐할 수 있습니다.
   이 프로그램은 빠진 시점을 채워 주지 않으니, 결측이 있으면 앞뒤 값으로 **선형 보간**해서 넣어 주세요.
2. **단위와 스케일**이 학습 데이터(수전 LP 전력량)와 같아야 합니다.
3. 값이 **누적값**인지 **구간 사용량**인지 확인하고, 누적값이면 `--cumulative` 를 붙이세요.

---

## 4. 다른 프로그램에 붙이기 (C API)

SMGW 운영 S/W 등에 직접 넣을 때는 `c/lstm_joint.c`, `c/lstm_joint.h`, `c/model_weights.h` 세 파일만 가져가면 됩니다.

```c
#include "lstm_joint.h"

float window[LJ_T][LJ_C];   /* [96][4], 오래된 시점이 window[0] */
/* ... 최근 24시간 구간 사용량(정규화 전 원래 값)을 채움 ... */

lj_result r;
lj_infer(window, &r);

if (r.is_attack)
    printf("공격 탐지: %s (확률 %.3f)\n", LJ_CLASS_NAMES[r.cls], r.p_attack);
```

- 정규화는 `lj_infer` 안에서 처리합니다. **원래 값 그대로** 넣으면 됩니다.
- 스레드 안전하지 않습니다 (내부 static 버퍼 사용). 여러 스레드에서 쓰면 호출을 직렬화하세요.
- 빌드 시 `-lm` (수학 라이브러리) 링크가 필요합니다.

---

## 5. 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `need >= 96 rows (got N)` | 데이터가 24시간보다 짧음. 96줄(`--cumulative` 면 97줄) 이상 넣기 |
| 결과가 아무것도 안 나옴 | 각 줄의 마지막 4칸이 숫자인지 확인. 공백으로만 구분된 파일은 지원 안 함 |
| `make ARM=1` 에서 컴파일러 옵션 에러 | 보드가 Cortex-A9 이 아니면 `make` (옵션 없이) 로 빌드 |
| `Permission denied` 로 실행 안 됨 | `chmod +x lstm_joint` |
| 정상 데이터인데 공격이 많이 나옴 | 결측 시점, 단위 차이, 누적값 여부를 확인 (3장 주의사항) |

---

## 6. 모델 갱신 방법 (PC 작업)

모델을 다시 학습해서 **weight만 바뀐 경우 C 코드는 수정할 필요가 없습니다.** 헤더 파일 하나만 다시 만들면 됩니다.

```bash
# 1) model/ 폴더의 파일을 새 학습 결과로 교체
#    model.pt (state_dict), scaler.json, threshold.json, class_names.json

# 2) weight → C 헤더 변환  (PC, torch 필요)
python tools/export_c.py

# 3) PC에서 빌드 후 PyTorch 와 결과가 같은지 검증 → PASS 확인
gcc -O2 -std=gnu99 -o c/lstm_joint.exe c/main.c c/lstm_joint.c -lm
python tools/verify_c.py --npy <main 브랜치>/data/processed/downstream/all_type/test/X.npy -n 300

# 4) 보드에 c/model_weights.h 를 다시 복사하고 보드에서 make ARM=1
```

| 바뀐 것 | C 코드 수정 |
|---|---|
| weight (재학습) | 불필요 |
| hidden 크기 · 레이어 수 · 클래스 수 | 불필요 (헤더 재생성으로 자동 반영, 단 head hidden ≤ 64) |
| 모델 종류 자체 (예: Transformer) | **필요** |

현재 weight 검증 결과 (test 구간 300개): 확률 최대 오차 0.000006, 공격 판정·유형 불일치 **0건**.

---

## 7. 폴더 구성

```
├── c/                    ← 보드에 올리는 것 (이것만 있으면 실행 가능)
│   ├── main.c            CSV 읽기 + 결과 출력 (명령줄 프로그램)
│   ├── lstm_joint.c/.h   추론 엔진
│   ├── model_weights.h   weight 내장 헤더 (자동 생성, 직접 수정 금지)
│   └── Makefile
├── examples/             정상 / 공격 샘플 CSV
├── model/                원본 weight 와 설정값 (PC 에서 헤더 만들 때 사용)
│   ├── model.pt          PyTorch weight
│   ├── config.json       입력 형식, 모델 크기
│   ├── scaler.json       채널별 정규화 값 (평균 / 표준편차)
│   ├── threshold.json    공격 판정 기준값
│   └── class_names.json  유형 번호 → 이름
├── tools/                PC 전용 스크립트 (Python + torch)
│   ├── export_c.py       model/ → c/model_weights.h
│   └── verify_c.py       C 결과 = PyTorch 결과 인지 검증
└── reference/            PyTorch 모델 정의 (검증 · 참고용)
```

---

## 부록: 모델 내부 동작

C 코드를 수정하거나 다른 언어로 옮길 때 참고하세요.

**처리 순서**

1. 입력 `(96, 4)` 를 채널별로 정규화: `x = (x - mean[c]) / std[c]` (`scaler.json`)
2. **Encoder**: 2층 양방향 LSTM (hidden 64). 마지막 층의 정방향 최종 hidden(t=95)과 역방향 최종 hidden(t=0)을 이어 붙여 `z (128)`
3. **탐지 head**: `Linear(128→32) → ReLU → Linear(32→1) → sigmoid` → `p_attack`
4. **분류 head**: `Linear(128→32) → ReLU → Linear(32→5)` → 가장 큰 값이 유형

학습 때 쓰인 전처리 (원본 → 입력): 누적값 차분(음수는 0) → 15분 리샘플링 · 선형 보간 → 정규화 → 96칸 윈도우.
이 중 **차분(`--cumulative`)과 정규화만 C 프로그램이 처리**하고, 리샘플링 · 보간은 입력 전에 맞춰야 합니다.

**LSTM 수식 (PyTorch 규약)** — weight 는 `(4H, 입력)` 모양이고 게이트 순서는 **i, f, g, o**. bias 는 `bias_ih + bias_hh` 를 합쳐서 사용.

```
i = σ(W_ii x + b_ii + W_hi h + b_hi)
f = σ(W_if x + b_if + W_hf h + b_hf)
g = tanh(W_ig x + b_ig + W_hg h + b_hg)
o = σ(W_io x + b_io + W_ho h + b_ho)
c' = f*c + i*g,   h' = o * tanh(c')
```

2층 LSTM 의 입력은 1층의 정방향/역방향 출력을 시점마다 이어 붙인 `(96, 128)` 입니다.
역방향은 t=95 → 0 순서로 처리합니다.

**출처**: `main` 브랜치 `checkpoints/lstm/joint/all_type/best.pt` (epoch 21), 파라미터 143,622개.
