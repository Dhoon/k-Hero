# Campus Power Meter Attack Detection (SSL Transformer)

네트워크로 측정값을 전송하는 교내 전력계(power meter)의 시계열 데이터를 대상으로, 사이버 공격에 의해 변조된 측정값을 탐지하는 프로젝트입니다.

Self-Supervised Learning(SSL)으로 정상 전력 시계열의 패턴을 먼저 학습한 뒤, 이 pretrained Transformer 인코더를 **완전히 freeze한 채**(linear probing) 두 개의 독립된 downstream head를 학습합니다.

1. **Attack Detection** — 공격 여부 판단 (binary: Normal vs Attack)
2. **Attack Classification** — 공격 유형 분류 (multi-class, Normal 제외)

두 head는 파라미터·loss·checkpoint가 완전히 분리된 별도의 모델입니다 (서로의 gradient에 영향을 주지 않음). 다만 encoder가 frozen이라 fold 하나당 encoder는 한 번만 통과시키고, 그 결과로 두 head를 같은 스크립트 안에서 함께 학습합니다 (아래 워크플로우 참고).

## 데이터

`일자/시간, 유효전력량, 지상무효전력량, 진상무효전력량, 피상전력량` 4채널, 15분 단위 시계열입니다. `configs/data/default.yaml`에서 윈도우 길이(L)와 전처리 방식을 정의합니다.

### 공격 유형 (5종)

| 유형 | 설명 |
|---|---|
| Scale Down | 일정 시간 정상값을 일정 비율로 축소 (magnitude만 작아짐, 형태는 유지) |
| Ramp | 일정 시간 값을 서서히 증가/감소 (점진적 이탈) |
| Pulse Plateau | 일정 시간 값을 정상보다 높게 유지 (지속되는 상승) |
| Replay | 과거 실제 정상 구간을 그대로 복사해 재전송 (값 자체는 정상 데이터) |
| Instant Spike | 1~2 timestep만 순간적으로 크게 튐 |

## 모델 구조

```
Power-meter window X_t (L, 4)
        ↓
Pretrained Transformer Encoder
        ↓
        z_t
        ↓
 ┌─────────────────────────────┐
 │                             │
Attack Detection 모델        Attack Classification 모델
(MeanPool+MaxPool→MLP→binary) (MeanPool+MaxPool→MLP→multi-class)
Normal(0) / Attack(1)        공격 유형 (Normal 제외 샘플만 학습)
```

Transformer Encoder는 pretrain 단계에서 먼저 학습되고, downstream에서는 이 encoder를 **freeze**(파라미터 업데이트 없음, linear probing)한 채로 Detection head와 Classification head를 각각 독립적으로 학습합니다. encoder가 fold·head에 관계없이 항상 동일하므로, fold 하나당 encoder forward는 한 번만 수행하고 그 z_t로 두 head를 함께 학습합니다.

## 디렉토리 구조

```
campus-power-ad/
├── configs/
│   ├── data/default.yaml                 # 전처리·윈도잉 설정 (4채널)
│   ├── pretrain/default.yaml             # masked reconstruction + forecasting joint 설정
│   └── downstream/
│       ├── attack_injection.yaml         # 5종 공격 주입 파라미터 + fold별 데이터 생성 설정
│       └── default.yaml                  # Detection+Classification head 설정 1개, --fold CLI 인자로 6개 fold 실행
├── data/
│   ├── raw/                              # 원본 xls (건드리지 않음, git 제외)
│   ├── interim/                          # 중간 산출물 (git 제외)
│   └── processed/
│       ├── scaler.joblib
│       ├── status_events/                # 계기별 상태정보 이벤트 분류 결과
│       ├── pretrain/                     # 레이블 없는 정상 데이터 (forecasting target 포함)
│       │   └── train/ val/ test/
│       └── downstream/                   # fold별 최종 데이터셋 (한 번 생성, 고정 시드, 재사용)
│           ├── all_type/                 #   Normal+5종, train/val/test
│           ├── unseen_scale_down/        #   Normal+4종(Scale Down 제외), train/val만
│           ├── unseen_ramp/
│           ├── unseen_pulse_plateau/
│           ├── unseen_replay/
│           └── unseen_instant_spike/
│               (Detection·Classification 모델 둘 다 이 fold 데이터를 공유.
│                Classification은 그중 Normal 제외 샘플만 사용)
├── src/adt/
│   ├── data/
│   │   ├── loaders.py, preprocessing.py, windowing.py, scalers.py, dataset.py
│   │   ├── attack_injection.py           # 5종 공격 주입 함수 (raw scale에서 동작)
│   │   └── labeling.py                   # fold별 데이터셋(all_type/unseen_*) 생성 오케스트레이션
│   ├── models/
│   │   ├── encoder.py, positional_encoding.py
│   │   └── heads/
│   │       ├── reconstruction_head.py    # pretrain 전용, downstream에서는 제거
│   │       ├── forecasting_head.py       # pretrain 전용, downstream에서는 제거
│   │       ├── detection_head.py         # MeanPool+MaxPool → MLP → binary
│   │       └── classification_head.py    # MeanPool+MaxPool → MLP → multi-class
│   ├── ssl/
│   │   ├── masking.py                    # segment 단위 curriculum masking (15%→40%)
│   │   └── losses.py                     # L_mask + lambda * L_forecast
│   ├── engine/
│   │   ├── pretrain.py
│   │   ├── train_downstream.py           # fold 하나당 encoder 1회 forward → detector+classifier 함께 학습
│   │   └── evaluate_downstream.py        # fold 하나당 encoder 1회 forward → detector+classifier 함께 평가
│   ├── utils/                            # seed, logger, checkpoint, metrics
│   └── inference/detect.py
├── scripts/                              # CLI 진입점 (각 engine 함수 호출)
├── notebooks/                            # 탐색적 데이터분석
├── docs/
│   └── pretrain_methodology.md           # 방법론 문서 (이 프로젝트의 최종 설계 기준)
├── checkpoints/
│   ├── pretrain/
│   └── downstream/
│       ├── all_type/
│       │   ├── detector/                 # Attack Detection 모델
│       │   └── classifier/               # Attack Classification 모델
│       ├── unseen_scale_down/
│       │   ├── detector/
│       │   └── classifier/
│       ├── unseen_ramp/
│       │   ├── detector/
│       │   └── classifier/
│       ├── unseen_pulse_plateau/
│       │   ├── detector/
│       │   └── classifier/
│       ├── unseen_replay/
│       │   ├── detector/
│       │   └── classifier/
│       └── unseen_instant_spike/
│           ├── detector/
│           └── classifier/
├── logs/                                 # checkpoints와 동일 구조
├── outputs/
│   ├── figures/                          # checkpoints와 동일 구조
│   └── scores/                           # checkpoints와 동일 구조
└── tests/
```

## 데이터 관리 정책

- `data/raw/`, `data/interim/`, `data/processed/`의 실제 파일은 **git에 올리지 않습니다** (`.gitignore`에서 제외, 폴더 구조 유지용 `.gitkeep`만 추적).
- 원본 xls는 로컬 또는 팀 공유 드라이브에만 보관하고, 새 컴퓨터에서 clone한 뒤에는 `data/raw/`에 원본 파일을 직접 복사해 넣어서 씁니다.
- git에는 **코드 + 설정(configs/*.yaml) + 문서**만 버전관리합니다. 체크포인트(`checkpoints/**/*.pt`)도 같은 이유로 제외됩니다.

## Pretraining

정상 시계열에 segment 단위 masking(15%→40% curriculum)을 적용한 **Masked Reconstruction**(주 objective)과, window 이후 미래 h step을 예측하는 **Forecasting**(보조 objective, weight λ≈0.1~0.2)을 같은 step에서 함께 학습합니다.

```
L_pretrain = L_mask + lambda * L_forecast
```

Pretraining이 끝나면 Reconstruction Head와 Forecasting Head는 버리고, Transformer Encoder만 downstream에서 재사용합니다.

## 평가 방법

**All-Type Evaluation**: 5종 공격을 모두 train/val/test에 사용해 Attack Detection(Normal vs Attack), Attack Classification(5-class) 성능을 확인합니다. 이미 알고 있는 패턴을 잘 맞추는지 보는 sanity check 성격이며, 미지의 공격에 대한 일반화를 보장하지 않습니다.

**Unseen-Attack Evaluation (Leave-One-Attack-Out)**: 5종 공격 중 하나를 downstream 학습에서 완전히 제외하고, 나머지 4종+Normal로 Attack Detection 모델을 학습합니다. 테스트에서 제외했던 공격을 처음 입력해서 "정상이 아니다"라고 판단할 수 있는지 확인합니다. 같은 fold의 Attack Classification 모델은 held-out 타입 없이(4-class) 학습되며, known type(4종) 분류 성능만 확인합니다 — held-out 타입을 분류하는 건 이 모델의 역할이 아닙니다. 5종 각각에 대해 반복하며, fold 6개 × 모델 2개(Detection+Classification) = 총 **12개** 모델을 학습합니다. Unseen fold의 Detection 결과가 실제 generalization 성능을 나타내는 핵심 지표입니다.

Replay는 값 자체가 실제 정상 데이터라 window 내부 값만으로는 정상과 구분이 안 됩니다. 탐지 가능한 신호는 replay 구간 경계의 값 불연속뿐이므로, downstream 데이터는 attack 경계를 포함하는 overlap window(stride < window 길이)로 구성합니다.

### Train / Val / Test 분할

fold별 최종 데이터셋은 고정 시드로 한 번만 생성해서 그대로 씁니다 (fold마다 다시 만들지 않음 — 같은 유형의 데이터는 fold 간에 바이트 단위로 동일해야 비교가 공정합니다). Detection과 Classification은 같은 fold 데이터를 공유하되, Classification은 그중 Normal을 제외한 샘플만 사용합니다.

- **all_type**: Normal + 5종 전부, train/val/test 다 이 구성
- **unseen_X**: Normal + (X 제외 4종), train/val만 생성 (X는 여기 등장 안 함 — val이 체크포인트 선택에 쓰이는데 여기 X가 섞이면 평가가 오염됨)
- **test는 all_type의 test 하나만 존재**하며, 6개 fold의 Detection 모델 평가에 전부 이걸로 평가합니다. unseen_X 모델 입장에서는 이 test set에 자기가 한 번도 못 본 X가 포함되어 있으므로, 거기서의 반응이 핵심 결과가 됩니다.

```
all_type:  train/val/test = Normal + 5종 전부
unseen_X:  train/val      = Normal + (X 제외 4종)    (test 없음, all_type/test 재사용)
```

## 워크플로우

```bash
# 1) 원본 데이터 전처리 + pretrain용 정상 데이터 생성
python scripts/prepare_data.py --config configs/data/default.yaml

# 2) SSL pretraining (masked reconstruction + forecasting joint)
python scripts/pretrain.py --config configs/pretrain/default.yaml

# 3) downstream용 fold별 데이터셋 생성 (attack injection, 고정 시드로 한 번만 실행)
python scripts/prepare_downstream_data.py --config configs/downstream/attack_injection.yaml

# 4) Attack Detection + Classification 학습
#    fold 하나당 encoder(frozen)를 1번만 통과시키고, 그 결과로 detector+classifier를 같은 스크립트 안에서 함께 학습
#    (a) 전체 6개 fold를 한 번에 (--fold 생략 또는 all)
python scripts/train_downstream.py --config configs/downstream/default.yaml
#    (b) fold 하나만 개별 실행하고 싶을 때
python scripts/train_downstream.py --config configs/downstream/default.yaml --fold all_type
python scripts/train_downstream.py --config configs/downstream/default.yaml --fold unseen_scale_down
python scripts/train_downstream.py --config configs/downstream/default.yaml --fold unseen_ramp
python scripts/train_downstream.py --config configs/downstream/default.yaml --fold unseen_pulse_plateau
python scripts/train_downstream.py --config configs/downstream/default.yaml --fold unseen_replay
python scripts/train_downstream.py --config configs/downstream/default.yaml --fold unseen_instant_spike

# 5) 평가 (Detection은 6개 fold 전부 all_type의 test set으로 평가, Classification은 fold별 known-type만)
#    (a) 전체 6개 fold를 한 번에 (--fold 생략 또는 all)
python scripts/evaluate_downstream.py --config configs/downstream/default.yaml
#    (b) fold 하나만 개별 실행하고 싶을 때
python scripts/evaluate_downstream.py --config configs/downstream/default.yaml --fold all_type
python scripts/evaluate_downstream.py --config configs/downstream/default.yaml --fold unseen_scale_down
python scripts/evaluate_downstream.py --config configs/downstream/default.yaml --fold unseen_ramp
python scripts/evaluate_downstream.py --config configs/downstream/default.yaml --fold unseen_pulse_plateau
python scripts/evaluate_downstream.py --config configs/downstream/default.yaml --fold unseen_replay
python scripts/evaluate_downstream.py --config configs/downstream/default.yaml --fold unseen_instant_spike

# 6) 새 데이터에 대한 추론
python scripts/infer.py --ckpt checkpoints/downstream/all_type/detector/best.pt --input data/raw/new_lp.xlsx
```

## 설계 원칙

- `models/encoder.py`의 Transformer 백본은 pretrain에서만 학습되고, downstream에서는 **완전히 freeze**됩니다 (linear probing — 파라미터 업데이트 없음). 12개 모델 전부 동일한 pretrained encoder를 그대로 씁니다.
- Attack Detection과 Attack Classification은 역할이 다른 별개의 모델입니다 — Detection은 "공격인가 아닌가", Classification은 "공격이라면 어떤 유형인가"만 담당하며, 파라미터·loss·checkpoint가 완전히 분리된 독립적인 모델입니다 (서로의 gradient에 영향을 주지 않음).
- encoder가 frozen이라 fold·head 조합과 무관하게 항상 같은 결과를 내므로, `train_downstream.py`가 **fold 하나당 encoder forward를 1번만** 수행하고 그 z_t로 Detection head와 Classification head를 같은 실행 안에서 함께 학습합니다 (스크립트 실행은 fold당 1번, 총 6번 — 모델 개수는 여전히 12개).
- 두 head 다 all_type 1개 + unseen fold 5개, 총 6개 fold로 학습·평가되므로 전체 모델은 12개입니다. config 파일은 1개(`configs/downstream/default.yaml`)뿐이고 `--fold` CLI 인자로 fold를 선택합니다 (생략 시 6개 fold 전체 순차 실행).
- `configs/downstream/attack_injection.yaml`이 공격 파라미터와 fold별 데이터 생성 방식의 단일 출처입니다. 모든 fold 데이터는 한 번만, 고정 시드로 생성하며 Detection·Classification이 이를 공유합니다.
- unseen fold의 train/val은 held-out 타입을 완전히 제외하고, Detection 평가의 test는 all_type의 test를 6개 모델이 공유합니다.