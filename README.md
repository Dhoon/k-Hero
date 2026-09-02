# Campus Power Meter Attack Detection (SSL Transformer)

네트워크로 측정값을 전송하는 교내 전력계(power meter)의 시계열 데이터를 대상으로, 사이버 공격에 의해 변조된 측정값을 탐지하는 프로젝트입니다.

Self-Supervised Learning(SSL)으로 정상 전력 시계열의 패턴을 먼저 학습한 뒤, 같은 Transformer 인코더를 공유하는 두 개의 downstream head로

1. 공격 여부 판단 (Attack Detection)
2. 공격 유형 분류 (Attack Classification)

를 수행합니다.

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
Attack Detection Head       Attack Classification Head
Normal(0) / Attack(1)       5 attack types
```

Transformer Encoder는 pretrain 단계에서 먼저 학습되고, 두 downstream head가 이 encoder를 공유합니다.

## 디렉토리 구조

```
campus-power-ad/
├── configs/
│   ├── data/default.yaml                 # 전처리·윈도잉 설정 (4채널)
│   ├── pretrain/default.yaml             # masked reconstruction + forecasting joint 설정
│   └── downstream/
│       ├── attack_injection.yaml         # 5종 공격 주입 파라미터 + pool 생성 설정 (단일 출처)
│       ├── detection/
│       │   └── default.yaml              # 1개 config, --held-out-type CLI 인자로 6개 fold 실행
│       └── classification/
│           └── known.yaml                # 5-class, 전부 사용
├── data/
│   ├── raw/                              # 원본 xls (건드리지 않음, git 제외)
│   ├── interim/                          # 중간 산출물 (git 제외)
│   └── processed/
│       ├── scaler.joblib
│       ├── status_events/                # 계기별 상태정보 이벤트 분류 결과
│       ├── pretrain/                     # 레이블 없는 정상 데이터 (forecasting target 포함)
│       │   └── train/ val/ test/
│       └── downstream/
│           ├── detection/                # 타입별 pool. fold마다 새로 안 만들고 전부 여기서 조합해서 재사용
│           │   ├── normal/               #   train/ val/ test/  (label=0)
│           │   ├── scale_down/           #   train/ val/ test/  (label=1, 경계 포함 overlap window)
│           │   ├── ramp/
│           │   ├── pulse_plateau/
│           │   ├── replay/
│           │   └── instant_spike/
│           └── classification/
│               └── known/                # 5-class 레이블
│                   (train/ val/ test/)
├── src/adt/
│   ├── data/
│   │   ├── loaders.py, preprocessing.py, windowing.py, scalers.py, dataset.py
│   │   ├── attack_injection.py           # 5종 공격 주입 함수 (raw scale에서 동작)
│   │   └── labeling.py                   # known/unseen fold 데이터 오케스트레이션
│   ├── models/
│   │   ├── encoder.py, positional_encoding.py
│   │   └── heads/
│   │       ├── reconstruction_head.py    # pretrain 전용, downstream에서는 제거
│   │       ├── forecasting_head.py       # pretrain 전용, downstream에서는 제거
│   │       ├── detection_head.py         # MeanPool+MaxPool → MLP → binary
│   │       └── classification_head.py    # 5-class MLP
│   ├── ssl/
│   │   ├── masking.py                    # segment 단위 curriculum masking (15%→40%)
│   │   └── losses.py                     # L_mask + lambda * L_forecast
│   ├── engine/
│   │   ├── pretrain.py
│   │   ├── train_detection.py            # known/unseen 6개 config 공용
│   │   ├── train_classification.py
│   │   ├── evaluate_detection.py
│   │   └── evaluate_classification.py
│   ├── utils/                            # seed, logger, checkpoint, metrics
│   └── inference/detect.py
├── scripts/                              # CLI 진입점 (각 engine 함수 호출)
├── notebooks/                            # 탐색적 데이터분석
├── checkpoints/
│   ├── pretrain/
│   └── downstream/
│       ├── detection/{known, unseen_scale_down, unseen_ramp, unseen_pulse_plateau, unseen_replay, unseen_instant_spike}/
│       └── classification/known/
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

**Known-Attack Evaluation**: 5종 공격을 모두 train/test에 사용해 Attack Detection(Normal vs Attack), Attack Classification(5-class) 성능을 확인합니다. 이미 알고 있는 패턴을 잘 맞추는지 보는 sanity check 성격이며, 미지의 공격에 대한 일반화를 보장하지 않습니다.

**Unseen-Attack Evaluation (Leave-One-Attack-Out)**: 5종 공격 중 하나를 downstream 학습에서 완전히 제외하고, 나머지 4종 + Normal로만 Attack Detection Head를 학습합니다. 테스트에서 제외했던 공격을 처음 입력해서, 정확히 어떤 공격인지는 몰라도 "정상이 아니다"라고 판단할 수 있는지를 확인합니다. 5종 각각에 대해 반복하며, Attack Detection Head는 known(1) + unseen fold(5) 총 **6개** 모델을 학습합니다. 이 결과가 실제 generalization 성능을 나타내는 핵심 지표입니다.

Replay는 값 자체가 실제 정상 데이터라 window 내부 값만으로는 정상과 구분이 안 됩니다. 탐지 가능한 신호는 replay 구간 경계의 값 불연속뿐이므로, `detection` 데이터는 attack 경계를 포함하는 overlap window(stride < window 길이)로 구성합니다.

### Train / Val / Test 분할

`data/processed/downstream/detection/`의 타입별 pool(normal, scale_down, ramp, pulse_plateau, replay, instant_spike)은 각각 train/val/test로 시간순 분할되어 있고, fold마다 이 pool을 조합해서 씁니다.

- **train, val**: fold 구성과 동일 (held-out 타입은 train/val 둘 다에서 제외). Val이 학습 중 체크포인트 선택에 쓰이는데, 여기에 held-out 타입이 섞이면 "미지의 공격을 얼마나 잘 잡는지"가 체크포인트 선택 단계에서 이미 반영되어 평가가 오염되기 때문.
- **test**: 6개 fold가 전부 공유하는 단일 세트(Normal + 5종 전부)로 평가. known 모델은 5종 전체 성능을, 각 unseen 모델은 자신의 held-out 타입 성능을 이 test set에서 확인.

```
known:     train/val = Normal + 5종 전부      test = Normal + 5종 전부 (공유)
unseen_X:  train/val = Normal + (X 제외 4종)   test = Normal + 5종 전부 (공유, X 성능이 핵심 지표)
```

## 워크플로우

```bash
# 1) 원본 데이터 전처리 + pretrain용 정상 데이터 생성
python scripts/prepare_data.py --config configs/data/default.yaml

# 2) SSL pretraining (masked reconstruction + forecasting joint)
python scripts/pretrain.py --config configs/pretrain/default.yaml

# 3) downstream용 타입별 pool 생성 (attack injection, normal + 5종 + classification, 한 번만 실행)
python scripts/prepare_downstream_data.py --config configs/downstream/attack_injection.yaml

# 4) Attack Detection 학습 (known 1개 + unseen 5개, 같은 config를 --held-out-type만 바꿔서 총 6번 실행)
python scripts/train_detection.py --config configs/downstream/detection/default.yaml
python scripts/train_detection.py --config configs/downstream/detection/default.yaml --held-out-type scale_down
python scripts/train_detection.py --config configs/downstream/detection/default.yaml --held-out-type ramp
python scripts/train_detection.py --config configs/downstream/detection/default.yaml --held-out-type pulse_plateau
python scripts/train_detection.py --config configs/downstream/detection/default.yaml --held-out-type replay
python scripts/train_detection.py --config configs/downstream/detection/default.yaml --held-out-type instant_spike

# 5) Attack Classification 학습
python scripts/train_classification.py --config configs/downstream/classification/known.yaml

# 6) 평가 (6개 모델 전부 공유 test set으로 평가)
python scripts/evaluate_detection.py --config configs/downstream/detection/default.yaml
python scripts/evaluate_detection.py --config configs/downstream/detection/default.yaml --held-out-type scale_down
# ... (나머지 unseen fold 동일)
python scripts/evaluate_classification.py --config configs/downstream/classification/known.yaml

# 7) 새 데이터에 대한 추론
python scripts/infer.py --ckpt checkpoints/downstream/detection/known/best.pt --input data/raw/new_lp.xlsx
```

## 설계 원칙

- `models/encoder.py`의 Transformer 백본은 pretrain과 두 downstream task 전체에서 **공유**됩니다. head만 교체됩니다.
- Attack Detection과 Attack Classification은 역할이 다른 별개의 head입니다 — Detection은 "공격인가 아닌가", Classification은 "공격이라면 어떤 유형인가"만 담당합니다.
- Detection Head는 known 1개 + unseen fold 5개, 총 6개 모델로 독립적으로 학습·평가됩니다. config 파일은 1개(`detection/default.yaml`)뿐이고, `--held-out-type` CLI 인자로 6개 fold를 구분해서 실행합니다.
- `configs/downstream/attack_injection.yaml`이 공격 파라미터와 타입별 pool 생성 방식의 단일 출처입니다. Normal + 5종 pool을 한 번만 만들어서 모든 fold가 재사용하며, fold별로 다시 생성하지 않습니다.
- Detection의 train/val은 fold 구성과 동일하게(held-out 타입 제외) 만들고, test는 6개 fold가 공유하는 단일 세트(Normal+5종)를 씁니다 — held-out 타입이 체크포인트 선택에 영향을 주지 않도록 하기 위함입니다.