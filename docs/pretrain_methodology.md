# 전력계 시계열 사이버 공격 탐지 — 방법론 (campus-power-ad)

## 1. 연구 목표

네트워크를 통해 측정값을 전송하는 전력계(power meter)의 시계열 데이터를 대상으로, 사이버 공격에 의해 변조된 측정값을 탐지한다.

Self-Supervised Learning(SSL)으로 정상 전력계 시계열의 temporal pattern을 먼저 학습한 뒤, Transformer 기반 모델로

1. 공격 여부를 판단하고 (Attack Detection)
2. 공격일 경우 어떤 종류의 공격인지 분류한다 (Attack Classification)

이 둘은 별도의 head를 가진 별도의 모델로 각각 독립적으로 학습한다 (아래 3번 참고).

---

## 2. 데이터와 공격 유형

전력계는 시간에 따라 다채널 측정값을 전송한다. 한 시점의 값은

```
x_t = [유효전력량, 지상무효전력량, 진상무효전력량, 피상전력량]   (4채널)
```

이고, 모델 입력은 한 시점이 아니라 길이 L의 시계열 window다.

```
X_t = [x_(t-L+1), ..., x_t]      shape: (L, 4)
```

### 공격 유형 (5종)

1. **Scale Down** — 일정 시간 동안 정상 전력값을 일정 비율로 축소. 시계열의 전체적인 모양은 비슷하지만 magnitude가 작아짐.
2. **Ramp** — 일정 시간 동안 전력값을 서서히 증가/감소. 순간적인 이상값이 아니라 정상 trajectory에서 점진적으로 벗어남.
3. **Pulse Plateau** — 일정 시간 동안 전력값을 정상보다 크게 증가시킨 상태로 유지. 순간 spike가 아니라 높은 값이 일정 구간 지속됨.
4. **Replay** — 과거에 실제로 측정됐던 정상 시계열 데이터를 현재 데이터 대신 재전송. 값 자체는 실제 정상 데이터이기 때문에 단순 값 범위 기반 탐지가 어려움.
5. **Instant Spike** — 1~2개의 짧은 timestep에서 전력값이 순간적으로 크게 증가.

---

## 3. 전체 모델 구조

```
Power-meter time-series window X_t   (L, 4)
        ↓
Pretrained Transformer Encoder
        ↓
Time-series representation z_t
        ↓
 ┌─────────────────────────────┐
 │                             │
Attack Detection 모델        Attack Classification 모델
(binary head)                 (multi-class head)
 │                             │
Normal(0) / Attack(1)      공격 유형
```

Transformer Encoder는 pretraining 단계에서 먼저 학습되고, 두 downstream 모델이 이 pretrained encoder를 동일하게 가져와서 각자의 head를 새로 붙여 **독립적으로** 학습한다. 같은 fold의 데이터를 쓰지만, Detection 모델과 Classification 모델은 서로 다른 학습 run(서로 다른 encoder 사본 + 서로 다른 head)이며, 하나의 encoder를 두 head가 공유하며 동시에 학습하는 구조가 아니다.

- **Attack Detection**: fold의 모든 샘플(Normal 포함) 사용, label은 (Normal이면 0, 공격이면 1)인 binary
- **Attack Classification**: fold에서 Normal을 제외한 공격 샘플만 사용, label은 그 fold에 존재하는 공격 유형 중 하나 (multi-class)

fold 하나당 이 두 모델이 각각 하나씩 나온다 (아래 7번 참고, 총 6개 fold × 2 head = 12개 모델).

---

## 4. Stage 1: Self-Supervised Pretraining

목적은 공격을 직접 분류하는 것이 아니라, 정상 전력 시계열이 가지는 temporal pattern과 dynamics를 Transformer가 먼저 학습하도록 하는 것이다. 두 objective를 joint learning한다.

### 4.1 Masked Reconstruction (주 objective)

정상 시계열 window의 일부 구간을 **segment 단위**로 masking한다. 인접 시점 몇 개만 가리면 주변 값 보간으로 trivial하게 풀려버려서 진짜 temporal structure를 배우지 못하므로, 연속된 구간(segment)을 통째로 가린다.

- masking 비율은 15% → 40%로 curriculum 방식으로 점진적으로 올린다 (학습 초반엔 쉬운 과제, 후반엔 어려운 과제).
- 가려진 위치는 0이 아니라 학습 가능한 [MASK] 벡터로 치환한다.

```
Masked normal time-series
        ↓
Transformer Encoder
        ↓
Reconstruction Head
        ↓
Masked values reconstruction
```

masked position에 대해서만 reconstruction loss(MSE)를 계산한다.

```
L_mask = (1/|M|) * sum_{i in M} || x_i - x_hat_i ||^2
```

### 4.2 Forecasting (보조 objective)

과거 시계열 window를 입력으로 사용하여, 그 이후의 미래 h timestep을 예측한다.

```
[x_(t-L+1), ..., x_t]
        ↓
Transformer Encoder
        ↓
Forecasting Head
        ↓
[x_(t+1), ..., x_(t+h)] 예측
```

Transformer가 단순히 빠진 값을 복원하는 것뿐 아니라, 정상적인 전력 시계열이 시간에 따라 어떤 방향으로 변화하는지를 학습하도록 한다. 이 objective를 위해서는 데이터 파이프라인이 window 뒤쪽의 실제 미래 h step도 함께 제공해야 한다 (기존 windowing이 window 안쪽만 잘라주던 것과 다름).

### 4.3 Joint Pretraining Loss

두 objective는 반드시 같은 step에서 함께 학습한다(순차적으로 학습하면 catastrophic forgetting 위험).

```
L_pretrain = L_mask + lambda * L_forecast
```

masked reconstruction이 주 objective이므로 forecasting loss의 weight는 상대적으로 작게 둔다. 초기 범위는 `lambda ≈ 0.1 ~ 0.2`이며, validation을 통해 조정한다.

Reconstruction과 forecasting은 공격을 판단하기 위한 최종 방법이 아니라 Transformer Encoder를 pretrain하기 위한 pretext task다. Pretraining이 끝나면 Reconstruction Head와 Forecasting Head는 downstream에서 제거하고, 학습된 Transformer Encoder만 가져와 사용한다.

---

## 5. Stage 2: Downstream Task 1 — Attack Detection

정상 데이터만 이용하는 one-class anomaly detection이 아니라, 공격 label을 이용한 **supervised binary classification**이다.

```
label:
Normal          → 0
Scale Down      → 1
Ramp            → 1
Pulse Plateau   → 1
Replay          → 1
Instant Spike   → 1
```

공격 종류는 중요하지 않고, "정상인가 공격인가"만 판단한다.

```
X_t
 ↓
Pretrained Transformer Encoder
 ↓
z_t = [h_1, ..., h_L]
 ↓
MeanPool(z_t) ; MaxPool(z_t)   ← concat
 ↓
MLP
 ↓
Normal(0) / Attack(1)
```

Mean pooling과 max pooling을 함께 쓰는 이유: Scale Down/Ramp/Pulse Plateau처럼 window 전체 패턴이 바뀌는 공격은 mean pooling이 잘 반영하고, Instant Spike처럼 1~2 step만 튀는 공격은 mean pooling에 신호가 희석되므로 max pooling이 필요하다. Binary Cross Entropy loss를 사용한다.

## 6. Stage 3: Downstream Task 2 — Attack Classification

공격 데이터가 어떤 공격 유형인지 분류한다. fold에 존재하는 공격 유형 수만큼 class를 가진다 (all_type은 5-class, unseen_X는 4-class — Normal은 이 head의 학습/출력에 포함되지 않는다).

```
X_t
 ↓
Pretrained Transformer Encoder
 ↓
z_t
 ↓
MeanPool(z_t) ; MaxPool(z_t)   ← concat
 ↓
MLP
 ↓
공격 유형 (해당 fold에 존재하는 타입 중 하나)
```

Attack Detection과 Attack Classification의 역할은 명확히 다르다.

- Attack Detection: 공격인가 아닌가?
- Attack Classification: 공격이라면 어떤 공격인가?

### Windowing 설계 (Replay 탐지를 위한 요구사항)

Replay는 값 자체가 실제 정상 데이터이기 때문에, window 하나만 놓고 보면 값의 magnitude나 trajectory로는 정상과 구분되지 않는다. 탐지 가능한 유일한 신호는 replay 구간의 시작/끝 지점에서 발생하는 **값의 불연속(경계)**이다.

이 신호를 모델이 볼 수 있으려면, window가 replay 구간의 경계를 포함해야 한다. 즉 window를 만들 때 stride를 window 길이보다 작게 잡아 overlap을 주어서, 공격 구간에 완전히 갇힌 window뿐 아니라 경계를 걸친 window도 충분히 존재하도록 해야 한다. training과 evaluation 양쪽 모두에 적용한다.

---

## 7. All-Type Evaluation

5종 공격을 모두 downstream 학습(train)과 평가(test)에 사용한다 (all_type fold). Detection 모델, Classification 모델 둘 다 이 fold로 학습·평가한다.

```
Training / Val / Test 공통 구성:
Normal, Scale Down, Ramp, Pulse Plateau, Replay, Instant Spike
```

평가 항목:

1. Attack Detection 성능 (Normal vs Attack)
2. Attack Classification 성능 (5-class accuracy / confusion matrix)

이 결과는 모델이 이미 알고 있는 공격 패턴을 잘 맞추는지 확인하는 sanity check 성격이며, 미지의 공격에 대한 일반화 성능을 보장하지 않는다.

---

## 8. Unseen-Attack Evaluation (Leave-One-Attack-Out)

학습하지 않은 새로운 공격 유형에 대해서도 Attack Detection이 가능한지 평가한다. 5개 공격 중 하나를 downstream training에서 완전히 제외하고, 나머지 4개 + Normal로 Attack Detection 모델을 학습한다. 이후 test에서 제외했던 공격을 처음 입력한다.

```
예: Replay를 unseen attack으로 설정

Training (Detection):
Normal → 0, Scale Down → 1, Ramp → 1, Pulse Plateau → 1, Instant Spike → 1
(Replay는 training에 전혀 사용하지 않음)

Test:
Replay 데이터를 처음 입력
```

목표는 Replay를 "Replay"라고 분류하는 것이 아니라, Attack Detection 모델이 `Replay → Attack(1)`이라고 판단할 수 있는지를 보는 것이다. 즉 "정확히 무슨 공격인지는 모르지만 정상은 아니다"를 판단할 수 있는지가 핵심이다. 이 fold의 Attack Classification 모델도 같은 이유로 Replay라는 class 자체 없이(4-class) 학습되며, 이 모델에 대해서는 known-type(4종) 분류 성능만 확인한다 — Replay 자체를 분류하는 건 애초에 이 모델의 역할이 아니다.

이 절차를 5개 공격 각각에 대해 반복한다.

```
1. Scale Down 제외    → Scale Down을 unseen attack으로 test
2. Ramp 제외          → Ramp를 unseen attack으로 test
3. Pulse Plateau 제외 → Pulse Plateau를 unseen attack으로 test
4. Replay 제외        → Replay를 unseen attack으로 test
5. Instant Spike 제외 → Instant Spike를 unseen attack으로 test
```

fold는 총 6개(all_type 1 + unseen 5)이고, fold마다 Detection 모델과 Classification 모델을 각각 독립적으로 학습하므로 총 **12개** 모델이 나온다. Unseen fold의 Attack Detection 결과가 미지의 공격에 대한 generalization 성능을 확인하는 핵심 실험이다.

### Train / Val / Test 분할

fold별 최종 데이터셋을 한 번의 고정 시드로 생성해서 그대로 사용한다 (fold마다 다시 생성하지 않음 — 같은 유형의 데이터는 fold 간에 동일해야 비교가 공정하다). Detection 모델과 Classification 모델은 같은 fold의 데이터를 공유하되, Classification은 그중 Normal을 제외한 샘플만 사용한다.

- **all_type**: Normal + 5종 전부. train/val/test 모두 이 구성.
- **unseen_X**: Normal + (X 제외 4종). train/val만 만든다 (X는 train/val 어디에도 등장하지 않음 — val이 체크포인트 선택에 쓰이는데 여기에 X가 섞이면 "미지의 공격을 얼마나 잘 잡는지"가 이미 반영되어 평가가 오염된다).
- **test는 all_type의 test 하나만 존재하며, 6개 fold의 Detection 모델(6개) 평가에 전부 이걸로 평가한다.** unseen_X 모델은 원래 이 test set에 X가 포함되어 있으므로, 그 부분에서 자신이 한 번도 못 본 X에 대해 어떻게 반응하는지가 핵심 결과가 된다.

```
all_type:     train/val/test = Normal + 5종 전부
unseen_X:     train/val      = Normal + (X 제외 4종)      (test 없음, all_type/test 재사용)
```

---

## 9. 전체 파이프라인 요약

```
[SSL Pretraining]
Normal power-meter time series (4채널)
        ↓
Segment Masking (curriculum 15% → 40%)
        ↓
Transformer Encoder
        ↓
 ┌─────────────────────┐
 │                     │
Masked Reconstruction  Forecasting
(주 objective)          (보조 objective, 미래 h step 예측)
 │                     │
 └──────────┬──────────┘
            ↓
L_pretrain = L_mask + lambda * L_forecast     (lambda ≈ 0.1~0.2)

            ↓  (Reconstruction/Forecasting Head 제거, Encoder만 유지)

[Downstream — fold(all_type + unseen 5개)마다 아래 두 모델을 독립적으로 학습]
Power-meter time-series window X_t
        ↓
Pretrained Transformer Encoder
        ↓
Representation z_t
        ↓
 ┌─────────────────────────────┐
 │                             │
Attack Detection 모델        Attack Classification 모델
(MeanPool+MaxPool→MLP→binary) (MeanPool+MaxPool→MLP→multi-class)
Normal / Attack               공격 유형 (Normal 제외 샘플만 학습)

[평가]
- All-Type Evaluation: 5종 모두 train/val/test에 사용 (sanity check)
- Unseen-Attack Evaluation: leave-one-attack-out, 5-fold 전부 수행
  → train/val은 fold별로 held-out 타입 제외, test는 all_type의 test 공유
  → fold 6개 × head 2개 = 총 12개 모델
```