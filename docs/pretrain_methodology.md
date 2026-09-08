# 전력계 시계열 사이버 공격 탐지 — 방법론 (campus-power-ad)

## 1. 연구 목표

네트워크를 통해 측정값을 전송하는 전력계(power meter)의 시계열 데이터를 대상으로, 사이버 공격에 의해 변조된 측정값을 탐지한다.

Self-Supervised Learning(SSL)으로 정상 전력계 시계열의 temporal pattern을 먼저 학습한 뒤,

1. 공격 여부를 판단하고 (Attack Detection)
2. 공격일 경우 어떤 종류의 공격인지 분류한다 (Attack Classification)

두 가지 백본 아키텍처를 병행 구현하여 비교한다.

- **LSTM Autoencoder 기반** (주 실험): BiLSTM encoder로 bottleneck을 학습, downstream에서 Sequential(Phase 1/2) 또는 Joint multi-task 방식으로 head를 학습
- **Transformer 기반**: Masked Reconstruction + Forecasting SSL, downstream에서 LayerNorm / block-level fine-tuning

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

### Windowing 설계 (Replay 탐지를 위한 요구사항)

Replay는 값 자체가 실제 정상 데이터이기 때문에, window 하나만 놓고 보면 값의 magnitude나 trajectory로는 정상과 구분되지 않는다. 탐지 가능한 유일한 신호는 replay 구간의 시작/끝 지점에서 발생하는 값의 불연속(경계)이다.

이 신호를 모델이 볼 수 있으려면 window가 replay 구간의 경계를 포함해야 한다. 즉 stride를 window 길이보다 작게 잡아 overlap을 주어서, 공격 구간에 완전히 갇힌 window뿐 아니라 경계를 걸친 window도 충분히 존재하도록 한다. training과 evaluation 양쪽 모두에 적용한다.

### Val/Test 분포 변형

각 fold마다 두 가지 분포 버전을 생성한다.

| 분할 | 비율 (Normal : Attack) | 용도 |
|------|----------------------|------|
| `val_50_50` / `test_50_50` | 1 : 1 균형 | 메인 threshold calibration 및 AUC-ROC 보고 |
| `val_9_1` / `test_9_1` | 9 : 1 실전 | 참고용 |

---

## 3. 평가 프로토콜 (공통)

### Fold 구성

- **all_type**: Normal + 5종 전부. train/val/test 모두 이 구성.
- **unseen_X**: Normal + (X 제외 4종). train/val만 생성 (X는 train/val 어디에도 등장하지 않음).
- test는 `all_type/test_50_50` 하나만 존재하며, 6개 fold의 Detection 모델 평가에 공유한다.

```
all_type:   train/val/test = Normal + 5종 전부
unseen_X:   train/val      = Normal + (X 제외 4종)      (test 없음, all_type/test 재사용)
```

fold 데이터는 고정 시드로 한 번만 생성하여 고정한다. 같은 유형의 데이터는 fold 간에 동일해야 비교가 공정하다.

### Threshold Calibration

val_50_50에서 F1-score가 최대가 되는 threshold를 탐색하여 `threshold.json`에 저장한다. 평가 시 자동으로 로드하여 optimal threshold 결과를 계산한다.

### Detection 보고 항목 (test_50_50 기준)

- default threshold (0.5): accuracy, precision, recall, F1
- optimal threshold (val calibration): accuracy, precision, recall, F1
- AUC-ROC, AUC-PR
- per-type recall (공격 유형별 탐지율, optimal threshold 기준)

### Classification 보고 항목

test_50_50 중 attack 샘플 + 해당 fold의 known type만 대상.

- accuracy, macro precision/recall/F1
- per-type precision/recall

---

## 4. LSTM 기반 파이프라인 (주 실험)

### 4.1 모델 구조

**LSTMEncoder** (공유 backbone):
```
X  (B, T, C)
 ↓
2-layer Bidirectional LSTM
 ↓
z = cat[h_fwd_last, h_bwd_last]   (B, 2*H)   ← bottleneck
```

**LSTMDetectionHead** (binary):
```
z (B, 2*H)  →  Linear(2*H, hidden) → ReLU → Dropout → Linear(hidden, 1) → (B,) logit
```

**LSTMClassificationHead** (K-class):
```
z (B, 2*H)  →  Linear(2*H, hidden) → ReLU → Dropout → Linear(hidden, K) → (B, K) logits
```

### 4.2 Stage 1: LSTM AE Pretraining

정상 시계열만 사용하여 LSTM Autoencoder를 학습한다. Encoder가 정상 전력 시계열의 temporal pattern을 bottleneck z에 압축하는 것이 목적이다.

```
X  (B, T, C)
 ↓
LSTMEncoder → z  (B, 2*H)
 ↓
LSTMDecoder (z를 T timestep 반복 입력 → LSTM → Linear)
 ↓
X_recon  (B, T, C)

L_pretrain = MSE(X_recon, X)   [정상 데이터 전체]
```

Decoder는 pretrain 전용이다. Downstream에서는 제거하고 Encoder만 재사용한다.

### 4.3 Stage 2-A: Sequential Downstream (Phase 1 → Phase 2)

두 Phase가 순차적으로 실행되며 checkpoint가 분리된다.

**Phase 1 — Detection**

pretrain encoder를 초기값으로, encoder + det_head를 함께 학습한다.

```
loss = det_loss_fn(det_logit, y_bin)   [전체 배치]
```

- `encoder_mode=unfreeze` (기본): encoder 전체 학습 가능 (Adam은 encoder+det_head 파라미터 전부)
- `encoder_mode=freeze`: encoder 동결, det_head만 학습

loss 함수 선택:
- `loss_type=bce` (기본): `BCEWithLogitsLoss(pos_weight=...)` — 클래스 불균형 보정
- `loss_type=focal`: `BinaryFocalLoss(γ=2.0, α=0.5)` — hard negative에 집중

val 기준: val_50_50 AUC-ROC 최고인 epoch → `best.pt` (encoder state_dict + det_head state_dict 함께 저장)

checkpoint 경로: `checkpoints/downstream_lstm/{loss_type}_{encoder_mode}/{fold}/detector/best.pt`

**Phase 2 — Classification**

Phase 1 best.pt에서 encoder를 로드하고 완전히 동결한다. cls_head만 학습한다.

```
attack_mask = (y_bin == 1)
loss = CrossEntropy(cls_logit[attack_mask], y_type[attack_mask])   [attack 샘플만]
```

val 기준: val_50_50 cls loss 최소.  
checkpoint 경로: `checkpoints/downstream_lstm/{mode_tag}/{fold}/classifier/best.pt`

Phase 2 encoder는 항상 Phase 1 best.pt에서 로드한다. Phase 1 `encoder_mode=unfreeze`여도 Phase 2는 encoder freeze다.

### 4.4 Stage 2-B: Joint Multi-task Downstream

encoder + det_head + cls_head를 단일 loss로 동시에 학습한다. encoder는 pretrain checkpoint에서 초기화하고 전체 unfreeze다.

```
det_loss = BCE(det_logit, y_bin)                           [전체 배치]
cls_loss = CE(cls_logit[attack], y_type[attack])           [attack 샘플만]
total_loss = w_det * det_loss + w_cls * cls_loss
```

`w_det=1.0`, `w_cls=0.4` (기본값, config에서 조정 가능).

z 계산은 한 번, det_head와 cls_head가 같은 z를 소비한다.

val 기준: val_50_50 Detection AUC-ROC 최고.  
checkpoint: `checkpoints/downstream_lstm/joint/{fold}/best.pt` (encoder + det_head + cls_head 한 파일)

Sequential과 Joint의 비교:

| 항목 | Sequential | Joint |
|------|------------|-------|
| encoder 사용 방식 | Phase 1: Detection 최적화, Phase 2에 전달 | Detection + Classification 동시 최적화 |
| checkpoint 수 | detector/best.pt + classifier/best.pt | best.pt 하나 |
| 추론 시 forward | encoder 1회 (두 head 공유 가능) | encoder 1회 |
| 장점 | 두 task 기준 분리, 조합 유연 | 단일 학습 run, z 공유 |

### 4.5 Training Mode 조합

Sequential downstream에서 `loss_type × encoder_mode` 4가지 조합이 가능하다.

| mode_tag | loss | encoder Phase 1 |
|----------|------|-----------------|
| `bce_unfreeze` (기본) | BCE + pos_weight | 전체 학습 |
| `bce_freeze` | BCE + pos_weight | 동결 |
| `focal_unfreeze` | Focal | 전체 학습 |
| `focal_freeze` | Focal | 동결 |

```bash
# 기본 (bce_unfreeze)
python scripts/train_downstream_lstm.py --fold all_type

# focal + freeze
python scripts/train_downstream_lstm.py --fold all_type \
    --loss_type focal --encoder_mode freeze

# Joint
python scripts/train_downstream_lstm_joint.py --fold all_type
```

평가 시 동일한 `--loss_type`/`--encoder_mode`를 지정해야 올바른 checkpoint 경로를 찾는다.

```bash
python scripts/evaluate_downstream_lstm.py --fold all_type --calib 50_50
python scripts/evaluate_downstream_lstm.py --fold all_type --calib 50_50 \
    --loss_type focal --encoder_mode freeze
python scripts/evaluate_downstream_lstm_joint.py --fold all_type
```

---

## 5. Transformer 기반 파이프라인

### 5.1 Stage 1: SSL Pretraining

두 objective를 joint learning한다.

**Masked Reconstruction (주 objective)**

정상 시계열 window의 일부 구간을 segment 단위로 masking한다. 연속된 구간(segment)을 통째로 가려 인접 보간으로 trivial하게 풀리는 것을 막는다.

- masking 비율: 고정 15%
- 가려진 위치는 학습 가능한 [MASK] 벡터로 치환

```
Masked input  →  Transformer Encoder  →  Reconstruction Head
L_mask = (1/|M|) * sum_{i ∈ M} ‖x_i - x̂_i‖²
```

**Forecasting (보조 objective)**

마스킹 없는 원본 window를 별도의 clean forward로 encoder에 통과시켜 미래 h timestep을 예측한다. 구현상 효율을 위해 masked input과 clean input을 배치 방향으로 cat해서 encoder 1회 호출한다.

```python
h = encoder(cat([x_masked, x_clean]))
h_masked, h_clean = h.split(B)
pred_recon  = recon_head(h_masked)
pred_future = forecast_head(h_clean)
```

```
L_pretrain = L_mask + λ * L_forecast     (λ ≈ 0.15)
```

Pretraining 후 Reconstruction Head와 Forecasting Head는 제거하고 Transformer Encoder만 downstream에서 재사용한다.

### 5.2 Stage 2: Downstream

pretrained Transformer Encoder에 Detection head와 Classification head를 각각 독립적으로 붙여 학습한다. 두 head는 파라미터·loss·checkpoint가 완전히 분리되어 서로의 gradient에 영향을 주지 않는다.

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
Normal(0)/Attack(1)  또는  공격 유형 K-class
```

MeanPool과 MaxPool을 함께 쓰는 이유: Scale Down/Ramp/Pulse Plateau처럼 window 전체 패턴이 바뀌는 공격은 mean pooling이 잘 반영하고, Instant Spike처럼 1~2 step만 튀는 공격은 mean pooling에 신호가 희석되므로 max pooling이 필요하다.

encoder fine-tuning 범위:
- `t1` (Tier 1): 전체 LayerNorm affine 파라미터만 학습 가능 (~2K params)
- `t2` (Tier 2): 전체 LayerNorm + 마지막 transformer block의 attention/FFN 전체 (~68K params)

---

## 6. All-Type Evaluation

5종 공격을 모두 downstream 학습(train)과 평가(test)에 사용한다 (all_type fold). 모델이 이미 알고 있는 공격 패턴을 잘 맞추는지 확인하는 sanity check 성격이며, 미지의 공격에 대한 일반화 성능을 보장하지 않는다.

---

## 7. Unseen-Attack Evaluation (Leave-One-Attack-Out)

학습하지 않은 새로운 공격 유형에 대해서도 Attack Detection이 가능한지 평가한다. 5개 공격 중 하나를 downstream training에서 완전히 제외하고, all_type/test set으로 그 공격에 대한 반응을 확인한다.

```
예: Replay를 unseen attack으로 설정

Training (Detection):
Normal → 0, Scale Down → 1, Ramp → 1, Pulse Plateau → 1, Instant Spike → 1
(Replay는 training에 전혀 사용하지 않음)

Test:
Replay 데이터를 처음 입력 → "Replay"라고 분류하는 것이 목표가 아니라
                             Attack(1)이라고 판단하는 것이 목표
```

이 절차를 5개 공격 각각에 대해 반복한다. fold 6개 × backbone × 학습 모드 수만큼 모델이 나온다.

Unseen fold의 Attack Detection per-type recall(held-out 공격 유형)이 핵심 결과 지표다.

---

## 8. 전체 파이프라인 요약

```
[LSTM 파이프라인]

LSTM AE Pretraining
  정상 시계열 X → LSTMEncoder → z → LSTMDecoder → X_recon
  L_pretrain = MSE(X_recon, X)
        ↓  (Decoder 제거, Encoder만 유지)

Downstream  — fold(all_type + unseen 5개)마다 아래 중 선택

  [Sequential]
  Phase 1 (Detection)
    encoder(init=pretrain) + det_head
    loss = det_loss_fn(det_logit, y_bin)   [BCE or Focal, encoder unfreeze or freeze]
    best ← val_50_50 AUC-ROC 최고
    checkpoint: {mode_tag}/{fold}/detector/best.pt  [encoder + det_head]

  Phase 2 (Classification)
    encoder(init=Phase1 best.pt, freeze) + cls_head
    loss = CE(cls_logit[attack], y_type[attack])
    best ← val_50_50 cls loss 최소
    checkpoint: {mode_tag}/{fold}/classifier/best.pt

  [Joint]
  encoder(init=pretrain, unfreeze) + det_head + cls_head
  total_loss = w_det * BCE(det_logit, y_bin) + w_cls * CE(cls_logit[attack], y_type[attack])
  best ← val_50_50 Detection AUC-ROC 최고
  checkpoint: joint/{fold}/best.pt  [encoder + det_head + cls_head]

[Transformer 파이프라인]

SSL Pretraining
  masked pass → Reconstruction Head  (주 objective)
  clean pass  → Forecasting Head     (보조 objective)
  L = L_mask + λ * L_forecast
        ↓  (heads 제거, Encoder만 유지)

Downstream  — fold마다 아래 두 모델 독립적으로 학습
  encoder(init=pretrain, fine-tune t1/t2) + Detection head    → Normal / Attack
  encoder(init=pretrain, fine-tune t1/t2) + Classification head → 공격 유형

[평가]
- threshold calibration: val_50_50 F1-max → threshold.json
- Detection 보고: test_50_50 (default 0.5 + optimal threshold + AUC-ROC + per-type recall)
- Classification 보고: test_50_50 attack + known type (macro F1, per-type prec/rec)
- All-Type: sanity check  |  Unseen-Attack: generalization 핵심 지표
```
