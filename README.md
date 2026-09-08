# Campus Power Meter Attack Detection

네트워크로 측정값을 전송하는 교내 전력계(power meter)의 시계열 데이터를 대상으로, 사이버 공격에 의해 변조된 측정값을 탐지·분류하는 프로젝트입니다.

두 가지 백본 아키텍처를 병행 구현합니다.

| 백본 | Pretrain | Downstream |
|------|----------|------------|
| **LSTM** (주 실험) | LSTM Autoencoder (MSE 재구성) | Sequential (Phase 1 Detection → Phase 2 Classification) 또는 Joint Multi-task |
| **Transformer** | Masked Reconstruction + Forecasting SSL | 두 head 독립 fine-tuning (t1/t2) |

현재 활발히 개발·실험 중인 파이프라인은 LSTM 계열입니다.

---

## 데이터

`유효전력량, 지상무효전력량, 진상무효전력량, 피상전력량` 4채널, 15분 단위 시계열입니다.  
`configs/data/default.yaml`에서 윈도우 길이(L)와 전처리 방식을 정의합니다.

### 공격 유형 (5종)

| 유형 | 설명 |
|------|------|
| Scale Down | 일정 시간 정상값을 일정 비율로 축소 (magnitude만 작아짐) |
| Ramp | 일정 시간 값을 서서히 증가/감소 (점진적 이탈) |
| Pulse Plateau | 일정 시간 값을 정상보다 높게 유지 (지속되는 상승) |
| Replay | 과거 실제 정상 구간을 그대로 복사해 재전송 (값 자체는 정상 데이터) |
| Instant Spike | 1~2 timestep만 순간적으로 크게 튐 |

### Val/Test 분포 변형

각 fold마다 두 가지 분포 버전을 생성합니다.

| 분할 | 비율 (Normal : Attack) | 용도 |
|------|----------------------|------|
| `val_50_50` / `test_50_50` | 1 : 1 균형 | 메인 threshold calibration 및 AUC-ROC 보고 |
| `val_9_1` / `test_9_1` | 9 : 1 실전 | 참고용 (현재 기본 평가는 50_50 기준) |

---

## LSTM 파이프라인 (주 실험)

### 모델 구조

```
Power-meter window X  (B, T, C)
        ↓
LSTMEncoder (2-layer BiLSTM)
        ↓
z  (B, 2*H)  — bottleneck
        ↓
 ┌───────────────────────────┐
 │                           │
LSTMDetectionHead        LSTMClassificationHead
(z → (B,) logit)         (z → (B, K) logits)
Normal(0) / Attack(1)    공격 유형 (K-class)
```

### 학습 모드

**Sequential (Phase 1 → Phase 2)**

Phase 1과 Phase 2가 독립적으로 실행되며 checkpoint가 분리됩니다.

| Phase | 학습 대상 | val 기준 |
|-------|----------|----------|
| Phase 1 (Detection) | encoder + det_head | val_50_50 AUC-ROC |
| Phase 2 (Classification) | cls_head만 (encoder freeze) | val_50_50 cls loss |

`--loss_type`과 `--encoder_mode`로 Phase 1 방식을 선택합니다.

| 옵션 | 값 | 설명 |
|------|----|------|
| `--loss_type` | `bce` (기본) | pos_weight-adjusted BCE |
|  | `focal` | Binary Focal Loss (γ=2.0, α=0.5) |
| `--encoder_mode` | `unfreeze` (기본) | Phase 1에서 encoder 전체 학습 |
|  | `freeze` | Phase 1에서 encoder 완전 동결 |

checkpoint 경로: `checkpoints/downstream_lstm/{loss_type}_{encoder_mode}/{fold}/`

**Joint Multi-task**

encoder + det_head + cls_head를 하나의 loss로 동시 학습합니다.

```
total_loss = w_det * BCE(det_logit, y_bin)
           + w_cls * CE(cls_logit[attack], y_type[attack])
```

`w_det=1.0`, `w_cls=0.4`는 `configs/downstream_lstm_joint/default.yaml`에서 조정합니다.  
checkpoint 경로: `checkpoints/downstream_lstm/joint/{fold}/best.pt`

---

## Transformer 파이프라인

```
Masked window X_masked  +  Clean window X_clean
        ↓                          ↓
Transformer Encoder (공유 weights, 배치 방향 cat 1회 호출)
        ↓                          ↓
Reconstruction Head            Forecasting Head
(주 objective)                  (보조 objective, h-step 예측)

L_pretrain = L_mask + λ * L_forecast     (λ ≈ 0.15)
```

Downstream에서는 pretrained encoder만 가져와 fine-tuning합니다.
- `t1`: 전체 LayerNorm affine만 (~2K params)
- `t2`: 전체 LayerNorm + 마지막 block attention/FFN (~68K params)

Attack Detection head와 Classification head는 파라미터·loss·checkpoint가 완전히 분리된 독립 모델입니다.

---

## 평가 방법

**All-Type Evaluation**: 5종 공격을 train/val/test에 전부 사용. Known 패턴 sanity check.

**Unseen-Attack Evaluation (Leave-One-Attack-Out)**: 공격 1종을 downstream train/val에서 완전 제외하고, all_type test set으로 그 공격에 대한 Detection 반응을 확인. fold 6개 × backbone × 학습 모드 수만큼 모델이 나옵니다.

**threshold calibration**: val_50_50 F1-max sweep → `threshold.json`에 저장.  
**Detection 보고**: test_50_50 (default thr=0.5 + optimal thr) + AUC-ROC + AUC-PR + per-type recall.  
**Classification 보고**: test_50_50 attack 샘플 중 known type만 대상, macro F1/acc + per-type prec/rec.

---

## 디렉토리 구조

```
campus-power-ad/
├── configs/
│   ├── data/default.yaml                      # 전처리·윈도잉 설정 (4채널)
│   ├── pretrain_lstm/default.yaml             # LSTM AE pretrain 설정
│   ├── pretrain_transformer/default.yaml      # Transformer SSL pretrain 설정
│   ├── downstream_lstm/default.yaml           # LSTM Sequential downstream 설정
│   ├── downstream_lstm_joint/default.yaml     # LSTM Joint downstream 설정
│   └── downstream_transformer/
│       ├── default.yaml                       # Transformer downstream 설정
│       └── attack_injection.yaml              # 5종 공격 파라미터
├── data/
│   ├── raw/                                   # 원본 xls (git 제외)
│   ├── interim/                               # 중간 산출물 (git 제외)
│   └── processed/
│       ├── scaler.joblib
│       ├── pretrain/                          # 정상 데이터 (train/val/test)
│       └── downstream/                        # fold별 데이터셋 (고정 시드, 1회 생성)
│           ├── all_type/                      # Normal + 5종
│           │   ├── train/  val_50_50/  val_9_1/  test_50_50/  test_9_1/
│           ├── unseen_scale_down/             # Normal + 4종 (train/val만)
│           ├── unseen_ramp/
│           ├── unseen_pulse_plateau/
│           ├── unseen_replay/
│           └── unseen_instant_spike/
├── src/adt/
│   ├── data/
│   │   ├── loaders.py, preprocessing.py, windowing.py, scalers.py, dataset.py
│   │   ├── attack_injection.py                # 5종 공격 주입 함수
│   │   └── labeling.py                        # fold별 데이터셋 생성 오케스트레이션
│   ├── models/
│   │   ├── lstm_ae.py                         # LSTMEncoder, LSTMDecoder, LSTMAutoencoder
│   │   │                                      #   + LSTMDetectionHead, LSTMClassificationHead
│   │   ├── lstm_joint.py                      # LSTMJoint (shared encoder → det + cls)
│   │   ├── transformer_encoder.py             # Transformer 백본
│   │   └── heads/                             # Transformer downstream heads
│   ├── engine/
│   │   ├── pretrain_lstm.py                   # LSTM AE 학습 루프
│   │   ├── train_downstream_lstm.py           # LSTM Sequential Phase 1+2 학습
│   │   ├── evaluate_downstream_lstm.py        # LSTM Sequential 평가
│   │   ├── train_downstream_lstm_joint.py     # LSTM Joint 학습
│   │   ├── evaluate_downstream_lstm_joint.py  # LSTM Joint 평가
│   │   ├── pretrain_transformer.py            # Transformer SSL 학습
│   │   ├── train_downstream_transformer.py    # Transformer downstream 학습
│   │   └── evaluate_downstream_transformer.py # Transformer downstream 평가
│   ├── ssl/                                   # masking.py, losses.py (Transformer 전용)
│   └── utils/                                 # seed, logger, checkpoint, metrics
├── scripts/
│   ├── prepare_data.py
│   ├── prepare_downstream_data.py
│   ├── pretrain_lstm.py
│   ├── pretrain_transformer.py
│   ├── train_downstream_lstm.py               # --loss_type / --encoder_mode / --fold
│   ├── evaluate_downstream_lstm.py            # --calib / --loss_type / --encoder_mode / --fold
│   ├── train_downstream_lstm_joint.py         # --fold
│   ├── evaluate_downstream_lstm_joint.py      # --fold
│   ├── train_downstream_transformer.py        # --mode t1/t2 / --fold
│   ├── evaluate_downstream_transformer.py     # --calib / --fold
│   ├── check_recon_error_auc.py               # recon error AUC 진단 스크립트
│   └── infer.py
├── tests/
│   ├── test_lstm.py, test_lstm_joint.py
│   ├── test_models.py, test_engine.py
│   ├── test_downstream.py, test_windowing.py
│   ├── test_anomaly_injection.py, test_labeling.py
├── checkpoints/
│   ├── pretrain_lstm/best.pt
│   ├── pretrain_transformer/best.pt
│   └── downstream_lstm/
│       ├── bce_unfreeze/{fold}/detector/best.pt   # encoder + det_head
│       │                      /classifier/best.pt  # cls_head
│       ├── bce_freeze/{fold}/...
│       ├── focal_unfreeze/{fold}/...
│       ├── focal_freeze/{fold}/...
│       └── joint/{fold}/best.pt                   # encoder + det_head + cls_head
├── logs/                                          # TensorBoard 로그 (checkpoints와 동일 구조)
├── outputs/
│   ├── scores_lstm/     scores_lstm_joint/        # metrics.json
│   └── figures_lstm/    figures_lstm_joint/        # per-type recall 차트
└── docs/
    └── pretrain_methodology.md
```

---

## 데이터 관리 정책

- `data/raw/`, `data/interim/`, `data/processed/`의 실제 파일은 git에 올리지 않습니다 (`.gitignore`에서 제외, 폴더 구조 유지용 `.gitkeep`만 추적).
- 체크포인트(`checkpoints/**/*.pt`)도 git에서 제외됩니다.
- 원본 xls는 로컬 또는 팀 공유 드라이브에만 보관하고, 새 컴퓨터에서 clone한 뒤 `data/raw/`에 원본 파일을 복사해 사용합니다.

---

## 워크플로우

### 공통: 데이터 준비

```bash
# 1) 원본 데이터 전처리
python scripts/prepare_data.py --config configs/data/default.yaml

# 2) fold별 downstream 데이터셋 생성 (고정 시드, 1회만 실행)
python scripts/prepare_downstream_data.py
```

### LSTM 파이프라인 (주 실험)

```bash
# 3) LSTM AE pretrain
python scripts/pretrain_lstm.py --config configs/pretrain_lstm/default.yaml

# 4-A) Sequential downstream 학습 (all_type fold, bce_unfreeze 모드)
python scripts/train_downstream_lstm.py --fold all_type

# 4-A-alt) 다른 모드 지정
python scripts/train_downstream_lstm.py --fold all_type --loss_type focal --encoder_mode freeze

# 4-B) Joint multi-task 학습
python scripts/train_downstream_lstm_joint.py --fold all_type

# 5-A) Sequential 평가 (val_50_50 calibration, test_50_50 보고)
python scripts/evaluate_downstream_lstm.py --fold all_type --calib 50_50

# 5-A-alt) 다른 모드 평가
python scripts/evaluate_downstream_lstm.py --fold all_type --calib 50_50 \
    --loss_type focal --encoder_mode freeze

# 5-B) Joint 평가
python scripts/evaluate_downstream_lstm_joint.py --fold all_type
```

**checkpoint 경로 (Sequential)**
```
checkpoints/downstream_lstm/
  bce_unfreeze/all_type/detector/best.pt    # encoder + det_head
                        /classifier/best.pt # cls_head
  focal_freeze/all_type/detector/best.pt
               ...
```

**checkpoint 경로 (Joint)**
```
checkpoints/downstream_lstm/joint/all_type/best.pt  # encoder + det_head + cls_head
```

### Transformer 파이프라인

```bash
# 3) Transformer SSL pretrain (masked reconstruction + forecasting)
python scripts/pretrain_transformer.py --config configs/pretrain_transformer/default.yaml

# 4) Downstream 학습 (--mode t1 또는 t2)
python scripts/train_downstream_transformer.py --fold all_type --mode t2

# 5) 평가
python scripts/evaluate_downstream_transformer.py --fold all_type --calib 50_50
```

---

## 설계 원칙

- **fold 구성**: `all_type`(5종 전부) + `unseen_{type}`(해당 공격 제외 4종) 각 5개 = 총 6 fold. test는 `all_type/test_{50_50|9_1}` 하나를 6 fold가 공유.
- **Sequential vs Joint**: Sequential은 Detection과 Classification의 최적 checkpoint 기준을 완전히 분리할 수 있고, Joint은 z space를 공유해 추론 비용을 절감하며 두 task가 상호 보완적으로 학습됩니다.
- **encoder_mode=freeze 사용 시**: encoder는 pretrain 가중치 그대로 동결. bottleneck z의 의미론이 변하지 않으므로 `check_recon_error_auc.py` 같은 재구성 오류 기반 진단이 안정적.
- **threshold calibration**: val_50_50 F1-max sweep으로 threshold를 결정하고 `threshold.json`에 저장. 평가 시 자동 로드.
- **Replay 탐지**: 값 자체가 정상 데이터이므로 구간 경계를 포함하는 overlap window(stride < window 길이)가 필수.
