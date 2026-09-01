# Pretrain 방법론 & 파이프라인 설계 (campus-power-ad)

> 2026-09-01 확정. Transformer 기반 오토인코더 구조 + Self-supervised pretraining → Target task(이상탐지) fine-tuning → Attack 시나리오(합성 이상치) 평가, 3단계 파이프라인.

## 1. Pretrain 기법: Masked Segment Reconstruction (masked autoencoding)

**기본 방향:** target task 자체가 "복원(reconstruction)"이므로, pretrain도 같은 계열(복원)로 간다. pretrain과 finetune 사이 objective가 어긋나면(예: pretrain은 contrastive, finetune은 reconstruction) 전이가 잘 안 될 수 있기 때문.

### 세부 설계

| 항목 | 결정 | 이유 |
|---|---|---|
| 마스킹 단위 | **segment(연속 구간) 위주** | 전력 데이터는 인접 시점끼리 상관관계가 강함. 랜덤 단일 시점만 가리면 옆 시점 보간으로 trivial하게 풀려서(shortcut) 일간/주간 패턴 같은 진짜 구조를 안 배움. 몇 스텝씩 뭉텅이로 가려야 함 |
| 채널 마스킹 | **병행 사용** | 유효/지상무효/진상무효/피상전력량 4채널이 물리적으로 연관(역률 관계)되어 있음. 채널 하나를 통째로 가려서 "다른 채널 보고 이 채널 값 유추"하게 만들면 채널 간 관계까지 학습됨 |
| mask ratio | **15% → 40% curriculum** | 처음엔 쉬운 과제로 시작해 점점 어렵게 올리면 학습이 안정적으로 붙음 |
| 보조 objective (선택) | forecasting loss, mask loss와 **joint(동시) 학습**, weight λ ≈ 0.1~0.2 | 전력 데이터의 강한 주기성을 반영. 순차적으로(먼저 A task 끝내고 B task) 학습하면 catastrophic forgetting 위험 있어 반드시 같은 step에서 가중합으로 합쳐서 학습 |

### 설계 근거 (trivial task 방지)

pretrain task가 "너무 쉽게" 풀리면(예: 인접 시점 평균으로 trivial하게 해결됨) 모델이 진짜 구조를 안 배우고도 loss만 낮아짐 — 이러면 pretrain이 사실상 의미가 없어짐. masking(특히 segment/channel 단위, 충분한 ratio)은 "일부러 정보를 안 보여줘서 진짜로 맥락을 이해해야만 풀리게" 만드는 장치. 반대로 target task(전체 복원, masking 없음)는 pretrain보다 쉬운 버전이며, pretrain에서 배운 표현을 그대로 이어받아 빠르게 적응하는 구조.

- pretrain task ≠ target task여야 의미있는 게 아니라, **pretrain task가 trivial shortcut으로 안 풀려야 의미있음**. BERT(MLM→분류/QA), MAE(masked patch 복원→classification)도 본질적으로 이 원칙을 따름. 우리 케이스(masked reconstruction → full reconstruction)는 같은 계열(복원)이라 오히려 전이가 더 직접적.

---

## 2. Transformer Input 구성

**입력 텐서:** `(batch, window_size=96, n_features=4)` — `windowing.py`가 만드는 텐서 그대로 (96 step = 15분×96 = 24시간, 채널: 유효/지상무효/진상무효/피상전력량).

### 토큰화 방식: timestep-as-token (시점 단위 토큰)

- 매 타임스텝(15분)이 토큰 하나. 시퀀스 길이 = 96.
- 각 타임스텝의 4채널 값을 **Linear(4 → d_model)** 로 투영해 토큰 벡터 생성 → 이 투영 단계에서 이미 채널 간 정보가 섞임. 채널 마스킹(채널 하나 가리고 다른 채널로 유추) 설계와 자연스럽게 맞물림.
- patch 단위(여러 스텝 묶기, PatchTST식)는 시퀀스가 훨씬 길어질 때(예: 1주일=672 step) 연산량/inductive bias 이유로 고려. 지금 96 규모에선 timestep 단위로 충분하고 구현도 단순함.

### 정규화 순서
`scalers.py`의 채널별(feature-wise) 정규화가 Linear 투영보다 **선행**되어야 함 (`configs/data/default.yaml`의 `scaler: standard`). 채널마다 값 범위가 크게 다름(유효전력량 vs 무효전력량)라 정규화 없이 넣으면 큰 값 채널이 학습을 지배함.

### Positional Encoding + 시간 특징(time feature) 임베딩
- 순서 정보(sinusoidal/learnable PE)만으로는 "몇 번째 토큰인지"는 알아도 "몇 시인지"는 모름 — 시계열엔 절대적 시간 정보(주기성)가 중요.
- **hour-of-day, day-of-week를 별도 learnable embedding(혹은 sin/cos 인코딩)으로 추가**해서 positional encoding과 더하거나 concat 권장.
- 이게 특히 중요한 이유: segment masking으로 몇 시간을 통째로 가리는데, "그 구간이 몇 시였는지" 정보가 없으면 새벽인지 낮인지도 모른 채 복원해야 해서 과제가 불필요하게 어려워짐. 절대 시간 정보가 있어야 "이 시간대는 보통 이런 패턴"이라는 주기적 사전지식을 활용 가능.

### 마스크 토큰 표현
- segment 마스킹: 가려진 타임스텝을 0으로 채우지 말고 **학습 가능한 [MASK] 벡터**로 치환 (BERT 스타일). 0으로 채우면 심야 시간대의 실제 낮은 값과 구분이 안 됨.
- 채널 마스킹: 해당 채널 값만 학습 가능한 마스크 값으로 치환한 뒤, 나머지 채널과 함께 Linear 투영.

### 흐름 요약

```
(batch, 96, 4) raw window
   → 채널별 정규화 (scalers.py, standard)
   → (마스킹 적용 시) segment/channel 위치를 학습 가능 [MASK] 값으로 치환
   → Linear(4 → d_model) per timestep
        + hour-of-day / day-of-week embedding
        + positional encoding
   → (batch, 96, d_model) 토큰 시퀀스
   → nn.TransformerEncoder(d_model, n_heads, n_layers, d_ff)
   → (batch, 96, d_model) 토큰별 표현
   → head (pretrain: reconstruction/forecast, target task: reconstruction anomaly head)
```

---

## 3. 3단계 파이프라인

### ① Pretrain (Self-supervised)
- **데이터:** 직접 전기 사용해서 얻은 실측 LP 데이터(정상). 레이블 불필요
- **목표:** Transformer 인코더가 "정상적인 전력 소비 패턴이 시간적으로/채널간에 어떻게 생겼는지" 학습
- **방법:** 위 1번 설계(masked segment + channel reconstruction, + 보조 forecasting)
- **산출물:** `checkpoints/pretrain/best.pt` (인코더 가중치)
- **주의:** 이 단계엔 attack이나 합성 이상치가 전혀 섞이지 않음 — 순수하게 "정상이 뭔지"만 배우는 단계

### ② Target task (Fine-tuning for anomaly detection)
- ①에서 학습한 인코더를 불러와 이상탐지 head(`anomaly_head.py`, reconstruction 기반)를 붙이고 정상 데이터로 이어서 학습/보정
- masking 없이 **전체 window를 그대로 복원**하는, 실제로 inference에서 쓸 objective 그 자체로 적응
- 재구성 오차 분포 기준으로 threshold(현재 percentile 99) 산출
- **산출물:** `checkpoints/finetune/best.pt` + threshold 값
- 이 단계도 여전히 정상 데이터만 사용. "이상치가 들어오면 오차가 커질 것"이라는 가정에 기반한 모델/기준을 완성하는 단계

### ③ Attack 시나리오 (합성 이상치 = 평가 전용)
- ①②는 학습, ③은 순수 **검증**
- 레이블 있는 실제 이상 사례가 없으므로, 정상 test 데이터에 인위적으로 spike / drop / flatline / drift / trend_break 시나리오를 주입해 "실제 어택이었다면" 상황을 흉내냄
- 이 결과는 학습(①②)에 절대 다시 들어가지 않음. 오직 "학습된 모델을 통과시켰을 때 threshold를 넘기는지, AUROC/AUPRC/F1/Precision@K가 얼마나 나오는지"를 재는 별도 트랙
- 설정: `configs/eval/synthetic.yaml`

---

## 요약

```
[정상 실측 LP 데이터]
        │
        ▼
①  Pretrain (masked segment+channel reconstruction, self-supervised)
        │  checkpoints/pretrain/best.pt
        ▼
②  Target task fine-tuning (full reconstruction, anomaly head, threshold 산출)
        │  checkpoints/finetune/best.pt
        ▼
   ─────────────── 여기까지 학습 파이프라인, 정상 데이터만 사용 ───────────────

   [별도 트랙] ③ Attack 시나리오 (합성 이상치 주입, 평가 전용)
        │  정상 test 데이터 + 인위적 spike/drop/flatline/drift/trend_break
        ▼
   학습된 모델(②)에 통과 → AUROC/AUPRC/F1/Precision@K로 채점
```

① 과 ②는 "정상이 뭔지 배우고 기준을 세우는" 학습 파이프라인이고, ③은 그 기준이 실제로 이상을 잘 잡아내는지 사후에 시험지 채점하듯 검증하는 별도 트랙 — 셋이 서로 데이터가 섞이지 않게 완전히 분리된 구조.

---

## 현재 config와의 차이 (참고용, 반영은 직접)

`configs/pretrain/default.yaml`에 지금 들어있는 기본값은 `mask_mode: segment` 단일값, 고정 `mask_ratio: 0.4`, forecasting 보조loss 필드 없음 상태. 위 설계를 반영하려면 아래 항목들을 추가/수정하면 됨.

- `mask_mode`를 segment와 channel을 함께 쓰도록 (예: 확률적으로 둘 중 하나 선택하거나 매 샘플마다 두 방식을 혼합)
- `mask_ratio`를 고정값 대신 curriculum 스케줄(예: `mask_ratio_start: 0.15`, `mask_ratio_end: 0.4`, `curriculum_epochs`)로 변경
- 보조 forecasting loss 활성화 옵션 (`aux_forecast.enabled`, `aux_forecast.weight` 등)
- `models/encoder.py`, `models/positional_encoding.py`에 위 2번 설계(시점별 Linear 투영, hour-of-day/day-of-week 임베딩, 학습 가능 [MASK] 벡터) 반영 필요 — 현재는 "채널 임베딩(Linear) + positional encoding"까지만 docstring에 명시되어 있고 time feature 임베딩/mask 벡터는 없는 상태

## 다음 논의 후보
- attack 시나리오 종류를 좀 더 현실적인 전력 이상 패턴으로 다듬기
- threshold 잡는 방식을 percentile 외 다른 방법(k-sigma 등)과 비교