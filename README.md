# Campus Power LP Anomaly Detection (Transformer + SSL Pretraining)

교내 수전/LP(Load Profile) 전력 데이터를 이용해 Transformer 기반 이상탐지를 수행하는 프로젝트입니다.
2단계로 구성됩니다.

1. **Self-supervised pretraining** (`src/adt/ssl`, `scripts/pretrain.py`)
   레이블 없는 정상 구간 위주의 LP 시계열로 Transformer 인코더를 masked reconstruction 등으로 사전학습합니다.
2. **Anomaly detection fine-tuning / inference** (`src/adt/models/anomaly_head.py`, `scripts/finetune.py`, `scripts/infer.py`)
   사전학습된 인코더 가중치를 불러와 이상탐지 헤드(재구성 오차 기반 혹은 지도학습 분류)를 얹어 학습·평가합니다.

## 데이터 컬럼 (예시: LP 수전 데이터)
`일자/시간, LP발생횟수, 유효전력량, 지상무효전력량, 진상무효전력량, 피상전력량` 등 15분 단위 다변량 시계열.
`configs/data/default.yaml`에서 사용할 컬럼과 윈도우 길이를 정의합니다.

## 디렉토리 구조

```
campus-power-ad/
├── configs/                 # 실험 설정 (yaml)
│   ├── data/                #   전처리/윈도잉 설정
│   ├── pretrain/             #   SSL 사전학습 설정
│   └── finetune/             #   이상탐지 파인튜닝 설정
├── data/
│   ├── raw/                 # 원본 엑셀/CSV (건드리지 않음)
│   ├── interim/              # 결측치 처리·리샘플링 등 중간 산출물
│   └── processed/            # 윈도우화된 텐서 (npy/parquet), train/val/test split
├── src/adt/                  # 패키지 소스
│   ├── data/                 # 로딩, 전처리, 윈도잉, 스케일링, Dataset/DataLoader
│   ├── models/                # Transformer 인코더(backbone), SSL head, 이상탐지 head
│   ├── ssl/                   # 마스킹 전략, SSL 손실함수
│   ├── engine/                 # 학습/평가 루프 (pretrain, finetune, evaluate)
│   ├── utils/                  # seed, logger, checkpoint, metrics
│   └── inference/               # 학습된 모델로 새 데이터 이상 점수 산출
├── scripts/                   # CLI 진입점 (각 engine 함수를 호출)
├── notebooks/                  # 탐색적 데이터분석(EDA)
├── checkpoints/
│   ├── pretrain/               # SSL 사전학습 가중치
│   └── finetune/                # 이상탐지 파인튜닝 가중치 (pretrain 가중치 로드해서 이어 학습)
├── logs/                        # 학습 로그 / tensorboard
├── outputs/
│   ├── figures/                  # 재구성 오차 플롯, 이상 구간 시각화
│   └── scores/                    # 이상 점수 csv 등 추론 결과
└── tests/                         # 유닛 테스트
```

## 워크플로우

```bash
# 1) 원본 데이터 전처리 + 윈도잉
python scripts/prepare_data.py --config configs/data/default.yaml

# 2) 정상 데이터 위주로 self-supervised pretraining
python scripts/pretrain.py --config configs/pretrain/default.yaml

# 3) 사전학습 인코더를 불러와 이상탐지 파인튜닝/보정
python scripts/finetune.py --config configs/finetune/default.yaml \
    --encoder-ckpt checkpoints/pretrain/best.pt

# 4) 평가
python scripts/evaluate.py --ckpt checkpoints/finetune/best.pt

# 5) 새 데이터에 대한 추론 (이상 점수 산출)
python scripts/infer.py --ckpt checkpoints/finetune/best.pt --input data/raw/new_lp.xlsx
```

## 설계 원칙
- `models/encoder.py`의 Transformer 백본은 pretrain/finetune 단계에서 **공유**됩니다. head만 교체됩니다.
- `checkpoints/pretrain`과 `checkpoints/finetune`을 분리해 두 단계의 결과를 독립적으로 재현/비교할 수 있습니다.
- `configs/`를 단계별로 분리해 하이퍼파라미터를 실험별로 버전관리하기 쉽게 했습니다.
