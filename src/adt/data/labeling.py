"""분류 학습용 Dataset — Leave-one-attack-type-out.

split별 역할:
  train/val : known_types(4종)만 주입 → 학습/검증
  test      : held_out_type(1종)만 주입 → 일반화 성능 평가

레이블 누수 방지를 위해 stride=window_size 비겹침 윈도우만 사용.
이미 있는 inject_* 함수를 재사용하고, "어떤 타입을 어느 split에 적용할지"만 오케스트레이션.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.adt.data.anomaly_injection import inject_synthetic_anomalies
from src.adt.data.scalers import StandardScalerND


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------

class ClassificationWindowDataset(Dataset):
    """(x_norm, time_feat, label) 튜플 Dataset.

    x_norm  : (T, C) float32, 정규화된 전력값
    time_feat: (T, 2) float32
    label   : float32  (0=정상, 1=주입됨)  BCEWithLogitsLoss 호환
    """

    def __init__(
        self,
        X: np.ndarray,         # (N, T, C)
        time_feat: np.ndarray, # (N, T, 2)
        labels: np.ndarray,    # (N,) int32 or float32
    ) -> None:
        self.X = torch.from_numpy(X.astype(np.float32))
        self.time_feat = torch.from_numpy(time_feat.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.time_feat[idx], self.labels[idx]

    # label numpy array (sampler 가중치 계산용)
    @property
    def label_array(self) -> np.ndarray:
        return self.labels.numpy()


# -------------------------------------------------------------------------
# 내부 헬퍼
# -------------------------------------------------------------------------

def _filter_injection_cfg(injection_cfg: dict, allowed_types: list[str]) -> dict:
    """anomaly_types 중 allowed_types만 남기고 확률 재정규화."""
    cfg = copy.deepcopy(injection_cfg)
    cfg["anomaly_types"] = [
        t for t in cfg["anomaly_types"] if t["type"] in allowed_types
    ]
    if not cfg["anomaly_types"]:
        raise ValueError(
            f"allowed_types={allowed_types}에 해당하는 attack 타입이 "
            f"injection_config에 없음"
        )
    total = sum(t["prob"] for t in cfg["anomaly_types"])
    for t in cfg["anomaly_types"]:
        t["prob"] = t["prob"] / total
    return cfg


# -------------------------------------------------------------------------
# 팩토리
# -------------------------------------------------------------------------

def build_classification_dataset(
    processed_dir: str | Path,
    split: str,
    attack_cfg: dict[str, Any],
    scaler: StandardScalerND,
    window_size: int = 96,
    seed: int = 42,
    verbose: bool = True,
) -> ClassificationWindowDataset:
    """split에 맞게 이상치를 주입한 ClassificationWindowDataset을 반환.

    Args:
        processed_dir: data/processed (X.npy, time_feat.npy 있는 루트)
        split        : "train" | "val" | "test"
        attack_cfg   : classification yaml의 attack_split 블록
                       { known_types, held_out_type, injection_config }
        scaler       : StandardScalerND (inverse_transform / transform 사용)
        window_size  : stride=window_size로 비겹침 윈도우 추출 (기본 96=24h)
        seed         : 재현성 시드
    """
    processed_dir = Path(processed_dir)
    split_dir = processed_dir / split

    X_all = np.load(split_dir / "X.npy")           # (N_stride1, T, C)
    tf_all = np.load(split_dir / "time_feat.npy")  # (N_stride1, T, 2)

    # 비겹침 윈도우
    X_clean = X_all[::window_size]
    tf_clean = tf_all[::window_size]
    N = len(X_clean)

    # split별 허용 attack type
    if split in ("train", "val"):
        allowed_types = list(attack_cfg["known_types"])
    else:  # test
        allowed_types = [attack_cfg["held_out_type"]]

    # injection config 로드 + 필터
    with open(attack_cfg["injection_config"], encoding="utf-8") as f:
        raw_inj_cfg = yaml.safe_load(f)
    filtered_cfg = _filter_injection_cfg(raw_inj_cfg, allowed_types)

    X_corrupted, labels = inject_synthetic_anomalies(
        X_clean, filtered_cfg, scaler, seed=seed
    )
    n_pos = int(labels.sum())

    if verbose:
        print(
            f"[ClassDataset] split={split:5s}  N={N:5d}  "
            f"types={allowed_types}  "
            f"injected={n_pos}/{N} ({n_pos / N * 100:.1f}%)"
        )

    return ClassificationWindowDataset(
        X_corrupted.astype(np.float32),
        tf_clean.astype(np.float32),
        labels.astype(np.int32),
    )


def build_classification_dataloader(
    dataset: ClassificationWindowDataset,
    batch_size: int,
    attack_ratio_per_batch: float = 0.3,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """WeightedRandomSampler로 배치 내 attack 비율을 attack_ratio_per_batch로 유지.

    positive가 없거나 negative가 없는 극단 케이스에서는 단순 shuffle DataLoader.
    """
    from torch.utils.data import WeightedRandomSampler

    labels = dataset.label_array
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    if n_pos == 0 or n_neg == 0:
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        )

    w_pos = attack_ratio_per_batch / n_pos
    w_neg = (1.0 - attack_ratio_per_batch) / n_neg
    weights = np.where(labels == 1, w_pos, w_neg).astype(np.float64)

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(dataset),
        replacement=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
