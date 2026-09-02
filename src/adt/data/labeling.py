"""분류/탐지 학습용 Dataset + downstream fold 생성 오케스트레이터.

# 기존 API (build_classification_dataset)
  split별 역할:
    train/val : known_types(4종)만 주입 → 학습/검증
    test      : held_out_type(1종)만 주입 → 일반화 성능 평가

# 새 API (generate_downstream_folds)
  all_type superset 한 번 생성 후 unseen_X fold는 필터링만 적용:
    - all_type   : Normal + 5종 전부, train/val/test
    - unseen_X   : Normal + 4종(X 제외),  train/val
  → fold 간 공유 샘플이 바이트 단위로 동일하게 유지됨
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.adt.data.attack_injection import (
    inject_instant_spike,
    inject_pulse_plateau,
    inject_ramp,
    inject_replay,
    inject_scale_down,
    inject_synthetic_anomalies,
)
from src.adt.data.scalers import StandardScalerND

# ── type label 상수 ────────────────────────────────────────────────────────
LABEL_NORMAL = -1  # Normal 윈도우의 type_label 값

TYPE_IDX: dict[str, int] = {
    "scale_down":    0,
    "ramp":          1,
    "pulse_plateau": 2,
    "replay":        3,
    "instant_spike": 4,
}

# generate_downstream_folds 의 fold 기본값 (yaml에서 오버라이드 가능)
_DEFAULT_FOLD_DEFS: list[dict] = [
    {"name": "all_type",             "unseen_type": None,            "splits": ["train", "val", "test"]},
    {"name": "unseen_scale_down",    "unseen_type": "scale_down",    "splits": ["train", "val"]},
    {"name": "unseen_ramp",          "unseen_type": "ramp",          "splits": ["train", "val"]},
    {"name": "unseen_pulse_plateau", "unseen_type": "pulse_plateau", "splits": ["train", "val"]},
    {"name": "unseen_replay",        "unseen_type": "replay",        "splits": ["train", "val"]},
    {"name": "unseen_instant_spike", "unseen_type": "instant_spike", "splits": ["train", "val"]},
]


# =========================================================================
# Dataset (기존 API)
# =========================================================================

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

    @property
    def label_array(self) -> np.ndarray:
        return self.labels.numpy()


# =========================================================================
# 내부 헬퍼 (기존)
# =========================================================================

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


# =========================================================================
# 팩토리 (기존 API — 하위 호환)
# =========================================================================

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


# =========================================================================
# 새 API — type_label 포함 주입 + fold 오케스트레이션
# =========================================================================

def _inject_with_type_labels(
    clean_windows_norm: np.ndarray,
    cfg: dict,
    scaler: StandardScalerND,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """attack 주입 후 type_label(-1=Normal, 0..4=attack 종류)을 함께 반환.

    Returns:
        corrupted_norm : (N, T, C) float32 — 정규화된 주입 결과
        type_labels    : (N,)  int32       — LABEL_NORMAL(-1) 또는 TYPE_IDX[type]
    """
    rng = np.random.default_rng(seed)
    N, T, C = clean_windows_norm.shape

    raw_windows = scaler.inverse_transform(clean_windows_norm)
    n_inject = max(1, int(N * cfg["injection_ratio"]))
    inject_indices = rng.choice(N, size=n_inject, replace=False)

    atypes = cfg["anomaly_types"]
    probs = np.array([a["prob"] for a in atypes], dtype=float)
    probs /= probs.sum()

    corrupted_raw = raw_windows.copy()
    type_labels = np.full(N, LABEL_NORMAL, dtype=np.int32)

    for idx in inject_indices:
        ti = int(rng.choice(len(atypes), p=probs))
        atype = atypes[ti]
        name = atype["type"]
        ch = int(rng.integers(0, C))
        w = corrupted_raw[idx]  # (T, C)

        if name == "scale_down":
            w, _ = inject_scale_down(
                w, ch, atype["scale_factor"], atype["duration_steps"], rng
            )
        elif name == "ramp":
            w, _ = inject_ramp(
                w, ch, atype["scale_start"], atype["trough_scale"],
                atype["duration_steps"], rng,
            )
        elif name == "pulse_plateau":
            w, _ = inject_pulse_plateau(
                w, ch, atype["magnitude"], atype["duration_steps"], rng
            )
        elif name == "replay":
            w, _ = inject_replay(
                w, ch, atype["duration_steps"], raw_windows, rng
            )
        elif name == "instant_spike":
            w, _ = inject_instant_spike(
                w, ch, atype["magnitude"], atype["duration_steps"], rng
            )
        else:
            continue  # 알 수 없는 타입은 건너뜀

        corrupted_raw[idx] = w
        type_labels[idx] = TYPE_IDX.get(name, LABEL_NORMAL)

    corrupted_norm = scaler.transform(corrupted_raw)
    return corrupted_norm, type_labels


def generate_downstream_folds(
    pretrain_dir: str | Path,
    output_dir: str | Path,
    cfg: dict,
    scaler: StandardScalerND,
) -> dict:
    """all_type superset → unseen_X 필터링으로 6개 fold를 한 번에 생성·저장.

    저장 구조::
        {output_dir}/{fold_name}/{split}/
            X.npy            (N, T, C) float32
            time_feat.npy    (N, T, 2) float32
            binary_label.npy (N,)      int32  — 0=Normal, 1=Attack
            type_label.npy   (N,)      int32  — LABEL_NORMAL(-1) or TYPE_IDX

    Returns:
        stats: { fold_name: { split: { total, normal, attacks: {type: count} } } }
    """
    pretrain_dir = Path(pretrain_dir)
    output_dir = Path(output_dir)
    base_seed = cfg.get("seed", 42)
    fold_defs = cfg.get("folds", _DEFAULT_FOLD_DEFS)

    # 모든 fold에서 필요한 split 집합
    needed_splits: set[str] = set()
    for fd in fold_defs:
        for s in fd.get("splits", []):
            needed_splits.add(s)

    # ── Step 1: split별 all_type superset 생성 ────────────────────────────
    # train → seed+0, val → seed+1, test → seed+2 로 분리
    _split_seed_offset = {"train": 0, "val": 1, "test": 2}
    superset: dict[str, dict[str, np.ndarray]] = {}

    for split in sorted(needed_splits):
        split_seed = base_seed + _split_seed_offset.get(split, 3)
        split_dir = pretrain_dir / split

        X_raw = np.load(split_dir / "X.npy")
        tf = np.load(split_dir / "time_feat.npy")

        corrupted, type_labels = _inject_with_type_labels(
            X_raw, cfg, scaler, seed=split_seed
        )
        binary_labels = (type_labels >= 0).astype(np.int32)

        superset[split] = {
            "X":            corrupted,
            "time_feat":    tf,
            "binary_label": binary_labels,
            "type_label":   type_labels,
        }

    # ── Step 2: fold별 필터링 + 저장 ─────────────────────────────────────
    stats: dict = {}

    for fd in fold_defs:
        fold_name = fd["name"]
        unseen_type: str | None = fd.get("unseen_type")
        splits = fd.get("splits", [])

        fold_stats: dict = {}
        for split in splits:
            data = superset[split]

            if unseen_type is not None:
                unseen_idx = TYPE_IDX[unseen_type]
                keep = data["type_label"] != unseen_idx
            else:
                keep = np.ones(len(data["X"]), dtype=bool)

            X_f  = data["X"][keep]
            tf_f = data["time_feat"][keep]
            bl_f = data["binary_label"][keep]
            tl_f = data["type_label"][keep]

            out_dir = output_dir / fold_name / split
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "X.npy",            X_f.astype(np.float32))
            np.save(out_dir / "time_feat.npy",    tf_f.astype(np.float32))
            np.save(out_dir / "binary_label.npy", bl_f.astype(np.int32))
            np.save(out_dir / "type_label.npy",   tl_f.astype(np.int32))

            n_normal = int((tl_f == LABEL_NORMAL).sum())
            attack_counts = {
                name: int((tl_f == idx).sum())
                for name, idx in TYPE_IDX.items()
                if int((tl_f == idx).sum()) > 0
            }
            fold_stats[split] = {
                "total":   len(X_f),
                "normal":  n_normal,
                "attacks": attack_counts,
            }

        stats[fold_name] = fold_stats

    return stats
