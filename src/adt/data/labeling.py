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

def _generate_attacks_only(
    clean_windows_norm: np.ndarray,
    time_feat: np.ndarray,
    n_target: int,
    cfg: dict,
    scaler: StandardScalerND,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """clean_windows_norm 풀에서 n_target개 attack 윈도우를 생성.

    원본 normal 윈도우는 수정하지 않고, base를 복원 추출해 attack 주입.
    cross-split leakage 없음 — base pool은 같은 split의 normal만 사용.

    Returns:
        X_attack  : (n_target, T, C) float32
        tf_attack : (n_target, T, 2) float32  — base window의 time_feat
        type_lbls : (n_target,)      int32    — TYPE_IDX 값
    """
    rng = np.random.default_rng(seed)
    N, T, C = clean_windows_norm.shape

    base_idx  = rng.choice(N, size=n_target, replace=True)
    raw_pool  = scaler.inverse_transform(clean_windows_norm)

    atypes = cfg["anomaly_types"]
    probs  = np.array([a["prob"] for a in atypes], dtype=float)
    probs /= probs.sum()

    X_atk_raw = raw_pool[base_idx].copy()
    tf_atk    = time_feat[base_idx].copy()
    type_lbls = np.empty(n_target, dtype=np.int32)

    for i in range(n_target):
        ti    = int(rng.choice(len(atypes), p=probs))
        atype = atypes[ti]
        name  = atype["type"]
        ch    = int(rng.integers(0, C))
        w     = X_atk_raw[i]

        if name == "scale_down":
            w, _ = inject_scale_down(w, ch, atype["scale_factor"], atype["duration_steps"], rng)
        elif name == "ramp":
            w, _ = inject_ramp(w, ch, atype["scale_start"], atype["trough_scale"],
                                atype["duration_steps"], rng)
        elif name == "pulse_plateau":
            w, _ = inject_pulse_plateau(w, ch, atype["magnitude"], atype["duration_steps"], rng)
        elif name == "replay":
            w, _ = inject_replay(w, ch, atype["duration_steps"], raw_pool, rng)
        elif name == "instant_spike":
            w, _ = inject_instant_spike(w, ch, atype["magnitude"], atype["duration_steps"], rng)
        else:
            fb   = next((a for a in atypes if a["type"] == "scale_down"), atypes[0])
            w, _ = inject_scale_down(w, ch, fb.get("scale_factor", [0.5, 0.7]),
                                      fb.get("duration_steps", [4, 12]), rng)
            name = "scale_down"

        X_atk_raw[i] = w
        type_lbls[i]  = TYPE_IDX.get(name, 0)

    return scaler.transform(X_atk_raw).astype(np.float32), tf_atk.astype(np.float32), type_lbls


def _subsample_attacks(
    X: np.ndarray,
    tf: np.ndarray,
    binary_label: np.ndarray,
    type_label: np.ndarray,
    normal_to_attack_ratio: float = 9.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normal은 유지, attack만 stratified 다운샘플해 normal:attack ≈ ratio:1 맞춤.

    Args:
        normal_to_attack_ratio: 9.0 → 9:1 (공격이 전체의 ~10%)
    """
    rng             = np.random.default_rng(seed)
    n_normal        = int((binary_label == 0).sum())
    n_attack_target = max(1, round(n_normal / normal_to_attack_ratio))

    attack_idx = np.where(binary_label == 1)[0]
    if len(attack_idx) <= n_attack_target:
        return X, tf, binary_label, type_label

    attack_tl    = type_label[attack_idx]
    unique_types = np.unique(attack_tl)
    n_avail      = len(attack_idx)

    selected: list[int] = []
    remaining = n_attack_target
    for i, t in enumerate(unique_types):
        mask_t  = (attack_tl == t)
        count_t = int(mask_t.sum())
        if i == len(unique_types) - 1:
            n_from_t = remaining
        else:
            prop_t   = count_t / n_avail
            n_from_t = max(0, round(n_attack_target * prop_t))
            remaining -= n_from_t
        n_sel = min(n_from_t, count_t)
        if n_sel > 0:
            chosen = rng.choice(attack_idx[mask_t], size=n_sel, replace=False)
            selected.extend(chosen.tolist())

    normal_idx = np.where(binary_label == 0)[0]
    keep = np.sort(np.concatenate([normal_idx, np.array(selected, dtype=np.int64)]))
    return X[keep], tf[keep], binary_label[keep], type_label[keep]


def _save_split_data(
    out_dir: Path,
    X: np.ndarray,
    tf: np.ndarray,
    binary_label: np.ndarray,
    type_label: np.ndarray,
) -> dict:
    """4개 npy 파일 저장 후 통계 dict 반환."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy",            X.astype(np.float32))
    np.save(out_dir / "time_feat.npy",    tf.astype(np.float32))
    np.save(out_dir / "binary_label.npy", binary_label.astype(np.int32))
    np.save(out_dir / "type_label.npy",   type_label.astype(np.int32))
    n_normal = int((type_label == LABEL_NORMAL).sum())
    return {
        "total":   len(X),
        "normal":  n_normal,
        "attacks": {
            name: int((type_label == idx).sum())
            for name, idx in TYPE_IDX.items()
            if int((type_label == idx).sum()) > 0
        },
    }


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
    """50:50 balanced downstream folds 생성·저장.

    split별 저장 구조::
        {output_dir}/{fold}/train/      — 50:50 (train은 하나만)
        {output_dir}/{fold}/val_50_50/  — 50:50 native
        {output_dir}/{fold}/val_9_1/    — 9:1  subsampled (실전 대표)
        {output_dir}/{fold}/test_50_50/ — all_type 전용, 50:50
        {output_dir}/{fold}/test_9_1/   — all_type 전용, 9:1

    attack base window는 동일 split의 normal pool에서만 추출
    (서로 다른 pretrain subdirectory → cross-split leakage 없음).

    Returns:
        stats: { fold_name: { split_dir_name: { total, normal, attacks } } }
    """
    pretrain_dir = Path(pretrain_dir)
    output_dir   = Path(output_dir)
    base_seed    = cfg.get("seed", 42)
    fold_defs    = cfg.get("folds", _DEFAULT_FOLD_DEFS)

    needed_splits: set[str] = set()
    for fd in fold_defs:
        for s in fd.get("splits", []):
            needed_splits.add(s)

    _split_seed_offset = {"train": 0, "val": 1, "test": 2}

    # ── leakage 검증 출력 ─────────────────────────────────────────────────
    print("\n[generate_downstream_folds] Normal base window counts per split:")
    for split in sorted(needed_splits):
        n = len(np.load(pretrain_dir / split / "X.npy", mmap_mode="r"))
        print(f"  {split:6s}: {n:,} normal windows  (from {pretrain_dir / split})")
    print("  → Each split loaded from a distinct pretrain subdirectory: no cross-split overlap.\n")

    # ── Step 1: split별 50:50 superset 생성 ──────────────────────────────
    superset: dict[str, dict[str, np.ndarray]] = {}

    for split in sorted(needed_splits):
        split_seed = base_seed + _split_seed_offset.get(split, 3)
        split_dir  = pretrain_dir / split

        X_clean  = np.load(split_dir / "X.npy")
        tf_clean = np.load(split_dir / "time_feat.npy")
        n_normal = len(X_clean)

        # n_normal 개 attack 생성 → 50:50
        X_atk, tf_atk, tl_atk = _generate_attacks_only(
            X_clean, tf_clean, n_normal, cfg, scaler, seed=split_seed
        )

        tl_normal = np.full(n_normal, LABEL_NORMAL, dtype=np.int32)
        bl_normal  = np.zeros(n_normal, dtype=np.int32)
        bl_atk     = np.ones(n_normal,  dtype=np.int32)

        superset[split] = {
            "X":            np.concatenate([X_clean, X_atk],   axis=0),
            "time_feat":    np.concatenate([tf_clean, tf_atk],  axis=0),
            "binary_label": np.concatenate([bl_normal, bl_atk], axis=0),
            "type_label":   np.concatenate([tl_normal, tl_atk], axis=0),
        }

    # ── Step 2: fold별 필터링 + 저장 ──────────────────────────────────────
    stats: dict = {}

    for fd in fold_defs:
        fold_name   = fd["name"]
        unseen_type: str | None = fd.get("unseen_type")
        splits      = fd.get("splits", [])
        fold_stats: dict = {}

        for split in splits:
            data = superset[split]
            tl   = data["type_label"]

            # unseen_type 필터 (normal은 LABEL_NORMAL=-1이라 항상 keep)
            if unseen_type is not None:
                keep = tl != TYPE_IDX[unseen_type]
            else:
                keep = np.ones(len(tl), dtype=bool)

            X_f  = data["X"][keep]
            tf_f = data["time_feat"][keep]
            bl_f = data["binary_label"][keep]
            tl_f = data["type_label"][keep]

            if split == "train":
                out_dir = output_dir / fold_name / "train"
                fold_stats["train"] = _save_split_data(out_dir, X_f, tf_f, bl_f, tl_f)
            else:
                # 50:50 native view
                out_50 = output_dir / fold_name / f"{split}_50_50"
                fold_stats[f"{split}_50_50"] = _save_split_data(out_50, X_f, tf_f, bl_f, tl_f)

                # 9:1 subsampled view (attack만 다운샘플, normal 유지)
                X_9, tf_9, bl_9, tl_9 = _subsample_attacks(
                    X_f, tf_f, bl_f, tl_f,
                    normal_to_attack_ratio=9.0,
                    seed=base_seed + _split_seed_offset.get(split, 3) + 100,
                )
                out_9 = output_dir / fold_name / f"{split}_9_1"
                fold_stats[f"{split}_9_1"] = _save_split_data(out_9, X_9, tf_9, bl_9, tl_9)

        stats[fold_name] = fold_stats

        print(f"\n[fold={fold_name}]")
        for split_key, s in fold_stats.items():
            n_atk = sum(s["attacks"].values())
            ratio = n_atk / max(s["total"], 1)
            type_str = "  ".join(f"{t}:{c}" for t, c in sorted(s["attacks"].items()))
            print(
                f"  {split_key:14s}: total={s['total']:,}  "
                f"normal={s['normal']:,}  attack={n_atk:,}  "
                f"ratio={ratio:.1%}  [{type_str}]"
            )

    return stats
