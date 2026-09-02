"""Downstream Detection + Classification 통합 학습 루프.

설계 원칙:
  - frozen encoder → encoder forward 1회 → z_t 공유
  - detection head: 전체 샘플, BCEWithLogitsLoss(pos_weight)
  - classification head: Attack 샘플만, CrossEntropyLoss, z_t 재사용
  - 두 head는 별도 optimizer/loss/checkpoint — gradient 교차 없음
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.adt.data.labeling import LABEL_NORMAL, TYPE_IDX
from src.adt.models.encoder import TimeSeriesTransformerEncoder
from src.adt.models.heads.detection_head import DetectionHead
from src.adt.models.heads.classification_head import ClassificationHead
from src.adt.utils.checkpoint import load_encoder_frozen, save_checkpoint
from src.adt.utils.seed import set_seed

# ── 상수 ──────────────────────────────────────────────────────────────────
ALL_FOLDS: list[str] = [
    "all_type",
    "unseen_scale_down",
    "unseen_ramp",
    "unseen_pulse_plateau",
    "unseen_replay",
    "unseen_instant_spike",
]

# fold → unseen attack type 이름 (None = 제외 없음)
FOLD_UNSEEN_TYPE: dict[str, str | None] = {
    "all_type":             None,
    "unseen_scale_down":    "scale_down",
    "unseen_ramp":          "ramp",
    "unseen_pulse_plateau": "pulse_plateau",
    "unseen_replay":        "replay",
    "unseen_instant_spike": "instant_spike",
}

# TYPE_IDX 역방향 (int → type_name)
IDX_TO_TYPE: dict[int, str] = {v: k for k, v in TYPE_IDX.items()}


# =========================================================================
# Dataset
# =========================================================================

class DownstreamFoldDataset(Dataset):
    """(X, time_feat, binary_label, type_label) 4-tuple Dataset."""

    def __init__(self, fold_split_dir: str | Path) -> None:
        d = Path(fold_split_dir)
        self.X           = torch.from_numpy(np.load(d / "X.npy").astype(np.float32))
        self.time_feat   = torch.from_numpy(np.load(d / "time_feat.npy").astype(np.float32))
        self.binary_label = torch.from_numpy(np.load(d / "binary_label.npy").astype(np.int64))
        self.type_label  = torch.from_numpy(np.load(d / "type_label.npy").astype(np.int64))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            self.X[idx],
            self.time_feat[idx],
            self.binary_label[idx],
            self.type_label[idx],
        )


# =========================================================================
# 클래스 정보 계산
# =========================================================================

def compute_class_info(
    type_labels: np.ndarray,
) -> tuple[list[int], dict[str, str], dict[int, int]]:
    """학습 데이터의 type_label에서 분류 헤드 클래스 정보를 계산.

    Args:
        type_labels: (N,) int, LABEL_NORMAL(-1) 포함 가능

    Returns:
        present_orig_idxs : sorted list of original TYPE_IDX values (attack only)
        class_names       : {str(class_idx): type_name}  — JSON 저장용
        type_to_class     : {orig_type_idx: class_idx}   — 학습 시 type_label 변환용
    """
    present_orig_idxs = sorted(set(
        int(t) for t in type_labels if t >= 0
    ))
    class_names = {
        str(cls_idx): IDX_TO_TYPE[orig_idx]
        for cls_idx, orig_idx in enumerate(present_orig_idxs)
    }
    type_to_class = {
        orig_idx: cls_idx
        for cls_idx, orig_idx in enumerate(present_orig_idxs)
    }
    return present_orig_idxs, class_names, type_to_class


def compute_pos_weight(binary_labels: np.ndarray) -> float:
    """Normal/Attack 개수 비율로 BCEWithLogitsLoss pos_weight 계산.

    pos_weight = n_neg / n_pos  (class imbalance 보정)
    """
    n_pos = int((binary_labels == 1).sum())
    n_neg = int((binary_labels == 0).sum())
    return float(n_neg / max(n_pos, 1))


# =========================================================================
# 단일 배치 학습 스텝 (테스트 가능하도록 독립 함수로 분리)
# =========================================================================

def _remap_type_labels(
    tl: torch.Tensor, type_to_class: dict[int, int]
) -> torch.Tensor:
    """original TYPE_IDX → consecutive class_idx 변환."""
    return torch.tensor(
        [type_to_class[t.item()] for t in tl], dtype=torch.long, device=tl.device
    )


def run_step(
    encoder: nn.Module,
    det_head: nn.Module,
    cls_head: nn.Module,
    x: torch.Tensor,        # (B, T, C)
    tf: torch.Tensor,       # (B, T, 2)
    bl: torch.Tensor,       # (B,) long
    tl: torch.Tensor,       # (B,) long
    det_loss_fn: nn.Module,
    cls_loss_fn: nn.Module,
    type_to_class: dict[int, int],
    det_optim: torch.optim.Optimizer,
    cls_optim: torch.optim.Optimizer,
) -> dict[str, Any]:
    """단일 배치 학습 스텝.

    encoder forward 1회 → z_t 공유:
      - detection head: 전체 배치
      - classification head: attack 샘플만 (z_t slicing, re-encode 없음)

    Returns:
        dict with keys: z_t, det_loss, cls_loss, attack_mask, z_attack
    """
    # ── encoder forward (frozen) ──────────────────────────────────────────
    with torch.no_grad():
        z_t = encoder(x, tf)   # (B, T, d_model)

    # ── Detection ─────────────────────────────────────────────────────────
    det_logit = det_head(z_t)              # (B,)
    det_loss = det_loss_fn(det_logit, bl.float())
    det_optim.zero_grad()
    det_loss.backward()
    det_optim.step()

    # ── Classification (attack 샘플만, z_t 재사용) ─────────────────────────
    attack_mask = (bl == 1)               # (B,) bool
    z_attack = z_t[attack_mask]           # z_t slice — re-encode 없음
    cls_loss_val: float | None = None
    if attack_mask.any():
        tl_attack = tl[attack_mask]
        cls_tgt = _remap_type_labels(tl_attack, type_to_class)
        cls_logit = cls_head(z_attack)    # (M, num_classes)
        cls_loss = cls_loss_fn(cls_logit, cls_tgt)
        cls_loss_val = cls_loss.item()
        cls_optim.zero_grad()
        cls_loss.backward()
        cls_optim.step()

    return {
        "z_t":         z_t.detach(),
        "det_loss":    det_loss.item(),
        "cls_loss":    cls_loss_val,
        "attack_mask": attack_mask,
        "z_attack":    z_attack.detach(),
    }


# =========================================================================
# epoch 단위 학습 / 검증
# =========================================================================

def _train_epoch(
    encoder, det_head, cls_head, loader,
    det_loss_fn, cls_loss_fn, type_to_class,
    det_optim, cls_optim, device,
) -> dict[str, float]:
    det_head.train()
    cls_head.train()
    det_sum = cls_sum = cls_n = n = 0

    for x, tf, bl, tl in loader:
        x, tf, bl, tl = (
            x.to(device), tf.to(device), bl.to(device), tl.to(device)
        )
        result = run_step(
            encoder, det_head, cls_head,
            x, tf, bl, tl,
            det_loss_fn, cls_loss_fn, type_to_class,
            det_optim, cls_optim,
        )
        det_sum += result["det_loss"]
        if result["cls_loss"] is not None:
            cls_sum += result["cls_loss"]
            cls_n += 1
        n += 1

    return {
        "det": det_sum / max(n, 1),
        "cls": cls_sum / max(cls_n, 1) if cls_n > 0 else float("nan"),
    }


@torch.no_grad()
def _val_epoch(
    encoder, det_head, cls_head, loader,
    det_loss_fn, cls_loss_fn, type_to_class, device,
) -> dict[str, float]:
    det_head.eval()
    cls_head.eval()
    det_sum = cls_sum = cls_n = n = 0

    for x, tf, bl, tl in loader:
        x, tf, bl, tl = (
            x.to(device), tf.to(device), bl.to(device), tl.to(device)
        )
        z_t = encoder(x, tf)

        det_logit = det_head(z_t)
        det_sum += det_loss_fn(det_logit, bl.float()).item()

        attack_mask = (bl == 1)
        if attack_mask.any():
            tl_attack = tl[attack_mask]
            cls_tgt = _remap_type_labels(tl_attack, type_to_class).to(device)
            cls_logit = cls_head(z_t[attack_mask])
            cls_sum += cls_loss_fn(cls_logit, cls_tgt).item()
            cls_n += 1
        n += 1

    return {
        "det": det_sum / max(n, 1),
        "cls": cls_sum / max(cls_n, 1) if cls_n > 0 else float("nan"),
    }


# =========================================================================
# fold 단위 학습
# =========================================================================

def _cosine_lr(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 0,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_fold(
    fold_name: str,
    cfg: dict[str, Any],
    encoder: nn.Module,
    device: torch.device,
    verbose: bool = True,
) -> None:
    """단일 fold의 detection + classification 헤드를 학습.

    Args:
        fold_name: e.g. "all_type", "unseen_scale_down"
        cfg      : downstream default.yaml 전체 dict
        encoder  : frozen encoder (load_encoder_frozen 이미 완료)
        device   : torch.device
    """
    downstream_dir = Path(cfg["downstream_dir"])
    fold_dir = downstream_dir / fold_name

    det_cfg = cfg["detection"]
    cls_cfg = cfg["classification"]
    model_cfg = cfg["model"]
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # ── 데이터 로드 ─────────────────────────────────────────────────────
    ds_train = DownstreamFoldDataset(fold_dir / "train")
    ds_val   = DownstreamFoldDataset(fold_dir / "val")

    train_tl = ds_train.type_label.numpy()
    _, class_names, type_to_class = compute_class_info(train_tl)
    num_classes = len(class_names)
    pw_float = compute_pos_weight(ds_train.binary_label.numpy())

    if verbose:
        print(
            f"[train] fold={fold_name}  num_classes={num_classes}  "
            f"train={len(ds_train):,}  val={len(ds_val):,}  "
            f"pos_weight={pw_float:.2f}"
        )

    train_loader = DataLoader(
        ds_train, batch_size=det_cfg["batch_size"], shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        ds_val, batch_size=det_cfg["batch_size"], shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )

    # ── 모델 초기화 ──────────────────────────────────────────────────────
    d_model = model_cfg["d_model"]
    det_head = DetectionHead(
        d_model=d_model,
        hidden_dim=det_cfg["hidden_dim"],
        dropout=det_cfg["dropout"],
    ).to(device)
    cls_head = ClassificationHead(
        d_model=d_model,
        num_classes=num_classes,
        hidden_dim=cls_cfg["hidden_dim"],
        dropout=cls_cfg["dropout"],
    ).to(device)

    # ── Loss ─────────────────────────────────────────────────────────────
    pw_tensor = torch.tensor([pw_float], device=device)
    det_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    cls_loss_fn = nn.CrossEntropyLoss()

    # ── Optimizer ────────────────────────────────────────────────────────
    det_optim = torch.optim.AdamW(
        det_head.parameters(),
        lr=det_cfg["lr"], weight_decay=det_cfg["weight_decay"],
    )
    cls_optim = torch.optim.AdamW(
        cls_head.parameters(),
        lr=cls_cfg["lr"], weight_decay=cls_cfg["weight_decay"],
    )

    det_epochs = det_cfg["epochs"]
    cls_epochs = cls_cfg["epochs"]
    epochs = max(det_epochs, cls_epochs)

    det_steps = det_epochs * len(train_loader)
    cls_steps = cls_epochs * len(train_loader)
    det_sched = _cosine_lr(det_optim, det_steps, det_cfg.get("warmup_steps", 0))
    cls_sched = _cosine_lr(cls_optim, cls_steps, cls_cfg.get("warmup_steps", 0))

    # ── 학습 루프 ────────────────────────────────────────────────────────
    det_ckpt_dir = Path(det_cfg["ckpt_dir"]) / fold_name / "detector"
    cls_ckpt_dir = Path(cls_cfg["ckpt_dir"]) / fold_name / "classifier"
    det_ckpt_dir.mkdir(parents=True, exist_ok=True)
    cls_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # class_names.json 저장 (classifier ckpt 디렉토리)
    class_names_path = cls_ckpt_dir / "class_names.json"
    class_names_path.write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    best_det_val = float("inf")
    best_cls_val = float("inf")

    for epoch in range(epochs):
        train_losses = _train_epoch(
            encoder, det_head, cls_head, train_loader,
            det_loss_fn, cls_loss_fn, type_to_class,
            det_optim, cls_optim, device,
        )
        val_losses = _val_epoch(
            encoder, det_head, cls_head, val_loader,
            det_loss_fn, cls_loss_fn, type_to_class, device,
        )

        if epoch < det_epochs:
            det_sched.step()
        if epoch < cls_epochs:
            cls_sched.step()

        det_is_best = val_losses["det"] < best_det_val
        cls_is_best = (
            not math.isnan(val_losses["cls"])
            and val_losses["cls"] < best_cls_val
        )
        if det_is_best:
            best_det_val = val_losses["det"]
        if cls_is_best:
            best_cls_val = val_losses["cls"]

        save_checkpoint(
            {"epoch": epoch + 1, "head": det_head.state_dict(),
             "optimizer": det_optim.state_dict(), "val_loss": val_losses["det"]},
            det_ckpt_dir, is_best=det_is_best,
        )
        save_checkpoint(
            {"epoch": epoch + 1, "head": cls_head.state_dict(),
             "num_classes": num_classes, "class_names": class_names,
             "optimizer": cls_optim.state_dict(), "val_loss": val_losses["cls"]},
            cls_ckpt_dir, is_best=cls_is_best,
        )

        if verbose:
            print(
                f"  ep{epoch+1:3d}  "
                f"train det={train_losses['det']:.4f} cls={train_losses['cls']:.4f}  "
                f"val det={val_losses['det']:.4f} cls={val_losses['cls']:.4f}"
                + ("  [det★]" if det_is_best else "")
                + ("  [cls★]" if cls_is_best else "")
            )

    if verbose:
        print(
            f"[train] fold={fold_name} done  "
            f"best val det={best_det_val:.4f}  cls={best_cls_val:.4f}"
        )


def train_all_folds(
    cfg: dict[str, Any],
    encoder: nn.Module,
    device: torch.device,
    folds: list[str] | None = None,
    verbose: bool = True,
) -> None:
    """지정된 fold 목록을 순차 학습. folds=None이면 ALL_FOLDS 전부."""
    targets = folds if folds is not None else ALL_FOLDS
    for fold in targets:
        if verbose:
            print(f"\n{'='*60}\n fold: {fold}\n{'='*60}")
        train_fold(fold, cfg, encoder, device, verbose=verbose)
