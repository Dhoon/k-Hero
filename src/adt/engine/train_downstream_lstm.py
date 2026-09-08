"""LSTM Downstream Detection + Classification 학습 루프.

설계 원칙:
  Phase 1 — Detection : encoder 완전 unfreeze + det_head 공동 학습.
                        best checkpoint 기준: val_50_50 AUC-ROC.
                        best 시 encoder_finetuned.pt 저장.
  Phase 2 — Classification : Phase 1 best encoder 로드 → 완전 freeze,
                              cls_head만 학습.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.adt.engine.train_downstream_transformer import (
    ALL_FOLDS,
    FOLD_UNSEEN_TYPE,
    IDX_TO_TYPE,
    DownstreamFoldDataset,
    _remap_type_labels,
    compute_class_info,
    compute_pos_weight,
)
from src.adt.models.lstm_ae import (
    LSTMClassificationHead,
    LSTMDetectionHead,
    LSTMEncoder,
)
from src.adt.utils.checkpoint import save_checkpoint
from src.adt.utils.logger import get_logger
from src.adt.utils.seed import set_seed


class _FocalLoss(nn.Module):
    """Binary Focal Loss (logit input, reduction=mean)."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.5) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs   = torch.sigmoid(logits)
        pt      = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()


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


@torch.no_grad()
def _val_det(
    encoder: nn.Module,
    det_head: nn.Module,
    loader: DataLoader,
    det_loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """val loss + AUC-ROC 반환. AUC가 불가능하면 nan."""
    encoder.eval()
    det_head.eval()
    logits_list, labels_list = [], []
    tot = n = 0
    for x, _, bl, _ in loader:
        x, bl = x.to(device), bl.to(device)
        _, z  = encoder(x)
        logit = det_head(z)
        tot  += det_loss_fn(logit, bl.float()).item()
        logits_list.append(logit.cpu())
        labels_list.append(bl.cpu())
        n += 1
    logits_np = torch.cat(logits_list).numpy()
    labels_np = torch.cat(labels_list).numpy().astype(int)
    try:
        auc = float(roc_auc_score(labels_np, logits_np))
    except Exception:
        auc = float("nan")
    return tot / max(n, 1), auc


@torch.no_grad()
def _val_cls(
    encoder: nn.Module,
    cls_head: nn.Module,
    loader: DataLoader,
    cls_loss_fn: nn.Module,
    type_to_class: dict[int, int],
    device: torch.device,
) -> float:
    encoder.eval()
    cls_head.eval()
    tot = cls_n = 0
    for x, _, bl, tl in loader:
        x, bl, tl = x.to(device), bl.to(device), tl.to(device)
        _, z  = encoder(x)
        attack_mask = bl == 1
        if not attack_mask.any():
            continue
        cls_tgt   = _remap_type_labels(tl[attack_mask], type_to_class).to(device)
        cls_logit = cls_head(z[attack_mask])
        tot      += cls_loss_fn(cls_logit, cls_tgt).item()
        cls_n    += 1
    return tot / max(cls_n, 1) if cls_n > 0 else float("nan")


def train_fold_lstm(
    fold_name: str,
    cfg: dict[str, Any],
    device: torch.device,
    verbose: bool = True,
) -> None:
    """단일 fold의 LSTM detection + classification 헤드를 학습."""
    downstream_dir = Path(cfg["downstream_dir"])
    fold_dir  = downstream_dir / fold_name
    det_cfg   = cfg["detection"]
    cls_cfg   = cfg["classification"]
    model_cfg = cfg["model"]
    seed      = cfg.get("seed", 42)
    set_seed(seed)

    log_dir = Path(det_cfg.get("log_dir", "logs/downstream_lstm")) / fold_name
    logger, writer = get_logger(log_dir, name=f"adt.downstream_lstm.{fold_name}")

    data_cfg_path = Path(cfg.get("data_config", "configs/data/default.yaml"))
    with open(data_cfg_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    n_features     = len(data_cfg["feature_cols"])
    hidden_dim     = model_cfg["hidden_dim"]
    num_layers     = model_cfg["num_layers"]
    bottleneck_dim = 2 * hidden_dim

    ds_train = DownstreamFoldDataset(fold_dir / "train")
    ds_val   = DownstreamFoldDataset(fold_dir / "val_50_50")  # AUC-ROC 기준

    train_tl = ds_train.type_label.numpy()
    _, class_names, type_to_class = compute_class_info(train_tl)
    num_classes = len(class_names)
    pw_float    = compute_pos_weight(ds_train.binary_label.numpy())

    logger.info(
        f"fold={fold_name}  num_classes={num_classes}  "
        f"train={len(ds_train):,}  val={len(ds_val):,}  "
        f"pos_weight={pw_float:.2f}"
    )

    train_loader = DataLoader(
        ds_train, batch_size=det_cfg["batch_size"], shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        ds_val, batch_size=det_cfg["batch_size"], shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    loss_type    = det_cfg.get("loss_type", "bce")
    encoder_mode = det_cfg.get("encoder_mode", "unfreeze")
    mode_tag = f"{loss_type}_{encoder_mode}"

    det_ckpt_dir = Path(det_cfg["ckpt_dir"]) / mode_tag / fold_name / "detector"
    cls_ckpt_dir = Path(cls_cfg["ckpt_dir"]) / mode_tag / fold_name / "classifier"
    det_ckpt_dir.mkdir(parents=True, exist_ok=True)
    cls_ckpt_dir.mkdir(parents=True, exist_ok=True)
    (cls_ckpt_dir / "class_names.json").write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if loss_type == "focal":
        det_loss_fn: nn.Module = _FocalLoss(
            gamma=det_cfg.get("focal_gamma", 2.0),
            alpha=det_cfg.get("focal_alpha", 0.5),
        )
    else:
        pw_tensor   = torch.tensor([pw_float], device=device)
        det_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    cls_loss_fn = nn.CrossEntropyLoss()

    logger.info(f"loss_type={loss_type}  encoder_mode={encoder_mode}")

    # ── Phase 1: Detection ────────────────────────────────────────────────
    encoder  = LSTMEncoder(n_features, hidden_dim, num_layers).to(device)
    det_head = LSTMDetectionHead(
        bottleneck_dim, det_cfg["hidden_dim"], det_cfg["dropout"]
    ).to(device)

    pretrain_ckpt = cfg.get("pretrain_ckpt")
    if pretrain_ckpt and Path(pretrain_ckpt).exists():
        state = torch.load(pretrain_ckpt, map_location="cpu")
        encoder.load_state_dict(state["encoder"])
        logger.info(f"encoder loaded from {pretrain_ckpt}")
    else:
        logger.warning("pretrain_ckpt not found — random init")

    for p in encoder.parameters():
        p.requires_grad_(encoder_mode == "unfreeze")

    if encoder_mode == "unfreeze":
        det_params = list(encoder.parameters()) + list(det_head.parameters())
    else:
        det_params = list(det_head.parameters())
    det_optim  = torch.optim.AdamW(
        det_params, lr=det_cfg["lr"], weight_decay=det_cfg["weight_decay"]
    )
    det_epochs = det_cfg["epochs"]
    det_sched  = _cosine_lr(
        det_optim,
        total_steps=det_epochs * len(train_loader),
        warmup_steps=det_cfg.get("warmup_steps", 0),
    )

    best_det_auc = float("-inf")

    for epoch in range(det_epochs):
        encoder.train()
        det_head.train()
        for x, _, bl, _ in train_loader:
            x, bl = x.to(device), bl.to(device)
            _, z  = encoder(x)
            logit = det_head(z)
            loss  = det_loss_fn(logit, bl.float())
            det_optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(det_params, 1.0)
            det_optim.step()
            det_sched.step()

        val_loss, val_auc = _val_det(encoder, det_head, val_loader, det_loss_fn, device)
        is_best = (not math.isnan(val_auc)) and (val_auc > best_det_auc)
        if is_best:
            best_det_auc = val_auc

        save_checkpoint(
            {
                "epoch":     epoch + 1,
                "encoder":   encoder.state_dict(),
                "head":      det_head.state_dict(),
                "optimizer": det_optim.state_dict(),
                "val_auc":   val_auc,
                "val_loss":  val_loss,
            },
            det_ckpt_dir,
            is_best=is_best,
        )

        auc_str = f"{val_auc:.4f}" if not math.isnan(val_auc) else " nan "
        logger.info(
            f"  [det] ep{epoch+1:3d}  "
            f"val_loss={val_loss:.4f}  auc={auc_str}"
            + ("  [★]" if is_best else "")
        )
        writer.add_scalar("det/val_loss", val_loss, epoch + 1)
        if not math.isnan(val_auc):
            writer.add_scalar("det/val_auc", val_auc, epoch + 1)

    logger.info(f"[det] done  best_auc={best_det_auc:.4f}")

    # ── Phase 2: Classification (Phase 1 best encoder freeze) ─────────────
    best_enc_path = det_ckpt_dir / "best.pt"
    if best_enc_path.exists():
        encoder.load_state_dict(
            torch.load(best_enc_path, map_location="cpu")["encoder"]
        )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    cls_head  = LSTMClassificationHead(
        bottleneck_dim, num_classes, cls_cfg["hidden_dim"], cls_cfg["dropout"]
    ).to(device)
    cls_optim = torch.optim.AdamW(
        cls_head.parameters(),
        lr=cls_cfg["lr"],
        weight_decay=cls_cfg["weight_decay"],
    )
    cls_epochs = cls_cfg["epochs"]
    cls_sched  = _cosine_lr(
        cls_optim,
        total_steps=cls_epochs * len(train_loader),
        warmup_steps=cls_cfg.get("warmup_steps", 0),
    )
    best_cls_val = float("inf")

    for epoch in range(cls_epochs):
        cls_head.train()
        for x, _, bl, tl in train_loader:
            x, bl, tl = x.to(device), bl.to(device), tl.to(device)
            with torch.no_grad():
                _, z = encoder(x)
            attack_mask = bl == 1
            if not attack_mask.any():
                continue
            cls_tgt   = _remap_type_labels(tl[attack_mask], type_to_class).to(device)
            cls_logit = cls_head(z[attack_mask])
            loss      = cls_loss_fn(cls_logit, cls_tgt)
            cls_optim.zero_grad(set_to_none=True)
            loss.backward()
            cls_optim.step()
            cls_sched.step()

        val_cls = _val_cls(encoder, cls_head, val_loader, cls_loss_fn, type_to_class, device)
        is_best = (not math.isnan(val_cls)) and (val_cls < best_cls_val)
        if is_best:
            best_cls_val = val_cls

        save_checkpoint(
            {
                "epoch":       epoch + 1,
                "head":        cls_head.state_dict(),
                "num_classes": num_classes,
                "class_names": class_names,
                "optimizer":   cls_optim.state_dict(),
                "val_loss":    val_cls,
            },
            cls_ckpt_dir,
            is_best=is_best,
        )

        logger.info(
            f"  [cls] ep{epoch+1:3d}  val={val_cls:.4f}"
            + ("  [★]" if is_best else "")
        )
        if not math.isnan(val_cls):
            writer.add_scalar("cls/val_loss", val_cls, epoch + 1)

    logger.info(f"[cls] done  best_val={best_cls_val:.4f}")
    writer.close()


def train_all_folds_lstm(
    cfg: dict[str, Any],
    device: torch.device,
    folds: list[str] | None = None,
    verbose: bool = True,
) -> None:
    targets = folds if folds is not None else ALL_FOLDS
    for fold in targets:
        if verbose:
            print(f"\n{'='*60}\n fold: {fold}\n{'='*60}")
        train_fold_lstm(fold, cfg, device, verbose=verbose)
