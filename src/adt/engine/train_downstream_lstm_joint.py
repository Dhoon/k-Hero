"""LSTM Joint Multi-task Detection+Classification 학습 루프.

설계 원칙:
  - encoder: AE pretrain checkpoint 초기화 → 전체 unfreeze
  - loss = w_det * BCE(det_logit, y_bin)
          + w_cls * CE(cls_logit[attack], y_type[attack])
  - checkpoint 선택 기준: val_50_50 Detection AUC-ROC 최고
  - 저장: checkpoints/downstream_lstm/joint/{fold}/best.pt
          (encoder + det_head + cls_head 한 파일)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.adt.engine.train_downstream_transformer import (
    ALL_FOLDS,
    DownstreamFoldDataset,
    _remap_type_labels,
    compute_class_info,
    compute_pos_weight,
)
from src.adt.models.lstm_joint import LSTMJoint
from src.adt.utils.checkpoint import save_checkpoint
from src.adt.utils.logger import get_logger
from src.adt.utils.seed import set_seed


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
def _val_joint(
    model: LSTMJoint,
    loader: DataLoader,
    det_loss_fn: nn.Module,
    cls_loss_fn: nn.Module,
    type_to_class: dict[int, int],
    w_det: float,
    w_cls: float,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """val loop.

    Returns:
        (avg_det_loss, avg_cls_loss, total_loss, det_auc_roc)
    """
    model.eval()
    det_logits_all, labels_all = [], []
    det_sum = cls_sum = n_det = n_cls = 0

    for x, _, bl, tl in loader:
        x, bl, tl = x.to(device), bl.to(device), tl.to(device)
        det_logit, cls_logit = model(x)

        det_sum += det_loss_fn(det_logit, bl.float()).item()
        n_det   += 1

        attack_mask = bl == 1
        if attack_mask.any():
            cls_tgt  = _remap_type_labels(tl[attack_mask], type_to_class).to(device)
            cls_sum += cls_loss_fn(cls_logit[attack_mask], cls_tgt).item()
            n_cls   += 1

        det_logits_all.append(det_logit.cpu())
        labels_all.append(bl.cpu())

    avg_det = det_sum / max(n_det, 1)
    avg_cls = cls_sum / max(n_cls, 1) if n_cls > 0 else float("nan")
    total   = w_det * avg_det + (w_cls * avg_cls if not math.isnan(avg_cls) else 0.0)

    logits_np = torch.cat(det_logits_all).numpy()
    labels_np = torch.cat(labels_all).numpy().astype(int)
    try:
        auc = float(roc_auc_score(labels_np, logits_np))
    except Exception:
        auc = float("nan")

    return avg_det, avg_cls, total, auc


def train_fold_lstm_joint(
    fold_name: str,
    cfg: dict[str, Any],
    device: torch.device,
    verbose: bool = True,
) -> None:
    """단일 fold Joint 학습."""
    downstream_dir = Path(cfg["downstream_dir"])
    fold_dir       = downstream_dir / fold_name
    train_cfg      = cfg["training"]
    heads_cfg      = cfg["heads"]
    model_cfg      = cfg["model"]
    seed           = cfg.get("seed", 42)
    set_seed(seed)

    log_dir = Path(train_cfg.get("log_dir", "logs/downstream_lstm_joint")) / fold_name
    logger, writer = get_logger(log_dir, name=f"adt.joint.{fold_name}")

    data_cfg_path = Path(cfg.get("data_config", "configs/data/default.yaml"))
    with open(data_cfg_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    n_features = len(data_cfg["feature_cols"])
    hidden_dim = model_cfg["hidden_dim"]
    num_layers = model_cfg["num_layers"]

    ds_train = DownstreamFoldDataset(fold_dir / "train")
    ds_val   = DownstreamFoldDataset(fold_dir / "val_50_50")

    train_tl = ds_train.type_label.numpy()
    _, class_names, type_to_class = compute_class_info(train_tl)
    num_classes = len(class_names)
    pw_float    = compute_pos_weight(ds_train.binary_label.numpy())

    logger.info(
        f"fold={fold_name}  num_classes={num_classes}  "
        f"train={len(ds_train):,}  val={len(ds_val):,}  pos_weight={pw_float:.2f}"
    )

    batch_size = train_cfg["batch_size"]
    train_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    ckpt_dir = Path(train_cfg["ckpt_dir"]) / fold_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "class_names.json").write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    w_det = float(train_cfg.get("w_det", 1.0))
    w_cls = float(train_cfg.get("w_cls", 0.4))
    logger.info(f"w_det={w_det}  w_cls={w_cls}")

    model = LSTMJoint(
        n_features, hidden_dim, num_layers, num_classes,
        det_hidden=heads_cfg["hidden_dim"],
        cls_hidden=heads_cfg["hidden_dim"],
        dropout=heads_cfg["dropout"],
    ).to(device)

    pretrain_ckpt = cfg.get("pretrain_ckpt")
    if pretrain_ckpt and Path(pretrain_ckpt).exists():
        state = torch.load(pretrain_ckpt, map_location="cpu")
        model.encoder.load_state_dict(state["encoder"])
        logger.info(f"encoder init from {pretrain_ckpt}")
    else:
        logger.warning("pretrain_ckpt not found — random init")

    for p in model.parameters():
        p.requires_grad_(True)

    pw_tensor   = torch.tensor([pw_float], device=device)
    det_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    cls_loss_fn = nn.CrossEntropyLoss()

    epochs    = train_cfg["epochs"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = _cosine_lr(
        optimizer,
        total_steps=epochs * len(train_loader),
        warmup_steps=train_cfg.get("warmup_steps", 0),
    )

    best_det_auc = float("-inf")

    for epoch in range(epochs):
        model.train()
        for x, _, bl, tl in train_loader:
            x, bl, tl = x.to(device), bl.to(device), tl.to(device)
            det_logit, cls_logit = model(x)

            det_loss = det_loss_fn(det_logit, bl.float())

            attack_mask = bl == 1
            if attack_mask.any():
                cls_tgt  = _remap_type_labels(tl[attack_mask], type_to_class).to(device)
                cls_loss = cls_loss_fn(cls_logit[attack_mask], cls_tgt)
            else:
                cls_loss = torch.tensor(0.0, device=device)

            loss = w_det * det_loss + w_cls * cls_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        val_det, val_cls, val_total, val_auc = _val_joint(
            model, val_loader, det_loss_fn, cls_loss_fn,
            type_to_class, w_det, w_cls, device,
        )

        is_best = (not math.isnan(val_auc)) and (val_auc > best_det_auc)
        if is_best:
            best_det_auc = val_auc

        save_checkpoint(
            {
                "epoch":       epoch + 1,
                "encoder":     model.encoder.state_dict(),
                "det_head":    model.det_head.state_dict(),
                "cls_head":    model.cls_head.state_dict(),
                "num_classes": num_classes,
                "class_names": class_names,
                "val_auc":     val_auc,
                "val_det_loss": val_det,
                "val_cls_loss": val_cls,
                "optimizer":   optimizer.state_dict(),
            },
            ckpt_dir,
            is_best=is_best,
        )

        auc_str = f"{val_auc:.4f}" if not math.isnan(val_auc) else " nan "
        cls_str = f"{val_cls:.4f}" if not math.isnan(val_cls) else " nan "
        logger.info(
            f"  ep{epoch+1:3d}  "
            f"det_loss={val_det:.4f}  cls_loss={cls_str}  total={val_total:.4f}  "
            f"det_auc={auc_str}"
            + ("  [★]" if is_best else "")
        )
        writer.add_scalar("val/det_loss",   val_det,   epoch + 1)
        writer.add_scalar("val/total_loss", val_total, epoch + 1)
        if not math.isnan(val_cls):
            writer.add_scalar("val/cls_loss", val_cls, epoch + 1)
        if not math.isnan(val_auc):
            writer.add_scalar("val/det_auc", val_auc, epoch + 1)

    logger.info(f"done  best_det_auc={best_det_auc:.4f}")
    writer.close()


def train_all_folds_lstm_joint(
    cfg: dict[str, Any],
    device: torch.device,
    folds: list[str] | None = None,
    verbose: bool = True,
) -> None:
    targets = folds if folds is not None else ALL_FOLDS
    for fold in targets:
        if verbose:
            print(f"\n{'='*60}\n fold: {fold}\n{'='*60}")
        train_fold_lstm_joint(fold, cfg, device, verbose=verbose)
