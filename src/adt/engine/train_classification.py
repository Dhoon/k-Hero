"""Classification 다운스트림 학습 루프 (Exp A / Exp B).

Exp A: encoder random init + ClassificationHead (baseline)
Exp B: SSL pretrain encoder + ClassificationHead (우리 방법)

학습 설정:
  - BCEWithLogitsLoss
  - WeightedRandomSampler (attack_ratio_per_batch 유지)
  - val AUROC 최대 시 best.pt 저장
  - Leave-one-attack-type-out: train/val → known_types, test → held_out_type
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.adt.data.classification_dataset import (
    build_classification_dataset,
    build_classification_dataloader,
)
from src.adt.data.scalers import StandardScalerND
from src.adt.models.encoder import TimeSeriesTransformerEncoder
from src.adt.models.heads.classification_head import ClassificationHead
from src.adt.utils.checkpoint import save_checkpoint
from src.adt.utils.logger import get_logger
from src.adt.utils.seed import set_seed


# -------------------------------------------------------------------------
# LR 스케줄러
# -------------------------------------------------------------------------

def _cosine_lr(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 0,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# -------------------------------------------------------------------------
# 단일 epoch
# -------------------------------------------------------------------------

def _train_epoch(
    encoder: nn.Module,
    head: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
) -> float:
    encoder.train()
    head.train()
    total_loss = 0.0
    trainable = [p for p in list(encoder.parameters()) + list(head.parameters()) if p.requires_grad]

    for x, time_feat, labels in tqdm(loader, desc=f"  cls ep{epoch+1}", leave=False, ncols=80):
        x = x.to(device)
        time_feat = time_feat.to(device)
        labels = labels.to(device)

        H = encoder(x, time_feat, mask=None)
        logits = head(H)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def _val_epoch(
    encoder: nn.Module,
    head: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """val loop → (loss, auroc)."""
    encoder.eval()
    head.eval()
    total_loss = 0.0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for x, time_feat, labels in loader:
        x = x.to(device)
        time_feat = time_feat.to(device)
        labels = labels.to(device)
        H = encoder(x, time_feat, mask=None)
        logits = head(H)
        total_loss += criterion(logits, labels).item()
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels)
    scores_np = 1.0 / (1.0 + np.exp(-logits_np))  # sigmoid

    try:
        auroc = float(roc_auc_score(labels_np, scores_np))
    except ValueError:
        auroc = 0.0

    return total_loss / len(loader), auroc


# -------------------------------------------------------------------------
# 메인 진입점
# -------------------------------------------------------------------------

def run(cfg: dict[str, Any], max_epochs: int | None = None) -> None:
    """Classification 학습 루프.

    Args:
        cfg       : classification yaml 전체 dict
        max_epochs: 덮어쓰기용 (smoke test 등)
    """
    train_cfg: dict = cfg["train"]
    head_cfg: dict = cfg.get("head", {})

    data_cfg_path = Path(cfg.get("data_config", "configs/data/default.yaml"))
    with open(data_cfg_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    processed_dir = Path(data_cfg["processed_dir"])
    n_features = len(data_cfg["feature_cols"])
    window_size = data_cfg.get("window_size", 96)
    epochs = max_epochs if max_epochs is not None else train_cfg["epochs"]
    freeze_encoder: bool = cfg.get("freeze_encoder", False)
    encoder_ckpt: str | None = cfg.get("encoder_ckpt", None)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger, writer = get_logger(train_cfg["log_dir"])
    logger.info(f"device={device}  epochs={epochs}  n_features={n_features}")
    logger.info(f"encoder_ckpt={encoder_ckpt}  freeze_encoder={freeze_encoder}")

    # --- 스케일러 ---
    scaler = StandardScalerND.load(processed_dir / "scaler.joblib")

    # --- attack split 설정 ---
    attack_cfg: dict = cfg["attack_split"]

    # --- Dataset / DataLoader ---
    train_ds = build_classification_dataset(
        processed_dir, "train", attack_cfg, scaler,
        window_size=window_size, seed=cfg.get("seed", 42),
    )
    val_ds = build_classification_dataset(
        processed_dir, "val", attack_cfg, scaler,
        window_size=window_size, seed=cfg.get("seed", 42) + 1,
    )

    train_loader = build_classification_dataloader(
        train_ds, train_cfg["batch_size"],
        attack_ratio_per_batch=train_cfg.get("attack_ratio_per_batch", 0.3),
        shuffle=True,
    )
    val_loader = build_classification_dataloader(
        val_ds, train_cfg["batch_size"],
        attack_ratio_per_batch=0.5,  # val은 균형 샘플링으로 AUROC 안정성
        shuffle=False,
    )
    logger.info(f"train batches={len(train_loader)}  val batches={len(val_loader)}")

    # --- Encoder 초기화 ---
    if encoder_ckpt is not None:
        pretrain_state = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
        pretrain_model_cfg = pretrain_state.get("cfg", {}).get("model", {})
        d_model = pretrain_model_cfg.get("d_model", 128)
        n_heads = pretrain_model_cfg.get("n_heads", 4)
        n_layers = pretrain_model_cfg.get("n_layers", 4)
        d_ff = pretrain_model_cfg.get("d_ff", 256)
        dropout = pretrain_model_cfg.get("dropout", 0.1)
    else:
        # Exp A: random init → model cfg는 head_cfg 또는 기본값
        model_cfg = cfg.get("model", {})
        d_model = model_cfg.get("d_model", 128)
        n_heads = model_cfg.get("n_heads", 4)
        n_layers = model_cfg.get("n_layers", 4)
        d_ff = model_cfg.get("d_ff", 256)
        dropout = model_cfg.get("dropout", 0.1)
        pretrain_state = None

    encoder = TimeSeriesTransformerEncoder(
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
    ).to(device)

    if pretrain_state is not None:
        encoder.load_state_dict(pretrain_state["encoder"])
        logger.info(f"encoder 로드 완료: {encoder_ckpt}")
    else:
        logger.info("encoder random init (Exp A)")

    if freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
        logger.info("encoder 고정 (head만 학습)")

    # --- Classification Head ---
    head = ClassificationHead(
        d_model=d_model,
        hidden_dim=head_cfg.get("hidden_dim", 64),
        dropout=head_cfg.get("dropout", 0.1),
        pooling=head_cfg.get("pooling", "mean_max"),
    ).to(device)

    n_enc = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in head.parameters() if p.requires_grad)
    logger.info(f"학습 파라미터: encoder={n_enc:,}  head={n_head:,}")

    # --- Loss / Optimizer / Scheduler ---
    pos_weight_val = train_cfg.get("pos_weight", None)
    pos_weight = (
        torch.tensor([pos_weight_val], device=device)
        if pos_weight_val is not None
        else None
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    trainable_params = [p for p in list(encoder.parameters()) + list(head.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    warmup = train_cfg.get("warmup_steps", 0)
    total_steps = epochs * len(train_loader)
    scheduler = _cosine_lr(optimizer, total_steps, warmup_steps=warmup)

    # --- 학습 루프 ---
    best_auroc = -1.0
    ckpt_dir = train_cfg["ckpt_dir"]

    for epoch in range(epochs):
        train_loss = _train_epoch(
            encoder, head, train_loader, optimizer, scheduler, criterion, device, epoch
        )
        val_loss, val_auroc = _val_epoch(encoder, head, val_loader, criterion, device)
        lr_now = scheduler.get_last_lr()[0]

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("auroc/val", val_auroc, epoch)
        writer.add_scalar("train/lr", lr_now, epoch)

        logger.info(
            f"epoch={epoch+1:3d}/{epochs}"
            f"  train={train_loss:.4f}"
            f"  val_loss={val_loss:.4f}"
            f"  val_auroc={val_auroc:.4f}"
            f"  lr={lr_now:.2e}"
        )

        is_best = val_auroc > best_auroc
        if is_best:
            best_auroc = val_auroc

        save_checkpoint(
            state={
                "epoch": epoch + 1,
                "encoder": encoder.state_dict(),
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_auroc": best_auroc,
                "cfg": cfg,
            },
            ckpt_dir=ckpt_dir,
            is_best=is_best,
        )

    logger.info(f"학습 완료. best_val_auroc={best_auroc:.4f}")
    writer.close()
