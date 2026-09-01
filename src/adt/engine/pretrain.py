"""Self-supervised pretraining 학습 루프."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from src.adt.data.dataset import build_dataloader
from src.adt.models.encoder import TimeSeriesTransformerEncoder
from src.adt.models.pretrain_head import MaskedReconstructionHead
from src.adt.ssl.losses import masked_reconstruction_loss
from src.adt.ssl.masking import generate_mask, get_mask_ratio
from src.adt.utils.checkpoint import save_checkpoint
from src.adt.utils.logger import get_logger
from src.adt.utils.seed import set_seed


# -------------------------------------------------------------------------
# 스케줄러
# -------------------------------------------------------------------------

def _cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """linear warmup → cosine decay 스케줄러."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# -------------------------------------------------------------------------
# 단일 epoch 학습 / 검증
# -------------------------------------------------------------------------

def _train_epoch(
    encoder: nn.Module,
    head: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    epoch: int,
    ssl_cfg: dict,
    grad_clip: float = 1.0,
) -> float:
    encoder.train()
    head.train()
    total_loss = 0.0
    params = list(encoder.parameters()) + list(head.parameters())

    for x, time_feat in tqdm(loader, desc=f"  train ep{epoch+1}", leave=False, ncols=80):
        x = x.to(device)
        time_feat = time_feat.to(device)

        mask = generate_mask(x, epoch, ssl_cfg)

        pred = head(encoder(x, time_feat, mask=mask))
        loss = masked_reconstruction_loss(pred, x, mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def _val_epoch(
    encoder: nn.Module,
    head: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    ssl_cfg: dict,
) -> float:
    encoder.eval()
    head.eval()
    total_loss = 0.0

    for x, time_feat in loader:
        x = x.to(device)
        time_feat = time_feat.to(device)
        mask = generate_mask(x, epoch, ssl_cfg)
        pred = head(encoder(x, time_feat, mask=mask))
        total_loss += masked_reconstruction_loss(pred, x, mask).item()

    return total_loss / len(loader)


# -------------------------------------------------------------------------
# 메인 학습 진입점
# -------------------------------------------------------------------------

def run(cfg: dict[str, Any], max_epochs: int | None = None) -> None:
    """Pretrain 학습 루프.

    Args:
        cfg       : pretrain yaml 전체 dict
        max_epochs: 덮어쓰기용 (smoke test 등에서 짧게 돌릴 때)
    """
    # --- 설정 파싱 ---
    train_cfg: dict = cfg["train"]
    model_cfg: dict = cfg["model"]
    ssl_cfg: dict = cfg["ssl"]

    # data config 로드 (processed_dir, feature_cols 등)
    data_cfg_path = Path(cfg.get("data_config", "configs/data/default.yaml"))
    with open(data_cfg_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    processed_dir = data_cfg["processed_dir"]
    n_features = len(data_cfg["feature_cols"])
    epochs = max_epochs if max_epochs is not None else train_cfg["epochs"]

    # --- 공통 설정 ---
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger, writer = get_logger(train_cfg["log_dir"])
    logger.info(f"device={device}  epochs={epochs}  n_features={n_features}")

    # --- 모델 ---
    encoder = TimeSeriesTransformerEncoder(
        n_features=n_features,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
    ).to(device)

    head = MaskedReconstructionHead(
        d_model=model_cfg["d_model"],
        n_features=n_features,
    ).to(device)

    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    logger.info(f"encoder params: {n_params:,}")

    # --- DataLoader ---
    train_loader = build_dataloader(processed_dir, "train", train_cfg["batch_size"], shuffle=True)
    val_loader = build_dataloader(processed_dir, "val", train_cfg["batch_size"], shuffle=False)
    logger.info(f"train batches={len(train_loader)}  val batches={len(val_loader)}")

    # --- Optimizer + Scheduler ---
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    total_steps = epochs * len(train_loader)
    scheduler = _cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=train_cfg["warmup_steps"],
        total_steps=total_steps,
    )

    # --- 학습 루프 ---
    best_val_loss = float("inf")
    ckpt_dir = train_cfg["ckpt_dir"]

    for epoch in range(epochs):
        train_loss = _train_epoch(
            encoder, head, train_loader, optimizer, scheduler,
            device, epoch, ssl_cfg,
        )
        val_loss = _val_epoch(encoder, head, val_loader, device, epoch, ssl_cfg)
        mask_ratio = get_mask_ratio(epoch, ssl_cfg)
        lr_now = scheduler.get_last_lr()[0]

        # 로깅
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("ssl/mask_ratio", mask_ratio, epoch)
        writer.add_scalar("train/lr", lr_now, epoch)

        logger.info(
            f"epoch={epoch+1:3d}/{epochs}"
            f"  train={train_loss:.4f}"
            f"  val={val_loss:.4f}"
            f"  mask_ratio={mask_ratio:.3f}"
            f"  lr={lr_now:.2e}"
        )

        # 체크포인트
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        save_checkpoint(
            state={
                "epoch": epoch + 1,
                "encoder": encoder.state_dict(),
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "cfg": cfg,
            },
            ckpt_dir=ckpt_dir,
            is_best=is_best,
        )

    logger.info(f"학습 완료. best_val_loss={best_val_loss:.4f}")
    writer.close()
