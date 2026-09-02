"""Target task fine-tuning 학습 루프.

pretrain에서 학습한 encoder를 불러와 anomaly_head를 붙이고,
masking 없이 전체 window를 복원하도록 학습한다.
복원 오차 = 이상 점수: 정상 구간은 오차가 작고 이상 구간은 오차가 크다.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from src.adt.data.dataset import build_dataloader
from src.adt.models.anomaly_head import ReconstructionAnomalyHead
from src.adt.models.encoder import TimeSeriesTransformerEncoder
from src.adt.utils.checkpoint import save_checkpoint
from src.adt.utils.logger import get_logger
from src.adt.utils.seed import set_seed


# -------------------------------------------------------------------------
# 스케줄러
# -------------------------------------------------------------------------

def _cosine_lr(optimizer: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """warmup 없는 cosine decay (pretrained 가중치이므로 warmup 불필요)."""
    def lr_lambda(step: int) -> float:
        progress = step / max(total_steps, 1)
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
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
) -> float:
    encoder.train()
    head.train()
    total_loss = 0.0
    trainable = [p for p in list(encoder.parameters()) + list(head.parameters()) if p.requires_grad]

    for x, time_feat in tqdm(loader, desc=f"  finetune ep{epoch+1}", leave=False, ncols=80):
        x = x.to(device)
        time_feat = time_feat.to(device)

        pred = head(encoder(x, time_feat, mask=None))
        loss = F.mse_loss(pred, x)

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
    device: torch.device,
) -> float:
    encoder.eval()
    head.eval()
    total_loss = 0.0

    for x, time_feat in loader:
        x = x.to(device)
        time_feat = time_feat.to(device)
        pred = head(encoder(x, time_feat, mask=None))
        total_loss += F.mse_loss(pred, x).item()

    return total_loss / len(loader)


# -------------------------------------------------------------------------
# threshold 계산
# -------------------------------------------------------------------------

@torch.no_grad()
def _compute_threshold(
    encoder: nn.Module,
    head: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    method: str,
    value: float,
) -> float:
    """val split 전체의 window별 재구성 오차 분포로 threshold 계산.

    Returns:
        threshold (float)
    """
    encoder.eval()
    head.eval()
    all_errors: list[torch.Tensor] = []

    for x, time_feat in loader:
        x = x.to(device)
        time_feat = time_feat.to(device)
        pred = head(encoder(x, time_feat, mask=None))
        # window별 MSE: (B, T, C) → mean over T, C → (B,)
        err = F.mse_loss(pred, x, reduction="none").mean(dim=(1, 2))
        all_errors.append(err.cpu())

    errors = torch.cat(all_errors).numpy()

    if method == "percentile":
        threshold = float(np.percentile(errors, value))
    elif method == "k_sigma":
        threshold = float(errors.mean() + value * errors.std())
    else:
        raise ValueError(f"알 수 없는 threshold_method: {method}")

    return threshold


# -------------------------------------------------------------------------
# 메인 진입점
# -------------------------------------------------------------------------

def run(cfg: dict[str, Any], max_epochs: int | None = None) -> None:
    """Finetune 학습 루프.

    Args:
        cfg       : finetune yaml 전체 dict
        max_epochs: 덮어쓰기용 (smoke test 등)
    """
    train_cfg: dict = cfg["train"]
    head_cfg: dict = cfg["head"]

    data_cfg_path = Path(cfg.get("data_config", "configs/data/default.yaml"))
    with open(data_cfg_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    processed_dir = data_cfg["processed_dir"]
    n_features = len(data_cfg["feature_cols"])
    epochs = max_epochs if max_epochs is not None else train_cfg["epochs"]
    freeze_encoder: bool = cfg.get("freeze_encoder", False)
    encoder_ckpt: str = cfg.get("encoder_ckpt", "checkpoints/pretrain/best.pt")

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger, writer = get_logger(train_cfg["log_dir"])
    logger.info(f"device={device}  epochs={epochs}  n_features={n_features}")
    logger.info(f"encoder_ckpt={encoder_ckpt}  freeze_encoder={freeze_encoder}")

    # --- pretrain 체크포인트 로드 ---
    pretrain_state = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)

    # --- 모델 ---
    # pretrain config에서 model 하이퍼파라미터 복원 (없으면 기본값)
    pretrain_model_cfg = pretrain_state.get("cfg", {}).get("model", {})
    d_model = pretrain_model_cfg.get("d_model", 128)
    n_heads = pretrain_model_cfg.get("n_heads", 4)
    n_layers = pretrain_model_cfg.get("n_layers", 4)
    d_ff = pretrain_model_cfg.get("d_ff", 256)
    dropout = pretrain_model_cfg.get("dropout", 0.1)

    encoder = TimeSeriesTransformerEncoder(
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
    ).to(device)
    encoder.load_state_dict(pretrain_state["encoder"])
    logger.info(f"encoder 로드 완료: {encoder_ckpt}")

    # pretrain head 가중치로 warm start 시도
    head = ReconstructionAnomalyHead.from_pretrain_state(
        d_model=d_model,
        n_features=n_features,
        head_state_dict=pretrain_state.get("head", {}),
    ).to(device)
    logger.info("anomaly_head 초기화 완료 (warm start from pretrain head)")

    if freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
        logger.info("encoder 고정 (head만 학습)")

    n_enc = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in head.parameters() if p.requires_grad)
    logger.info(f"학습 파라미터: encoder={n_enc:,}  head={n_head:,}")

    # --- DataLoader (test는 사용 안 함) ---
    train_loader = build_dataloader(processed_dir, "train", train_cfg["batch_size"], shuffle=True)
    val_loader = build_dataloader(processed_dir, "val", train_cfg["batch_size"], shuffle=False)
    logger.info(f"train batches={len(train_loader)}  val batches={len(val_loader)}")

    # --- Optimizer + Scheduler ---
    trainable_params = [p for p in list(encoder.parameters()) + list(head.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    total_steps = epochs * len(train_loader)
    scheduler = _cosine_lr(optimizer, total_steps)

    # --- 학습 루프 ---
    best_val_loss = float("inf")
    ckpt_dir = train_cfg["ckpt_dir"]

    for epoch in range(epochs):
        train_loss = _train_epoch(encoder, head, train_loader, optimizer, scheduler, device, epoch)
        val_loss = _val_epoch(encoder, head, val_loader, device)
        lr_now = scheduler.get_last_lr()[0]

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("train/lr", lr_now, epoch)

        logger.info(
            f"epoch={epoch+1:3d}/{epochs}"
            f"  train={train_loss:.4f}"
            f"  val={val_loss:.4f}"
            f"  lr={lr_now:.2e}"
        )

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

    # --- threshold 계산 (val split 기준) ---
    # best 모델 로드
    best_state = torch.load(Path(ckpt_dir) / "best.pt", map_location=device, weights_only=False)
    encoder.load_state_dict(best_state["encoder"])
    head.load_state_dict(best_state["head"])

    threshold = _compute_threshold(
        encoder, head, val_loader, device,
        method=head_cfg.get("threshold_method", "percentile"),
        value=head_cfg.get("threshold_value", 99.0),
    )
    logger.info(f"threshold ({head_cfg.get('threshold_method','percentile')} "
                f"{head_cfg.get('threshold_value',99.0)}): {threshold:.6f}")

    threshold_path = Path(ckpt_dir) / "threshold.json"
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump({
            "threshold": threshold,
            "method": head_cfg.get("threshold_method", "percentile"),
            "value": head_cfg.get("threshold_value", 99.0),
            "best_val_loss": best_val_loss,
        }, f, indent=2)
    logger.info(f"threshold 저장: {threshold_path}")

    writer.close()
