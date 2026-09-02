"""Self-supervised pretraining 학습 루프.

Joint objective:
  L_pretrain = L_mask + forecast_loss_weight * L_forecast

  - L_mask    : masked reconstruction loss (마스킹된 위치에서만 MSE)
  - L_forecast: forecasting loss (마지막 타임스텝 h_L → 미래 h step MSE)
  - encoder는 masked input을 한 번만 통과 → 두 head가 같은 encoder 출력을 공유
"""
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
from src.adt.models.heads.pretrain_head import MaskedReconstructionHead
from src.adt.models.heads.forecasting_head import ForecastingHead
from src.adt.ssl.losses import masked_reconstruction_loss, forecast_loss, pretrain_joint_loss
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
    recon_head: nn.Module,
    forecast_head: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    epoch: int,
    ssl_cfg: dict,
    forecast_loss_weight: float,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    encoder.train()
    recon_head.train()
    forecast_head.train()
    total_loss = mask_loss_sum = forecast_loss_sum = 0.0
    params = list(encoder.parameters()) + list(recon_head.parameters()) + list(forecast_head.parameters())

    for x, time_feat, future_target in tqdm(loader, desc=f"  train ep{epoch+1}", leave=False, ncols=80):
        x = x.to(device)
        time_feat = time_feat.to(device)
        future_target = future_target.to(device)

        mask = generate_mask(x, epoch, ssl_cfg)

        # 단일 encoder forward — reconstruction과 forecasting이 같은 출력을 공유
        enc_out = encoder(x, time_feat, mask=mask)   # (B, T, d_model)

        pred_recon = recon_head(enc_out)             # (B, T, C)
        pred_future = forecast_head(enc_out)         # (B, h, C)

        l_mask = masked_reconstruction_loss(pred_recon, x, mask)
        l_forecast = forecast_loss(pred_future, future_target)
        loss = pretrain_joint_loss(l_mask, l_forecast, forecast_loss_weight)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        mask_loss_sum += l_mask.item()
        forecast_loss_sum += l_forecast.item()

    n = len(loader)
    return {
        "total": total_loss / n,
        "mask": mask_loss_sum / n,
        "forecast": forecast_loss_sum / n,
    }


@torch.no_grad()
def _val_epoch(
    encoder: nn.Module,
    recon_head: nn.Module,
    forecast_head: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    ssl_cfg: dict,
    forecast_loss_weight: float,
) -> dict[str, float]:
    encoder.eval()
    recon_head.eval()
    forecast_head.eval()
    total_loss = mask_loss_sum = forecast_loss_sum = 0.0

    for x, time_feat, future_target in loader:
        x = x.to(device)
        time_feat = time_feat.to(device)
        future_target = future_target.to(device)

        mask = generate_mask(x, epoch, ssl_cfg)

        enc_out = encoder(x, time_feat, mask=mask)
        pred_recon = recon_head(enc_out)
        pred_future = forecast_head(enc_out)

        l_mask = masked_reconstruction_loss(pred_recon, x, mask)
        l_forecast = forecast_loss(pred_future, future_target)
        loss = pretrain_joint_loss(l_mask, l_forecast, forecast_loss_weight)

        total_loss += loss.item()
        mask_loss_sum += l_mask.item()
        forecast_loss_sum += l_forecast.item()

    n = len(loader)
    return {
        "total": total_loss / n,
        "mask": mask_loss_sum / n,
        "forecast": forecast_loss_sum / n,
    }


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

    # data config 로드 (processed_dir, feature_cols, forecast_horizon 등)
    data_cfg_path = Path(cfg.get("data_config", "configs/data/default.yaml"))
    with open(data_cfg_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    processed_dir = str(Path(data_cfg["processed_dir"]) / "pretrain")
    n_features = len(data_cfg["feature_cols"])
    forecast_horizon: int = data_cfg.get("forecast_horizon", 0)
    forecast_loss_weight: float = ssl_cfg.get("forecast_loss_weight", 0.0)
    epochs = max_epochs if max_epochs is not None else train_cfg["epochs"]

    # --- 공통 설정 ---
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger, writer = get_logger(train_cfg["log_dir"])
    logger.info(
        f"device={device}  epochs={epochs}  n_features={n_features}"
        f"  forecast_horizon={forecast_horizon}  forecast_loss_weight={forecast_loss_weight}"
    )

    # --- 모델 ---
    encoder = TimeSeriesTransformerEncoder(
        n_features=n_features,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
    ).to(device)

    recon_head = MaskedReconstructionHead(
        d_model=model_cfg["d_model"],
        n_features=n_features,
    ).to(device)

    forecast_head = ForecastingHead(
        d_model=model_cfg["d_model"],
        forecast_horizon=forecast_horizon,
        n_features=n_features,
    ).to(device)

    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    logger.info(f"encoder params: {n_params:,}")

    # --- DataLoader ---
    train_loader = build_dataloader(processed_dir, "train", train_cfg["batch_size"], shuffle=True)
    val_loader = build_dataloader(processed_dir, "val", train_cfg["batch_size"], shuffle=False)
    logger.info(f"train batches={len(train_loader)}  val batches={len(val_loader)}")

    # --- Optimizer + Scheduler ---
    params = (
        list(encoder.parameters())
        + list(recon_head.parameters())
        + list(forecast_head.parameters())
    )
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
        train_losses = _train_epoch(
            encoder, recon_head, forecast_head, train_loader,
            optimizer, scheduler, device, epoch, ssl_cfg, forecast_loss_weight,
        )
        val_losses = _val_epoch(
            encoder, recon_head, forecast_head, val_loader,
            device, epoch, ssl_cfg, forecast_loss_weight,
        )
        mask_ratio = get_mask_ratio(epoch, ssl_cfg)
        lr_now = scheduler.get_last_lr()[0]

        # 로깅
        for tag, v in [("train", train_losses), ("val", val_losses)]:
            writer.add_scalar(f"loss/{tag}", v["total"], epoch)
            writer.add_scalar(f"loss/{tag}_mask", v["mask"], epoch)
            writer.add_scalar(f"loss/{tag}_forecast", v["forecast"], epoch)
        writer.add_scalar("ssl/mask_ratio", mask_ratio, epoch)
        writer.add_scalar("train/lr", lr_now, epoch)

        logger.info(
            f"epoch={epoch+1:3d}/{epochs}"
            f"  train={train_losses['total']:.4f}"
            f"  (mask={train_losses['mask']:.4f} fc={train_losses['forecast']:.4f})"
            f"  val={val_losses['total']:.4f}"
            f"  (mask={val_losses['mask']:.4f} fc={val_losses['forecast']:.4f})"
            f"  mask_ratio={mask_ratio:.3f}  lr={lr_now:.2e}"
        )

        # 체크포인트 — encoder + 두 head 모두 저장 (downstream은 encoder만 사용)
        is_best = val_losses["total"] < best_val_loss
        if is_best:
            best_val_loss = val_losses["total"]

        save_checkpoint(
            state={
                "epoch": epoch + 1,
                "encoder": encoder.state_dict(),
                "recon_head": recon_head.state_dict(),
                "forecast_head": forecast_head.state_dict(),
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
