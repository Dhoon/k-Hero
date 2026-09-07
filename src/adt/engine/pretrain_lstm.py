"""LSTM Autoencoder pretrain 루프 — 순수 MSE reconstruction, 마스킹 없음."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from src.adt.data.dataset import build_dataloader
from src.adt.models.lstm_ae import LSTMAutoencoder
from src.adt.utils.checkpoint import save_checkpoint
from src.adt.utils.logger import get_logger
from src.adt.utils.seed import set_seed


def _cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _train_epoch(
    ae: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    ae.train()
    total = 0.0
    for x, _, _ in tqdm(loader, desc="  train", leave=False, ncols=80):
        x = x.to(device)
        x_recon = ae(x)
        loss = F.mse_loss(x_recon, x)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(ae.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def _val_epoch(
    ae: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    ae.eval()
    total = 0.0
    for x, _, _ in loader:
        x = x.to(device)
        total += F.mse_loss(ae(x), x).item()
    return total / max(len(loader), 1)


def run(cfg: dict[str, Any], max_epochs: int | None = None) -> None:
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    data_cfg_path = Path(cfg.get("data_config", "configs/data/default.yaml"))
    with open(data_cfg_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    processed_dir = str(Path(data_cfg["processed_dir"]) / "pretrain")
    n_features = len(data_cfg["feature_cols"])
    epochs = max_epochs if max_epochs is not None else train_cfg["epochs"]

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger, writer = get_logger(train_cfg["log_dir"], name="adt.pretrain_lstm")
    logger.info(f"device={device}  epochs={epochs}  n_features={n_features}")

    ae = LSTMAutoencoder(
        n_features=n_features,
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
    ).to(device)

    n_params = sum(p.numel() for p in ae.parameters() if p.requires_grad)
    logger.info(f"LSTMAutoencoder trainable params: {n_params:,}")

    train_loader = build_dataloader(
        processed_dir, "train", train_cfg["batch_size"], shuffle=True
    )
    val_loader = build_dataloader(
        processed_dir, "val", train_cfg["batch_size"], shuffle=False
    )
    logger.info(
        f"train batches={len(train_loader)}  val batches={len(val_loader)}"
    )

    optimizer = torch.optim.AdamW(
        ae.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    total_steps = epochs * len(train_loader)
    scheduler = _cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=train_cfg.get("warmup_steps", 0),
        total_steps=total_steps,
    )

    best_val_loss = float("inf")
    ckpt_dir = train_cfg["ckpt_dir"]
    grad_clip = train_cfg.get("grad_clip", 1.0)

    for epoch in range(epochs):
        tr_loss  = _train_epoch(ae, train_loader, optimizer, scheduler, device, grad_clip)
        val_loss = _val_epoch(ae, val_loader, device)

        writer.add_scalar("loss/train", tr_loss,  epoch)
        writer.add_scalar("loss/val",   val_loss, epoch)
        writer.add_scalar("train/lr",   scheduler.get_last_lr()[0], epoch)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        save_checkpoint(
            state={
                "epoch":         epoch + 1,
                "encoder":       ae.encoder.state_dict(),
                "decoder":       ae.decoder.state_dict(),
                "model":         ae.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "scheduler":     scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "cfg":           cfg,
            },
            ckpt_dir=ckpt_dir,
            is_best=is_best,
        )

        logger.info(
            f"epoch={epoch+1:3d}/{epochs}"
            f"  train_mse={tr_loss:.4f}"
            f"  val_mse={val_loss:.4f}"
            + ("  [★ best]" if is_best else "")
        )

    logger.info(f"완료. best_val_loss={best_val_loss:.4f}")
    writer.close()
