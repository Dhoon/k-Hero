"""체크포인트 저장 / 로드 유틸."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    state: dict[str, Any],
    ckpt_dir: str | Path,
    is_best: bool = False,
) -> None:
    """체크포인트를 last.pt로 저장하고 best일 때는 best.pt도 별도 저장.

    Args:
        state   : 저장할 dict (epoch, encoder, head, optimizer, scheduler, ...)
        ckpt_dir: 저장 디렉토리
        is_best : True면 best.pt도 갱신
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    last_path = ckpt_dir / "last.pt"
    torch.save(state, last_path)

    if is_best:
        best_path = ckpt_dir / "best.pt"
        shutil.copyfile(last_path, best_path)


def load_checkpoint(
    ckpt_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
) -> tuple[int, float]:
    """저장된 체크포인트에서 모델 (+ optimizer/scheduler) 가중치 복원.

    Returns:
        (start_epoch, best_val_loss)
    """
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["encoder"])

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])

    return state.get("epoch", 0), state.get("best_val_loss", float("inf"))
