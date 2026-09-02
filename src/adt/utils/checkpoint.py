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


def load_encoder_frozen(
    ckpt_path: str | Path | None,
    encoder: nn.Module,
    allow_random_init: bool = False,
) -> nn.Module:
    """pretrain checkpoint에서 encoder 가중치를 로드하고 eval+freeze.

    Args:
        ckpt_path        : pretrain best.pt 경로
        encoder          : TimeSeriesTransformerEncoder 인스턴스
        allow_random_init: True면 checkpoint 없을 때 random init으로 진행
                           (테스트/smoke 전용). False(기본값)면 FileNotFoundError.

    Returns:
        eval() + requires_grad=False 상태의 encoder (in-place 수정 후 반환)

    Raises:
        FileNotFoundError: ckpt_path가 None이거나 파일이 없고 allow_random_init=False일 때
    """
    if ckpt_path is None or not Path(ckpt_path).exists():
        if not allow_random_init:
            missing = str(ckpt_path) if ckpt_path is not None else "<None>"
            raise FileNotFoundError(
                f"Pretrain checkpoint not found: {missing}\n"
                "SSL pretrain 없이 downstream을 학습하면 결과가 무의미합니다.\n"
                "테스트/smoke 용도라면 allow_random_init=True를 명시적으로 전달하세요."
            )
    else:
        state = torch.load(ckpt_path, map_location="cpu")
        # pretrain 체크포인트는 {"encoder": state_dict, ...} 구조
        enc_state = state.get("encoder", state)
        encoder.load_state_dict(enc_state, strict=True)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    return encoder
