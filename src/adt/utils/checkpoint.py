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


def load_encoder_for_finetune(
    ckpt_path: str | Path,
    encoder: nn.Module,
    mode: str = "layernorm_only",
    num_unfrozen_blocks: int = 1,
) -> tuple[nn.Module, list]:
    """pretrain checkpoint 로드 후 mode에 따라 일부 파라미터만 unfreeze.

    Args:
        ckpt_path           : pretrain best.pt 경로
        encoder             : 가중치를 채울 encoder 인스턴스 (in-place 수정)
        mode                : "layernorm_only" — LayerNorm affine만 학습 가능
                              "last_block"     — 전체 LN + 마지막 N개 block attention/FFN
        num_unfrozen_blocks : mode="last_block"일 때 unfreeze할 block 수 (기본 1)

    Returns:
        (encoder, trainable_params)

    Raises:
        FileNotFoundError: checkpoint 파일이 없을 때
        ValueError       : 지원하지 않는 mode
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pretrain checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    enc_state = state.get("encoder", state)
    encoder.load_state_dict(enc_state, strict=True)

    # 전체 freeze 먼저
    for p in encoder.parameters():
        p.requires_grad_(False)

    trainable_params: list = []

    if mode == "layernorm_only":
        n_modules = n_params = 0
        for name, m in encoder.named_modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters(recurse=False):
                    p.requires_grad_(True)
                    trainable_params.append(p)
                    n_params += p.numel()
                n_modules += 1
        print(
            f"[finetune] mode={mode}  "
            f"unfreeze: {n_modules} LayerNorm modules  {n_params:,} params"
        )

    elif mode == "last_block":
        # (a) 전체 LayerNorm unfreeze (Tier 1 효과 유지)
        n_ln_modules = n_ln_params = 0
        for name, m in encoder.named_modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters(recurse=False):
                    p.requires_grad_(True)
                    trainable_params.append(p)
                    n_ln_params += p.numel()
                n_ln_modules += 1

        # (b) 마지막 num_unfrozen_blocks개 layer의 attention/FFN (LN 제외 — 중복 방지)
        layers = encoder.transformer_encoder.layers  # type: ignore[attr-defined]
        n_total = len(layers)
        n_unfreeze = min(num_unfrozen_blocks, n_total)
        unfreeze_indices = list(range(n_total - n_unfreeze, n_total))

        n_blk_params = 0
        for idx in unfreeze_indices:
            for name, m in layers[idx].named_modules():
                if isinstance(m, nn.LayerNorm):
                    continue  # (a)에서 이미 처리
                for p in m.parameters(recurse=False):
                    if not p.requires_grad:
                        p.requires_grad_(True)
                        trainable_params.append(p)
                        n_blk_params += p.numel()

        print(
            f"[finetune] mode={mode}  "
            f"LayerNorm 전체 {n_ln_modules}개 {n_ln_params:,}params"
            f" + block{unfreeze_indices} attention/FFN {n_blk_params:,}params"
            f"  총 {n_ln_params + n_blk_params:,}params"
        )

    else:
        raise ValueError(
            f"지원하지 않는 finetune mode: {mode!r}. "
            "'layernorm_only' 또는 'last_block'만 지원."
        )

    encoder.train()
    return encoder, trainable_params


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
