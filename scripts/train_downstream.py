"""Downstream Detection + Classification 학습 CLI.

사용법::

    # Tier 1 (LayerNorm only)
    python scripts/train_downstream.py --fold all_type --mode t1

    # Tier 2 (전체 LN + 마지막 block attention/FFN)
    python scripts/train_downstream.py --fold all_type --mode t2

    # yaml 기본값 그대로
    python scripts/train_downstream.py --fold all_type

    # config 지정
    python scripts/train_downstream.py --config configs/downstream/default.yaml --fold all_type --mode t1
"""
from __future__ import annotations

import argparse
import yaml
from pathlib import Path

import torch

from src.adt.models.encoder import TimeSeriesTransformerEncoder
from src.adt.utils.checkpoint import load_encoder_frozen
from src.adt.engine.train_downstream import ALL_FOLDS, train_fold, train_all_folds

_MODE_MAP = {
    "t1": "layernorm_only",
    "t2": "last_block",
}


def _load_cfg(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(
    config: str = "configs/downstream/default.yaml",
    fold: str | None = None,
    mode: str | None = None,
) -> None:
    cfg = _load_cfg(config)

    # --mode 옵션이 있으면 yaml의 finetune.mode를 덮어씀
    if mode is not None:
        ft_mode = _MODE_MAP.get(mode, mode)
        cfg.setdefault("finetune", {})["mode"] = ft_mode
        cfg["finetune"]["enabled"] = True
        print(f"[train_downstream] finetune mode override: {mode} → {ft_mode}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ft_mode_str = cfg.get("finetune", {}).get("mode", "layernorm_only")
    print(f"[train_downstream] device={device}  finetune.mode={ft_mode_str}")

    model_cfg = cfg["model"]
    encoder = TimeSeriesTransformerEncoder(
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
    )
    encoder = load_encoder_frozen(cfg.get("pretrain_ckpt"), encoder).to(device)

    if fold is None or fold.lower() == "all":
        train_all_folds(cfg, encoder, device)
    else:
        if fold not in ALL_FOLDS:
            raise ValueError(f"알 수 없는 fold: {fold!r}\n유효: {ALL_FOLDS}")
        train_fold(fold, cfg, encoder, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downstream train")
    parser.add_argument(
        "--config", default="configs/downstream/default.yaml",
        help="downstream config yaml 경로",
    )
    parser.add_argument(
        "--fold", default=None,
        help="학습할 fold 이름 (생략 or 'all'이면 전부)",
    )
    parser.add_argument(
        "--mode", default=None, choices=["t1", "t2"],
        help="t1=layernorm_only  t2=last_block (생략하면 yaml 값 사용)",
    )
    args = parser.parse_args()
    main(config=args.config, fold=args.fold, mode=args.mode)
