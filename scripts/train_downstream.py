"""Downstream Detection + Classification 학습 CLI.

사용법::

    # 전체 6개 fold 학습
    python scripts/train_downstream.py

    # 특정 fold만
    python scripts/train_downstream.py --fold all_type
    python scripts/train_downstream.py --fold unseen_replay

    # config 지정
    python scripts/train_downstream.py --config configs/downstream/default.yaml
"""
from __future__ import annotations

import argparse
import yaml
from pathlib import Path

import torch

from src.adt.models.encoder import TimeSeriesTransformerEncoder
from src.adt.utils.checkpoint import load_encoder_frozen
from src.adt.engine.train_downstream import ALL_FOLDS, train_fold, train_all_folds


def _load_cfg(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(
    config: str = "configs/downstream/default.yaml",
    fold: str | None = None,
) -> None:
    cfg = _load_cfg(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_downstream] device={device}  config={config}")

    # encoder 로드 + freeze
    model_cfg = cfg["model"]
    encoder = TimeSeriesTransformerEncoder(
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
    )
    encoder = load_encoder_frozen(cfg.get("pretrain_ckpt"), encoder).to(device)
    n_frozen = sum(p.numel() for p in encoder.parameters())
    print(f"[train_downstream] encoder frozen params={n_frozen:,}")

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
        help="downstream config yaml 경로"
    )
    parser.add_argument(
        "--fold", default=None,
        help="학습할 fold 이름 (생략 or 'all'이면 전부)"
    )
    args = parser.parse_args()
    main(config=args.config, fold=args.fold)
