"""Downstream Detection + Classification 평가 CLI.

사용법::

    # 전체 6개 fold 평가
    python scripts/evaluate_downstream.py

    # 특정 fold만
    python scripts/evaluate_downstream.py --fold unseen_replay

    # config 지정
    python scripts/evaluate_downstream.py --config configs/downstream/default.yaml
"""
from __future__ import annotations

import argparse
import yaml
from pathlib import Path

import torch

from src.adt.models.encoder import TimeSeriesTransformerEncoder
from src.adt.utils.checkpoint import load_encoder_frozen
from src.adt.engine.evaluate_downstream import ALL_FOLDS, evaluate_fold, evaluate_all_folds


def _load_cfg(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(
    config: str = "configs/downstream/default.yaml",
    fold: str | None = None,
) -> None:
    cfg = _load_cfg(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate_downstream] device={device}  config={config}")

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
        evaluate_all_folds(cfg, encoder, device)
    else:
        if fold not in ALL_FOLDS:
            raise ValueError(f"알 수 없는 fold: {fold!r}\n유효: {ALL_FOLDS}")
        evaluate_fold(fold, cfg, encoder, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downstream evaluate")
    parser.add_argument(
        "--config", default="configs/downstream/default.yaml",
        help="downstream config yaml 경로"
    )
    parser.add_argument(
        "--fold", default=None,
        help="평가할 fold 이름 (생략 or 'all'이면 전부)"
    )
    args = parser.parse_args()
    main(config=args.config, fold=args.fold)
