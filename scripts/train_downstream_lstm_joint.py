"""LSTM Joint Multi-task Downstream 학습 CLI."""
from __future__ import annotations

import argparse

import torch
import yaml

from src.adt.engine.train_downstream_lstm_joint import (
    ALL_FOLDS,
    train_all_folds_lstm_joint,
    train_fold_lstm_joint,
)


def main(
    config: str = "configs/downstream_lstm_joint/default.yaml",
    fold: str | None = None,
) -> None:
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = cfg["training"]
    print(
        f"[train_downstream_lstm_joint] config={config}  device={device}"
        f"  w_det={train_cfg.get('w_det', 1.0)}"
        f"  w_cls={train_cfg.get('w_cls', 0.4)}"
    )

    if fold is None or fold.lower() == "all":
        train_all_folds_lstm_joint(cfg, device)
    else:
        if fold not in ALL_FOLDS:
            raise ValueError(f"알 수 없는 fold: {fold!r}\n유효: {ALL_FOLDS}")
        train_fold_lstm_joint(fold, cfg, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Joint downstream training")
    parser.add_argument("--config", default="configs/downstream_lstm_joint/default.yaml")
    parser.add_argument("--fold",   default=None, help="fold name or 'all'")
    args = parser.parse_args()
    main(args.config, args.fold)
