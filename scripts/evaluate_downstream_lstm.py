"""LSTM Downstream 평가 CLI."""
from __future__ import annotations

import argparse

import torch
import yaml

from src.adt.engine.evaluate_downstream_lstm import (
    ALL_FOLDS,
    evaluate_all_folds_lstm,
    evaluate_fold_lstm,
)


def main(
    config: str = "configs/downstream_lstm/default.yaml",
    fold: str | None = None,
) -> None:
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate_downstream_lstm] config={config}  device={device}")

    if fold is None or fold.lower() == "all":
        evaluate_all_folds_lstm(cfg, device)
    else:
        if fold not in ALL_FOLDS:
            raise ValueError(f"알 수 없는 fold: {fold!r}\n유효: {ALL_FOLDS}")
        evaluate_fold_lstm(fold, cfg, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM downstream evaluation")
    parser.add_argument("--config", default="configs/downstream_lstm/default.yaml")
    parser.add_argument("--fold",   default=None, help="fold name or 'all'")
    args = parser.parse_args()
    main(args.config, args.fold)
