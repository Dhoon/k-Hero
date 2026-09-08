"""LSTM Joint Downstream 평가 CLI."""
from __future__ import annotations

import argparse

import torch
import yaml

from src.adt.engine.evaluate_downstream_lstm_joint import (
    ALL_FOLDS,
    evaluate_all_folds_lstm_joint,
    evaluate_fold_lstm_joint,
)


def main(
    config: str = "configs/downstream_lstm_joint/default.yaml",
    fold: str | None = None,
) -> None:
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[evaluate_downstream_lstm_joint] config={config}  device={device}"
        f"  ckpt_dir={cfg['training']['ckpt_dir']}"
    )

    if fold is None or fold.lower() == "all":
        evaluate_all_folds_lstm_joint(cfg, device)
    else:
        if fold not in ALL_FOLDS:
            raise ValueError(f"알 수 없는 fold: {fold!r}\n유효: {ALL_FOLDS}")
        evaluate_fold_lstm_joint(fold, cfg, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Joint downstream evaluation")
    parser.add_argument("--config", default="configs/downstream_lstm_joint/default.yaml")
    parser.add_argument("--fold",   default=None, help="fold name or 'all'")
    args = parser.parse_args()
    main(args.config, args.fold)
