"""LSTM Autoencoder pretraining CLI."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.adt.engine.pretrain_lstm import run


def main(
    config: str = "configs/pretrain_lstm/default.yaml",
    max_epochs: int | None = None,
) -> None:
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"[pretrain_lstm] config={config}")
    run(cfg, max_epochs=max_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM AE pretrain")
    parser.add_argument("--config", default="configs/pretrain_lstm/default.yaml")
    parser.add_argument("--max_epochs", type=int, default=None)
    args = parser.parse_args()
    main(config=args.config, max_epochs=args.max_epochs)
