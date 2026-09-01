"""CLI: self-supervised pretraining 실행.

Usage:
    python scripts/pretrain.py
    python scripts/pretrain.py --config configs/pretrain/default.yaml
    python scripts/pretrain.py --config configs/pretrain/default.yaml --max-epochs 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.adt.engine.pretrain import run


def main() -> None:
    parser = argparse.ArgumentParser(description="LP 전력 데이터 Self-supervised Pretraining")
    parser.add_argument(
        "--config",
        default="configs/pretrain/default.yaml",
        help="pretrain 설정 파일 (default: configs/pretrain/default.yaml)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="최대 epoch 수 덮어쓰기 (smoke test 등 단축 실행용)",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run(cfg, max_epochs=args.max_epochs)


if __name__ == "__main__":
    main()
