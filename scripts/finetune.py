"""CLI: target task fine-tuning 실행.

Usage:
    python scripts/finetune.py
    python scripts/finetune.py --config configs/finetune/default.yaml
    python scripts/finetune.py --config configs/finetune/default.yaml --max-epochs 3
    python scripts/finetune.py --encoder-ckpt checkpoints/pretrain/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.adt.engine.finetune import run


def main() -> None:
    parser = argparse.ArgumentParser(description="LP 전력 데이터 Fine-tuning (이상탐지 head)")
    parser.add_argument(
        "--config",
        default="configs/finetune/default.yaml",
        help="finetune 설정 파일 (default: configs/finetune/default.yaml)",
    )
    parser.add_argument(
        "--encoder-ckpt",
        default=None,
        help="pretrain encoder 체크포인트 경로 덮어쓰기 (default: yaml의 encoder_ckpt)",
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

    if args.encoder_ckpt is not None:
        cfg["encoder_ckpt"] = args.encoder_ckpt

    run(cfg, max_epochs=args.max_epochs)


if __name__ == "__main__":
    main()
