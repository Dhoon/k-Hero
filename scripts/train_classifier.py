"""CLI: Classification 다운스트림 학습 (Exp A / Exp B).

Usage:
    # Exp A: random init encoder
    python scripts/train_classification.py --config configs/downstream/classification/exp_a_no_pretrain.yaml

    # Exp B: SSL pretrain encoder
    python scripts/train_classification.py --config configs/downstream/classification/exp_b_ssl_pretrain.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.adt.engine.downstream_classification import run


def main() -> None:
    parser = argparse.ArgumentParser(description="LP 전력 데이터 Classification 학습 (Exp A / Exp B)")
    parser.add_argument(
        "--config",
        default="configs/downstream/classification/exp_b_ssl_pretrain.yaml",
        help="classification 설정 파일",
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
