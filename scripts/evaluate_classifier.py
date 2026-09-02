"""CLI: Classification 모델 평가 (Exp A / Exp B).

Usage:
    # Exp A: random init encoder
    python scripts/evaluate_classification.py --config configs/downstream/classification/exp_a_no_pretrain.yaml

    # Exp B: SSL pretrain encoder
    python scripts/evaluate_classification.py --config configs/downstream/classification/exp_b_ssl_pretrain.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.adt.engine.evaluate_classification import run


def main() -> None:
    parser = argparse.ArgumentParser(description="LP 전력 데이터 Classification 평가 (Exp A / Exp B)")
    parser.add_argument(
        "--config",
        default="configs/downstream/classification/exp_b_ssl_pretrain.yaml",
        help="classification 설정 파일",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run(cfg)


if __name__ == "__main__":
    main()
