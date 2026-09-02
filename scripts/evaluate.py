"""CLI: 합성 이상치 기반 평가 실행.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --config configs/finetune/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.adt.engine.evaluate import run


def main() -> None:
    parser = argparse.ArgumentParser(description="합성 이상치 기반 이상탐지 평가")
    parser.add_argument(
        "--config",
        default="configs/finetune/default.yaml",
        help="finetune 설정 파일 (eval 절 포함)",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run(cfg)


if __name__ == "__main__":
    main()
