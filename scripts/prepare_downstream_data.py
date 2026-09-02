"""downstream fold 데이터 생성 스크립트.

사용법::

    python scripts/prepare_downstream_data.py \
        [--config configs/downstream/attack_injection.yaml]

동작:
    1. attack_injection.yaml 에서 경로/시드/주입 파라미터/fold 정의를 읽음
    2. pretrain_dir 의 stride=1 윈도우에 5종 attack 주입 (type_label 포함)
    3. all_type superset → unseen_X fold 필터링으로 6개 fold 생성
    4. data/processed/downstream/{fold_name}/{split}/ 에 저장
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import yaml

from src.adt.data.labeling import generate_downstream_folds
from src.adt.data.scalers import StandardScalerND


def _load_cfg(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _print_stats(stats: dict) -> None:
    print("\n" + "=" * 65)
    print(f"{'fold':<24} {'split':<6} {'total':>7} {'normal':>7} {'attack':>7} {'types'}")
    print("=" * 65)
    for fold_name, splits in stats.items():
        for split, info in splits.items():
            n_attack = info["total"] - info["normal"]
            atk_ratio = n_attack / max(info["total"], 1) * 100
            type_str = "  ".join(
                f"{t}:{cnt}" for t, cnt in sorted(info["attacks"].items())
            )
            print(
                f"{fold_name:<24} {split:<6} {info['total']:>7,} "
                f"{info['normal']:>7,} {n_attack:>7,} ({atk_ratio:.1f}%)  "
                f"{type_str}"
            )
    print("=" * 65)


def main(config_path: str = "configs/downstream/attack_injection.yaml") -> None:
    cfg = _load_cfg(config_path)

    pretrain_dir = Path(cfg["pretrain_dir"])
    scaler_path  = Path(cfg["scaler_path"])
    output_dir   = Path(cfg["output_dir"])

    if not pretrain_dir.exists():
        raise FileNotFoundError(
            f"pretrain_dir not found: {pretrain_dir}\n"
            "先に scripts/prepare_data.py を実行してください."
        )
    if not scaler_path.exists():
        raise FileNotFoundError(f"scaler not found: {scaler_path}")

    raw = joblib.load(scaler_path)
    if isinstance(raw, dict):
        # prepare_data.py 가 dict {"mean": ..., "std": ...} 로 저장한 경우
        scaler = StandardScalerND()
        scaler.mean_ = raw["mean"]
        scaler.std_  = raw["std"]
    else:
        scaler = raw
    print(f"[downstream] scaler loaded from {scaler_path}")
    print(
        f"[downstream] pretrain_dir : {pretrain_dir}\n"
        f"[downstream] output_dir   : {output_dir}\n"
        f"[downstream] injection_ratio : {cfg.get('injection_ratio', 0.1)}\n"
        f"[downstream] seed         : {cfg.get('seed', 42)}\n"
        f"[downstream] folds        : {len(cfg.get('folds', []))} defined"
    )

    t0 = time.perf_counter()
    stats = generate_downstream_folds(
        pretrain_dir=pretrain_dir,
        output_dir=output_dir,
        cfg=cfg,
        scaler=scaler,
    )
    elapsed = time.perf_counter() - t0

    _print_stats(stats)
    print(f"\n[downstream] done in {elapsed:.1f}s — saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate downstream fold data")
    parser.add_argument(
        "--config",
        default="configs/downstream/attack_injection.yaml",
        help="path to attack_injection.yaml",
    )
    args = parser.parse_args()
    main(args.config)
