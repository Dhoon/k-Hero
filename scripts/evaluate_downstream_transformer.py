"""Downstream Detection + Classification 평가 CLI.

사용법::

    # 전체 6개 fold 평가
    python scripts/evaluate_downstream.py

    # 특정 fold만
    python scripts/evaluate_downstream.py --fold unseen_replay

    # config 지정
    python scripts/evaluate_downstream.py --config configs/downstream_transformer/default.yaml
"""
from __future__ import annotations

import argparse
import yaml
from pathlib import Path

import torch

from src.adt.models.transformer_encoder import TimeSeriesTransformerEncoder
from src.adt.utils.checkpoint import load_encoder_frozen
from src.adt.engine.evaluate_downstream_transformer import ALL_FOLDS, evaluate_fold, evaluate_all_folds
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

fm.fontManager.addfont(font_path)

font_prop = fm.FontProperties(fname=font_path)
font_name = font_prop.get_name()

print("Matplotlib font name:", font_name)

matplotlib.rcParams["font.family"] = font_name
matplotlib.rcParams["axes.unicode_minus"] = False

def _load_cfg(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(
    config: str = "configs/downstream_transformer/default.yaml",
    fold: str | None = None,
    calib: str = "9_1",
) -> None:
    cfg = _load_cfg(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate_downstream] device={device}  config={config}  calib=val_{calib}")

    model_cfg = cfg["model"]
    encoder = TimeSeriesTransformerEncoder(
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
    )
    encoder = load_encoder_frozen(cfg.get("pretrain_ckpt"), encoder).to(device)

    if fold is None or fold.lower() == "all":
        evaluate_all_folds(cfg, encoder, device, calib=calib)
    else:
        if fold not in ALL_FOLDS:
            raise ValueError(f"알 수 없는 fold: {fold!r}\n유효: {ALL_FOLDS}")
        evaluate_fold(fold, cfg, encoder, device, calib=calib)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downstream evaluate")
    parser.add_argument(
        "--config", default="configs/downstream_transformer/default.yaml",
        help="downstream config yaml 경로"
    )
    parser.add_argument(
        "--fold", default=None,
        help="평가할 fold 이름 (생략 or 'all'이면 전부)"
    )
    parser.add_argument(
        "--calib", default="9_1", choices=["9_1", "50_50"],
        help="threshold 캘리브레이션 기준 분포 (default: 9_1)"
    )
    args = parser.parse_args()
    main(config=args.config, fold=args.fold, calib=args.calib)
