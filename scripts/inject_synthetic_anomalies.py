"""CLI: 합성 이상치를 주입해 레이블 있는 평가셋 생성.

핵심 설계:
  data/processed/test/X.npy는 stride=1로 겹치는 윈도우라서 그대로 쓰면
  하나의 이상 이벤트가 수백 개 윈도우에 중복돼 AUROC/F1이 왜곡된다.
  → test split에서 stride=window_size(96)로 1개씩 건너뛰어 비겹침 윈도우만 사용.

Usage:
    python scripts/inject_synthetic_anomalies.py
    python scripts/inject_synthetic_anomalies.py --config configs/eval/synthetic.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.adt.data.anomaly_injection import inject_synthetic_anomalies


def main() -> None:
    parser = argparse.ArgumentParser(description="합성 이상치 주입 → synthetic_eval 데이터셋 생성")
    parser.add_argument(
        "--config",
        default="configs/eval/synthetic.yaml",
        help="synthetic 이상치 설정 파일",
    )
    parser.add_argument(
        "--data-config",
        default="configs/data/default.yaml",
        help="데이터 설정 파일 (window_size 등)",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(args.data_config, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    window_size: int = data_cfg["window_size"]
    processed_dir = Path(data_cfg["processed_dir"])
    source_split: str = cfg.get("source_split", "test")
    output_dir = Path(cfg["output_dir"])
    seed: int = cfg.get("seed", 42)

    # --- test split 로드 ---
    split_dir = processed_dir / source_split
    X_all = np.load(split_dir / "X.npy")          # (N, T, C), stride=1 overlapping
    tf_all = np.load(split_dir / "time_feat.npy") # (N, T, 2)

    # stride=1 → stride=window_size 로 비겹침 샘플링
    X_clean = X_all[::window_size]
    tf_clean = tf_all[::window_size]
    N = len(X_clean)

    print(f"[inject] {source_split} split: {len(X_all)} 윈도우 (stride=1) "
          f"→ {N} 비겹침 윈도우 (stride={window_size})")

    # --- 이상치 주입 ---
    X_corrupted, labels = inject_synthetic_anomalies(X_clean, cfg, seed=seed)
    n_pos = int(labels.sum())
    print(f"[inject] 주입 윈도우: {n_pos}/{N} "
          f"({n_pos/N*100:.1f}%)  seed={seed}")

    # --- 저장 ---
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X.npy", X_corrupted.astype(np.float32))
    np.save(output_dir / "clean_X.npy", X_clean.astype(np.float32))
    np.save(output_dir / "labels.npy", labels.astype(np.int32))
    np.save(output_dir / "time_feat.npy", tf_clean.astype(np.float32))

    print(f"[inject] 저장 완료 → {output_dir}/")
    print(f"  X.npy        {X_corrupted.shape}")
    print(f"  clean_X.npy  {X_clean.shape}")
    print(f"  labels.npy   {labels.shape}  (1={n_pos}, 0={N - n_pos})")
    print(f"  time_feat.npy {tf_clean.shape}")


if __name__ == "__main__":
    main()
