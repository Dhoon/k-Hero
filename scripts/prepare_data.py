"""CLI: 원본 LP 데이터 → 전처리 → 채널별 정규화 → 슬라이딩 윈도우 → data/processed/ 저장.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --config configs/data/default.yaml

출력 구조:
    data/processed/
        pretrain/
            train/
                X.npy             (N_train, window_size, n_features)
                time_feat.npy     (N_train, window_size, 2)  ← hour_of_day, day_of_week
                future_target.npy (N_train, forecast_horizon, n_features)
            val/
                X.npy, time_feat.npy, future_target.npy
            test/
                X.npy, time_feat.npy, future_target.npy
        scaler.joblib       ← train으로만 fit한 StandardScalerND
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

from src.adt.data.loaders import load_all_meters, check_송전_nonzero
from src.adt.data.preprocessing import preprocess_meter
from src.adt.data.scalers import StandardScalerND
from src.adt.data.windowing import process_meter_segments, concat_meter_splits


# -------------------------------------------------------------------------
# 헬퍼
# -------------------------------------------------------------------------

def _load_cfg(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_split(split_dir: Path, arrays: dict) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "X.npy", arrays["X"])
    np.save(split_dir / "time_feat.npy", arrays["time_feat"])
    np.save(split_dir / "future_target.npy", arrays["future_target"])


def _report_split(name: str, arrays: dict, feature_names: list[str]) -> None:
    X = arrays["X"]
    tf = arrays["time_feat"]
    ft = arrays["future_target"]
    print(f"\n  [{name}]  windows={X.shape[0]}  X={X.shape}  future_target={ft.shape}")
    nan_count = int(np.isnan(X).sum())
    print(f"    NaN 개수: {nan_count}")
    print(f"    채널별 정규화 후 통계 (mean / std):")
    for i, col in enumerate(feature_names):
        m = float(np.nanmean(X[..., i]))
        s = float(np.nanstd(X[..., i]))
        print(f"      {col:30s}  mean={m:+.4f}  std={s:.4f}")
    print(f"    time_feat shape: {tf.shape}  (hour_of_day, day_of_week)")


# -------------------------------------------------------------------------
# 메인
# -------------------------------------------------------------------------

def main(config_path: str = "configs/data/default.yaml") -> None:
    cfg = _load_cfg(config_path)

    raw_dir = Path(cfg["raw_dir"])
    processed_dir = Path(cfg["processed_dir"]) / "pretrain"
    feature_cols: list[str] = cfg["feature_cols"]
    resample_freq: str = cfg.get("resample_freq", "15min")
    fill_method: str = cfg.get("fill_method", "linear")
    gap_threshold_hours: float = cfg.get("gap_threshold_hours", 6.0)
    window_size: int = cfg["window_size"]
    stride: int = cfg.get("stride", 1)
    forecast_horizon: int = cfg.get("forecast_horizon", 0)
    ratios: list[float] = cfg["train_val_test_split"]

    print("=" * 60)
    print("  LP 데이터 준비 파이프라인")
    print("=" * 60)
    print(f"  config          : {config_path}")
    print(f"  raw_dir         : {raw_dir}")
    print(f"  feature         : {feature_cols}")
    print(f"  window          : size={window_size}, stride={stride}, forecast_horizon={forecast_horizon}")
    print(f"  split           : {ratios}")
    print(f"  gap_threshold   : {gap_threshold_hours}h")

    # ------------------------------------------------------------------
    # 0. processed_dir 초기화 (stale 파일 방지)
    # ------------------------------------------------------------------
    import shutil
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        print(f"\n[초기화] {processed_dir} 삭제 완료")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. 로드 + meter_id 검증
    # ------------------------------------------------------------------
    print("\n[1/5] xlsx 로드")
    meter_dfs_raw = load_all_meters(raw_dir)
    check_송전_nonzero(meter_dfs_raw)

    # ------------------------------------------------------------------
    # 2. 전처리 (계량기별 → segment 리스트)
    # ------------------------------------------------------------------
    print("\n[2/5] 전처리 (datetime 파싱 → gap 분리 → segment별 리샘플링)")
    meter_segments: dict[str, list] = {}
    scaler_dir = Path(cfg["processed_dir"])
    for meter_id, df_raw in meter_dfs_raw.items():
        segments = preprocess_meter(
            df_raw,
            freq=resample_freq,
            method=fill_method,
            gap_threshold_hours=gap_threshold_hours,
            window_size=window_size,
            meter_id=meter_id,
            status_events_dir=scaler_dir / "status_events",
            diff_features=feature_cols,   # 적산값 → 구간 사용량 변환
        )
        meter_segments[meter_id] = segments
        total_rows = sum(len(s) for s in segments)
        print(
            f"\n  meter={meter_id}  유효segment={len(segments)}개  "
            f"합계rows={total_rows}"
        )

    # ------------------------------------------------------------------
    # 3. 슬라이딩 윈도우 + split (segment별, 후 meter 간 concat)
    # ------------------------------------------------------------------
    print("\n[3/5] 슬라이딩 윈도우 생성 + train/val/test 분할")
    all_meter_splits = []
    for meter_id, segments in meter_segments.items():
        if not segments:
            print(f"  [스킵] meter={meter_id}: 유효 segment 없음")
            continue
        splits = process_meter_segments(
            segments, feature_cols, window_size, stride, ratios, forecast_horizon
        )
        n_train = splits["train"]["X"].shape[0]
        n_val = splits["val"]["X"].shape[0]
        n_test = splits["test"]["X"].shape[0]
        print(
            f"  meter={meter_id}  train={n_train}  val={n_val}  test={n_test}"
        )
        if n_train + n_val + n_test == 0:
            print(f"  [경고] meter={meter_id}: 생성된 윈도우 0개. 스킵.")
            continue
        all_meter_splits.append(splits)

    if not all_meter_splits:
        print("[오류] 사용 가능한 계량기 데이터가 없습니다. 중단.")
        sys.exit(1)

    combined = concat_meter_splits(all_meter_splits)
    print(
        f"\n  합산  train={combined['train']['X'].shape[0]}"
        f"  val={combined['val']['X'].shape[0]}"
        f"  test={combined['test']['X'].shape[0]}"
    )

    # ------------------------------------------------------------------
    # 4. Scaler fit (train 전용) → transform 전체 (X + future_target)
    # ------------------------------------------------------------------
    print("\n[4/5] 채널별 StandardScaler fit (train 전용) + transform")
    scaler = StandardScalerND()
    combined["train"]["X"] = scaler.fit_transform(combined["train"]["X"])
    combined["val"]["X"] = scaler.transform(combined["val"]["X"])
    combined["test"]["X"] = scaler.transform(combined["test"]["X"])

    # future_target도 같은 채널 공간이므로 동일 scaler로 정규화
    for split_name in ("train", "val", "test"):
        combined[split_name]["future_target"] = scaler.transform(
            combined[split_name]["future_target"]
        )

    scaler_path = scaler_dir / "scaler.joblib"
    scaler.save(scaler_path)
    print(f"  scaler 저장: {scaler_path}")
    scaler.report(feature_names=feature_cols)

    # ------------------------------------------------------------------
    # 5. 저장
    # ------------------------------------------------------------------
    print("\n[5/5] data/processed/pretrain/ 저장")
    for split_name in ("train", "val", "test"):
        split_dir = processed_dir / split_name
        _save_split(split_dir, combined[split_name])
        print(
            f"  {split_name}/X.npy              {combined[split_name]['X'].shape}"
            f"  {combined[split_name]['X'].nbytes / 1e6:.1f} MB"
        )
        print(
            f"  {split_name}/time_feat.npy      {combined[split_name]['time_feat'].shape}"
        )
        print(
            f"  {split_name}/future_target.npy  {combined[split_name]['future_target'].shape}"
            f"  {combined[split_name]['future_target'].nbytes / 1e6:.1f} MB"
        )

    # ------------------------------------------------------------------
    # 검증 요약
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  검증 요약")
    print("=" * 60)
    for split_name in ("train", "val", "test"):
        _report_split(split_name, combined[split_name], feature_cols)

    print("\n[완료] 데이터 준비 파이프라인 종료.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LP 데이터 전처리 파이프라인")
    parser.add_argument(
        "--config",
        default="configs/data/default.yaml",
        help="데이터 설정 파일 경로 (default: configs/data/default.yaml)",
    )
    args = parser.parse_args()
    main(config_path=args.config)
