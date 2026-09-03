"""Detection head threshold 진단 스크립트.

진단 항목:
  1. val set sigmoid 확률 분포 (Normal / Attack 그룹별 mean/median/percentile)
  2. AUC-ROC + AUC-PR (threshold 무관 순위 지표)
  3. val set threshold sweep (0.05~0.95) -> best-F1 threshold -> test set 재평가
  4. 전체 6개 fold 요약 테이블

test set: 항상 all_type/test (6개 fold 공유)
val set : 각 fold 고유 val

사용법::
    python scripts/diagnose_threshold.py
    python scripts/diagnose_threshold.py --config configs/downstream/default.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
)
from torch.utils.data import DataLoader

from src.adt.models.encoder import TimeSeriesTransformerEncoder
from src.adt.models.heads.detection_head import DetectionHead
from src.adt.utils.checkpoint import load_encoder_frozen
from src.adt.engine.train_downstream import (
    ALL_FOLDS, DownstreamFoldDataset,
)


# ─────────────────────────────────────────────────────────────────────────────
# 추론 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _infer_detection(
    encoder: nn.Module,
    det_head: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Detection head 전체 추론.

    Returns:
        probs : (N,) float32  sigmoid 확률
        labels: (N,) int32    binary_label
    """
    probs_list, label_list = [], []
    for x, tf, bl, _ in loader:
        x, tf = x.to(device), tf.to(device)
        z_t   = encoder(x, tf)
        logit = det_head(z_t)            # (B,)
        probs_list.append(torch.sigmoid(logit).cpu())
        label_list.append(bl)
    probs  = torch.cat(probs_list).numpy().astype(np.float32)
    labels = torch.cat(label_list).numpy().astype(np.int32)
    return probs, labels


# ─────────────────────────────────────────────────────────────────────────────
# 단일 fold 진단
# ─────────────────────────────────────────────────────────────────────────────

def _pct_str(arr: np.ndarray) -> str:
    return (
        f"p5={np.percentile(arr, 5):.3f}  "
        f"p25={np.percentile(arr, 25):.3f}  "
        f"median={np.median(arr):.3f}  "
        f"p75={np.percentile(arr, 75):.3f}  "
        f"p95={np.percentile(arr, 95):.3f}"
    )


def diagnose_fold(
    fold_name: str,
    cfg: dict,
    encoder: nn.Module,
    device: torch.device,
) -> dict:
    """단일 fold 진단. 결과 dict 반환."""
    downstream_dir = Path(cfg["downstream_dir"])
    det_cfg        = cfg["detection"]
    model_cfg      = cfg["model"]

    # ── Detection head 로드 ───────────────────────────────────────────────
    det_ckpt = Path(det_cfg["ckpt_dir"]) / fold_name / "detector" / "best.pt"
    det_head = DetectionHead(
        d_model   = model_cfg["d_model"],
        hidden_dim= det_cfg["hidden_dim"],
        dropout   = 0.0,
    ).to(device)
    if det_ckpt.exists():
        state = torch.load(det_ckpt, map_location="cpu")
        det_head.load_state_dict(state["head"])
        print(f"  [OK] detector ckpt loaded: {det_ckpt}")
    else:
        print(f"  [WARN] detector ckpt not found: {det_ckpt} -- using random init")
    det_head.eval()

    bs = det_cfg["batch_size"]

    # ── (1) val set 추론 ──────────────────────────────────────────────────
    val_dir = downstream_dir / fold_name / "val"
    if not val_dir.exists():
        print(f"  [SKIP] val dir not found: {val_dir}")
        return {}

    ds_val   = DownstreamFoldDataset(val_dir)
    val_loader = DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=0)
    val_probs, val_labels = _infer_detection(encoder, det_head, val_loader, device)

    n_mask = (val_labels == 0)
    a_mask = (val_labels == 1)
    p_normal = val_probs[n_mask]
    p_attack = val_probs[a_mask]

    print(f"\n  -- val set sigmoid 분포 (N={len(val_labels):,}) --")
    print(f"  Normal ({n_mask.sum():,})  mean={p_normal.mean():.4f}  {_pct_str(p_normal)}")
    print(f"  Attack ({a_mask.sum():,})  mean={p_attack.mean():.4f}  {_pct_str(p_attack)}")

    overlap = (p_normal.mean() + p_attack.mean()) / 2
    sep     = p_attack.mean() - p_normal.mean()
    print(f"  Attack-Normal mean gap: {sep:+.4f}  ({'분리됨' if sep > 0.1 else '거의 겹침'})")

    # ── (2) AUC-ROC / AUC-PR ─────────────────────────────────────────────
    auc_roc = float(roc_auc_score(val_labels, val_probs))
    auc_pr  = float(average_precision_score(val_labels, val_probs))
    print(f"\n  AUC-ROC : {auc_roc:.4f}  (0.5=random, 1.0=perfect)")
    print(f"  AUC-PR  : {auc_pr:.4f}   (baseline={a_mask.mean():.3f} = attack ratio)")

    # ── (3) val threshold sweep -> best F1 ───────────────────────────────
    thresholds = np.arange(0.05, 1.00, 0.05)
    best_thr, best_f1 = 0.5, -1.0
    rows = []
    for thr in thresholds:
        pred = (val_probs >= thr).astype(np.int32)
        f1   = float(f1_score(val_labels, pred, zero_division=0))
        prec = float(precision_score(val_labels, pred, zero_division=0))
        rec  = float(recall_score(val_labels, pred, zero_division=0))
        rows.append((thr, prec, rec, f1))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    print(f"\n  -- val threshold sweep --")
    print(f"  {'thr':>6}  {'prec':>6}  {'rec':>6}  {'F1':>6}")
    for thr, prec, rec, f1 in rows:
        mark = " <-- best" if abs(thr - best_thr) < 1e-6 else ""
        print(f"  {thr:>6.2f}  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}{mark}")
    print(f"\n  best val threshold: {best_thr:.2f}  (F1={best_f1:.4f})")

    # ── (4) test set 재평가 (all_type/test 고정) ─────────────────────────
    test_dir = downstream_dir / "all_type" / "test"
    result_test: dict = {}
    if test_dir.exists():
        ds_test    = DownstreamFoldDataset(test_dir)
        test_loader = DataLoader(ds_test, batch_size=bs, shuffle=False, num_workers=0)
        test_probs, test_labels = _infer_detection(encoder, det_head, test_loader, device)

        # default threshold=0.5
        pred_05  = (test_probs >= 0.5).astype(np.int32)
        f1_05    = float(f1_score(test_labels, pred_05, zero_division=0))
        prec_05  = float(precision_score(test_labels, pred_05, zero_division=0))
        rec_05   = float(recall_score(test_labels, pred_05, zero_division=0))
        auc_test = float(roc_auc_score(test_labels, test_probs))

        # best threshold from val
        pred_best = (test_probs >= best_thr).astype(np.int32)
        f1_best   = float(f1_score(test_labels, pred_best, zero_division=0))
        prec_best = float(precision_score(test_labels, pred_best, zero_division=0))
        rec_best  = float(recall_score(test_labels, pred_best, zero_division=0))

        print(f"\n  -- test set (all_type/test) --")
        print(f"  AUC-ROC (test): {auc_test:.4f}")
        print(f"  thr=0.50  prec={prec_05:.3f}  rec={rec_05:.3f}  F1={f1_05:.4f}")
        print(f"  thr={best_thr:.2f}  prec={prec_best:.3f}  rec={rec_best:.3f}  F1={f1_best:.4f}  <- val-best")

        result_test = {
            "auc_roc_test": auc_test,
            "f1_05":        f1_05,
            "prec_05":      prec_05,
            "rec_05":       rec_05,
            "best_thr":     best_thr,
            "f1_best":      f1_best,
            "prec_best":    prec_best,
            "rec_best":     rec_best,
        }
    else:
        print(f"  [SKIP] test dir not found: {test_dir}")

    return {
        "auc_roc_val": auc_roc,
        "auc_pr_val":  auc_pr,
        "best_thr":    best_thr,
        "best_f1_val": best_f1,
        **result_test,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 전체 fold 요약
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(summary: dict[str, dict]) -> None:
    print("\n" + "=" * 90)
    print("SUMMARY — 전체 fold Detection 진단")
    print("=" * 90)
    hdr = (
        f"{'fold':<24} {'AUC-ROC(val)':>13} {'AUC-PR(val)':>12} "
        f"{'best_thr':>9} {'F1(val)':>8} "
        f"{'AUC-ROC(tst)':>13} {'F1@0.5(tst)':>12} {'F1@best(tst)':>13}"
    )
    print(hdr)
    print("-" * 90)
    for fold, r in summary.items():
        if not r:
            print(f"  {fold:<24} -- skipped --")
            continue
        print(
            f"  {fold:<24} "
            f"{r.get('auc_roc_val', float('nan')):>13.4f} "
            f"{r.get('auc_pr_val',  float('nan')):>12.4f} "
            f"{r.get('best_thr',    float('nan')):>9.2f} "
            f"{r.get('best_f1_val', float('nan')):>8.4f} "
            f"{r.get('auc_roc_test',float('nan')):>13.4f} "
            f"{r.get('f1_05',       float('nan')):>12.4f} "
            f"{r.get('f1_best',     float('nan')):>13.4f}"
        )
    print("=" * 90)
    print("  AUC-ROC: 0.5=random  1.0=perfect")
    print("  AUC-PR : baseline = attack ratio (~0.10 for all_type)")
    print("  F1@best: val-best threshold 적용한 test 결과 (threshold 과적합 주의)")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main(config: str = "configs/downstream/default.yaml") -> None:
    cfg    = yaml.safe_load(open(config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  config={config}")

    # encoder 로드 (frozen)
    model_cfg = cfg["model"]
    ds_dir    = Path(cfg["downstream_dir"]) / "all_type" / "train"
    n_feat    = int(np.load(ds_dir / "X.npy", mmap_mode="r").shape[-1])
    encoder   = TimeSeriesTransformerEncoder(
        n_features=n_feat,
        d_model   =model_cfg["d_model"],
        n_heads   =model_cfg["n_heads"],
        n_layers  =model_cfg["n_layers"],
        d_ff      =model_cfg["d_ff"],
        dropout   =0.0,
    )
    ckpt_path = Path(cfg.get("pretrain_ckpt", "checkpoints/pretrain/best.pt"))
    encoder   = load_encoder_frozen(ckpt_path, encoder).to(device)
    print(f"encoder loaded from {ckpt_path}\n")

    summary: dict[str, dict] = {}
    for fold in ALL_FOLDS:
        print("=" * 60)
        print(f"fold: {fold}")
        print("=" * 60)
        result = diagnose_fold(fold, cfg, encoder, device)
        summary[fold] = result

    print_summary(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/downstream/default.yaml")
    args = parser.parse_args()
    main(args.config)
