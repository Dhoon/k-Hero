"""LSTM Joint Detection+Classification 평가 루프.

평가 원칙:
  - threshold 탐색: val_50_50 F1-max sweep
  - Detection 보고: test_50_50 (default 0.5 + optimal threshold)
  - Classification: test_50_50, attack + known type만 대상
  - checkpoint: checkpoints/downstream_lstm/joint/{fold}/best.pt
                (encoder + det_head + cls_head 한 파일)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.adt.data.labeling import TYPE_IDX
from src.adt.engine.evaluate_downstream_lstm import _infer_lstm, _infer_filtered_lstm
from src.adt.engine.evaluate_downstream_transformer import (
    _TYPE_ORDER,
    find_best_threshold,
    plot_per_type_recall,
)
from src.adt.engine.train_downstream_transformer import (
    ALL_FOLDS,
    FOLD_UNSEEN_TYPE,
    IDX_TO_TYPE,
    DownstreamFoldDataset,
    _remap_type_labels,
)
from src.adt.models.lstm_joint import LSTMJoint
from src.adt.utils.logger import get_logger


def _measure_latency_joint(
    model: LSTMJoint,
    T: int,
    C: int,
    device: torch.device,
    n_iters: int = 100,
) -> dict[str, float]:
    dummy = torch.randn(1, T, C, device=device)
    model.eval()

    with torch.no_grad():
        for _ in range(10):
            model(dummy)

    cpu_times: list[float] = []
    with torch.no_grad():
        for _ in range(n_iters):
            t0 = time.perf_counter()
            model(dummy)
            cpu_times.append((time.perf_counter() - t0) * 1000)

    result: dict[str, float] = {"cpu_ms": sum(cpu_times) / len(cpu_times)}

    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)
        gpu_times: list[float] = []
        with torch.no_grad():
            for _ in range(n_iters):
                starter.record()
                model(dummy)
                ender.record()
                torch.cuda.synchronize()
                gpu_times.append(starter.elapsed_time(ender))
        result["gpu_ms"] = sum(gpu_times) / len(gpu_times)

    return result


def _calibrate_threshold_joint(
    fold_name: str,
    cfg: dict,
    model: LSTMJoint,
    device: torch.device,
    verbose: bool = True,
) -> tuple[float, float]:
    """val_50_50 F1-max threshold 탐색 → threshold.json 저장."""
    downstream_dir = Path(cfg["downstream_dir"])
    train_cfg      = cfg["training"]
    val_dir  = downstream_dir / fold_name / "val_50_50"
    ds_val   = DownstreamFoldDataset(val_dir)
    loader   = DataLoader(
        ds_val, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=0
    )

    logits_np, bl_np, _ = _infer_lstm(model.encoder, model.det_head, loader, device)
    probs = torch.sigmoid(torch.from_numpy(logits_np)).numpy()

    _pos = probs[bl_np == 1]
    _neg = probs[bl_np == 0]
    if len(_pos) > 0 and len(_neg) > 0:
        print(
            f"[diag/{fold_name}] positive n={len(_pos)}"
            f"  min={_pos.min():.4f}  median={np.median(_pos):.4f}"
            f"  max={_pos.max():.4f}  >0.99={(_pos>0.99).mean():.3f}"
        )
        print(
            f"[diag/{fold_name}] negative n={len(_neg)}"
            f"  median={np.median(_neg):.4f}"
            f"  p90={np.percentile(_neg,90):.4f}  <0.01={(_neg<0.01).mean():.3f}"
        )

    threshold, val_f1 = find_best_threshold(probs, bl_np)

    ckpt_dir = Path(train_cfg["ckpt_dir"]) / fold_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "threshold.json").write_text(
        json.dumps({"threshold": threshold, "val_f1": val_f1}, indent=2),
        encoding="utf-8",
    )
    if verbose:
        print(
            f"[joint threshold/{fold_name}] val_50_50  "
            f"threshold={threshold:.4f}  val_f1={val_f1:.4f}"
        )
    return threshold, val_f1


def evaluate_fold_lstm_joint(
    fold_name: str,
    cfg: dict[str, Any],
    device: torch.device,
    verbose: bool = True,
) -> dict[str, Any]:
    """단일 fold Joint 평가.  test_50_50 Detection + Classification."""
    downstream_dir = Path(cfg["downstream_dir"])
    train_cfg      = cfg["training"]
    heads_cfg      = cfg["heads"]
    model_cfg      = cfg["model"]

    log_dir = Path(train_cfg.get("log_dir", "logs/downstream_lstm_joint")) / fold_name
    logger, writer = get_logger(
        log_dir, name=f"adt.eval_joint.{fold_name}", log_file="eval.log"
    )

    data_cfg_path = Path(cfg.get("data_config", "configs/data/default.yaml"))
    with open(data_cfg_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    n_features = len(data_cfg["feature_cols"])
    T          = int(data_cfg.get("window_size", 96))
    hidden_dim = model_cfg["hidden_dim"]
    num_layers = model_cfg["num_layers"]

    ds_test_50 = DownstreamFoldDataset(downstream_dir / "all_type" / "test_50_50")
    loader_50  = DataLoader(
        ds_test_50, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=0
    )

    ckpt_dir   = Path(train_cfg["ckpt_dir"]) / fold_name
    best_path  = ckpt_dir / "best.pt"

    # class_names: checkpoint 우선, 없으면 class_names.json fallback
    if best_path.exists():
        ckpt_data = torch.load(best_path, map_location="cpu")
        class_names: dict[str, str] = ckpt_data.get(
            "class_names",
            {str(i): IDX_TO_TYPE[i] for i in range(5)},
        )
        num_classes = ckpt_data.get("num_classes", len(class_names))
    else:
        cn_path = ckpt_dir / "class_names.json"
        class_names = (
            json.loads(cn_path.read_text(encoding="utf-8"))
            if cn_path.exists()
            else {str(i): IDX_TO_TYPE[i] for i in range(5)}
        )
        num_classes = len(class_names)
        ckpt_data   = {}

    cls_idx_to_name = {int(k): v for k, v in class_names.items()}

    model = LSTMJoint(
        n_features, hidden_dim, num_layers, num_classes,
        det_hidden=heads_cfg["hidden_dim"],
        cls_hidden=heads_cfg["hidden_dim"],
        dropout=heads_cfg["dropout"],
    ).to(device)

    if best_path.exists():
        model.encoder.load_state_dict(ckpt_data["encoder"])
        model.det_head.load_state_dict(ckpt_data["det_head"])
        model.cls_head.load_state_dict(ckpt_data["cls_head"])
        logger.info(f"loaded from {best_path}")
    else:
        logger.warning(f"best.pt not found: {best_path}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ── 파라미터 수 + latency ─────────────────────────────────────────────
    n_params  = sum(p.numel() for p in model.parameters())
    latency   = _measure_latency_joint(model, T, n_features, device)
    lat_str   = f"cpu={latency['cpu_ms']:.3f}ms"
    if "gpu_ms" in latency:
        lat_str += f"  gpu={latency['gpu_ms']:.3f}ms"
    logger.info(f"params={n_params:,}  latency (batch=1, 100iter): {lat_str}")

    # ── threshold calibration (val_50_50) ─────────────────────────────────
    threshold, val_f1 = _calibrate_threshold_joint(fold_name, cfg, model, device, verbose)

    # ── recall-target 테이블 (val_50_50) ──────────────────────────────────
    _val50_dir = downstream_dir / fold_name / "val_50_50"
    if _val50_dir.exists():
        from sklearn.metrics import precision_recall_curve as _prc
        _ds_v50 = DownstreamFoldDataset(_val50_dir)
        _ld_v50 = DataLoader(_ds_v50, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=0)
        _lg_v50, _bl_v50, _ = _infer_lstm(model.encoder, model.det_head, _ld_v50, device)
        _pr_v50 = torch.sigmoid(torch.from_numpy(_lg_v50)).numpy()
        _precs, _recs, _thrs = _prc(_bl_v50, _pr_v50)
        _precs, _recs = _precs[:-1], _recs[:-1]
        logger.info("[recall-target / val_50_50]")
        for _tgt in [0.70, 0.80, 0.85, 0.90, 0.95]:
            _idx = np.where(_recs >= _tgt)[0]
            if len(_idx) == 0:
                logger.info(f"  recall>={_tgt:.2f}: 도달 불가")
            else:
                _best = _idx[np.argmax(_thrs[_idx])]
                logger.info(
                    f"  recall>={_tgt:.2f}: thr={_thrs[_best]:.4f}"
                    f"  precision={_precs[_best]:.4f}  recall={_recs[_best]:.4f}"
                )

    unseen_type = FOLD_UNSEEN_TYPE.get(fold_name)

    # ── Detection eval (test_50_50) ────────────────────────────────────────
    logits_50, bl_50, tl_50 = _infer_lstm(model.encoder, model.det_head, loader_50, device)
    probs_50     = torch.sigmoid(torch.from_numpy(logits_50)).numpy()
    pred_default = (probs_50 >= 0.5).astype(np.int32)
    pred_optimal = (probs_50 >  threshold).astype(np.int32)

    def _m(pred: np.ndarray) -> dict[str, float]:
        return {
            "accuracy":  float(accuracy_score(bl_50, pred)),
            "precision": float(precision_score(bl_50, pred, zero_division=0)),
            "recall":    float(recall_score(bl_50, pred, zero_division=0)),
            "f1":        float(f1_score(bl_50, pred, zero_division=0)),
        }

    try:
        auc_roc = float(roc_auc_score(bl_50, probs_50))
    except Exception:
        auc_roc = float("nan")
    try:
        auc_pr = float(average_precision_score(bl_50, probs_50))
    except Exception:
        auc_pr = float("nan")

    per_type_recall: dict[str, float] = {}
    for type_name, orig_idx in TYPE_IDX.items():
        mask = tl_50 == orig_idx
        if mask.sum() > 0:
            per_type_recall[type_name] = float(pred_optimal[mask].mean())

    det_metrics: dict[str, Any] = {
        **_m(pred_optimal),
        "threshold":       threshold,
        "val_f1":          val_f1,
        "auc_roc":         auc_roc,
        "auc_pr":          auc_pr,
        "default_thr":     _m(pred_default),
        "per_type_recall": per_type_recall,
    }

    logger.info(f"Detection [test_50_50]  thr={threshold:.4f}  val_f1={val_f1:.4f}")
    logger.info(
        f"  default(0.50): acc={det_metrics['default_thr']['accuracy']:.3f}  "
        f"prec={det_metrics['default_thr']['precision']:.3f}  "
        f"rec={det_metrics['default_thr']['recall']:.3f}  "
        f"F1={det_metrics['default_thr']['f1']:.3f}"
    )
    logger.info(
        f"  optimal:       acc={det_metrics['accuracy']:.3f}  "
        f"prec={det_metrics['precision']:.3f}  "
        f"rec={det_metrics['recall']:.3f}  "
        f"F1={det_metrics['f1']:.3f}"
    )
    logger.info(f"  AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}")
    logger.info("  per-type recall (optimal threshold):")
    for tn in [t for t in _TYPE_ORDER if t in per_type_recall]:
        mark = "  ◀ UNSEEN" if tn == unseen_type else ""
        logger.info(f"    {tn:20s} recall={per_type_recall[tn]:.3f}{mark}")
    writer.add_scalar("eval/auc_roc_50_50", auc_roc)

    # ── Classification (test_50_50, known attack types) ───────────────────
    known_type_names = set(class_names.values())
    known_orig_idxs  = {TYPE_IDX[n] for n in known_type_names if n in TYPE_IDX}
    cls_mask = np.array(
        [bl_50[i] == 1 and tl_50[i] in known_orig_idxs for i in range(len(bl_50))]
    )

    if cls_mask.sum() == 0:
        cls_metrics: dict[str, Any] = {
            "accuracy": float("nan"),
            "per_class": {},
            "confusion_matrix": [],
        }
    else:
        tl_cls      = tl_50[cls_mask]
        orig_to_cls = {TYPE_IDX[name]: int(k) for k, name in class_names.items()}
        cls_targets = np.array([orig_to_cls[t] for t in tl_cls], dtype=np.int32)

        cls_logits_all, _, _ = _infer_filtered_lstm(
            model.encoder, model.cls_head, ds_test_50,
            cls_mask, train_cfg["batch_size"], device,
        )
        cls_preds = cls_logits_all.argmax(axis=1)

        precs = precision_score(
            cls_targets, cls_preds, average=None,
            labels=list(range(num_classes)), zero_division=0,
        )
        recs = recall_score(
            cls_targets, cls_preds, average=None,
            labels=list(range(num_classes)), zero_division=0,
        )
        macro_prec = float(precision_score(cls_targets, cls_preds, average="macro", zero_division=0))
        macro_rec  = float(recall_score(cls_targets, cls_preds, average="macro", zero_division=0))
        macro_f1   = float(f1_score(cls_targets, cls_preds, average="macro", zero_division=0))
        cls_metrics = {
            "accuracy":   float(accuracy_score(cls_targets, cls_preds)),
            "macro_prec": macro_prec,
            "macro_rec":  macro_rec,
            "macro_f1":   macro_f1,
            "per_class": {
                cls_idx_to_name.get(i, str(i)): {
                    "precision": float(precs[i]),
                    "recall":    float(recs[i]),
                }
                for i in range(num_classes)
            },
            "confusion_matrix": confusion_matrix(
                cls_targets, cls_preds, labels=list(range(num_classes))
            ).tolist(),
        }
        logger.info(
            f"Classification  acc={cls_metrics['accuracy']:.3f}  "
            f"macro: prec={macro_prec:.3f}  rec={macro_rec:.3f}  F1={macro_f1:.3f}  "
            f"(N={cls_mask.sum()})"
        )
        for cn, cm_val in cls_metrics["per_class"].items():
            logger.info(
                f"    {cn:20s} "
                f"prec={cm_val['precision']:.3f}  rec={cm_val['recall']:.3f}"
            )

    # ── 결과 저장 ─────────────────────────────────────────────────────────
    out_root = Path(cfg.get("output_dir", "outputs/scores_lstm_joint"))
    for task, metrics in [
        ("detector_50_50", {**det_metrics, "n_params": n_params, "latency": latency}),
        ("classifier",     cls_metrics),
    ]:
        out_dir = out_root / fold_name / task
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    fig_dir = Path(cfg.get("figure_dir", "outputs/figures_lstm_joint")) / fold_name
    plot_per_type_recall(fold_name, per_type_recall, fig_dir, unseen_type)

    writer.close()
    return {
        "detection_50_50": det_metrics,
        "classification":  cls_metrics,
        "n_params":        n_params,
        "latency":         latency,
    }


def evaluate_all_folds_lstm_joint(
    cfg: dict[str, Any],
    device: torch.device,
    folds: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, dict]:
    targets = folds if folds is not None else ALL_FOLDS
    results: dict[str, dict] = {}
    for fold in targets:
        if verbose:
            print(f"\n{'='*60}\n fold: {fold}\n{'='*60}")
        results[fold] = evaluate_fold_lstm_joint(fold, cfg, device, verbose=verbose)
    return results
