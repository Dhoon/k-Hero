"""LSTM Downstream Detection + Classification 평가 루프.

평가 원칙:
  - threshold 탐색: val_9_1 (실전 분포) → F1-max sweep
  - Detection 보고: test_50_50 (native balanced) + test_9_1 (field-representative) 두 view
  - Classification: test_50_50 사용, attack 샘플 + known type만 대상
  - 추가 출력: 파라미터 수 (encoder + det_head), CPU/GPU latency (batch=1, 100회 평균)
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
from src.adt.engine.evaluate_downstream import find_best_threshold, plot_per_type_recall
from src.adt.engine.train_downstream import (
    ALL_FOLDS,
    FOLD_UNSEEN_TYPE,
    IDX_TO_TYPE,
    DownstreamFoldDataset,
    _remap_type_labels,
    compute_class_info,
)
from src.adt.models.lstm_ae import (
    LSTMClassificationHead,
    LSTMDetectionHead,
    LSTMEncoder,
)


# ── 추론 헬퍼 ────────────────────────────────────────────────────────────────

@torch.no_grad()
def _infer_lstm(
    encoder: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """전체 loader 전방향 추론.

    Returns:
        logits_np : (N,)   detection raw logit
        bl_np     : (N,)   binary_label
        tl_np     : (N,)   type_label
    """
    encoder.eval()
    head.eval()
    logits_list, bl_list, tl_list = [], [], []
    for x, _, bl, tl in loader:
        x      = x.to(device)
        _, z   = encoder(x)
        logit  = head(z)
        logits_list.append(logit.cpu())
        bl_list.append(bl)
        tl_list.append(tl)
    return (
        torch.cat(logits_list).numpy(),
        torch.cat(bl_list).numpy().astype(np.int32),
        torch.cat(tl_list).numpy().astype(np.int32),
    )


@torch.no_grad()
def _infer_filtered_lstm(
    encoder: nn.Module,
    head: nn.Module,
    dataset: DownstreamFoldDataset,
    mask: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """mask가 True인 샘플만 head에 통과 (classification 전용)."""
    indices = np.where(mask)[0]
    logits_list: list[torch.Tensor] = []
    encoder.eval()
    head.eval()
    for i in range(0, len(indices), batch_size):
        idx_batch = indices[i: i + batch_size]
        xs        = torch.stack([dataset.X[j] for j in idx_batch]).to(device)
        _, z      = encoder(xs)
        logits_list.append(head(z).cpu())
    logits = torch.cat(logits_list).numpy()
    bl     = dataset.binary_label[indices].numpy().astype(np.int32)
    tl     = dataset.type_label[indices].numpy().astype(np.int32)
    return logits, bl, tl


# ── Latency 측정 ──────────────────────────────────────────────────────────────

def _measure_latency(
    encoder: nn.Module,
    det_head: nn.Module,
    T: int,
    C: int,
    device: torch.device,
    n_iters: int = 100,
) -> dict[str, float]:
    """단일 sample (batch=1) forward latency 측정."""
    dummy = torch.randn(1, T, C, device=device)
    encoder.eval()
    det_head.eval()

    # warmup
    with torch.no_grad():
        for _ in range(10):
            _, z = encoder(dummy)
            det_head(z)

    cpu_times: list[float] = []
    with torch.no_grad():
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _, z = encoder(dummy)
            det_head(z)
            cpu_times.append((time.perf_counter() - t0) * 1000)

    result: dict[str, float] = {"cpu_ms": sum(cpu_times) / len(cpu_times)}

    if device.type == "cuda":
        import torch.cuda
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)
        gpu_times: list[float] = []
        with torch.no_grad():
            for _ in range(n_iters):
                starter.record()
                _, z = encoder(dummy)
                det_head(z)
                ender.record()
                torch.cuda.synchronize()
                gpu_times.append(starter.elapsed_time(ender))
        result["gpu_ms"] = sum(gpu_times) / len(gpu_times)

    return result


# ── Threshold calibration ────────────────────────────────────────────────────

def _calibrate_threshold_lstm(
    fold_name: str,
    cfg: dict,
    encoder: nn.Module,
    det_head: nn.Module,
    device: torch.device,
    verbose: bool = True,
) -> tuple[float, float]:
    """val_9_1 기준 F1-max threshold 탐색 → threshold.json 저장."""
    downstream_dir = Path(cfg["downstream_dir"])
    det_cfg = cfg["detection"]
    val_dir = downstream_dir / fold_name / "val_9_1"
    ds_val  = DownstreamFoldDataset(val_dir)
    loader  = DataLoader(
        ds_val, batch_size=det_cfg["batch_size"], shuffle=False, num_workers=0
    )

    logits_np, bl_np, _ = _infer_lstm(encoder, det_head, loader, device)
    probs = torch.sigmoid(torch.from_numpy(logits_np)).numpy()
    threshold, val_f1 = find_best_threshold(probs, bl_np)

    det_ckpt_dir = Path(det_cfg["ckpt_dir"]) / fold_name / "detector"
    det_ckpt_dir.mkdir(parents=True, exist_ok=True)
    (det_ckpt_dir / "threshold.json").write_text(
        json.dumps({"threshold": threshold, "val_f1": val_f1}, indent=2),
        encoding="utf-8",
    )
    if verbose:
        print(
            f"[lstm threshold/{fold_name}] val_9_1  "
            f"threshold={threshold:.4f}  val_f1={val_f1:.4f}"
        )
    return threshold, val_f1


# ── fold 단위 평가 ────────────────────────────────────────────────────────────

def evaluate_fold_lstm(
    fold_name: str,
    cfg: dict[str, Any],
    device: torch.device,
    verbose: bool = True,
) -> dict[str, Any]:
    """단일 fold 평가.  test_50_50 + test_9_1 두 view + 분류 + latency."""
    downstream_dir = Path(cfg["downstream_dir"])
    det_cfg   = cfg["detection"]
    cls_cfg   = cfg["classification"]
    model_cfg = cfg["model"]

    data_cfg_path = Path(cfg.get("data_config", "configs/data/default.yaml"))
    with open(data_cfg_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    n_features     = len(data_cfg["feature_cols"])
    T              = int(data_cfg.get("window_size", 96))
    hidden_dim     = model_cfg["hidden_dim"]
    num_layers     = model_cfg["num_layers"]
    bottleneck_dim = 2 * hidden_dim

    # ── test sets (공유, all_type 아래) ──────────────────────────────────
    ds_test_50 = DownstreamFoldDataset(downstream_dir / "all_type" / "test_50_50")
    ds_test_9  = DownstreamFoldDataset(downstream_dir / "all_type" / "test_9_1")
    loader_50  = DataLoader(
        ds_test_50, batch_size=det_cfg["batch_size"], shuffle=False, num_workers=0
    )
    loader_9 = DataLoader(
        ds_test_9, batch_size=det_cfg["batch_size"], shuffle=False, num_workers=0
    )

    det_ckpt_dir = Path(det_cfg["ckpt_dir"]) / fold_name / "detector"
    cls_ckpt_dir = Path(cls_cfg["ckpt_dir"]) / fold_name / "classifier"

    # class_names
    class_names_path = cls_ckpt_dir / "class_names.json"
    if class_names_path.exists():
        class_names: dict[str, str] = json.loads(
            class_names_path.read_text(encoding="utf-8")
        )
    else:
        class_names = {str(i): IDX_TO_TYPE[i] for i in range(5)}
    num_classes    = len(class_names)
    cls_idx_to_name = {int(k): v for k, v in class_names.items()}

    # ── 모델 로드 ────────────────────────────────────────────────────────
    encoder  = LSTMEncoder(n_features, hidden_dim, num_layers).to(device)
    det_head = LSTMDetectionHead(
        bottleneck_dim, det_cfg["hidden_dim"], det_cfg["dropout"]
    ).to(device)
    cls_head = LSTMClassificationHead(
        bottleneck_dim, num_classes, cls_cfg["hidden_dim"], cls_cfg["dropout"]
    ).to(device)

    enc_path = det_ckpt_dir / "encoder_finetuned.pt"
    if enc_path.exists():
        encoder.load_state_dict(torch.load(enc_path, map_location="cpu")["encoder"])
        if verbose:
            print(f"[lstm eval/{fold_name}] finetuned encoder: {enc_path}")

    det_best = det_ckpt_dir / "best.pt"
    if det_best.exists():
        det_head.load_state_dict(torch.load(det_best, map_location="cpu")["head"])

    cls_best = cls_ckpt_dir / "best.pt"
    if cls_best.exists():
        cls_head.load_state_dict(torch.load(cls_best, map_location="cpu")["head"])

    encoder.eval()
    det_head.eval()
    cls_head.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # ── 파라미터 수 + latency ─────────────────────────────────────────────
    n_enc   = sum(p.numel() for p in encoder.parameters())
    n_det   = sum(p.numel() for p in det_head.parameters())
    n_total = n_enc + n_det
    latency = _measure_latency(encoder, det_head, T, n_features, device)

    if verbose:
        print(
            f"[lstm eval/{fold_name}] params: enc={n_enc:,}  "
            f"det_head={n_det:,}  total={n_total:,}"
        )
        lat_str = f"cpu={latency['cpu_ms']:.3f}ms"
        if "gpu_ms" in latency:
            lat_str += f"  gpu={latency['gpu_ms']:.3f}ms"
        print(f"[lstm eval/{fold_name}] latency (batch=1, {100}iter): {lat_str}")

    # ── threshold (val_9_1) ───────────────────────────────────────────────
    threshold, val_f1 = _calibrate_threshold_lstm(
        fold_name, cfg, encoder, det_head, device, verbose=verbose
    )

    unseen_type = FOLD_UNSEEN_TYPE.get(fold_name)

    def _run_det_eval(
        logits_np: np.ndarray, bl_np: np.ndarray, tl_np: np.ndarray
    ) -> dict[str, Any]:
        probs_v      = torch.sigmoid(torch.from_numpy(logits_np)).numpy()
        pred_default = (probs_v >= 0.5).astype(np.int32)
        pred_optimal = (probs_v >  threshold).astype(np.int32)

        def _m(pred: np.ndarray) -> dict[str, float]:
            return {
                "accuracy":  float(accuracy_score(bl_np, pred)),
                "precision": float(precision_score(bl_np, pred, zero_division=0)),
                "recall":    float(recall_score(bl_np, pred, zero_division=0)),
                "f1":        float(f1_score(bl_np, pred, zero_division=0)),
            }

        try:
            auc_roc = float(roc_auc_score(bl_np, probs_v))
        except Exception:
            auc_roc = float("nan")
        try:
            auc_pr = float(average_precision_score(bl_np, probs_v))
        except Exception:
            auc_pr = float("nan")

        ptr: dict[str, float] = {}
        for type_name, orig_idx in TYPE_IDX.items():
            mask = tl_np == orig_idx
            if mask.sum() > 0:
                ptr[type_name] = float(pred_optimal[mask].mean())

        return {
            **_m(pred_optimal),
            "threshold":       threshold,
            "val_f1":          val_f1,
            "auc_roc":         auc_roc,
            "auc_pr":          auc_pr,
            "default_thr":     _m(pred_default),
            "per_type_recall": ptr,
        }

    logits_50, bl_50, tl_50 = _infer_lstm(encoder, det_head, loader_50, device)
    logits_9,  bl_9,  tl_9  = _infer_lstm(encoder, det_head, loader_9,  device)
    det_metrics_50 = _run_det_eval(logits_50, bl_50, tl_50)
    det_metrics_9  = _run_det_eval(logits_9,  bl_9,  tl_9)
    per_type_recall = det_metrics_50["per_type_recall"]

    if verbose:
        for label, dm in [
            ("test_50_50 (native)", det_metrics_50),
            ("test_9_1  (field)",   det_metrics_9),
        ]:
            print(
                f"[lstm eval/{fold_name}] Detection [{label}]  "
                f"thr={threshold:.4f}  val_f1={val_f1:.4f}"
            )
            print(
                f"  default(0.50): acc={dm['default_thr']['accuracy']:.3f}  "
                f"prec={dm['default_thr']['precision']:.3f}  "
                f"rec={dm['default_thr']['recall']:.3f}  "
                f"F1={dm['default_thr']['f1']:.3f}"
            )
            print(
                f"  optimal:       acc={dm['accuracy']:.3f}  "
                f"prec={dm['precision']:.3f}  "
                f"rec={dm['recall']:.3f}  "
                f"F1={dm['f1']:.3f}"
            )
            print(f"  AUC-ROC={dm['auc_roc']:.4f}  AUC-PR={dm['auc_pr']:.4f}")
        print("  per-type recall (test_50_50, optimal threshold):")
        for tn, r in sorted(per_type_recall.items()):
            mark = "  ◀ UNSEEN" if tn == unseen_type else ""
            print(f"    {tn:20s} recall={r:.3f}{mark}")

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
            encoder, cls_head, ds_test_50, cls_mask, det_cfg["batch_size"], device
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
        cls_metrics = {
            "accuracy": float(accuracy_score(cls_targets, cls_preds)),
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
        if verbose:
            print(
                f"[lstm eval/{fold_name}] Classification  "
                f"acc={cls_metrics['accuracy']:.3f}  "
                f"(N={cls_mask.sum()})"
            )

    # ── 결과 저장 ─────────────────────────────────────────────────────────
    out_root = Path(cfg.get("output_dir", "outputs/scores_lstm"))
    for task, metrics in [
        ("detector_50_50", {**det_metrics_50, "n_params": n_total, "latency": latency}),
        ("detector_9_1",   det_metrics_9),
        ("classifier",     cls_metrics),
    ]:
        out_dir = out_root / fold_name / task
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ── Figure ────────────────────────────────────────────────────────────
    fig_dir = Path(cfg.get("figure_dir", "outputs/figures_lstm")) / fold_name
    plot_per_type_recall(fold_name, per_type_recall, fig_dir, unseen_type)

    return {
        "detection_50_50": det_metrics_50,
        "detection_9_1":   det_metrics_9,
        "classification":  cls_metrics,
        "n_params":        n_total,
        "latency":         latency,
    }


def evaluate_all_folds_lstm(
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
        results[fold] = evaluate_fold_lstm(fold, cfg, device, verbose=verbose)
    return results
