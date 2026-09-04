"""Downstream Detection + Classification 통합 평가 루프.

평가 원칙:
  - test set: 항상 all_type/test (6개 fold 공유)
  - Detection 평가: 전체 샘플 → accuracy/precision/recall/F1
                    + type_label별 per-type detection recall
  - Classification 평가: all_type/test에서 Normal 제외
                         + 해당 fold가 모르는 타입(unseen_X)도 제외
                         → per-class precision/recall + confusion matrix
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as _fm
    _MPL_OK = True
    _KO_FONT_OK = False
    _KO_FONTS = ["Malgun Gothic", "NanumGothic", "Apple SD Gothic Neo"]
    _avail = {f.name for f in _fm.fontManager.ttflist}
    for _f in _KO_FONTS:
        if _f in _avail:
            matplotlib.rcParams["font.family"] = _f
            _KO_FONT_OK = True
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
except ImportError:
    _MPL_OK = False
    _KO_FONT_OK = False

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
    precision_recall_curve, roc_auc_score, average_precision_score,
)

from src.adt.data.labeling import TYPE_IDX
from src.adt.models.encoder import TimeSeriesTransformerEncoder
from src.adt.models.heads.detection_head import DetectionHead
from src.adt.models.heads.classification_head import ClassificationHead
from src.adt.engine.train_downstream import (
    ALL_FOLDS, FOLD_UNSEEN_TYPE, IDX_TO_TYPE,
    DownstreamFoldDataset,
)


def _build_encoder(cfg: dict, device: torch.device) -> nn.Module:
    """cfg[model]로 encoder 인스턴스 생성 (가중치 미로드, random init)."""
    mc = cfg["model"]
    dc = cfg.get("data_config", "configs/data/default.yaml")
    import yaml
    from pathlib import Path as _Path
    data_cfg = yaml.safe_load(open(dc, encoding="utf-8"))
    n_features = len(data_cfg["feature_cols"])
    return TimeSeriesTransformerEncoder(
        n_features=n_features,
        d_model=mc["d_model"],
        n_heads=mc["n_heads"],
        n_layers=mc["n_layers"],
        d_ff=mc["d_ff"],
        dropout=mc["dropout"],
    ).to(device)

# =========================================================================
# Threshold 탐색 / 저장
# =========================================================================

def find_best_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """val set sigmoid 확률 → F1 최대 threshold 탐색.

    Args:
        probs : (N,) sigmoid 확률 (0~1)
        labels: (N,) int binary label

    Returns:
        (best_threshold, best_f1)
    """
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    # precision_recall_curve: len(thresholds) == len(precisions) - 1
    denom = precisions[:-1] + recalls[:-1]
    f1s = np.where(denom > 0, 2 * precisions[:-1] * recalls[:-1] / denom, 0.0)
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def calibrate_threshold(
    fold_name: str,
    cfg: dict[str, Any],
    eval_encoder: nn.Module,
    det_head: nn.Module,
    device: torch.device,
    verbose: bool = True,
) -> tuple[float, float]:
    """val set으로 최적 detection threshold 탐색 후 threshold.json 저장.

    threshold는 val set에서만 결정 — test set 절대 사용 금지 (data leakage).

    Returns:
        (threshold, val_f1)  — sigmoid 확률 기준 threshold
    """
    downstream_dir = Path(cfg["downstream_dir"])
    det_cfg = cfg["detection"]

    val_dir = downstream_dir / fold_name / "val"
    ds_val = DownstreamFoldDataset(val_dir)
    val_loader = DataLoader(
        ds_val, batch_size=det_cfg["batch_size"], shuffle=False, num_workers=0
    )

    logits_np, bl_np, _ = _infer(eval_encoder, det_head, val_loader, device)
    probs = torch.sigmoid(torch.from_numpy(logits_np)).numpy()

    threshold, val_f1 = find_best_threshold(probs, bl_np)

    det_ckpt_dir = Path(det_cfg["ckpt_dir"]) / fold_name / "detector"
    det_ckpt_dir.mkdir(parents=True, exist_ok=True)
    thr_path = det_ckpt_dir / "threshold.json"
    thr_path.write_text(
        json.dumps({"threshold": threshold, "val_f1": val_f1}, indent=2),
        encoding="utf-8",
    )

    if verbose:
        print(
            f"[threshold/{fold_name}] val best threshold={threshold:.4f}  "
            f"val_f1={val_f1:.4f}  → saved: {thr_path}"
        )
    return threshold, val_f1


# =========================================================================
# 추론 헬퍼
# =========================================================================

@torch.no_grad()
def _infer(
    encoder: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """encoder → head forward 전체 반환.

    Returns:
        logits_all : (N,) float (detection) or (N, C) float (classification)
        bl_all     : (N,) int  binary_label
        tl_all     : (N,) int  type_label
    """
    logits_list, bl_list, tl_list = [], [], []
    for x, tf, bl, tl in loader:
        x, tf = x.to(device), tf.to(device)
        z_t = encoder(x, tf)
        logit = head(z_t)  # (B,) or (B, C)
        logits_list.append(logit.cpu())
        bl_list.append(bl)
        tl_list.append(tl)

    logits_all = torch.cat(logits_list, dim=0).numpy()
    bl_all     = torch.cat(bl_list, dim=0).numpy().astype(np.int32)
    tl_all     = torch.cat(tl_list, dim=0).numpy().astype(np.int32)
    return logits_all, bl_all, tl_all


# =========================================================================
# fold 단위 평가
# =========================================================================

def evaluate_fold(
    fold_name: str,
    cfg: dict[str, Any],
    encoder: nn.Module,
    device: torch.device,
    verbose: bool = True,
) -> dict[str, Any]:
    """단일 fold의 detection + classification 평가.

    Returns:
        {
          "detection":      { "accuracy", "precision", "recall", "f1",
                              "per_type_recall": {type_name: recall} },
          "classification": { "accuracy", "per_class": {type_name: {"precision","recall"}},
                              "confusion_matrix": list[list[int]] }
        }
    """
    downstream_dir = Path(cfg["downstream_dir"])
    det_cfg = cfg["detection"]
    cls_cfg = cfg["classification"]
    model_cfg = cfg["model"]

    # ── test set 로드 (항상 all_type/test) ──────────────────────────────
    test_dir = downstream_dir / "all_type" / "test"
    ds_test = DownstreamFoldDataset(test_dir)
    test_loader = DataLoader(
        ds_test, batch_size=det_cfg["batch_size"], shuffle=False, num_workers=0
    )

    # ── checkpoint 경로 ───────────────────────────────────────────────────
    det_ckpt_dir = Path(det_cfg["ckpt_dir"]) / fold_name / "detector"
    cls_ckpt_dir = Path(cls_cfg["ckpt_dir"]) / fold_name / "classifier"
    d_model = model_cfg["d_model"]

    # ── class_names.json 로드 ─────────────────────────────────────────────
    class_names_path = cls_ckpt_dir / "class_names.json"
    if class_names_path.exists():
        class_names: dict[str, str] = json.loads(
            class_names_path.read_text(encoding="utf-8")
        )
    else:
        class_names = {str(i): IDX_TO_TYPE[i] for i in range(5)}

    num_classes = len(class_names)
    # class_idx → original type_name
    cls_idx_to_name = {int(k): v for k, v in class_names.items()}
    # original type_name → class_idx
    name_to_cls_idx = {v: int(k) for k, v in class_names.items()}

    # ── Detection 헤드 로드 ───────────────────────────────────────────────
    det_head = DetectionHead(
        d_model=d_model,
        hidden_dim=det_cfg["hidden_dim"],
        dropout=det_cfg["dropout"],
    ).to(device)

    det_best = det_ckpt_dir / "best.pt"
    if det_best.exists():
        state = torch.load(det_best, map_location="cpu")
        det_head.load_state_dict(state["head"])
    det_head.eval()

    # ── Classification 헤드 로드 ─────────────────────────────────────────
    cls_head = ClassificationHead(
        d_model=d_model,
        num_classes=num_classes,
        hidden_dim=cls_cfg["hidden_dim"],
        dropout=cls_cfg["dropout"],
    ).to(device)

    cls_best = cls_ckpt_dir / "best.pt"
    if cls_best.exists():
        state = torch.load(cls_best, map_location="cpu")
        cls_head.load_state_dict(state["head"])
    cls_head.eval()

    # ── encoder 선택: fold-specific finetuned 우선, 없으면 passed encoder ───
    ft_enc_path = det_ckpt_dir / "encoder_finetuned.pt"
    if ft_enc_path.exists():
        eval_encoder = _build_encoder(cfg, device)
        state = torch.load(ft_enc_path, map_location="cpu")
        eval_encoder.load_state_dict(state["encoder"], strict=True)
        eval_encoder.eval()
        for p in eval_encoder.parameters():
            p.requires_grad_(False)
        if verbose:
            print(f"[eval/{fold_name}] finetuned encoder loaded: {ft_enc_path}")
    else:
        eval_encoder = encoder

    # ── Detection 평가 ────────────────────────────────────────────────────
    det_logits, bl_all, tl_all = _infer(eval_encoder, det_head, test_loader, device)
    probs = torch.sigmoid(torch.from_numpy(det_logits)).numpy()  # (N,) 0~1

    # val set으로 최적 threshold 탐색 (data leakage 방지 — test 미사용)
    threshold, val_f1 = calibrate_threshold(
        fold_name, cfg, eval_encoder, det_head, device, verbose=verbose
    )

    pred_default = (probs >= 0.5).astype(np.int32)
    pred_optimal = (probs >  threshold).astype(np.int32)

    def _det_metrics(pred: np.ndarray) -> dict[str, float]:
        return {
            "accuracy":  float(accuracy_score(bl_all, pred)),
            "precision": float(precision_score(bl_all, pred, zero_division=0)),
            "recall":    float(recall_score(bl_all, pred, zero_division=0)),
            "f1":        float(f1_score(bl_all, pred, zero_division=0)),
        }

    metrics_default = _det_metrics(pred_default)
    metrics_optimal = _det_metrics(pred_optimal)

    try:
        auc_roc = float(roc_auc_score(bl_all, probs))
    except Exception:
        auc_roc = float("nan")
    try:
        auc_pr = float(average_precision_score(bl_all, probs))
    except Exception:
        auc_pr = float("nan")

    per_type_recall: dict[str, float] = {}
    unseen_type = FOLD_UNSEEN_TYPE.get(fold_name)
    for type_name, orig_idx in TYPE_IDX.items():
        type_mask = (tl_all == orig_idx)
        if type_mask.sum() > 0:
            per_type_recall[type_name] = float(pred_optimal[type_mask].mean())

    det_metrics = {
        **metrics_optimal,
        "threshold":       threshold,
        "val_f1":          val_f1,
        "auc_roc":         auc_roc,
        "auc_pr":          auc_pr,
        "default_thr":     metrics_default,
        "per_type_recall": per_type_recall,
    }

    if verbose:
        print(f"[eval/{fold_name}] Detection  threshold={threshold:.4f}  (val_f1={val_f1:.4f})")
        print(
            f"  default(0.50):           acc={metrics_default['accuracy']:.3f}  "
            f"prec={metrics_default['precision']:.3f}  "
            f"rec={metrics_default['recall']:.3f}  "
            f"F1={metrics_default['f1']:.3f}"
        )
        print(
            f"  optimal({threshold:.4f}): acc={metrics_optimal['accuracy']:.3f}  "
            f"prec={metrics_optimal['precision']:.3f}  "
            f"rec={metrics_optimal['recall']:.3f}  "
            f"F1={metrics_optimal['f1']:.3f}"
        )
        print(f"  AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}")
        print("  per-type recall (optimal threshold):")
        for tn, r in sorted(per_type_recall.items()):
            mark = "  ◀ UNSEEN" if tn == unseen_type else ""
            print(f"    {tn:20s} recall={r:.3f}{mark}")

    # ── Classification 평가 ───────────────────────────────────────────────
    # known types: class_names에 있는 type만 평가
    known_type_names = set(class_names.values())
    known_orig_idxs = {TYPE_IDX[n] for n in known_type_names if n in TYPE_IDX}

    # test 샘플 중 Attack이고 known type인 것만
    cls_mask = np.array([
        bl_all[i] == 1 and tl_all[i] in known_orig_idxs
        for i in range(len(bl_all))
    ])

    if cls_mask.sum() == 0:
        cls_metrics: dict[str, Any] = {
            "accuracy": float("nan"),
            "per_class": {},
            "confusion_matrix": [],
        }
    else:
        # class_names의 type_name → class_idx 역변환으로 target 계산
        tl_cls = tl_all[cls_mask]
        # orig_type_idx → class_idx
        orig_to_cls = {TYPE_IDX[name]: int(k) for k, name in class_names.items()}
        cls_targets = np.array([orig_to_cls[t] for t in tl_cls], dtype=np.int32)

        cls_logits_all, cls_bl, cls_tl = _infer_filtered(
            eval_encoder, cls_head, ds_test, cls_mask, det_cfg["batch_size"], device
        )
        cls_preds = cls_logits_all.argmax(axis=1)

        cls_metrics = {
            "accuracy": float(accuracy_score(cls_targets, cls_preds)),
            "per_class": {},
            "confusion_matrix": confusion_matrix(
                cls_targets, cls_preds,
                labels=list(range(num_classes))
            ).tolist(),
        }

        precs = precision_score(
            cls_targets, cls_preds, average=None, labels=list(range(num_classes)),
            zero_division=0,
        )
        recs = recall_score(
            cls_targets, cls_preds, average=None, labels=list(range(num_classes)),
            zero_division=0,
        )
        for cls_i in range(num_classes):
            name = cls_idx_to_name.get(cls_i, str(cls_i))
            cls_metrics["per_class"][name] = {
                "precision": float(precs[cls_i]),
                "recall":    float(recs[cls_i]),
            }

        if verbose:
            print(
                f"[eval/{fold_name}] Classification  "
                f"acc={cls_metrics['accuracy']:.3f}  "
                f"(N={cls_mask.sum()})"
            )
            for cn, cm_val in cls_metrics["per_class"].items():
                print(
                    f"    {cn:20s} "
                    f"prec={cm_val['precision']:.3f}  rec={cm_val['recall']:.3f}"
                )

    results = {"detection": det_metrics, "classification": cls_metrics}

    # ── 결과 저장 ─────────────────────────────────────────────────────────
    out_root = Path(cfg.get("output_dir", "outputs/scores"))
    for task, metrics in [("detector", det_metrics), ("classifier", cls_metrics)]:
        out_dir = out_root / fold_name / task
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ── Figure ────────────────────────────────────────────────────────────
    fig_dir = Path(cfg.get("figure_dir", "outputs/figures")) / fold_name
    plot_per_type_recall(fold_name, per_type_recall, fig_dir, unseen_type)

    return results


@torch.no_grad()
def _infer_filtered(
    encoder: nn.Module,
    head: nn.Module,
    dataset: DownstreamFoldDataset,
    mask: np.ndarray,      # (N,) bool
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """mask가 True인 샘플만 head에 통과시켜 logits 반환."""
    indices = np.where(mask)[0]
    logits_list = []
    head.eval()
    for i in range(0, len(indices), batch_size):
        idx_batch = indices[i: i + batch_size]
        xs  = torch.stack([dataset.X[j]          for j in idx_batch]).to(device)
        tfs = torch.stack([dataset.time_feat[j]  for j in idx_batch]).to(device)
        z_t = encoder(xs, tfs)
        logits_list.append(head(z_t).cpu())
    logits = torch.cat(logits_list, dim=0).numpy()
    bl = dataset.binary_label[indices].numpy().astype(np.int32)
    tl = dataset.type_label[indices].numpy().astype(np.int32)
    return logits, bl, tl


# =========================================================================
# Figure 생성
# =========================================================================

_KO_LABEL = {
    "instant_spike": "instant_spike\n(급격한 절대적 이상)",
    "pulse_plateau": "pulse_plateau\n(장시간 완만한 상승)",
    "scale_down":    "scale_down\n(사용량 축소 조작)",
    "ramp":          "ramp\n(점진적 드리프트)",
    "replay":        "replay\n(재생 공격)",
}
_EN_LABEL = {
    "instant_spike": "instant_spike\n(abrupt absolute spike)",
    "pulse_plateau": "pulse_plateau\n(sustained high plateau)",
    "scale_down":    "scale_down\n(magnitude reduction)",
    "ramp":          "ramp\n(gradual drift)",
    "replay":        "replay\n(replay attack)",
}
_TYPE_ORDER = ["instant_spike", "pulse_plateau", "scale_down", "ramp", "replay"]


def _bar_color(recall_pct: float) -> str:
    if recall_pct >= 70:
        return "#1B3A6B"
    if recall_pct >= 40:
        return "#3B6BC1"
    return "#9BAFD4"


def plot_per_type_recall(
    fold_name: str,
    per_type_recall: dict[str, float],
    out_dir: Path,
    unseen_type: str | None = None,
) -> None:
    """공격 유형별 detection recall 수평 바차트를 out_dir/per_type_recall.png에 저장."""
    if not _MPL_OK:
        print("[figure] matplotlib 없음 — figure 생략")
        return

    types   = [t for t in _TYPE_ORDER if t in per_type_recall]
    recalls = [per_type_recall[t] * 100 for t in types]
    _label_map = _KO_LABEL if _KO_FONT_OK else _EN_LABEL
    labels  = [_label_map.get(t, t) for t in types]
    colors  = [_bar_color(r) for r in recalls]

    fig, ax = plt.subplots(figsize=(9, max(3, len(types) * 0.95 + 1.2)))
    bars = ax.barh(labels, recalls, color=colors, height=0.55)

    for bar, r in zip(bars, recalls):
        ax.text(
            bar.get_width() + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{r:.1f}%",
            va="center", ha="left", fontsize=11, fontweight="bold",
        )

    ax.axvline(50, color="#555", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.set_xlim(0, 115)
    if _KO_FONT_OK:
        ax.set_xlabel("탐지율 (%)", fontsize=12)
        ax.set_title(f"합성 공격 유형별 탐지율\n({fold_name})", fontsize=14, fontweight="bold", pad=12)
    else:
        ax.set_xlabel("Detection Rate (%)", fontsize=12)
        ax.set_title(f"Detection Rate by Attack Type\n({fold_name})", fontsize=14, fontweight="bold", pad=12)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "per_type_recall.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out_path}")


# =========================================================================
# 전체 fold 평가
# =========================================================================

def evaluate_all_folds(
    cfg: dict[str, Any],
    encoder: nn.Module,
    device: torch.device,
    folds: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, dict]:
    """지정된 fold 목록을 순차 평가. folds=None이면 ALL_FOLDS 전부."""
    targets = folds if folds is not None else ALL_FOLDS
    results: dict[str, dict] = {}
    for fold in targets:
        if verbose:
            print(f"\n{'='*60}\n fold: {fold}\n{'='*60}")
        results[fold] = evaluate_fold(fold, cfg, encoder, device, verbose=verbose)
    return results
