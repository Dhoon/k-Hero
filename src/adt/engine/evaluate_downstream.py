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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

from src.adt.data.labeling import TYPE_IDX
from src.adt.models.heads.detection_head import DetectionHead
from src.adt.models.heads.classification_head import ClassificationHead
from src.adt.engine.train_downstream import (
    ALL_FOLDS, FOLD_UNSEEN_TYPE, IDX_TO_TYPE,
    DownstreamFoldDataset,
)

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
    threshold: float = 0.5,
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

    # ── Detection 평가 ────────────────────────────────────────────────────
    det_logits, bl_all, tl_all = _infer(encoder, det_head, test_loader, device)
    pred_binary = (det_logits >= 0.0).astype(np.int32)  # logit threshold=0

    det_metrics = {
        "accuracy":  float(accuracy_score(bl_all, pred_binary)),
        "precision": float(precision_score(bl_all, pred_binary, zero_division=0)),
        "recall":    float(recall_score(bl_all, pred_binary, zero_division=0)),
        "f1":        float(f1_score(bl_all, pred_binary, zero_division=0)),
    }

    # per-type detection recall (Attack type별 탐지율)
    per_type_recall: dict[str, float] = {}
    unseen_type = FOLD_UNSEEN_TYPE.get(fold_name)
    for type_name, orig_idx in TYPE_IDX.items():
        type_mask = (tl_all == orig_idx)
        if type_mask.sum() > 0:
            recall_val = float(pred_binary[type_mask].mean())
            per_type_recall[type_name] = recall_val

    det_metrics["per_type_recall"] = per_type_recall

    if verbose:
        print(
            f"[eval/{fold_name}] Detection  "
            f"acc={det_metrics['accuracy']:.3f}  "
            f"prec={det_metrics['precision']:.3f}  "
            f"rec={det_metrics['recall']:.3f}  "
            f"F1={det_metrics['f1']:.3f}"
        )
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

        # Classification 추론: cls_mask 샘플만 다시 forward
        # (전체 test loader에서 한 번에 얻은 logits을 index로 뽑기 위해 별도 run)
        cls_logits_all, _, _ = _infer(encoder, cls_head, test_loader, device)
        # _infer returns logits for ALL test samples with head=cls_head;
        # cls_head expects Attack-only input, but we ran on all.
        # Safer: re-run only on cls_mask samples.
        cls_logits_all, cls_bl, cls_tl = _infer_filtered(
            encoder, cls_head, ds_test, cls_mask, det_cfg["batch_size"], device
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
