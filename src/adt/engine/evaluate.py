"""합성 이상치 기반 평가 루프.

checkpoints/finetune/best.pt + threshold.json 을 로드해서
data/processed/synthetic_eval/ 데이터의 윈도우별 재구성 오차를 이상 점수로 삼고
labels.npy 를 정답으로 AUROC/AUPRC/F1/Precision@K 를 계산한다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.adt.models.anomaly_head import ReconstructionAnomalyHead
from src.adt.models.encoder import TimeSeriesTransformerEncoder


# -------------------------------------------------------------------------
# 이상 점수 계산
# -------------------------------------------------------------------------

@torch.no_grad()
def _compute_scores(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    X: np.ndarray,
    time_feat: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """윈도우별 재구성 오차 (MSE) → anomaly score (N,)."""
    encoder.eval()
    head.eval()
    scores: list[np.ndarray] = []
    N = len(X)

    for i in range(0, N, batch_size):
        x_b = torch.from_numpy(X[i:i + batch_size]).to(device)
        tf_b = torch.from_numpy(time_feat[i:i + batch_size]).to(device)
        pred = head(encoder(x_b, tf_b, mask=None))
        err = F.mse_loss(pred, x_b, reduction="none").mean(dim=(1, 2))
        scores.append(err.cpu().numpy())

    return np.concatenate(scores)


# -------------------------------------------------------------------------
# 지표 계산
# -------------------------------------------------------------------------

def _precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    top_k = np.argsort(scores)[::-1][:k]
    return float(labels[top_k].sum() / k)


def compute_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    metrics: list[str],
) -> dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        roc_auc_score,
    )

    preds = (scores > threshold).astype(int)
    results: dict[str, float] = {}

    for m in metrics:
        if m == "auroc":
            results["auroc"] = float(roc_auc_score(labels, scores))
        elif m == "auprc":
            results["auprc"] = float(average_precision_score(labels, scores))
        elif m == "f1":
            results["f1"] = float(f1_score(labels, preds, zero_division=0))
        elif m == "precision_at_k":
            k = int(labels.sum())
            results["precision_at_k"] = _precision_at_k(scores, labels, k) if k > 0 else 0.0

    return results


# -------------------------------------------------------------------------
# 시각화
# -------------------------------------------------------------------------

def _save_figures(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    figures_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    figures_dir.mkdir(parents=True, exist_ok=True)

    # 점수 분포 히스토그램
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores[labels == 0], bins=60, alpha=0.6, label="Normal", color="steelblue")
    ax.hist(scores[labels == 1], bins=60, alpha=0.6, label="Anomaly (injected)", color="tomato")
    ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold={threshold:.4f}")
    ax.set_xlabel("Reconstruction MSE")
    ax.set_ylabel("Count")
    ax.set_title("Anomaly Score Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "score_distribution.png", dpi=150)
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_curve.png", dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------------
# 메인 진입점
# -------------------------------------------------------------------------

def run(cfg: dict[str, Any]) -> dict[str, float]:
    """합성 이상치 평가 실행.

    Args:
        cfg: finetune yaml 전체 dict (eval 절 포함)

    Returns:
        metrics dict
    """
    train_cfg: dict = cfg["train"]
    eval_cfg: dict = cfg.get("eval", {})

    data_cfg_path = Path(cfg.get("data_config", "configs/data/default.yaml"))
    with open(data_cfg_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    syn_cfg_path = Path(eval_cfg.get("synthetic_eval_config", "configs/eval/synthetic.yaml"))
    with open(syn_cfg_path, encoding="utf-8") as f:
        syn_cfg = yaml.safe_load(f)

    ckpt_dir = Path(train_cfg["ckpt_dir"])
    syn_eval_dir = Path(syn_cfg["output_dir"])
    metrics_list: list[str] = eval_cfg.get("metrics", ["auroc", "auprc", "f1", "precision_at_k"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 체크포인트 로드 ---
    ckpt_path = ckpt_dir / "best.pt"
    threshold_path = ckpt_dir / "threshold.json"
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    with open(threshold_path, encoding="utf-8") as f:
        thr_info = json.load(f)
    threshold: float = thr_info["threshold"]

    pretrain_model_cfg = state.get("cfg", {}).get("model", {})
    n_features = len(data_cfg["feature_cols"])
    d_model = pretrain_model_cfg.get("d_model", 128)

    encoder = TimeSeriesTransformerEncoder(
        n_features=n_features,
        d_model=d_model,
        n_heads=pretrain_model_cfg.get("n_heads", 4),
        n_layers=pretrain_model_cfg.get("n_layers", 4),
        d_ff=pretrain_model_cfg.get("d_ff", 256),
        dropout=pretrain_model_cfg.get("dropout", 0.1),
    ).to(device)
    encoder.load_state_dict(state["encoder"])

    head = ReconstructionAnomalyHead(d_model=d_model, n_features=n_features).to(device)
    head.load_state_dict(state["head"])

    print(f"[eval] ckpt={ckpt_path}  threshold={threshold:.6f}")

    # --- 데이터 로드 ---
    X = np.load(syn_eval_dir / "X.npy")
    time_feat = np.load(syn_eval_dir / "time_feat.npy")
    labels = np.load(syn_eval_dir / "labels.npy")
    print(f"[eval] synthetic_eval: {len(X)} 윈도우  anomaly={labels.sum()}/{len(labels)}")

    # --- 이상 점수 계산 ---
    scores = _compute_scores(encoder, head, X, time_feat, device)

    # --- 지표 ---
    results = compute_metrics(scores, labels, threshold, metrics_list)
    for k, v in results.items():
        print(f"[eval] {k}: {v:.4f}")

    # --- 시각화 ---
    figures_dir = Path("outputs/figures")
    _save_figures(scores, labels, threshold, figures_dir)
    print(f"[eval] 그래프 저장 → {figures_dir}/")

    # --- CSV 저장 ---
    scores_dir = Path("outputs/scores")
    scores_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame({
        "score": scores,
        "pred": (scores > threshold).astype(int),
        "label": labels,
    })
    csv_path = scores_dir / "window_scores.csv"
    df.to_csv(csv_path, index=True, index_label="window_idx")
    print(f"[eval] 점수 CSV 저장 → {csv_path}")

    # --- 결과 JSON ---
    result_path = scores_dir / "metrics.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({**results, "threshold": threshold}, f, indent=2)
    print(f"[eval] 지표 저장 → {result_path}")

    return results
