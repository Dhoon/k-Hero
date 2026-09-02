"""Classification 모델 평가 루프 (Exp A / Exp B).

held_out_type(instant_spike)만 test split에 주입한 테스트셋으로 평가.
Exp C(reconstruction)와 동일한 attack 분포 → 3-way 공정 비교 가능.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.adt.data.classification_dataset import (
    build_classification_dataset,
    build_classification_dataloader,
)
from src.adt.data.scalers import StandardScalerND
from src.adt.models.encoder import TimeSeriesTransformerEncoder
from src.adt.models.heads.classification_head import ClassificationHead


# -------------------------------------------------------------------------
# 추론
# -------------------------------------------------------------------------

@torch.no_grad()
def _infer(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (scores, labels) as numpy arrays."""
    encoder.eval()
    head.eval()
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for x, time_feat, labels in loader:
        x = x.to(device)
        time_feat = time_feat.to(device)
        H = encoder(x, time_feat, mask=None)
        logits = head(H)
        scores = torch.sigmoid(logits).cpu().numpy()
        all_scores.append(scores)
        all_labels.append(labels.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


# -------------------------------------------------------------------------
# 지표
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

    preds = (scores >= threshold).astype(int)
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

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores[labels == 0], bins=60, alpha=0.6, label="Normal", color="steelblue")
    ax.hist(scores[labels == 1], bins=60, alpha=0.6, label="Anomaly (injected)", color="tomato")
    ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold={threshold:.3f}")
    ax.set_xlabel("Anomaly Score (sigmoid)")
    ax.set_ylabel("Count")
    ax.set_title("Classification Score Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "score_distribution.png", dpi=150)
    plt.close(fig)

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
    """Classification 모델 평가.

    Args:
        cfg: classification yaml 전체 dict (eval 절 포함)

    Returns:
        metrics dict
    """
    train_cfg: dict = cfg["train"]
    head_cfg: dict = cfg.get("head", {})
    eval_cfg: dict = cfg.get("eval", {})
    attack_cfg: dict = cfg["attack_split"]

    data_cfg_path = Path(cfg.get("data_config", "configs/data/default.yaml"))
    with open(data_cfg_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    processed_dir = Path(data_cfg["processed_dir"])
    n_features = len(data_cfg["feature_cols"])
    window_size = data_cfg.get("window_size", 96)

    metrics_list: list[str] = eval_cfg.get("metrics", ["auroc", "auprc", "f1", "precision_at_k"])
    threshold: float = eval_cfg.get("threshold", 0.5)
    figures_dir = Path(eval_cfg.get("figure_dir", "outputs/figures/classification"))
    scores_dir = Path(eval_cfg.get("output_dir", "outputs/scores/classification"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 체크포인트 로드 ---
    ckpt_dir = Path(train_cfg["ckpt_dir"])
    ckpt_path = ckpt_dir / "best.pt"
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"[eval] ckpt={ckpt_path}")

    encoder_ckpt: str | None = cfg.get("encoder_ckpt", None)
    if encoder_ckpt is not None:
        pretrain_state = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
        pretrain_model_cfg = pretrain_state.get("cfg", {}).get("model", {})
        d_model = pretrain_model_cfg.get("d_model", 128)
        n_heads = pretrain_model_cfg.get("n_heads", 4)
        n_layers = pretrain_model_cfg.get("n_layers", 4)
        d_ff = pretrain_model_cfg.get("d_ff", 256)
        dropout = pretrain_model_cfg.get("dropout", 0.1)
    else:
        model_cfg = cfg.get("model", {})
        d_model = model_cfg.get("d_model", 128)
        n_heads = model_cfg.get("n_heads", 4)
        n_layers = model_cfg.get("n_layers", 4)
        d_ff = model_cfg.get("d_ff", 256)
        dropout = model_cfg.get("dropout", 0.1)

    encoder = TimeSeriesTransformerEncoder(
        n_features=n_features, d_model=d_model,
        n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, dropout=dropout,
    ).to(device)
    encoder.load_state_dict(state["encoder"])

    head = ClassificationHead(
        d_model=d_model,
        hidden_dim=head_cfg.get("hidden_dim", 64),
        dropout=head_cfg.get("dropout", 0.1),
        pooling=head_cfg.get("pooling", "mean_max"),
    ).to(device)
    head.load_state_dict(state["head"])

    # --- 테스트셋: held_out_type만 주입 ---
    scaler = StandardScalerND.load(processed_dir / "scaler.joblib")
    test_ds = build_classification_dataset(
        processed_dir, "test", attack_cfg, scaler,
        window_size=window_size, seed=cfg.get("seed", 42) + 2,
    )
    test_loader = build_classification_dataloader(
        test_ds, batch_size=256,
        attack_ratio_per_batch=0.5,
        shuffle=False,
    )
    print(f"[eval] test 윈도우={len(test_ds)}  held_out_type={attack_cfg['held_out_type']}")

    # --- 추론 ---
    scores, labels = _infer(encoder, head, test_loader, device)

    # --- 지표 ---
    results = compute_metrics(scores, labels, threshold, metrics_list)
    for k, v in results.items():
        print(f"[eval] {k}: {v:.4f}")

    # --- 시각화 ---
    _save_figures(scores, labels, threshold, figures_dir)
    print(f"[eval] 그래프 저장 → {figures_dir}/")

    # --- CSV 저장 ---
    import pandas as pd
    scores_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "score": scores,
        "pred": (scores >= threshold).astype(int),
        "label": labels,
    })
    csv_path = scores_dir / "window_scores.csv"
    df.to_csv(csv_path, index=True, index_label="window_idx")
    print(f"[eval] 점수 CSV 저장 → {csv_path}")

    # --- 결과 JSON ---
    result_path = scores_dir / "metrics.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            **results,
            "threshold": threshold,
            "held_out_type": attack_cfg["held_out_type"],
        }, f, indent=2)
    print(f"[eval] 지표 저장 → {result_path}")

    return results
