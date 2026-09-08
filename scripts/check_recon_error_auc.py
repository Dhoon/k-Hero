"""Reconstruction error + bottleneck → LogisticRegression AUC-ROC 검증.

encoder: downstream bce_unfreeze best.pt (finetuned)
decoder: pretrain_ckpt (원본 AE decoder)
data   : downstream/{fold}/train  +  val_50_50
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.adt.engine.train_downstream_transformer import DownstreamFoldDataset
from src.adt.models.lstm_ae import LSTMDecoder, LSTMEncoder


@torch.no_grad()
def extract_bottleneck_and_error(
    loader: DataLoader,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    decoder.eval()
    zs, errors, ys = [], [], []
    for x, _, bl, _ in loader:
        x = x.to(device)
        T = x.shape[1]
        _, z   = encoder(x)
        x_hat  = decoder(z, T)
        err    = ((x_hat - x) ** 2).mean(dim=[1, 2])
        zs.append(z.cpu().numpy())
        errors.append(err.cpu().numpy())
        ys.append(bl.numpy())
    z_all   = np.concatenate(zs)
    err_all = np.concatenate(errors)
    y_all   = np.concatenate(ys)
    features = np.concatenate([z_all, err_all[:, None]], axis=1)
    return features, y_all


def main(
    config: str = "configs/downstream_lstm/default.yaml",
    fold: str = "all_type",
    encoder_ckpt: str | None = None,
) -> None:
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(cfg.get("data_config", "configs/data/default.yaml"), encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg  = cfg["model"]
    det_cfg    = cfg["detection"]
    n_features = len(data_cfg["feature_cols"])
    hidden_dim = model_cfg["hidden_dim"]
    num_layers = model_cfg["num_layers"]
    bottleneck = 2 * hidden_dim
    batch_size = det_cfg["batch_size"]

    # ── 모델 로드 ────────────────────────────────────────────────────────────
    encoder = LSTMEncoder(n_features, hidden_dim, num_layers).to(device)
    decoder = LSTMDecoder(bottleneck, n_features, num_layers=1).to(device)

    # encoder: --encoder_ckpt 우선, 없으면 bce_unfreeze/{fold}/detector/best.pt fallback
    default_det_ckpt = Path(det_cfg["ckpt_dir"]) / "bce_unfreeze" / fold / "detector" / "best.pt"
    det_ckpt = Path(encoder_ckpt) if encoder_ckpt is not None else default_det_ckpt
    if det_ckpt.exists():
        ckpt = torch.load(det_ckpt, map_location="cpu")
        key = "encoder" if "encoder" in ckpt else None
        if key:
            encoder.load_state_dict(ckpt[key])
            print(f"encoder loaded from {det_ckpt}")
        else:
            legacy = det_ckpt.parent / "encoder_finetuned.pt"
            encoder.load_state_dict(torch.load(legacy, map_location="cpu")["encoder"])
            print(f"encoder loaded from (legacy) {legacy}")
    else:
        print(f"[warn] {det_ckpt} 없음 — pretrain encoder 사용")
        pretrain = torch.load(cfg["pretrain_ckpt"], map_location="cpu")
        encoder.load_state_dict(pretrain["encoder"])

    # decoder: pretrain AE
    pretrain_ckpt = cfg.get("pretrain_ckpt")
    if pretrain_ckpt and Path(pretrain_ckpt).exists():
        decoder.load_state_dict(torch.load(pretrain_ckpt, map_location="cpu")["decoder"])
        print(f"decoder loaded from {pretrain_ckpt}")
    else:
        print("[warn] pretrain_ckpt 없음 — random decoder (결과 무의미)")

    # ── 데이터로더 ───────────────────────────────────────────────────────────
    downstream_dir = Path(cfg["downstream_dir"])
    train_loader = DataLoader(
        DownstreamFoldDataset(downstream_dir / fold / "train"),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    val_loader = DataLoader(
        DownstreamFoldDataset(downstream_dir / fold / "val_50_50"),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    # ── 특징 추출 ────────────────────────────────────────────────────────────
    print("extracting train...")
    feat_train, y_train = extract_bottleneck_and_error(train_loader, encoder, decoder, device)
    print("extracting val_50_50...")
    feat_val, y_val = extract_bottleneck_and_error(val_loader, encoder, decoder, device)
    print(f"train={feat_train.shape}  val={feat_val.shape}  pos_rate(val)={y_val.mean():.3f}")

    # ── z + recon_error ──────────────────────────────────────────────────────
    clf1 = LogisticRegression(max_iter=1000).fit(feat_train, y_train)
    auc1 = roc_auc_score(y_val, clf1.predict_proba(feat_val)[:, 1])
    print(f"z + recon_error  AUC-ROC = {auc1:.4f}")

    # ── recon_error 단독 ─────────────────────────────────────────────────────
    clf2 = LogisticRegression(max_iter=1000).fit(feat_train[:, -1:], y_train)
    auc2 = roc_auc_score(y_val, clf2.predict_proba(feat_val[:, -1:])[:, 1])
    print(f"recon_error only AUC-ROC = {auc2:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/downstream_lstm/default.yaml")
    parser.add_argument("--fold",   default="all_type")
    parser.add_argument(
        "--encoder_ckpt",
        default=None,
        help="encoder best.pt 경로 (기본: checkpoints/downstream_lstm/bce_unfreeze/{fold}/detector/best.pt)",
    )
    args = parser.parse_args()
    main(args.config, args.fold, args.encoder_ckpt)
