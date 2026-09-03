"""Pretrain checkpoint 품질 진단 스크립트 (실제 masking 조건 반영).

진단 항목:
  DIAG 1 -- pretrain/val에서 Normal 200개, 실제 masking 적용
            마스킹된 위치에서만 MSE (학습과 동일 조건)
            trivial baseline = 비마스킹 구간 평균으로 마스킹 구간 예측
  DIAG 2 -- downstream/all_type/train Normal 500 + Attack 500
            동일 masking 조건 → masked MSE → AUC-ROC
  DIAG 3 -- pretrain/val 300+개, forecast error vs last-step trivial baseline

masking 설정은 configs/pretrain/default.yaml ssl 섹션을 그대로 사용.
epoch=99 (curriculum 완료 이후 고정 구간, mask_ratio_end 적용).

사용법::
    python scripts/diagnose_pretrain.py
    python scripts/diagnose_pretrain.py --pretrain_config configs/pretrain/default.yaml
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.adt.models.encoder import TimeSeriesTransformerEncoder
from src.adt.models.heads.reconstruction_head import ReconstructionAnomalyHead
from src.adt.models.heads.forecasting_head import ForecastingHead
from src.adt.ssl.masking import generate_mask
from src.adt.ssl.losses import masked_reconstruction_loss


# ---------------------------------------------------------------------------
# 체크포인트 로드
# ---------------------------------------------------------------------------

def load_pretrain_models(ckpt_path: Path, device: torch.device):
    """best.pt -> (encoder, recon_head, forecast_head, n_features, d_model, fh)."""
    state = torch.load(ckpt_path, map_location="cpu")
    enc_state  = state["encoder"]
    rh_state   = state["recon_head"]
    fh_state   = state["forecast_head"]

    # 구조 추론
    n_features       = rh_state["proj.weight"].shape[0]
    d_model          = rh_state["proj.weight"].shape[1]
    forecast_horizon = fh_state["mlp.2.weight"].shape[0] // n_features

    enc_keys = list(enc_state.keys())
    n_layers = max(
        int(k.split(".")[2])
        for k in enc_keys
        if k.startswith("transformer_encoder.layers.")
    ) + 1
    d_ff = enc_state["transformer_encoder.layers.0.linear1.weight"].shape[0]
    n_heads = (
        enc_state["transformer_encoder.layers.0.self_attn.in_proj_weight"].shape[0] // 3
    ) // (d_model // 4)

    print(f"[ckpt] n_features={n_features}  d_model={d_model}  "
          f"n_layers={n_layers}  d_ff={d_ff}  n_heads={n_heads}  "
          f"forecast_horizon={forecast_horizon}")

    encoder = TimeSeriesTransformerEncoder(
        n_features=n_features, d_model=d_model,
        n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, dropout=0.0,
    )
    encoder.load_state_dict(enc_state, strict=True)
    encoder.eval().to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)

    recon_head = ReconstructionAnomalyHead(d_model=d_model, n_features=n_features)
    recon_head.load_state_dict(rh_state, strict=True)
    recon_head.eval().to(device)

    forecast_head = ForecastingHead(
        d_model=d_model, forecast_horizon=forecast_horizon, n_features=n_features,
    )
    forecast_head.load_state_dict(fh_state, strict=True)
    forecast_head.eval().to(device)

    return encoder, recon_head, forecast_head, n_features, d_model, forecast_horizon


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _pct(arr: np.ndarray, label: str) -> str:
    if len(arr) == 0:
        return f"  {label}: N=0"
    return (
        f"  {label} (N={len(arr):>4})  "
        f"mean={arr.mean():.6f}  "
        f"p25={np.percentile(arr,25):.6f}  "
        f"med={np.median(arr):.6f}  "
        f"p75={np.percentile(arr,75):.6f}  "
        f"p95={np.percentile(arr,95):.6f}"
    )


def _masked_mse_batch(
    encoder: nn.Module,
    recon_head: nn.Module,
    xb: torch.Tensor,
    tfb: torch.Tensor,
    ssl_cfg: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """마스킹 적용 후 masked MSE + trivial baseline MSE 반환.

    Returns:
        model_mse  : (B,) -- 마스킹된 위치에서 model 예측 MSE
        trivial_mse: (B,) -- 마스킹된 위치에서 trivial(비마스킹 평균) MSE
    """
    xb  = xb.to(device)
    tfb = tfb.to(device)

    mask = generate_mask(xb, ssl_cfg)   # (B,T) or (B,T,C)

    # encoder forward with mask
    z    = encoder(xb, tfb, mask=mask)
    recon = recon_head(z)                            # (B, T, C)

    # mask -> (B, T, C) bool
    if mask.dim() == 2:
        mask3d = mask.unsqueeze(-1).expand_as(xb)   # (B, T, C)
    else:
        mask3d = mask                                # (B, T, C)

    B = xb.shape[0]
    model_mse   = np.zeros(B, dtype=np.float32)
    trivial_mse = np.zeros(B, dtype=np.float32)

    xb_np    = xb.cpu().numpy()       # (B, T, C)
    recon_np = recon.cpu().numpy()
    mask_np  = mask3d.cpu().numpy()   # (B, T, C) bool

    for i in range(B):
        m = mask_np[i]                # (T, C)
        n_pos = m.sum()
        if n_pos == 0:
            continue

        orig    = xb_np[i]           # (T, C)
        pred    = recon_np[i]

        # model MSE at masked positions
        model_mse[i] = float(np.mean((orig[m] - pred[m]) ** 2))

        # trivial: per-channel mean of UN-masked positions
        not_m = ~m                   # (T, C)
        triv  = np.zeros_like(orig)  # (T, C)
        for ch in range(orig.shape[1]):
            unmasked_vals = orig[:, ch][not_m[:, ch]]
            ch_mean = float(unmasked_vals.mean()) if len(unmasked_vals) > 0 else 0.0
            triv[:, ch] = ch_mean
        trivial_mse[i] = float(np.mean((orig[m] - triv[m]) ** 2))

    return model_mse, trivial_mse


# ---------------------------------------------------------------------------
# DIAG 1: 정성적 + 정량적 복원 품질 (pretrain/val Normal)
# ---------------------------------------------------------------------------

@torch.no_grad()
def diag_reconstruction_quality(
    encoder: nn.Module,
    recon_head: nn.Module,
    pretrain_val_dir: Path,
    ssl_cfg: dict,
    device: torch.device,
    n_samples: int = 200,
    batch_size: int = 64,
    n_qual: int = 5,
) -> None:
    print("\n" + "=" * 70)
    print("DIAG 1 -- 복원 품질 (masked positions only, Normal pretrain/val)")
    mask_ratio = ssl_cfg.get("mask_ratio", 0.40)
    print(f"  masking: mask_ratio={mask_ratio:.2f}  "
          f"mode={ssl_cfg.get('mask_mode','mixed')}  "
          f"guard_tail={ssl_cfg.get('mask_guard_tail',0)}")
    print("=" * 70)

    X_all  = np.load(pretrain_val_dir / "X.npy",         mmap_mode="r")
    tf_all = np.load(pretrain_val_dir / "time_feat.npy", mmap_mode="r")
    n = min(n_samples, len(X_all))
    idxs = np.random.default_rng(0).choice(len(X_all), n, replace=False)

    all_model_mse   = []
    all_trivial_mse = []

    for start in range(0, n, batch_size):
        end  = min(start + batch_size, n)
        bi   = idxs[start:end]
        xb   = torch.tensor(X_all[bi],  dtype=torch.float32)
        tfb  = torch.tensor(tf_all[bi], dtype=torch.float32)
        m_mse, t_mse = _masked_mse_batch(
            encoder, recon_head, xb, tfb, ssl_cfg, device
        )
        all_model_mse.append(m_mse)
        all_trivial_mse.append(t_mse)

    m_mse = np.concatenate(all_model_mse)
    t_mse = np.concatenate(all_trivial_mse)
    ratio = m_mse / np.clip(t_mse, 1e-12, None)

    print(f"\n  N={n}  (masked-position MSE 기준)\n")
    print(_pct(m_mse,   "model MSE  "))
    print(_pct(t_mse,   "trivial MSE"))
    print(_pct(ratio,   "ratio(m/t) "))
    print()
    med_r = float(np.median(ratio))
    if med_r < 0.7:
        print(f"  [OK]       median ratio={med_r:.3f} < 0.7  -> 모델이 trivial baseline보다 유의미하게 낫다")
    elif med_r < 1.0:
        print(f"  [MARGINAL] median ratio={med_r:.3f}  -> trivial보다 낫지만 격차가 작다")
    else:
        print(f"  [WARN]     median ratio={med_r:.3f} >= 1.0 -> trivial baseline 이하 (pretrain 미학습 의심)")

    # ── 정성 샘플: n_qual개 window, 마스킹 위치 원본/복원 값 출력 ──────────
    print(f"\n  -- 정성 샘플 {n_qual}개 (masked timestep 원본 vs 복원) --")
    qual_idxs = idxs[:n_qual]
    torch.manual_seed(7)

    for wi, idx in enumerate(qual_idxs):
        xb  = torch.tensor(X_all[idx:idx+1],  dtype=torch.float32)
        tfb = torch.tensor(tf_all[idx:idx+1], dtype=torch.float32)
        xb_dev = xb.to(device)
        tfb_dev = tfb.to(device)

        mask = generate_mask(xb_dev, ssl_cfg)
        z    = encoder(xb_dev, tfb_dev, mask=mask)
        recon = recon_head(z).squeeze(0).cpu().numpy()   # (T, C)
        orig  = xb.squeeze(0).numpy()

        if mask.dim() == 2:
            mask3d = mask.unsqueeze(-1).expand_as(xb_dev)
        else:
            mask3d = mask
        m_np = mask3d.squeeze(0).cpu().numpy()   # (T, C)

        masked_t = np.where(m_np[:, 0])[0]       # ch0 기준 마스킹된 timestep 인덱스
        n_ch = orig.shape[1]
        n_show = min(10, len(masked_t))           # 최대 10 timestep 출력

        m_mse_i = float(np.mean((orig[m_np] - recon[m_np]) ** 2))
        print(f"\n  [w{wi+1}] idx={idx}  masked_positions={m_np.sum()}  "
              f"masked_MSE={m_mse_i:.6f}")
        for ch in range(min(2, n_ch)):
            mt_ch = np.where(m_np[:, ch])[0]
            print(f"    ch{ch}  {'t':>5}  {'orig':>9}  {'recon':>9}  {'|err|':>8}  bar")
            for t in mt_ch[:n_show]:
                o = orig[t, ch]
                r = recon[t, ch]
                bar = "#" * int(min(abs(o - r) / (max(abs(o), 0.1)) * 20, 20))
                print(f"    ch{ch}  {t:>5}  {o:>9.4f}  {r:>9.4f}  {abs(o-r):>8.4f}  {bar}")


# ---------------------------------------------------------------------------
# DIAG 2: Normal vs Attack masked reconstruction error + AUC-ROC
# ---------------------------------------------------------------------------

TYPE_IDX = {
    "scale_down":    0,
    "ramp":          1,
    "pulse_plateau": 2,
    "replay":        3,
    "instant_spike": 4,
}


class NpyDataset(Dataset):
    def __init__(self, X, tf, labels, type_labels):
        self.X  = torch.tensor(X,  dtype=torch.float32)
        self.tf = torch.tensor(tf, dtype=torch.float32)
        self.bl = torch.tensor(labels,      dtype=torch.int32)
        self.tl = torch.tensor(type_labels, dtype=torch.int32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.tf[i], self.bl[i], self.tl[i]


@torch.no_grad()
def diag_recon_error_distribution(
    encoder: nn.Module,
    recon_head: nn.Module,
    downstream_dir: Path,
    ssl_cfg: dict,
    device: torch.device,
    n_per_group: int = 500,
    batch_size: int = 64,
) -> None:
    print("\n" + "=" * 70)
    print("DIAG 2 -- Normal vs Attack masked reconstruction error + AUC-ROC")
    mask_ratio = ssl_cfg.get("mask_ratio", 0.40)
    print(f"  masking: mask_ratio={mask_ratio:.2f}  "
          f"mode={ssl_cfg.get('mask_mode','mixed')}")
    print("=" * 70)

    train_dir = downstream_dir / "all_type" / "train"
    X_all  = np.load(train_dir / "X.npy",            mmap_mode="r")
    tf_all = np.load(train_dir / "time_feat.npy",    mmap_mode="r")
    bl_all = np.load(train_dir / "binary_label.npy", mmap_mode="r")
    tl_all = np.load(train_dir / "type_label.npy",   mmap_mode="r")

    rng = np.random.default_rng(42)
    normal_idx = np.where(bl_all == 0)[0]
    attack_idx = np.where(bl_all == 1)[0]
    sel_n = rng.choice(normal_idx, min(n_per_group, len(normal_idx)), replace=False)
    sel_a = rng.choice(attack_idx, min(n_per_group, len(attack_idx)), replace=False)
    sel   = np.concatenate([sel_n, sel_a])

    X_s  = X_all[sel]
    tf_s = tf_all[sel]
    bl_s = bl_all[sel]
    tl_s = tl_all[sel]

    all_model_mse = []
    all_labels    = []
    type_mse      = {name: [] for name in TYPE_IDX}

    for start in range(0, len(sel), batch_size):
        end  = min(start + batch_size, len(sel))
        xb   = torch.tensor(X_s[start:end],  dtype=torch.float32)
        tfb  = torch.tensor(tf_s[start:end], dtype=torch.float32)
        lb   = bl_s[start:end]
        tlb  = tl_s[start:end]

        m_mse, _ = _masked_mse_batch(
            encoder, recon_head, xb, tfb, ssl_cfg, device
        )
        all_model_mse.append(m_mse)
        all_labels.append(lb)

        for bi in range(len(xb)):
            for name, tidx in TYPE_IDX.items():
                if int(tlb[bi]) == tidx:
                    type_mse[name].append(float(m_mse[bi]))

    all_mse    = np.concatenate(all_model_mse)
    all_labels = np.concatenate(all_labels)

    n_mask = (all_labels == 0)
    a_mask = (all_labels == 1)

    print("\n  -- masked-position MSE per group --")
    print(_pct(all_mse[n_mask], "Normal        "))
    print(_pct(all_mse[a_mask], "Attack (all)  "))
    print("\n  -- per attack type --")
    for name in TYPE_IDX:
        arr = np.array(type_mse[name])
        print(_pct(arr, f"  {name:<20}"))

    if a_mask.sum() > 0 and n_mask.sum() > 0:
        try:
            auc = roc_auc_score(all_labels, all_mse)
            print(f"\n  AUC-ROC (masked recon error as anomaly score): {auc:.4f}")
            print(f"  (baseline linear-probing AUC: ~0.56-0.61)")
        except Exception as e:
            print(f"\n  AUC-ROC error: {e}")

        print("\n  -- per-type AUC-ROC (Normal vs each attack type) --")
        for name, tidx in TYPE_IDX.items():
            mask_t = np.array([int(tl_s[i]) == tidx for i in range(len(tl_s))])
            sel2   = n_mask | mask_t
            if all_labels[sel2].sum() == 0 or (1 - all_labels[sel2]).sum() == 0:
                print(f"    {name:<20} -- skip")
                continue
            try:
                a2 = roc_auc_score(all_labels[sel2], all_mse[sel2])
                print(f"    {name:<20} AUC-ROC={a2:.4f}  "
                      f"(N_normal={n_mask.sum()}, N_attack={mask_t.sum()})")
            except Exception:
                print(f"    {name:<20} -- AUC error")
    else:
        print("\n  [SKIP] normal or attack sample 없음")


# ---------------------------------------------------------------------------
# DIAG 3: Forecast error vs trivial baseline
# ---------------------------------------------------------------------------

@torch.no_grad()
def diag_forecast_error(
    encoder: nn.Module,
    forecast_head: nn.Module,
    pretrain_val_dir: Path,
    ssl_cfg: dict,
    device: torch.device,
    n_samples: int = 400,
    batch_size: int = 64,
) -> None:
    print("\n" + "=" * 70)
    print("DIAG 3 -- Forecast error vs trivial last-step baseline")
    print("=" * 70)

    X_all  = np.load(pretrain_val_dir / "X.npy",              mmap_mode="r")
    tf_all = np.load(pretrain_val_dir / "time_feat.npy",      mmap_mode="r")
    ft_all = np.load(pretrain_val_dir / "future_target.npy",  mmap_mode="r")

    n = min(n_samples, len(X_all))
    idxs = np.random.default_rng(1).choice(len(X_all), n, replace=False)

    model_mse_list   = []
    trivial_mse_list = []

    for start in range(0, n, batch_size):
        end  = min(start + batch_size, n)
        bi   = idxs[start:end]
        xb   = torch.tensor(X_all[bi],  dtype=torch.float32).to(device)
        tfb  = torch.tensor(tf_all[bi], dtype=torch.float32).to(device)
        ftb  = torch.tensor(ft_all[bi], dtype=torch.float32).to(device)

        # forecast는 masking 없이 (h_L anchor 보호 목적이므로 mask 불필요)
        z    = encoder(xb, tfb)
        pred = forecast_head(z)          # (B, h, C)

        model_mse   = ((pred - ftb) ** 2).mean(dim=(1, 2)).cpu().numpy()
        last_step   = xb[:, -1:, :].expand_as(ftb)
        trivial_mse = ((last_step - ftb) ** 2).mean(dim=(1, 2)).cpu().numpy()

        model_mse_list.append(model_mse)
        trivial_mse_list.append(trivial_mse)

    m_mse = np.concatenate(model_mse_list)
    t_mse = np.concatenate(trivial_mse_list)
    ratio = m_mse / np.clip(t_mse, 1e-12, None)

    print(f"\n  N={n}  forecast_horizon={ft_all.shape[1]}\n")
    print(_pct(m_mse,  "model MSE   "))
    print(_pct(t_mse,  "trivial MSE "))
    print(_pct(ratio,  "ratio(m/t)  "))
    med_r = float(np.median(ratio))
    print()
    if med_r < 0.9:
        print(f"  [OK]       median ratio={med_r:.3f} < 0.9  -> forecast 학습됨")
    elif med_r < 1.1:
        print(f"  [MARGINAL] median ratio={med_r:.3f}  -> trivial과 비슷한 수준")
    else:
        print(f"  [WARN]     median ratio={med_r:.3f} >= 1.1 -> trivial baseline 이하 (forecast 미학습)")

    # 채널별 breakdown
    print(f"\n  -- per-channel forecast MSE (model vs trivial) --")
    ch_model_list   = []
    ch_trivial_list = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bi  = idxs[start:end]
        xb  = torch.tensor(X_all[bi],  dtype=torch.float32).to(device)
        tfb = torch.tensor(tf_all[bi], dtype=torch.float32).to(device)
        ftb = torch.tensor(ft_all[bi], dtype=torch.float32).to(device)
        z   = encoder(xb, tfb)
        pred = forecast_head(z)
        last = xb[:, -1:, :].expand_as(ftb)
        ch_model_list.append(((pred - ftb) ** 2).mean(dim=1).cpu().numpy())    # (B, C)
        ch_trivial_list.append(((last - ftb) ** 2).mean(dim=1).cpu().numpy())

    m_ch = np.concatenate(ch_model_list,   axis=0).mean(axis=0)   # (C,)
    t_ch = np.concatenate(ch_trivial_list, axis=0).mean(axis=0)
    for ch in range(len(m_ch)):
        r = m_ch[ch] / max(t_ch[ch], 1e-12)
        print(f"    ch{ch}  model={m_ch[ch]:.6f}  trivial={t_ch[ch]:.6f}  ratio={r:.3f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(
    pretrain_config: str = "configs/pretrain/default.yaml",
    downstream_config: str = "configs/downstream/default.yaml",
) -> None:
    pt_cfg   = yaml.safe_load(open(pretrain_config,   encoding="utf-8"))
    ds_cfg   = yaml.safe_load(open(downstream_config, encoding="utf-8"))
    ssl_cfg  = pt_cfg["ssl"]

    mask_ratio = ssl_cfg.get("mask_ratio", 0.40)
    print(f"[setup] pretrain_config={pretrain_config}")
    print(f"[setup] mask_ratio={mask_ratio:.2f}  (fixed)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")

    ckpt_path = Path(ds_cfg.get("pretrain_ckpt", "checkpoints/pretrain/best.pt"))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    encoder, recon_head, forecast_head, n_feat, d_model, fh = \
        load_pretrain_models(ckpt_path, device)

    pretrain_val_dir = Path("data/processed/pretrain/val")
    downstream_dir   = Path(ds_cfg.get("downstream_dir", "data/processed/downstream"))

    diag_reconstruction_quality(
        encoder, recon_head, pretrain_val_dir, ssl_cfg, device,
        n_samples=200,
    )
    diag_recon_error_distribution(
        encoder, recon_head, downstream_dir, ssl_cfg, device,
        n_per_group=500,
    )
    diag_forecast_error(
        encoder, forecast_head, pretrain_val_dir, ssl_cfg, device,
        n_samples=400,
    )

    print("\n" + "=" * 70)
    print("진단 완료")
    print("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain_config",  default="configs/pretrain/default.yaml")
    p.add_argument("--downstream_config", default="configs/downstream/default.yaml")
    args = p.parse_args()
    main(args.pretrain_config, args.downstream_config)
