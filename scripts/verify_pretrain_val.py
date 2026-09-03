"""pretrain best.pt val loss 재현 검증.

기존 _val_epoch() 함수를 그대로 임포트해서 호출.
학습 로그와 수치가 일치하는지 확인한다.

사용법::
    python scripts/verify_pretrain_val.py
    python scripts/verify_pretrain_val.py --config configs/pretrain/default.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

# 기존 함수 그대로 재사용 ─ 새 코드 없음
from src.adt.engine.pretrain import _val_epoch
from src.adt.data.dataset import build_dataloader
from src.adt.models.encoder import TimeSeriesTransformerEncoder
from src.adt.models.heads.reconstruction_head import MaskedReconstructionHead
from src.adt.models.heads.forecasting_head import ForecastingHead


def main(config: str = "configs/pretrain/default.yaml") -> None:
    cfg       = yaml.safe_load(open(config, encoding="utf-8"))
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    ssl_cfg   = cfg["ssl"]

    data_cfg_path = Path(cfg.get("data_config", "configs/data/default.yaml"))
    data_cfg      = yaml.safe_load(open(data_cfg_path, encoding="utf-8"))
    n_features        = len(data_cfg["feature_cols"])
    forecast_horizon  = data_cfg.get("forecast_horizon", 0)
    forecast_loss_weight = float(ssl_cfg.get("forecast_loss_weight", 0.0))
    processed_dir     = str(Path(data_cfg["processed_dir"]) / "pretrain")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # ── 체크포인트 로드 ────────────────────────────────────────────────────
    ckpt_path = Path(train_cfg["ckpt_dir"]) / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")

    saved_epoch      = int(state.get("epoch", "?"))
    saved_best_loss  = float(state.get("best_val_loss", float("nan")))
    print(f"[ckpt] saved_epoch={saved_epoch}  best_val_loss(recorded)={saved_best_loss:.4f}")

    # ── 모델 생성 + 가중치 로드 ───────────────────────────────────────────
    encoder = TimeSeriesTransformerEncoder(
        n_features=n_features,
        d_model   =model_cfg["d_model"],
        n_heads   =model_cfg["n_heads"],
        n_layers  =model_cfg["n_layers"],
        d_ff      =model_cfg["d_ff"],
        dropout   =model_cfg["dropout"],
    ).to(device)

    recon_head = MaskedReconstructionHead(
        d_model   =model_cfg["d_model"],
        n_features=n_features,
    ).to(device)

    forecast_head = ForecastingHead(
        d_model         =model_cfg["d_model"],
        forecast_horizon=forecast_horizon,
        n_features      =n_features,
    ).to(device)

    # strict=True — 불일치 시 RuntimeError 로 즉시 노출
    enc_missing, enc_unexp = encoder.load_state_dict(
        state["encoder"], strict=True
    )
    rh_missing,  rh_unexp  = recon_head.load_state_dict(
        state["recon_head"], strict=True
    )
    fh_missing,  fh_unexp  = forecast_head.load_state_dict(
        state["forecast_head"], strict=True
    )
    print(f"[load] encoder      missing={enc_missing}  unexpected={enc_unexp}")
    print(f"[load] recon_head   missing={rh_missing}   unexpected={rh_unexp}")
    print(f"[load] forecast_head missing={fh_missing}  unexpected={fh_unexp}")

    # ── DataLoader ────────────────────────────────────────────────────────
    val_loader = build_dataloader(
        processed_dir,
        "val",
        batch_size  =train_cfg["batch_size"],
        shuffle     =False,
        num_workers =0,
    )
    print(f"[data] val batches={len(val_loader)}  batch_size={train_cfg['batch_size']}")

    # ── _val_epoch 호출 ───────────────────────────────────────────────────
    # saved_epoch은 학습 후 +1 저장된 값이므로, 마지막으로 돌았던 epoch index는
    # saved_epoch - 1.  mask_ratio는 curriculum 기준으로 이 epoch에 해당하는 값 사용.
    epoch_idx = max(0, saved_epoch - 1)

    print(f"\n[run] _val_epoch(epoch={epoch_idx})  "
          f"forecast_loss_weight={forecast_loss_weight}")
    print("      (masking은 stochastic -> 완전 동일값 보장 불가, 근사 일치 확인)")

    losses = _val_epoch(
        encoder, recon_head, forecast_head,
        val_loader, device,
        epoch_idx, ssl_cfg, forecast_loss_weight,
    )

    # ── 비교 출력 ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("결과 비교")
    print("=" * 60)
    print(f"  {'항목':<20} {'학습 로그(참고)':>18} {'지금 재계산':>14}")
    print("-" * 60)
    refs = {"total": 0.0958, "mask": 0.0900, "forecast": 0.0385}
    for key in ("total", "mask", "forecast"):
        ref = refs[key]
        now = losses[key]
        diff = abs(now - ref)
        flag = ""
        if diff > ref * 0.10:
            flag = "  <-- 10% 초과 차이"
        print(f"  L_{key:<16} {ref:>18.4f} {now:>14.4f}{flag}")
    print("=" * 60)
    print()

    total_diff = abs(losses["total"] - refs["total"])
    if total_diff < refs["total"] * 0.10:
        print("[OK]   val loss가 학습 로그와 10% 이내 일치 -> 체크포인트/모델 정상")
        print("       DIAG 1~3 스크립트와 validate() 차이를 아래에서 비교하세요.")
        print()
        print("  validate() vs diagnose_pretrain.py 주요 차이점:")
        print("  1. epoch 인자: validate() epoch=saved_epoch-1 / DIAG은 curriculum_epochs(50) 고정")
        print(f"     -> mask_ratio 차이 가능 (validate epoch={epoch_idx} vs DIAG epoch=50)")
        print("  2. DIAG 1 trivial baseline: 비마스킹 구간 채널 평균 예측")
        print("     -> validate()는 trivial 계산 없음 (손실 함수만 계산)")
        print("  3. DIAG 2: downstream 데이터(공격 포함) 사용 / validate()는 pretrain val만")
        print("  4. DIAG 3: forecast_head에 mask 없이 encoder 통과")
        print("     -> validate()도 동일 (forecast head는 mask 입력과 무관, h_L 사용)")
        print("  5. scaler: 양쪽 모두 정규화된 X 그대로 사용 (역정규화 없음)")
    else:
        print("[WARN] val loss가 학습 로그와 10% 이상 차이 -> 로딩/설정 문제 의심")
        print("       위 [load] 줄에서 missing/unexpected key 확인 필요")
        print("       ssl_cfg, n_features, forecast_horizon이 학습 때와 동일한지 확인")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pretrain/default.yaml")
    args = p.parse_args()
    main(args.config)
