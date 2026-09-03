"""downstream 학습 부진 원인 진단 스크립트.

진단 항목:
  1. encoder checkpoint 존재 여부 / 파일 크기 / 실제 로드 확인
  2. all_type/train 배치에서 Normal vs Attack pooled feature 분포 비교
  3. detection head 파라미터가 학습 중 실제로 바뀌는지 (norm 변화)
  4. LR 스케줄러 실효 LR 추적 (warmup 설정 버그 탐지)

사용법::
    PYTHONPATH=. python scripts/diagnose_downstream.py \
        [--config configs/downstream/default.yaml]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.adt.models.encoder import TimeSeriesTransformerEncoder
from src.adt.models.heads.detection_head import DetectionHead
from src.adt.utils.checkpoint import load_encoder_frozen
from src.adt.engine.train_downstream import (
    DownstreamFoldDataset, compute_class_info, compute_pos_weight,
    run_step, _cosine_lr,
)
from src.adt.models.heads.classification_head import ClassificationHead


def _load_cfg(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 진단 1: encoder checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def diag_checkpoint(cfg: dict) -> nn.Module | None:
    print("\n" + "=" * 60)
    print("DIAG 1: encoder checkpoint")
    print("=" * 60)

    ckpt_path = Path(cfg.get("pretrain_ckpt", "checkpoints/pretrain/best.pt"))
    print(f"  설정 경로 : {ckpt_path.resolve()}")

    if not ckpt_path.exists():
        print("  [FAIL] 파일 없음 - random init 폴백 또는 에러 발생")
        return None

    size_mb = ckpt_path.stat().st_size / 1024 / 1024
    print(f"  파일 크기 : {size_mb:.2f} MB", end="")
    if size_mb < 0.1:
        print("  ← [WARN] 너무 작음 (빈 파일?)")
    else:
        print("  [OK]")

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict):
        keys = list(state.keys())
        print(f"  checkpoint keys : {keys}")
        if "encoder" not in state:
            print("  [WARN] 'encoder' 키 없음 - state_dict 전체를 encoder로 간주")
        else:
            enc_state = state["encoder"]
            n_params = sum(v.numel() for v in enc_state.values())
            print(f"  encoder params  : {n_params:,}")
            # 파라미터가 모두 0인지 확인
            norms = [v.norm().item() for v in enc_state.values()]
            print(f"  param norm (mean/min/max): "
                  f"{np.mean(norms):.4f} / {min(norms):.4f} / {max(norms):.4f}")
            if np.mean(norms) < 1e-6:
                print("  [WARN] 파라미터 norm이 거의 0 - pretrain이 제대로 저장 안 됐을 수 있음")
        if "best_val_loss" in state:
            print(f"  best_val_loss   : {state['best_val_loss']:.6f}")
        if "epoch" in state:
            print(f"  saved at epoch  : {state['epoch']}")
    else:
        print("  [WARN] checkpoint가 dict가 아닌 형태")

    # 실제 로드
    model_cfg = cfg["model"]
    encoder = TimeSeriesTransformerEncoder(
        n_features=4,  # 실제 데이터에서 추론하지 않고 cfg에서 가져올 수 없어 임시 4
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=model_cfg["d_ff"],
        dropout=0.0,
    )
    try:
        encoder = load_encoder_frozen(ckpt_path, encoder)
        print("  load_encoder_frozen: [OK] 정상 로드 완료")
    except Exception as e:
        print(f"  load_encoder_frozen: [FAIL] {e}")
        return None

    # n_features 재확인 (실제 데이터와 맞는지)
    ds_dir = Path(cfg["downstream_dir"]) / "all_type" / "train"
    if ds_dir.exists():
        X = np.load(ds_dir / "X.npy", mmap_mode="r")
        n_feat_data = X.shape[-1]
        enc_in_dim = model_cfg["d_model"]  # proxy
        print(f"  데이터 n_features: {n_feat_data} (config d_model={model_cfg['d_model']})")
        if n_feat_data != 4:
            # n_features를 데이터에서 읽어 encoder 재생성
            encoder2 = TimeSeriesTransformerEncoder(
                n_features=n_feat_data,
                d_model=model_cfg["d_model"],
                n_heads=model_cfg["n_heads"],
                n_layers=model_cfg["n_layers"],
                d_ff=model_cfg["d_ff"],
                dropout=0.0,
            )
            try:
                encoder2 = load_encoder_frozen(ckpt_path, encoder2)
                encoder = encoder2
                print(f"  n_features={n_feat_data}로 encoder 재로드 [OK]")
            except Exception as e:
                print(f"  n_features={n_feat_data}로 재로드 실패: {e}")

    return encoder


# ─────────────────────────────────────────────────────────────────────────────
# 진단 2: Normal vs Attack pooled feature 분포
# ─────────────────────────────────────────────────────────────────────────────

def diag_features(encoder: nn.Module, cfg: dict) -> None:
    print("\n" + "=" * 60)
    print("DIAG 2: Normal vs Attack pooled feature 분포")
    print("=" * 60)

    ds_dir = Path(cfg["downstream_dir"]) / "all_type" / "train"
    if not ds_dir.exists():
        print(f"  [SKIP] {ds_dir} 없음 - prepare_downstream_data.py 먼저 실행 필요")
        return

    ds = DownstreamFoldDataset(ds_dir)
    bl = ds.binary_label.numpy()
    n_idx = np.where(bl == 0)[0]
    a_idx = np.where(bl == 1)[0]
    print(f"  전체 샘플: {len(ds):,}  Normal: {len(n_idx):,}  Attack: {len(a_idx):,}")

    # 각 16개씩 샘플링
    rng = np.random.default_rng(0)
    n_sample = 16
    ni = rng.choice(n_idx, min(n_sample, len(n_idx)), replace=False)
    ai = rng.choice(a_idx, min(n_sample, len(a_idx)), replace=False)

    def _pool(idxs):
        xs  = torch.stack([ds.X[i]         for i in idxs])
        tfs = torch.stack([ds.time_feat[i]  for i in idxs])
        with torch.no_grad():
            z = encoder(xs, tfs)           # (N, T, d_model)
            pooled = torch.cat([z.mean(1), z.max(1).values], dim=-1)  # (N, 2*d)
        return pooled.numpy()

    n_feat = _pool(ni)   # (16, 2*d_model)
    a_feat = _pool(ai)

    print(f"\n  pooled feature shape: {n_feat.shape}")

    # 분산
    n_std = n_feat.std(axis=0).mean()
    a_std = a_feat.std(axis=0).mean()
    print(f"  Normal  feature std (mean over dims): {n_std:.6f}")
    print(f"  Attack  feature std (mean over dims): {a_std:.6f}")
    if max(n_std, a_std) < 1e-4:
        print("  [WARN] 분산이 극히 작음 - encoder output이 거의 상수 → 학습 불가")

    # Normal/Attack 간 코사인 유사도
    n_mean = n_feat.mean(0)
    a_mean = a_feat.mean(0)
    cos_sim = float(
        np.dot(n_mean, a_mean) / (np.linalg.norm(n_mean) * np.linalg.norm(a_mean) + 1e-9)
    )
    l2_dist = float(np.linalg.norm(n_mean - a_mean))
    print(f"\n  Normal mean vs Attack mean:")
    print(f"    코사인 유사도 : {cos_sim:.6f}  (1.0 = 구분 불가, 낮을수록 분리됨)")
    print(f"    L2 거리       : {l2_dist:.6f}")
    if cos_sim > 0.999:
        print("  [WARN] 코사인 유사도 ≈ 1.0 → Normal/Attack feature 거의 동일")
        print("         encoder가 random init이거나 pretrain이 제대로 안 됐을 가능성 큼")

    # 샘플별 L2 (Normal 16개 × Attack 16개 쌍별 거리 평균)
    dists = np.array([
        np.linalg.norm(n_feat[i] - a_feat[j])
        for i in range(len(ni)) for j in range(len(ai))
    ])
    print(f"  pairwise L2 (N×A): mean={dists.mean():.4f}  std={dists.std():.4f}")

    # 같은 클래스 내 pairwise
    nn_dists = np.array([
        np.linalg.norm(n_feat[i] - n_feat[j])
        for i in range(len(ni)) for j in range(i+1, len(ni))
    ])
    aa_dists = np.array([
        np.linalg.norm(a_feat[i] - a_feat[j])
        for i in range(len(ai)) for j in range(i+1, len(ai))
    ])
    if len(nn_dists):
        print(f"  intra Normal  L2: mean={nn_dists.mean():.4f}")
    if len(aa_dists):
        print(f"  intra Attack  L2: mean={aa_dists.mean():.4f}")

    if len(nn_dists) and len(dists):
        ratio = dists.mean() / (nn_dists.mean() + 1e-9)
        print(f"  inter/intra 비율: {ratio:.3f}  (>1이면 클래스 간 분리됨, <1이면 같이 뭉침)")


# ─────────────────────────────────────────────────────────────────────────────
# 진단 2b: 공격 유형별 Normal vs Attack 분리도
# ─────────────────────────────────────────────────────────────────────────────

def diag_features_per_type(encoder: nn.Module, cfg: dict, n_sample: int = 32) -> None:
    """Normal vs 각 공격 유형별 pooled feature 분리도.

    inter/intra < 1 이면 그 유형이 Normal과 뭉쳐 있어 linear probing으로 구분 불가.
    LayerNorm이 magnitude를 정규화하는 특성상 Scale Down / Pulse Plateau가 특히 취약할 것으로 예상.
    """
    print("\n" + "=" * 60)
    print("DIAG 2b: 공격 유형별 Normal vs Attack 분리도")
    print("=" * 60)

    ds_dir = Path(cfg["downstream_dir"]) / "all_type" / "train"
    if not ds_dir.exists():
        print(f"  [SKIP] {ds_dir} 없음")
        return

    ds = DownstreamFoldDataset(ds_dir)
    bl = ds.binary_label.numpy()
    tl = ds.type_label.numpy()
    n_idx = np.where(bl == 0)[0]

    def _pool(idxs):
        xs  = torch.stack([ds.X[i]         for i in idxs])
        tfs = torch.stack([ds.time_feat[i]  for i in idxs])
        with torch.no_grad():
            z      = encoder(xs, tfs)
            pooled = torch.cat([z.mean(1), z.max(1).values], dim=-1)
        return pooled.numpy()

    rng = np.random.default_rng(42)

    # Normal 기준
    ni     = rng.choice(n_idx, min(n_sample, len(n_idx)), replace=False)
    n_feat = _pool(ni)
    n_mean = n_feat.mean(0)
    nn_dists = np.array([
        np.linalg.norm(n_feat[i] - n_feat[j])
        for i in range(len(ni)) for j in range(i + 1, len(ni))
    ])
    intra_n = nn_dists.mean() if len(nn_dists) else 0.0

    from src.adt.data.labeling import TYPE_IDX as _TYPE_IDX

    header = f"  {'attack type':<22} {'cos_sim':>8} {'inter_L2':>10} {'intra_N_L2':>11} {'inter/intra':>12}"
    print(f"\n{header}")
    print("  " + "-" * 67)

    for type_name, orig_idx in sorted(_TYPE_IDX.items(), key=lambda x: x[1]):
        a_idx = np.where((bl == 1) & (tl == orig_idx))[0]
        if len(a_idx) < 4:
            print(f"  {type_name:<22} [샘플 부족: {len(a_idx)}개]")
            continue

        ai     = rng.choice(a_idx, min(n_sample, len(a_idx)), replace=False)
        a_feat = _pool(ai)
        a_mean = a_feat.mean(0)

        cos_sim = float(
            np.dot(n_mean, a_mean)
            / (np.linalg.norm(n_mean) * np.linalg.norm(a_mean) + 1e-9)
        )
        inter_dists = np.array([
            np.linalg.norm(n_feat[i] - a_feat[j])
            for i in range(len(ni)) for j in range(len(ai))
        ])
        inter_mean = inter_dists.mean()
        ratio      = inter_mean / (intra_n + 1e-9)

        flag = "  ← 분리 어려움 (LayerNorm magnitude 소거 의심)" if ratio < 1.0 else ""
        print(
            f"  {type_name:<22} {cos_sim:>8.4f} {inter_mean:>10.4f} "
            f"{intra_n:>11.4f} {ratio:>12.3f}{flag}"
        )

    print(f"\n  기준: inter/intra > 1 이면 분리됨, < 1 이면 Normal과 뭉침")
    print(f"  Normal intra L2 (기준값): {intra_n:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 진단 3: detection head 파라미터 norm 변화
# ─────────────────────────────────────────────────────────────────────────────

def diag_head_learning(encoder: nn.Module, cfg: dict) -> None:
    print("\n" + "=" * 60)
    print("DIAG 3: detection head 파라미터 변화 (10 step)")
    print("=" * 60)

    ds_dir = Path(cfg["downstream_dir"]) / "all_type" / "train"
    if not ds_dir.exists():
        print(f"  [SKIP] {ds_dir} 없음")
        return

    ds = DownstreamFoldDataset(ds_dir)
    det_cfg = cfg["detection"]
    cls_cfg = cfg["classification"]
    model_cfg = cfg["model"]

    train_tl = ds.type_label.numpy()
    _, class_names, type_to_class = compute_class_info(train_tl)
    pw = compute_pos_weight(ds.binary_label.numpy())

    d_model = model_cfg["d_model"]
    det_head = DetectionHead(d_model=d_model, hidden_dim=det_cfg["hidden_dim"], dropout=0.0)
    cls_head = ClassificationHead(d_model=d_model, num_classes=len(class_names),
                                  hidden_dim=cls_cfg["hidden_dim"], dropout=0.0)
    det_head.train(); cls_head.train()

    det_optim = torch.optim.AdamW(det_head.parameters(), lr=det_cfg["lr"],
                                  weight_decay=det_cfg["weight_decay"])
    cls_optim = torch.optim.AdamW(cls_head.parameters(), lr=cls_cfg["lr"],
                                  weight_decay=cls_cfg["weight_decay"])

    det_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]))
    cls_loss_fn = nn.CrossEntropyLoss()

    # 파라미터 초기 norm
    def _param_norm(m):
        return sum(p.norm().item() ** 2 for p in m.parameters()) ** 0.5

    norms_before = _param_norm(det_head)
    print(f"  det_head param norm - before: {norms_before:.6f}")

    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
    N_STEPS = 10
    det_losses = []
    cls_losses = []

    for step, (x, tf, bl, tl) in enumerate(loader):
        if step >= N_STEPS:
            break
        result = run_step(
            encoder, det_head, cls_head,
            x, tf, bl, tl,
            det_loss_fn, cls_loss_fn, type_to_class,
            det_optim, cls_optim,
        )
        det_losses.append(result["det_loss"])
        if result["cls_loss"] is not None:
            cls_losses.append(result["cls_loss"])

    norms_after = _param_norm(det_head)
    delta = abs(norms_after - norms_before)
    print(f"  det_head param norm - after : {norms_after:.6f}  delta={delta:.6f}")
    if delta < 1e-7:
        print("  [WARN] det_head 파라미터가 전혀 안 바뀜 - gradient 흐름 문제")
    else:
        print("  [OK] det_head 파라미터 업데이트 확인")

    print(f"\n  detection loss 추이 ({N_STEPS} step):")
    for i, l in enumerate(det_losses):
        print(f"    step {i+1:2d}: {l:.6f}")
    if len(det_losses) > 1:
        trend = det_losses[-1] - det_losses[0]
        print(f"  loss 변화: {trend:+.6f}  ({'감소 중' if trend < 0 else '증가/정체'})")

    if cls_losses:
        print(f"\n  classification loss 추이:")
        for i, l in enumerate(cls_losses):
            print(f"    step {i+1:2d}: {l:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# 진단 4: LR 스케줄러 실효 LR
# ─────────────────────────────────────────────────────────────────────────────

def diag_scheduler(cfg: dict) -> None:
    print("\n" + "=" * 60)
    print("DIAG 4: LR 스케줄러 실효 LR")
    print("=" * 60)

    det_cfg = cfg["detection"]
    ds_dir = Path(cfg["downstream_dir"]) / "all_type" / "train"

    # 배치 수 추정
    if ds_dir.exists():
        n_samples = len(np.load(ds_dir / "X.npy", mmap_mode="r"))
        n_batches = math.ceil(n_samples / det_cfg["batch_size"])
    else:
        n_batches = 320  # 가정

    epochs = det_cfg["epochs"]
    warmup_steps = det_cfg.get("warmup_steps", 0)
    base_lr = det_cfg["lr"]

    # 현재 코드: scheduler.step()을 epoch당 1회 호출
    total_steps_epoch = epochs  # step()이 실제로 몇 번 불리는가

    print(f"\n  [현재 코드] epoch당 scheduler.step() 1회")
    print(f"  epochs={epochs}  warmup_steps(설정)={warmup_steps}  base_lr={base_lr}")
    print(f"  step() 총 호출 횟수: {total_steps_epoch}")
    print(f"\n  epoch별 실효 LR:  (total_steps={epochs * n_batches} 기준으로 LambdaLR 계산)")

    # _cosine_lr은 total_steps=epochs*n_batches로 만들어지지만
    # step()은 epoch당 1회라서 실제 step 번호는 0..epochs-1
    # → LambdaLR의 last_epoch가 step() 횟수로 움직임
    d = nn.Linear(1, 1)
    optim = torch.optim.AdamW(d.parameters(), lr=base_lr)
    total_steps_sched = epochs * n_batches
    sched = _cosine_lr(optim, total_steps_sched, warmup_steps)

    print(f"  {'epoch':>6}  {'sched_step':>10}  {'lr_scale':>10}  {'eff_lr':>12}")
    warmup_done_epoch = None
    for ep in range(epochs):
        lr_scale = sched.get_last_lr()[0] if ep > 0 else sched.get_last_lr()[0]
        eff_lr = base_lr * lr_scale
        flag = ""
        if ep < warmup_steps and (ep + 1) >= warmup_steps:
            flag = " ← warmup 끝"
            warmup_done_epoch = ep + 1
        print(f"  {ep+1:>6}  {ep:>10}  {lr_scale:>10.6f}  {eff_lr:>12.2e}{flag}")
        sched.step()

    if warmup_steps > total_steps_epoch:
        print(f"\n  [BUG] warmup_steps={warmup_steps} > 총 step() 호출 수({total_steps_epoch})")
        print(f"        → warmup이 끝나지 않고 학습 종료!")
        print(f"        → LR이 base_lr의 최대 {total_steps_epoch/warmup_steps*100:.0f}%까지만 도달")
        print(f"\n  수정 방법: scheduler.step()을 epoch당이 아니라 batch당 호출하거나,")
        print(f"            warmup_steps를 epoch 단위로 변경 (예: warmup_steps: 3)")
    elif warmup_steps == 0:
        print(f"\n  [OK] warmup_steps=0 → 첫 epoch부터 base_lr 사용")
    else:
        print(f"\n  warmup {warmup_done_epoch}epoch에 완료, 이후 cosine decay")

    # batch당 step()이 호출되는 경우 비교 (수정된 코드의 실제 동작)
    # get_last_lr()은 이미 base_lr이 곱해진 LR을 반환함 - 다시 곱하지 말 것
    print(f"\n  [참고] batch당 step() 호출 시 (수정된 코드):")
    d2 = nn.Linear(1, 1)
    optim2 = torch.optim.AdamW(d2.parameters(), lr=base_lr)
    sched2 = _cosine_lr(optim2, total_steps_sched, warmup_steps)
    milestones = set([0, warmup_steps - 1, warmup_steps, warmup_steps + 1,
                      total_steps_sched // 2, total_steps_sched - 1])
    print(f"  {'step':>8}  {'lr (optim)':>12}  {'% of base_lr':>14}")
    for step_i in range(total_steps_sched):
        if step_i in milestones:
            # optimizer.param_groups[0]['lr']이 실제 LR
            actual_lr = optim2.param_groups[0]["lr"]
            pct = actual_lr / base_lr * 100
            note = " ← warmup 끝 (여기서 100%)" if step_i == warmup_steps else ""
            print(f"  {step_i:>8}  {actual_lr:>12.6e}  {pct:>13.1f}%{note}")
        sched2.step()


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main(config: str = "configs/downstream/default.yaml") -> None:
    cfg = _load_cfg(config)

    encoder = diag_checkpoint(cfg)

    if encoder is not None:
        # n_features를 데이터에서 읽어 확인
        ds_dir = Path(cfg["downstream_dir"]) / "all_type" / "train"
        if ds_dir.exists():
            n_feat = np.load(ds_dir / "X.npy", mmap_mode="r").shape[-1]
            model_cfg = cfg["model"]
            enc_check = TimeSeriesTransformerEncoder(
                n_features=n_feat,
                d_model=model_cfg["d_model"],
                n_heads=model_cfg["n_heads"],
                n_layers=model_cfg["n_layers"],
                d_ff=model_cfg["d_ff"],
                dropout=0.0,
            )
            ckpt_path = Path(cfg.get("pretrain_ckpt", "checkpoints/pretrain/best.pt"))
            try:
                encoder = load_encoder_frozen(ckpt_path, enc_check)
            except Exception:
                pass  # 이미 diag_checkpoint에서 보고함

        diag_features(encoder, cfg)
        diag_features_per_type(encoder, cfg)
        diag_head_learning(encoder, cfg)

    diag_scheduler(cfg)

    print("\n" + "=" * 60)
    print("진단 완료")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/downstream/default.yaml")
    args = parser.parse_args()
    main(args.config)
