"""LR 스케줄러 실효값 검증.

optimizer.param_groups[0]['lr']을 직접 읽어 시뮬레이션.
(get_last_lr()은 base_lr이 이미 곱해진 값을 반환하므로, 진단 스크립트에서
 다시 base_lr을 곱해 출력했던 것은 표기 오류였음 — 값 자체는 정상)

사용법::
    PYTHONPATH=. python scripts/verify_lr.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import yaml

from src.adt.engine.train_downstream import _cosine_lr


def main(config: str = "configs/downstream/default.yaml") -> None:
    cfg = yaml.safe_load(open(config, encoding="utf-8"))
    det_cfg = cfg["detection"]

    # 실제 데이터 기준 배치 수
    train_dir = Path(cfg["downstream_dir"]) / "all_type" / "train"
    if train_dir.exists():
        n_samples = len(np.load(train_dir / "X.npy", mmap_mode="r"))
        n_batches = math.ceil(n_samples / det_cfg["batch_size"])
    else:
        n_batches = 321  # fallback: 82,114 / 256 ≈ 321
        print(f"[warn] downstream data 없음 — n_batches={n_batches}로 가정")

    epochs       = det_cfg["epochs"]
    warmup_steps = det_cfg.get("warmup_steps", 0)
    base_lr      = det_cfg["lr"]
    total_steps  = epochs * n_batches

    print(f"n_batches={n_batches}  epochs={epochs}  "
          f"total_steps={total_steps}  warmup_steps={warmup_steps}  base_lr={base_lr:.2e}")
    print(f"warmup covers {warmup_steps/n_batches:.2f} epochs\n")

    m = torch.nn.Linear(1, 1)
    optim = torch.optim.AdamW(m.parameters(), lr=base_lr)
    sched = _cosine_lr(optim, total_steps, warmup_steps)

    # 확인할 step 목록
    checkpoints = sorted({
        0, 1, warmup_steps // 2, warmup_steps - 1, warmup_steps, warmup_steps + 1,
        500, 1000, total_steps // 4, total_steps // 2, total_steps - 1,
    })
    checkpoints = [s for s in checkpoints if 0 <= s < total_steps]

    print(f"{'step':>8}  {'lr':>12}  {'% of base_lr':>14}  note")
    print("-" * 55)

    ptr = 0
    for s in range(total_steps):
        if s == checkpoints[ptr]:
            lr = optim.param_groups[0]["lr"]
            pct = lr / base_lr * 100
            note = ""
            if s == warmup_steps:
                note = "← warmup 끝 (여기서 base_lr에 도달해야 정상)"
            elif s == 0:
                note = "← warmup 시작 (0이어야 정상)"
            elif s < warmup_steps:
                note = "← warmup 중"
            print(f"{s:>8}  {lr:>12.6e}  {pct:>13.1f}%  {note}")
            ptr += 1
            if ptr >= len(checkpoints):
                break
        sched.step()

    print()
    # 최종 확인
    lr_at_warmup_end = base_lr * (warmup_steps / max(warmup_steps, 1))
    print(f"이론값 — step={warmup_steps}에서 기대 lr : {base_lr:.6e}  (100.0%)")
    print(f"         step=0에서 기대 lr         : 0.000000e+00  (0.0%)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/downstream/default.yaml")
    main(p.parse_args().config)
