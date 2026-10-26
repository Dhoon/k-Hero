"""C 구현(c/lstm_joint) 출력이 PyTorch 와 같은지 검증 (PC에서 실행, torch 필요).

사용법::

    python tools/verify_c.py --exe c/lstm_joint.exe \
        --npy ../campus-power-ad/data/processed/downstream/all_type/test/X.npy -n 300

--npy 는 정규화된 (N, T, C) window 배열 (main 브랜치 data/processed 산출물).
생략하면 random window 를 쓴다. 각 window 를 원본 스케일로 되돌려 CSV 로 C 에 넣고
p_attack / class logit 을 PyTorch 결과와 비교한다.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
from lstm_joint import LSTMJoint  # noqa: E402


def _load_model() -> LSTMJoint:
    cfg = json.loads((ROOT / "model" / "config.json").read_text(encoding="utf-8"))["model"]
    m = LSTMJoint(
        n_features=4, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"],
        num_classes=cfg["num_classes"], det_hidden=cfg["det_head_hidden"],
        cls_hidden=cfg["cls_head_hidden"], dropout=0.0,
    )
    m.load_state_dict(torch.load(ROOT / "model" / "model.pt", map_location="cpu"))
    return m.eval()


def _run_c(exe: str, rows: np.ndarray, extra: list[str]) -> list[str]:
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("a,b,c,d\n")
        for r in rows:
            f.write(",".join(f"{v:.9g}" for v in r) + "\n")
        path = f.name
    out = subprocess.run([exe, "--no-header", *extra, path], capture_output=True, text=True, check=True).stdout
    Path(path).unlink()
    return out.strip().split("\t")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=str(ROOT / "c" / "lstm_joint.exe"))
    ap.add_argument("--npy", default=None)
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sc = json.loads((ROOT / "model" / "scaler.json").read_text(encoding="utf-8"))
    mean, std = np.array(sc["mean"], np.float32), np.array(sc["std"], np.float32)
    thr = json.loads((ROOT / "model" / "threshold.json").read_text())["threshold"]

    rng = np.random.default_rng(args.seed)
    if args.npy:
        X = np.load(args.npy, mmap_mode="r")
        X = np.asarray(X[np.sort(rng.choice(len(X), size=min(args.n, len(X)), replace=False))])
    else:
        X = rng.standard_normal((args.n, 96, 4)).astype(np.float32)
    raw = X * std + mean   # C 는 정규화 전 값을 받는다

    model = _load_model()
    with torch.no_grad():
        det, cls = model(torch.from_numpy(raw.astype(np.float32) - mean) / torch.from_numpy(std))
    p_ref = torch.sigmoid(det).numpy()
    cls_ref = cls.numpy()

    max_dp = max_dl = 0.0
    n_flip_attack = n_flip_cls = 0
    for i in range(len(raw)):
        f = _run_c(args.exe, raw[i], [])
        p_c, att_c = float(f[1]), int(f[2])
        logits_c = np.array([float(v) for v in f[4:]])
        max_dp = max(max_dp, abs(p_c - p_ref[i]))
        max_dl = max(max_dl, float(np.abs(logits_c - cls_ref[i]).max()))
        n_flip_attack += att_c != int(p_ref[i] > thr)
        n_flip_cls += int(logits_c.argmax()) != int(cls_ref[i].argmax())

    # --cumulative 경로: 음수 없는 window 하나를 적산값으로 바꿔 동일 결과인지
    cum_ok = None
    nonneg = [i for i in range(len(raw)) if (raw[i] >= 0).all()]
    if nonneg:
        i = nonneg[0]
        cum = np.vstack([np.zeros((1, 4), np.float32), np.cumsum(raw[i], axis=0)])
        f = _run_c(args.exe, cum, ["--cumulative"])
        cum_ok = abs(float(f[1]) - p_ref[i]) < 1e-3

    n_att = int((p_ref > thr).sum())
    print(f"windows={len(raw)}  (ref attack={n_att}, normal={len(raw) - n_att})")
    print(f"max |p_attack diff|   = {max_dp:.2e}")
    print(f"max |cls logit diff|  = {max_dl:.2e}")
    print(f"is_attack mismatches  = {n_flip_attack}")
    print(f"class argmax mismatch = {n_flip_cls}")
    print(f"--cumulative check    = {cum_ok}")
    ok = max_dp < 1e-4 and max_dl < 1e-3 and n_flip_attack == 0 and n_flip_cls == 0 and cum_ok is not False
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
