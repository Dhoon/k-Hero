"""model/ 의 weight·메타데이터 → c/model_weights.h 생성 (PC에서 실행, torch 필요).

사용법::

    python tools/export_c.py

LSTM bias는 bias_ih + bias_hh 를 미리 합쳐서 내보낸다.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
OUT = ROOT / "c" / "model_weights.h"


def _arr(name: str, t: torch.Tensor) -> str:
    flat = t.detach().float().contiguous().view(-1).tolist()
    body = ",\n".join(
        "  " + ", ".join(f"{v:.9g}f" for v in flat[i:i + 8])
        for i in range(0, len(flat), 8)
    )
    return f"static const float {name}[{len(flat)}] = {{\n{body}\n}};\n"


def main() -> None:
    cfg = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    scaler = json.loads((MODEL_DIR / "scaler.json").read_text(encoding="utf-8"))
    thr = json.loads((MODEL_DIR / "threshold.json").read_text(encoding="utf-8"))
    names = json.loads((MODEL_DIR / "class_names.json").read_text(encoding="utf-8"))
    sd = torch.load(MODEL_DIR / "model.pt", map_location="cpu")

    m = cfg["model"]
    T, C = cfg["input"]["window_size"], cfg["input"]["n_features"]
    H, L, K = m["hidden_dim"], m["num_layers"], m["num_classes"]
    DH, CH = m["det_head_hidden"], m["cls_head_hidden"]

    parts = [
        "/* 자동 생성 파일 — tools/export_c.py 로 다시 만들 것. 직접 수정 금지. */\n",
        "#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n",
        f"#define LJ_T {T}\n#define LJ_C {C}\n#define LJ_H {H}\n#define LJ_L {L}\n",
        f"#define LJ_K {K}\n#define LJ_DH {DH}\n#define LJ_CH {CH}\n\n",
        f"static const float LJ_THRESHOLD = {thr['threshold']:.9g}f;\n",
        "static const char *const LJ_CLASS_NAMES[LJ_K] = {"
        + ", ".join(f'"{names[str(i)]}"' for i in range(K)) + "};\n\n",
        _arr("LJ_MEAN", torch.tensor(scaler["mean"])),
        _arr("LJ_STD", torch.tensor(scaler["std"])),
        "\n",
    ]

    # encoder: layer l, direction d (0=fwd, 1=bwd)
    for l in range(L):
        for d, sfx in enumerate(["", "_reverse"]):
            p = f"encoder.lstm.{{}}_l{l}{sfx}"
            parts.append(_arr(f"LJ_WIH_{l}_{d}", sd[p.format("weight_ih")]))
            parts.append(_arr(f"LJ_WHH_{l}_{d}", sd[p.format("weight_hh")]))
            parts.append(_arr(f"LJ_B_{l}_{d}", sd[p.format("bias_ih")] + sd[p.format("bias_hh")]))

    for head, tag in [("det_head", "DET"), ("cls_head", "CLS")]:
        parts.append(_arr(f"LJ_{tag}_W1", sd[f"{head}.net.0.weight"]))
        parts.append(_arr(f"LJ_{tag}_B1", sd[f"{head}.net.0.bias"]))
        parts.append(_arr(f"LJ_{tag}_W2", sd[f"{head}.net.3.weight"]))
        parts.append(_arr(f"LJ_{tag}_B2", sd[f"{head}.net.3.bias"]))

    # layer/direction 별 포인터 테이블
    parts.append("\nstatic const float *const LJ_WIH[LJ_L][2] = {"
                 + ", ".join(f"{{LJ_WIH_{l}_0, LJ_WIH_{l}_1}}" for l in range(L)) + "};\n")
    parts.append("static const float *const LJ_WHH[LJ_L][2] = {"
                 + ", ".join(f"{{LJ_WHH_{l}_0, LJ_WHH_{l}_1}}" for l in range(L)) + "};\n")
    parts.append("static const float *const LJ_B[LJ_L][2] = {"
                 + ", ".join(f"{{LJ_B_{l}_0, LJ_B_{l}_1}}" for l in range(L)) + "};\n")
    parts.append("\n#endif /* MODEL_WEIGHTS_H */\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(parts), encoding="utf-8", newline="\n")
    print(f"saved: {OUT}  ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
