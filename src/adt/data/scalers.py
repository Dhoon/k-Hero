"""채널별(feature-wise) StandardScaler — (N, T, C) 텐서용.

train split으로만 fit한 뒤 저장해두고,
val / test / inference 단계에서 동일 객체를 로드해 transform.

사용 예시:
    scaler = StandardScalerND()
    X_train = scaler.fit_transform(X_train_raw)
    X_val   = scaler.transform(X_val_raw)
    scaler.save("data/processed/scaler.joblib")

    # 나중에
    scaler = StandardScalerND.load("data/processed/scaler.joblib")
    X_new   = scaler.transform(X_new_raw)
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


class StandardScalerND:
    """채널(C) 축 기준 StandardScaler.  입력 shape: (N, T, C) 또는 (T, C)."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # fit / transform
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "StandardScalerND":
        """X: (N, T, C) 또는 (T, C). 채널별 mean/std를 train 데이터로 계산."""
        arr = X.reshape(-1, X.shape[-1])          # → (N*T, C) 또는 (T, C)
        self.mean_ = arr.mean(axis=0)             # (C,)
        self.std_ = arr.std(axis=0)               # (C,)
        # 표준편차 0인 채널 방어 (상수 채널)
        self.std_ = np.where(self.std_ == 0.0, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """정규화.  fit() 미호출 시 AssertionError."""
        assert self.mean_ is not None, "fit() 먼저 호출 필요"
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None
        return (X * self.std_ + self.mean_).astype(np.float32)

    # ------------------------------------------------------------------
    # 저장 / 로드
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """joblib으로 mean/std 저장."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"mean": self.mean_, "std": self.std_}, path)

    @classmethod
    def load(cls, path: str | Path) -> "StandardScalerND":
        """저장된 scaler 로드."""
        data = joblib.load(path)
        scaler = cls()
        scaler.mean_ = data["mean"]
        scaler.std_ = data["std"]
        return scaler

    # ------------------------------------------------------------------
    # 디버그 출력
    # ------------------------------------------------------------------

    def report(self, feature_names: list[str] | None = None) -> None:
        """채널별 mean / std 출력."""
        assert self.mean_ is not None
        print("\n[Scaler 통계 — 정규화 후 채널별 mean/std]")
        for i, (m, s) in enumerate(zip(self.mean_, self.std_)):
            name = feature_names[i] if feature_names else f"ch{i}"
            print(f"  {name:30s}  mean={m:+.4f}  std={s:.4f}")
