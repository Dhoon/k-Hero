"""test_labeling.py — downstream fold 생성 파이프라인 테스트.

검증 항목:
  1. _inject_with_type_labels 기본 동작 (shape, dtype, normal=-1)
  2. 재현성 (동일 seed → 동일 결과)
  3. generate_downstream_folds: unseen_X fold에 X 타입 없음
  4. generate_downstream_folds: 공유 샘플이 all_type와 바이트 단위 동일
  5. generate_downstream_folds: binary_label ↔ type_label 일관성
"""
from __future__ import annotations

import numpy as np
import pytest

from src.adt.data.labeling import (
    LABEL_NORMAL,
    TYPE_IDX,
    _inject_with_type_labels,
    generate_downstream_folds,
)
from src.adt.data.scalers import StandardScalerND


# =========================================================================
# 공통 픽스처
# =========================================================================

N, T, C = 200, 96, 4  # 작은 합성 데이터


@pytest.fixture
def identity_scaler() -> StandardScalerND:
    """mean=0, std=1 항등 스케일러 — inverse/transform 모두 no-op."""
    sc = StandardScalerND()
    sc.mean_ = np.zeros(C, dtype=np.float32)
    sc.std_  = np.ones(C, dtype=np.float32)
    return sc


@pytest.fixture
def injection_cfg() -> dict:
    """injection_ratio=0.8 로 모든 타입이 충분히 등장하게 설정."""
    return {
        "seed": 42,
        "injection_ratio": 0.8,
        "anomaly_types": [
            {"type": "scale_down",    "prob": 0.2, "scale_factor": [0.3, 0.7],   "duration_steps": [2, 8]},
            {"type": "ramp",          "prob": 0.2, "scale_start": 1.0,
             "trough_scale": [0.2, 0.6], "duration_steps": [2, 8]},
            {"type": "pulse_plateau", "prob": 0.2, "magnitude": [1.5, 2.5],      "duration_steps": [2, 8]},
            {"type": "replay",        "prob": 0.2, "duration_steps": [2, 8]},
            {"type": "instant_spike", "prob": 0.2, "magnitude": [0.9, 1.3],      "duration_steps": [1, 2]},
        ],
    }


@pytest.fixture
def fold_cfg(injection_cfg) -> dict:
    """2-fold 설정 (all_type + unseen_scale_down) — 테스트 속도용."""
    return {
        **injection_cfg,
        "folds": [
            {"name": "all_type",          "unseen_type": None,         "splits": ["train", "val"]},
            {"name": "unseen_scale_down", "unseen_type": "scale_down", "splits": ["train", "val"]},
        ],
    }


@pytest.fixture
def pretrain_dir(tmp_path, identity_scaler) -> "Path":
    """train/val 각 N=200 윈도우를 가진 임시 pretrain_dir."""
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        split_dir = tmp_path / "pretrain" / split
        split_dir.mkdir(parents=True)
        X = rng.standard_normal((N, T, C)).astype(np.float32)
        tf = np.zeros((N, T, 2), dtype=np.float32)
        np.save(split_dir / "X.npy", X)
        np.save(split_dir / "time_feat.npy", tf)
    return tmp_path / "pretrain"


# =========================================================================
# 1. _inject_with_type_labels
# =========================================================================

class TestInjectWithTypeLabels:
    def test_output_shapes(self, injection_cfg, identity_scaler):
        X = np.random.randn(N, T, C).astype(np.float32)
        corrupted, tl = _inject_with_type_labels(X, injection_cfg, identity_scaler, seed=0)
        assert corrupted.shape == (N, T, C)
        assert tl.shape == (N,)

    def test_type_label_dtype(self, injection_cfg, identity_scaler):
        X = np.random.randn(N, T, C).astype(np.float32)
        _, tl = _inject_with_type_labels(X, injection_cfg, identity_scaler, seed=0)
        assert tl.dtype == np.int32

    def test_normal_label_is_minus_one(self, injection_cfg, identity_scaler):
        X = np.random.randn(N, T, C).astype(np.float32)
        _, tl = _inject_with_type_labels(X, injection_cfg, identity_scaler, seed=0)
        # 모든 레이블이 -1(Normal) 또는 0..4(attack) 여야 함
        assert ((tl == LABEL_NORMAL) | (tl >= 0)).all()

    def test_type_indices_valid(self, injection_cfg, identity_scaler):
        X = np.random.randn(N, T, C).astype(np.float32)
        _, tl = _inject_with_type_labels(X, injection_cfg, identity_scaler, seed=0)
        valid = set([LABEL_NORMAL]) | set(TYPE_IDX.values())
        assert set(tl.tolist()).issubset(valid)

    def test_normal_windows_unchanged(self, injection_cfg, identity_scaler):
        """Normal 레이블 윈도우는 입력값과 동일해야 함."""
        X = np.abs(np.random.randn(N, T, C).astype(np.float32)) + 1.0
        corrupted, tl = _inject_with_type_labels(X, injection_cfg, identity_scaler, seed=0)
        normal_mask = tl == LABEL_NORMAL
        assert normal_mask.any(), "Normal 샘플이 하나도 없음 (injection_ratio 확인 필요)"
        np.testing.assert_allclose(corrupted[normal_mask], X[normal_mask], atol=1e-5)

    def test_reproducibility(self, injection_cfg, identity_scaler):
        X = np.random.randn(N, T, C).astype(np.float32)
        c1, t1 = _inject_with_type_labels(X, injection_cfg, identity_scaler, seed=42)
        c2, t2 = _inject_with_type_labels(X, injection_cfg, identity_scaler, seed=42)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(t1, t2)

    def test_different_seeds_differ(self, injection_cfg, identity_scaler):
        X = np.random.randn(N, T, C).astype(np.float32)
        _, t1 = _inject_with_type_labels(X, injection_cfg, identity_scaler, seed=1)
        _, t2 = _inject_with_type_labels(X, injection_cfg, identity_scaler, seed=2)
        assert not np.array_equal(t1, t2), "다른 seed인데 동일한 레이블 (의심스러움)"


# =========================================================================
# 2. generate_downstream_folds
# =========================================================================

class TestGenerateDownstreamFolds:
    def _run(self, tmp_path, pretrain_dir, fold_cfg, identity_scaler):
        out = tmp_path / "downstream"
        stats = generate_downstream_folds(pretrain_dir, out, fold_cfg, identity_scaler)
        return out, stats

    # ── fold 필터링 정합성 ──────────────────────────────────────────────

    def test_unseen_type_absent_from_fold(
        self, tmp_path, pretrain_dir, fold_cfg, identity_scaler
    ):
        """unseen_scale_down fold의 type_label에 scale_down(=0) 없어야 함."""
        out, _ = self._run(tmp_path, pretrain_dir, fold_cfg, identity_scaler)
        sd_idx = TYPE_IDX["scale_down"]
        for split in ("train", "val"):
            tl = np.load(out / "unseen_scale_down" / split / "type_label.npy")
            assert sd_idx not in tl, \
                f"scale_down(idx={sd_idx}) found in unseen_scale_down/{split}"

    def test_all_type_contains_all_attack_types(
        self, tmp_path, pretrain_dir, fold_cfg, identity_scaler
    ):
        """높은 injection_ratio로 all_type/train에 모든 attack 타입이 나타나야 함."""
        out, _ = self._run(tmp_path, pretrain_dir, fold_cfg, identity_scaler)
        tl = np.load(out / "all_type" / "train" / "type_label.npy")
        for name, idx in TYPE_IDX.items():
            assert idx in tl, f"'{name}'(idx={idx}) not found in all_type/train"

    # ── 공유 샘플 동일성 ───────────────────────────────────────────────

    def test_shared_samples_byte_identical(
        self, tmp_path, pretrain_dir, fold_cfg, identity_scaler
    ):
        """unseen_scale_down = all_type 에서 scale_down 제거한 것이므로 동일해야 함."""
        out, _ = self._run(tmp_path, pretrain_dir, fold_cfg, identity_scaler)

        X_all = np.load(out / "all_type" / "train" / "X.npy")
        tl_all = np.load(out / "all_type" / "train" / "type_label.npy")

        X_unseen = np.load(out / "unseen_scale_down" / "train" / "X.npy")
        tl_unseen = np.load(out / "unseen_scale_down" / "train" / "type_label.npy")

        sd_idx = TYPE_IDX["scale_down"]
        keep_mask = tl_all != sd_idx

        np.testing.assert_array_equal(X_unseen, X_all[keep_mask])
        np.testing.assert_array_equal(tl_unseen, tl_all[keep_mask])

    # ── binary_label 일관성 ────────────────────────────────────────────

    def test_binary_label_consistent_with_type_label(
        self, tmp_path, pretrain_dir, fold_cfg, identity_scaler
    ):
        """binary_label[i] == (type_label[i] >= 0)."""
        out, _ = self._run(tmp_path, pretrain_dir, fold_cfg, identity_scaler)
        for fold_name in ("all_type", "unseen_scale_down"):
            bl = np.load(out / fold_name / "train" / "binary_label.npy")
            tl = np.load(out / fold_name / "train" / "type_label.npy")
            expected = (tl >= 0).astype(np.int32)
            np.testing.assert_array_equal(bl, expected, err_msg=f"{fold_name}/train")

    # ── 저장 파일 존재 확인 ────────────────────────────────────────────

    def test_all_expected_files_saved(
        self, tmp_path, pretrain_dir, fold_cfg, identity_scaler
    ):
        out, _ = self._run(tmp_path, pretrain_dir, fold_cfg, identity_scaler)
        for fold_name, fd in [("all_type", fold_cfg["folds"][0]),
                               ("unseen_scale_down", fold_cfg["folds"][1])]:
            for split in fd["splits"]:
                for fname in ("X.npy", "time_feat.npy",
                              "binary_label.npy", "type_label.npy"):
                    assert (out / fold_name / split / fname).exists(), \
                        f"missing: {fold_name}/{split}/{fname}"

    # ── stats 반환값 ───────────────────────────────────────────────────

    def test_stats_counts_match_saved_files(
        self, tmp_path, pretrain_dir, fold_cfg, identity_scaler
    ):
        out, stats = self._run(tmp_path, pretrain_dir, fold_cfg, identity_scaler)
        for fold_name in ("all_type", "unseen_scale_down"):
            for split in ("train", "val"):
                tl = np.load(out / fold_name / split / "type_label.npy")
                reported_total = stats[fold_name][split]["total"]
                assert reported_total == len(tl), \
                    f"stats total mismatch for {fold_name}/{split}"
