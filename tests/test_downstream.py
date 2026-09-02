"""test_downstream.py — Downstream 학습/평가 파이프라인 테스트.

더미 encoder checkpoint로 실제 학습/평가 없이 구조 정합성만 확인.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.adt.models.encoder import TimeSeriesTransformerEncoder
from src.adt.models.heads.detection_head import DetectionHead
from src.adt.models.heads.classification_head import ClassificationHead
from src.adt.utils.checkpoint import load_encoder_frozen
from src.adt.data.labeling import TYPE_IDX, LABEL_NORMAL
from src.adt.engine.train_downstream import (
    ALL_FOLDS,
    FOLD_UNSEEN_TYPE,
    DownstreamFoldDataset,
    compute_class_info,
    compute_pos_weight,
    run_step,
    train_fold,
    train_all_folds,
)
from src.adt.engine.evaluate_downstream import (
    evaluate_fold,
    evaluate_all_folds,
    _infer_filtered,
)

# ── 공통 파라미터 ──────────────────────────────────────────────────────────
B, T, C, D = 8, 96, 4, 128   # batch, timesteps, channels, d_model
N_TRAIN, N_VAL, N_TEST = 200, 50, 50


# =========================================================================
# 픽스처
# =========================================================================

@pytest.fixture(scope="module")
def encoder() -> TimeSeriesTransformerEncoder:
    m = TimeSeriesTransformerEncoder(
        n_features=C, d_model=D, n_heads=4, n_layers=2, d_ff=256, dropout=0.0
    )
    m.eval()
    return m


@pytest.fixture
def dummy_ckpt(tmp_path, encoder) -> Path:
    """encoder state_dict를 저장한 더미 pretrain checkpoint."""
    ckpt = {"encoder": encoder.state_dict(), "epoch": 1, "best_val_loss": 0.5}
    p = tmp_path / "best.pt"
    torch.save(ckpt, p)
    return p


def _make_fold_data(
    root: Path,
    fold_name: str,
    splits: list[str],
    n_per_split: int = N_TRAIN,
    type_labels_present: list[int] | None = None,
) -> None:
    """테스트용 fold 데이터를 tmp_path 아래에 생성."""
    if type_labels_present is None:
        type_labels_present = list(range(5))  # 0..4 all types

    rng = np.random.default_rng(0)
    for split in splits:
        d = root / fold_name / split
        d.mkdir(parents=True, exist_ok=True)
        X  = rng.standard_normal((n_per_split, T, C)).astype(np.float32)
        tf = np.zeros((n_per_split, T, 2), dtype=np.float32)
        # 절반 Normal, 절반 Attack
        n_normal = n_per_split // 2
        n_attack = n_per_split - n_normal
        bl = np.zeros(n_per_split, dtype=np.int32)
        bl[n_normal:] = 1
        tl = np.full(n_per_split, LABEL_NORMAL, dtype=np.int32)
        for i in range(n_attack):
            tl[n_normal + i] = type_labels_present[i % len(type_labels_present)]
        np.save(d / "X.npy",            X)
        np.save(d / "time_feat.npy",    tf)
        np.save(d / "binary_label.npy", bl)
        np.save(d / "type_label.npy",   tl)


@pytest.fixture
def small_cfg(tmp_path) -> dict:
    """최소 epoch(1)으로 설정된 downstream config."""
    ds_dir = str(tmp_path / "downstream")
    ckpt_dir = str(tmp_path / "ckpts")
    return {
        "downstream_dir": ds_dir,
        "output_dir": str(tmp_path / "outputs"),
        "seed": 42,
        "model": {
            "d_model": D, "n_heads": 4, "n_layers": 2,
            "d_ff": 256, "dropout": 0.0,
        },
        "detection": {
            "hidden_dim": 32, "dropout": 0.0,
            "lr": 1e-3, "weight_decay": 1e-5,
            "epochs": 1, "batch_size": 32, "warmup_steps": 0,
            "ckpt_dir": ckpt_dir, "log_dir": str(tmp_path / "logs"),
        },
        "classification": {
            "hidden_dim": 32, "dropout": 0.0,
            "lr": 1e-3, "weight_decay": 1e-5,
            "epochs": 1, "batch_size": 32, "warmup_steps": 0,
            "ckpt_dir": ckpt_dir, "log_dir": str(tmp_path / "logs"),
        },
    }


# =========================================================================
# 1. Head 출력 shape
# =========================================================================

class TestHeadShapes:
    def test_detection_head_output_shape(self):
        head = DetectionHead(d_model=D, hidden_dim=32)
        H = torch.randn(B, T, D)
        out = head(H)
        assert out.shape == (B,), f"Expected (B,), got {out.shape}"

    def test_classification_head_output_shape_5class(self):
        head = ClassificationHead(d_model=D, num_classes=5, hidden_dim=32)
        H = torch.randn(B, T, D)
        out = head(H)
        assert out.shape == (B, 5), f"Expected (B, 5), got {out.shape}"

    def test_classification_head_output_shape_4class(self):
        head = ClassificationHead(d_model=D, num_classes=4, hidden_dim=32)
        H = torch.randn(B, T, D)
        out = head(H)
        assert out.shape == (B, 4), f"Expected (B, 4), got {out.shape}"

    def test_detection_head_no_nan(self):
        head = DetectionHead(d_model=D, hidden_dim=32)
        out = head(torch.randn(B, T, D))
        assert not out.isnan().any()

    def test_classification_head_no_nan(self):
        head = ClassificationHead(d_model=D, num_classes=5, hidden_dim=32)
        out = head(torch.randn(B, T, D))
        assert not out.isnan().any()


# =========================================================================
# 2. load_encoder_frozen — eval + freeze
# =========================================================================

class TestEncoderFreeze:
    def test_encoder_eval_after_freeze(self, dummy_ckpt):
        enc = TimeSeriesTransformerEncoder(
            n_features=C, d_model=D, n_heads=4, n_layers=2, d_ff=256
        )
        enc.train()
        enc = load_encoder_frozen(dummy_ckpt, enc)
        assert not enc.training, "encoder should be in eval mode"

    def test_encoder_no_grad_after_freeze(self, dummy_ckpt):
        enc = TimeSeriesTransformerEncoder(
            n_features=C, d_model=D, n_heads=4, n_layers=2, d_ff=256
        )
        enc = load_encoder_frozen(dummy_ckpt, enc)
        for p in enc.parameters():
            assert not p.requires_grad, "All encoder params should be frozen"

    def test_encoder_params_unchanged_after_train_step(self, encoder):
        """학습 step 후 encoder 파라미터 값이 변하지 않아야 함."""
        det_head = DetectionHead(d_model=D, hidden_dim=32)
        cls_head = ClassificationHead(d_model=D, num_classes=2, hidden_dim=32)
        det_optim = torch.optim.SGD(det_head.parameters(), lr=1e-2)
        cls_optim = torch.optim.SGD(cls_head.parameters(), lr=1e-2)

        # encoder 파라미터 스냅샷
        enc_before = {
            n: p.clone().detach() for n, p in encoder.named_parameters()
        }

        # 배치 설정: 4 normal + 4 attack (type 0 and 1)
        x  = torch.randn(B, T, C)
        tf = torch.zeros(B, T, 2)
        bl = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        tl = torch.tensor([-1, -1, -1, -1, 0, 0, 1, 1], dtype=torch.long)
        type_to_class = {0: 0, 1: 1}

        det_loss_fn = nn.BCEWithLogitsLoss()
        cls_loss_fn = nn.CrossEntropyLoss()

        run_step(
            encoder, det_head, cls_head,
            x, tf, bl, tl,
            det_loss_fn, cls_loss_fn, type_to_class,
            det_optim, cls_optim,
        )

        # encoder 파라미터 확인
        for n, p in encoder.named_parameters():
            torch.testing.assert_close(
                p, enc_before[n],
                msg=f"encoder param '{n}' changed after train step!",
            )


# =========================================================================
# 3. z_t 재사용 + Normal 제외
# =========================================================================

class TestRunStep:
    def _make_batch(self):
        """4 Normal + 4 Attack (type 0/1) 배치."""
        x  = torch.randn(B, T, C)
        tf = torch.zeros(B, T, 2)
        bl = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        tl = torch.tensor([-1, -1, -1, -1, 0, 0, 1, 1], dtype=torch.long)
        return x, tf, bl, tl

    def test_zt_reuse_for_classification(self, encoder):
        """z_attack는 z_t[attack_mask]와 동일해야 함 (re-encode 없음)."""
        det = DetectionHead(d_model=D, hidden_dim=32)
        cls = ClassificationHead(d_model=D, num_classes=2, hidden_dim=32)
        det_o = torch.optim.SGD(det.parameters(), lr=1e-3)
        cls_o = torch.optim.SGD(cls.parameters(), lr=1e-3)
        x, tf, bl, tl = self._make_batch()

        result = run_step(
            encoder, det, cls, x, tf, bl, tl,
            nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(),
            {0: 0, 1: 1}, det_o, cls_o,
        )
        z_t   = result["z_t"]
        z_att = result["z_attack"]
        mask  = result["attack_mask"]

        # z_attack == z_t[attack_mask]
        torch.testing.assert_close(z_att, z_t[mask])

    def test_encoder_called_once_per_batch(self, encoder):
        """배치 당 encoder forward 1번만 호출됨을 검증."""
        det = DetectionHead(d_model=D, hidden_dim=32)
        cls = ClassificationHead(d_model=D, num_classes=2, hidden_dim=32)
        det_o = torch.optim.SGD(det.parameters(), lr=1e-3)
        cls_o = torch.optim.SGD(cls.parameters(), lr=1e-3)
        x, tf, bl, tl = self._make_batch()

        with patch.object(encoder, "forward", wraps=encoder.forward) as mock_fwd:
            run_step(
                encoder, det, cls, x, tf, bl, tl,
                nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(),
                {0: 0, 1: 1}, det_o, cls_o,
            )
            assert mock_fwd.call_count == 1, (
                f"encoder.forward called {mock_fwd.call_count} times (expected 1)"
            )

    def test_normal_excluded_from_classification(self, encoder):
        """z_attack에 Normal 샘플이 포함되지 않아야 함."""
        det = DetectionHead(d_model=D, hidden_dim=32)
        cls = ClassificationHead(d_model=D, num_classes=2, hidden_dim=32)
        det_o = torch.optim.SGD(det.parameters(), lr=1e-3)
        cls_o = torch.optim.SGD(cls.parameters(), lr=1e-3)
        x, tf, bl, tl = self._make_batch()

        result = run_step(
            encoder, det, cls, x, tf, bl, tl,
            nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(),
            {0: 0, 1: 1}, det_o, cls_o,
        )
        attack_mask = result["attack_mask"]
        # Normal samples (bl==0) must NOT be in attack_mask
        normal_mask = (bl == 0)
        assert not (attack_mask & normal_mask).any(), (
            "Normal 샘플이 classification 입력에 포함됨"
        )
        # Number of attack samples in z_attack
        n_attack = int(attack_mask.sum())
        assert result["z_attack"].shape[0] == n_attack


# =========================================================================
# 4. compute_class_info + compute_pos_weight
# =========================================================================

class TestClassInfo:
    def test_all_type_has_5_classes(self):
        tl = np.array([-1, 0, 1, 2, 3, 4, 0, 1, -1], dtype=np.int32)
        _, class_names, type_to_class = compute_class_info(tl)
        assert len(class_names) == 5

    def test_unseen_fold_has_4_classes(self):
        # unseen_scale_down: scale_down(0) 없음
        tl = np.array([-1, 1, 2, 3, 4, 1, 2, 3, -1], dtype=np.int32)
        _, class_names, type_to_class = compute_class_info(tl)
        assert len(class_names) == 4
        # class_names 값에 "scale_down"이 없어야 함
        names = list(class_names.values())
        from src.adt.data.labeling import TYPE_IDX as TI
        assert "scale_down" not in names

    def test_class_names_json_serializable(self):
        tl = np.array([0, 1, 2, 3, 4, -1], dtype=np.int32)
        _, class_names, _ = compute_class_info(tl)
        # JSON 직렬화 가능한지
        s = json.dumps(class_names)
        loaded = json.loads(s)
        assert loaded == class_names

    def test_type_to_class_remapping(self):
        # ramp(1), pulse_plateau(2), replay(3), instant_spike(4)만 있을 때
        tl = np.array([1, 2, 3, 4, -1], dtype=np.int32)
        _, _, type_to_class = compute_class_info(tl)
        assert type_to_class[1] == 0   # ramp → class 0
        assert type_to_class[2] == 1
        assert type_to_class[3] == 2
        assert type_to_class[4] == 3

    def test_pos_weight_correct(self):
        bl = np.array([0] * 9 + [1] * 1, dtype=np.int32)  # 9:1 imbalance
        pw = compute_pos_weight(bl)
        assert abs(pw - 9.0) < 1e-6, f"Expected 9.0, got {pw}"

    def test_pos_weight_balanced(self):
        bl = np.array([0] * 5 + [1] * 5, dtype=np.int32)
        pw = compute_pos_weight(bl)
        assert abs(pw - 1.0) < 1e-6


# =========================================================================
# 5. train_fold — class_names.json 저장 + 체크포인트 생성
# =========================================================================

class TestTrainFold:
    def _setup_fold(self, tmp_path, small_cfg, fold_name, n=N_TRAIN, types=None):
        ds_dir = Path(small_cfg["downstream_dir"])
        present = types if types is not None else list(range(5))
        _make_fold_data(ds_dir, fold_name, ["train", "val"], n, present)

    def test_class_names_json_saved(self, tmp_path, small_cfg, encoder):
        fold = "all_type"
        self._setup_fold(tmp_path, small_cfg, fold, types=list(range(5)))
        train_fold(fold, small_cfg, encoder, torch.device("cpu"), verbose=False)
        cls_ckpt_dir = (
            Path(small_cfg["classification"]["ckpt_dir"]) / fold / "classifier"
        )
        assert (cls_ckpt_dir / "class_names.json").exists()

    def test_all_type_class_names_has_5(self, tmp_path, small_cfg, encoder):
        fold = "all_type"
        self._setup_fold(tmp_path, small_cfg, fold, types=list(range(5)))
        train_fold(fold, small_cfg, encoder, torch.device("cpu"), verbose=False)
        cls_ckpt_dir = (
            Path(small_cfg["classification"]["ckpt_dir"]) / fold / "classifier"
        )
        cn = json.loads((cls_ckpt_dir / "class_names.json").read_text(encoding="utf-8"))
        assert len(cn) == 5

    def test_unseen_fold_class_names_has_4(self, tmp_path, small_cfg, encoder):
        fold = "unseen_scale_down"
        # scale_down(0) 제외
        self._setup_fold(tmp_path, small_cfg, fold, types=[1, 2, 3, 4])
        train_fold(fold, small_cfg, encoder, torch.device("cpu"), verbose=False)
        cls_ckpt_dir = (
            Path(small_cfg["classification"]["ckpt_dir"]) / fold / "classifier"
        )
        cn = json.loads((cls_ckpt_dir / "class_names.json").read_text(encoding="utf-8"))
        assert len(cn) == 4
        assert "scale_down" not in cn.values()

    def test_checkpoints_saved(self, tmp_path, small_cfg, encoder):
        fold = "all_type"
        self._setup_fold(tmp_path, small_cfg, fold, types=list(range(5)))
        train_fold(fold, small_cfg, encoder, torch.device("cpu"), verbose=False)
        det_dir = Path(small_cfg["detection"]["ckpt_dir"]) / fold / "detector"
        cls_dir = Path(small_cfg["classification"]["ckpt_dir"]) / fold / "classifier"
        assert (det_dir / "last.pt").exists()
        assert (cls_dir / "last.pt").exists()


# =========================================================================
# 6. fold 순회 — train_all_folds / evaluate_all_folds
# =========================================================================

class TestFoldIteration:
    def _setup_all_folds(self, small_cfg) -> None:
        ds_dir = Path(small_cfg["downstream_dir"])
        for fold_name in ALL_FOLDS:
            unseen = FOLD_UNSEEN_TYPE.get(fold_name)
            if unseen is not None:
                types = [i for i in range(5) if i != TYPE_IDX[unseen]]
            else:
                types = list(range(5))
            splits = ["train", "val"] if fold_name != "all_type" else ["train", "val", "test"]
            _make_fold_data(ds_dir, fold_name, splits, n_per_split=64, type_labels_present=types)

    def test_train_all_folds_iterates_6_times(self, tmp_path, small_cfg, encoder):
        self._setup_all_folds(small_cfg)
        call_log: list[str] = []
        original_fn = train_fold

        def tracking_fn(fold_name, cfg, enc, dev, **kwargs):
            call_log.append(fold_name)
            return original_fn(fold_name, cfg, enc, dev, verbose=False)

        with patch(
            "src.adt.engine.train_downstream.train_fold",
            side_effect=tracking_fn,
        ):
            train_all_folds(small_cfg, encoder, torch.device("cpu"), verbose=False)

        assert len(call_log) == 6, f"Expected 6 calls, got {len(call_log)}: {call_log}"
        assert set(call_log) == set(ALL_FOLDS)

    def test_train_single_fold_iterates_once(self, tmp_path, small_cfg, encoder):
        self._setup_all_folds(small_cfg)
        call_log: list[str] = []
        original_fn = train_fold

        def tracking_fn(fold_name, cfg, enc, dev, **kwargs):
            call_log.append(fold_name)
            return original_fn(fold_name, cfg, enc, dev, verbose=False)

        with patch(
            "src.adt.engine.train_downstream.train_fold",
            side_effect=tracking_fn,
        ):
            train_all_folds(
                small_cfg, encoder, torch.device("cpu"),
                folds=["all_type"], verbose=False
            )

        assert call_log == ["all_type"], f"Expected ['all_type'], got {call_log}"


# =========================================================================
# 7. evaluate_fold — unseen_X 분류 시 X 타입 제외
# =========================================================================

class TestEvaluateFiltering:
    def _setup_and_train(self, tmp_path, small_cfg, encoder):
        """all_type/test + all 6 folds를 만들고 all_type만 train."""
        ds_dir = Path(small_cfg["downstream_dir"])
        for fold_name in ALL_FOLDS:
            unseen = FOLD_UNSEEN_TYPE.get(fold_name)
            types = (
                [i for i in range(5) if i != TYPE_IDX[unseen]]
                if unseen else list(range(5))
            )
            splits = ["train", "val"] if fold_name != "all_type" else ["train", "val", "test"]
            _make_fold_data(ds_dir, fold_name, splits, n_per_split=64, type_labels_present=types)

        # all_type만 학습 (class_names.json 생성을 위해)
        train_fold("all_type", small_cfg, encoder, torch.device("cpu"), verbose=False)

        # unseen_scale_down도 학습 (class_names에 scale_down 없어야 함)
        train_fold("unseen_scale_down", small_cfg, encoder, torch.device("cpu"), verbose=False)

    def test_unseen_type_excluded_from_classification_eval(
        self, tmp_path, small_cfg, encoder
    ):
        """unseen_scale_down classifier 평가 시 scale_down(orig_idx=0) 제외."""
        self._setup_and_train(tmp_path, small_cfg, encoder)

        # evaluate_fold에서 사용하는 _infer_filtered를 intercept
        collected_tl: list[int] = []
        original_fn = _infer_filtered

        def capturing_fn(enc, head, ds, mask, batch_size, device):
            result = original_fn(enc, head, ds, mask, batch_size, device)
            _, _, tl = result
            collected_tl.extend(tl.tolist())
            return result

        with patch(
            "src.adt.engine.evaluate_downstream._infer_filtered",
            side_effect=capturing_fn,
        ):
            evaluate_fold(
                "unseen_scale_down", small_cfg, encoder,
                torch.device("cpu"), verbose=False,
            )

        # scale_down의 orig_idx = 0 — 평가 샘플에 없어야 함
        sd_idx = TYPE_IDX["scale_down"]
        assert sd_idx not in collected_tl, (
            f"scale_down(type_label={sd_idx}) found in unseen_scale_down eval samples"
        )
