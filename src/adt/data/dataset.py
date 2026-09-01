"""torch Dataset / DataLoader 정의."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class LPWindowDataset(Dataset):
    """data/processed/{split}/X.npy + time_feat.npy 를 로드하는 Dataset.

    pretrain 단계: 레이블 없이 (x, time_feat) 튜플만 반환.
    finetune 단계: 레이블이 필요하면 subclass에서 확장.
    """

    def __init__(self, processed_dir: str | Path, split: str) -> None:
        split_dir = Path(processed_dir) / split
        x_path = split_dir / "X.npy"
        tf_path = split_dir / "time_feat.npy"

        if not x_path.exists():
            raise FileNotFoundError(
                f"{x_path} 없음. scripts/prepare_data.py 를 먼저 실행하세요."
            )

        self.X = torch.from_numpy(np.load(x_path))                # (N, T, C)
        self.time_feat = torch.from_numpy(np.load(tf_path))       # (N, T, 2)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.time_feat[idx]


def build_dataloader(
    processed_dir: str | Path,
    split: str,
    batch_size: int,
    shuffle: bool | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """LPWindowDataset을 감싸는 DataLoader 생성.

    Args:
        shuffle: None이면 split='train'일 때만 True.
                 배치 셔플이지 시계열 내부 순서를 섞는 게 아니라 leakage 없음.
    """
    if shuffle is None:
        shuffle = split == "train"

    dataset = LPWindowDataset(processed_dir, split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
