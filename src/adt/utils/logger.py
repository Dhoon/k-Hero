"""로깅 유틸 (콘솔 + 파일 + TensorBoard)."""
from __future__ import annotations

import logging
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def get_logger(log_dir: str | Path) -> tuple[logging.Logger, SummaryWriter]:
    """콘솔 + 파일 Logger와 TensorBoard SummaryWriter를 함께 반환.

    Returns:
        (logger, writer)
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("adt.pretrain")
    logger.setLevel(logging.INFO)

    # 핸들러 중복 추가 방지
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        fh = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    writer = SummaryWriter(log_dir=str(log_dir))
    return logger, writer
