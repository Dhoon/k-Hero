"""로깅 유틸 (콘솔 + 파일 + TensorBoard)."""
from __future__ import annotations

import logging
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def get_logger(
    log_dir: str | Path,
    name: str = "adt.pretrain",
    log_file: str = "train.log",
) -> tuple[logging.Logger, SummaryWriter]:
    """콘솔 + 파일 Logger와 TensorBoard SummaryWriter를 함께 반환.

    Args:
        log_dir  : 로그 파일 및 TensorBoard 이벤트 저장 경로
        name     : logging.getLogger 이름 (모듈별로 다르게 지정해 분리)
        log_file : 파일 핸들러 파일명

    Returns:
        (logger, writer)
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 핸들러 중복 추가 방지
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        fh = logging.FileHandler(log_dir / log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    writer = SummaryWriter(log_dir=str(log_dir))
    return logger, writer
