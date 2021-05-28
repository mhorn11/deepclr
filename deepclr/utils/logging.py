from datetime import datetime
import logging
import os
import sys
from typing import Optional

from tensorboardX import SummaryWriter


def create_logger(name: Optional[str] = None, save_dir: Optional[str] = None, distributed_rank: int = 0) \
        -> logging.Logger:
    """Create python logger and set formatter."""
    # get logger and set log level
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger

    # handlers
    if not logger.hasHandlers():
        # formatter
        if name is None:
            formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        else:
            formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

        # stdout
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # file
        if save_dir:
            filename = datetime.now().strftime('log_%Y%m%d_%H%M%S.txt')
            fh = logging.FileHandler(os.path.join(save_dir, filename), mode='w')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


def create_summary_writer(log_dir: str) -> SummaryWriter:
    """Create tensorboard summary writer."""
    writer = SummaryWriter(log_dir)
    return writer
