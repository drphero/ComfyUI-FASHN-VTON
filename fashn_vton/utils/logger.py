"""Logging utilities for FASHN VTON with colored console output."""

import json
import logging
from typing import Optional

from .common import exists


class CustomFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[94m",
        "INFO": "\033[0m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "CRITICAL": "\033[1;91m",
    }
    RESET = "\033[0m"

    def __init__(self, timestamp: bool = False, datefmt: str = "%Y-%m-%d %H:%M:%S"):
        fmt = "%(name)s - %(levelname)s - %(message)s"
        if timestamp:
            fmt = "%(asctime)s - " + fmt
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        original_msg = record.msg
        if isinstance(original_msg, dict):
            record.msg = json.dumps(original_msg, indent=4, sort_keys=True)
        else:
            record.msg = original_msg

        formatted_msg = super().format(record)
        levelname = record.levelname
        color_prefix = self.COLORS.get(levelname, self.COLORS["INFO"])
        return color_prefix + formatted_msg + self.RESET


def setup_logger(
    name: str, timestamp: bool = False, level: Optional[int] = None
) -> logging.Logger:
    logger = logging.getLogger(name)

    if exists(level):
        logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = CustomFormatter(timestamp=timestamp)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False

    return logger
