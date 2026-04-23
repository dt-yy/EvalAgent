from __future__ import annotations

import logging
import os
from typing import Any

_LOGGING_CONFIGURED = False


def configure_logging(default_level: str = "INFO") -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    level_name = os.getenv("LOG_LEVEL", default_level).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def kv_to_text(**kwargs: Any) -> str:
    parts = [f"{key}={value}" for key, value in kwargs.items()]
    return " ".join(parts)

