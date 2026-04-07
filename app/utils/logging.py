from __future__ import annotations

import json
from typing import Any, Dict

from app.utils.config import LOG_LEVEL, LOG_STRUCTURED


_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30}


def _should_log(level: str) -> bool:
    return _LEVELS.get(level, 20) >= _LEVELS.get(LOG_LEVEL, 20)


def log(level: str, message: str, **fields: Any) -> None:
    if not _should_log(level):
        return
    if LOG_STRUCTURED:
        payload: Dict[str, Any] = {"level": level, "message": message, **fields}
        print(json.dumps(payload, ensure_ascii=False))
        return
    print(f"[{level}] {message}")
    if fields:
        for key, value in fields.items():
            print(f"  {key}={value}")


def debug(message: str, **fields: Any) -> None:
    log("DEBUG", message, **fields)


def info(message: str, **fields: Any) -> None:
    log("INFO", message, **fields)


def warn(message: str, **fields: Any) -> None:
    log("WARN", message, **fields)
