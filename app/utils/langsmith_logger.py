from __future__ import annotations

from typing import Any, Dict, Optional

import os

from app.utils.config import DEBUG, LOG_STRUCTURED
from app.utils.logging import debug, info

try:
    from langsmith import Client
except Exception:  # pragma: no cover - optional dependency
    Client = None


class LangSmithLogger:
    def __init__(self, project: Optional[str] = None) -> None:
        tracing = _env_truthy("LANGSMITH_TRACING")
        api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
        if api_key == "your_langsmith_api_key_here":
            api_key = ""
        if Client and tracing and api_key:
            self.client = Client()
        else:
            self.client = None
        self.project = project

    def log_event(self, name: str, payload: Dict[str, Any]) -> None:
        if DEBUG:
            debug(f"LangSmith event: {name}", payload=payload)
        if not self.client:
            return
        try:
            self.client.create_run(
                name=name,
                inputs=payload,
                project_name=self.project,
                run_type="tool",
            )
        except Exception as exc:
            if DEBUG:
                info(f"LangSmith logging failed: {exc}")


def _env_truthy(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}
