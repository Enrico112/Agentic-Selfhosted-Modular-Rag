from __future__ import annotations

from typing import Any, Dict, Optional

from app.utils.config import DEBUG, LOG_STRUCTURED
from app.utils.logging import debug, info

try:
    from langsmith import Client
except Exception:  # pragma: no cover - optional dependency
    Client = None


class LangSmithLogger:
    def __init__(self, project: Optional[str] = None) -> None:
        self.client = Client() if Client else None
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
