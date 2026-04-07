from __future__ import annotations

from typing import Any, Dict, Optional
from datetime import datetime, timezone
from uuid import uuid4

import os

from app.utils.config import LOG_LEVEL, LOG_STRUCTURED
from app.utils.logging import debug, info, warn

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
        self.parent_run_id: Optional[str] = None
        if LOG_LEVEL == "DEBUG":
            debug("LangSmith logger init", file=__file__, project=self.project, tracing=tracing)

    def start_trace(self, name: str, inputs: Dict[str, Any]) -> None:
        if not self.client:
            return
        try:
            run_id = str(uuid4())
            if LOG_LEVEL == "DEBUG":
                debug("LangSmith parent run id generated", run_id=run_id)
            self.client.create_run(
                id=run_id,
                name=name,
                inputs=inputs,
                project_name=self.project,
                run_type="chain",
                start_time=datetime.now(timezone.utc),
            )
            self.parent_run_id = run_id
            if LOG_LEVEL == "DEBUG":
                debug("LangSmith parent run created", run_id=self.parent_run_id, name=name)
        except Exception as exc:
            if LOG_LEVEL == "DEBUG":
                info(f"LangSmith start_trace failed: {exc}")

    def end_trace(self, outputs: Optional[Dict[str, Any]] = None) -> None:
        if not self.client or not self.parent_run_id:
            return
        try:
            self.client.update_run(
                self.parent_run_id,
                outputs=outputs or {},
                end_time=datetime.now(timezone.utc),
            )
            if LOG_LEVEL == "DEBUG":
                debug("LangSmith parent run closed", run_id=self.parent_run_id)
        except Exception as exc:
            if LOG_LEVEL == "DEBUG":
                info(f"LangSmith end_trace failed: {exc}")

    def log_event(self, name: str, payload: Dict[str, Any]) -> None:
        if LOG_LEVEL == "DEBUG":
            debug(f"LangSmith event: {name}", payload=payload)
        if not self.client:
            return
        try:
            run_id = str(uuid4())
            self.client.create_run(
                id=run_id,
                name=name,
                inputs=payload,
                project_name=self.project,
                run_type="tool",
                parent_run_id=self.parent_run_id,
                start_time=datetime.now(timezone.utc),
            )
            if LOG_LEVEL == "DEBUG":
                debug("LangSmith child run created", run_id=run_id, name=name)
            if run_id:
                self.client.update_run(
                    run_id,
                    end_time=datetime.now(timezone.utc),
                )
                if LOG_LEVEL == "DEBUG":
                    debug("LangSmith child run closed", run_id=run_id, name=name)
        except Exception as exc:
            if LOG_LEVEL == "DEBUG":
                info(f"LangSmith logging failed: {exc}")


def _env_truthy(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def configure_langsmith_tracing(use_api: bool) -> None:
    if not use_api:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGSMITH_API_KEY"] = ""
        info("LangSmith API disabled; writing local traces only.")
        return
    if not _env_truthy("LANGSMITH_TRACING"):
        warn("LANGSMITH_TRACING is not enabled. LangSmith runs will not be captured.")


def _extract_run_id(run: Any) -> Optional[str]:
    # Deprecated: create_run returns None in this langsmith version.
    if not run:
        return None
    run_id = getattr(run, "id", None)
    if run_id:
        return str(run_id)
    if isinstance(run, dict):
        for key in ("id", "run_id"):
            if key in run:
                return str(run[key])
    return None
