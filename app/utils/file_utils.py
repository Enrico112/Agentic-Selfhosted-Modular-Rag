from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple


def file_signature(path: Path) -> Dict[str, object]:
    stat = path.stat()
    return {"mtime": stat.st_mtime, "size": stat.st_size}


def load_state(state_path: Path) -> Dict[str, object]:
    if not state_path.exists():
        return {"files": {}}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {"files": {}}


def save_state(state_path: Path, state: Dict[str, object]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def detect_changes(data_dir: Path, state_path: Path) -> Tuple[bool, Dict[str, object]]:
    state = load_state(state_path)
    current_files: Dict[str, object] = {}
    for path in sorted(data_dir.glob("*.md")):
        current_files[str(path)] = file_signature(path)
    changed = current_files != state.get("files", {})
    return changed, {"files": current_files}
