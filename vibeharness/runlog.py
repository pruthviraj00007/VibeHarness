"""Per-run logging into a hidden ``.vibe/`` folder in the workspace.

Each run writes two files, timestamped:
  - ``<stamp>.json``  full structured log INCLUDING the reasoning trace of every
                      turn (for later analysis / prompt-and-model improvement)
  - ``<stamp>.md``    a human-readable transcript

Logs live alongside the work (the current workspace) so each project keeps its
own history.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from .config import Config
from .agent import RunResult


def _hide(path: Path) -> None:
    """Best-effort: also set the OS 'hidden' attribute on Windows."""
    if os.name == "nt":
        try:
            import ctypes
            FILE_ATTRIBUTE_HIDDEN = 0x02
            ctypes.windll.kernel32.SetFileAttributesW(str(path), FILE_ATTRIBUTE_HIDDEN)
        except Exception:
            pass


class RunLogger:
    def __init__(self, workspace: Path | str):
        self.dir = Path(workspace) / ".vibe"

    def write(self, task: str, config: Config, result: RunResult, started: datetime) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        _hide(self.dir)
        stamp = started.strftime("%Y%m%d_%H%M%S")

        payload = {
            "task": task,
            "started_at": started.isoformat(timespec="seconds"),
            "model": config.model,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_steps": config.max_steps,
            "finished": result.finished,
            "final_summary": result.final_summary,
            "turns": result.to_dict()["turns"],   # includes per-turn reasoning traces
        }
        json_path = self.dir / f"{stamp}.json"
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.dir / f"{stamp}.md").write_text(result.transcript(), encoding="utf-8")
        return json_path
