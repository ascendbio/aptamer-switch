"""Where the current request writes its output.

Everything used to write into one `out/` directory, and the interface picks up
figures by modification time. With one user that is fine. With two it is a
correctness bug rather than a queueing one: concurrent runs overwrite each
other's plates, and whoever polls first sees whichever figure landed last —
someone else's design, presented as their own, with nothing to indicate it.

The active directory is a context variable, so it follows an async request
through `asyncio.to_thread` and into the tools without being threaded through
every signature by hand. Unset, it is the shared `out/`, which keeps the CLI and
every script behaving exactly as before.
"""

from __future__ import annotations

import contextvars
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "out"
_current: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "workspace", default=None)


def current() -> Path:
    """The directory this request should write to."""
    path = _current.get() or ROOT
    path.mkdir(parents=True, exist_ok=True)
    return path


def use(session_id: str) -> Path:
    """Point the current context at a session's own directory."""
    safe = re.sub(r"[^A-Za-z0-9_-]", "", str(session_id))[:32] or "shared"
    path = ROOT / "sessions" / safe
    path.mkdir(parents=True, exist_ok=True)
    _current.set(path)
    return path


def reset() -> None:
    _current.set(None)
