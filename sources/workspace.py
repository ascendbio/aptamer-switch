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


# The agent names its own target argument, and it varies it freely within one
# run: a single IL-6 query produced plates, figures and demo files under "IL-6",
# "interleukin-6" and "IL-6 core16-30". Those are one biomarker, and treating
# them as three identities split the outputs three ways and, because the target
# is part of the design cache key, made every repeat recompute from scratch.
#
# So the session pins the name the user typed, and everything downstream uses it.
_target: contextvars.ContextVar[str] = contextvars.ContextVar("target", default="")

_ANNOTATION = re.compile(
    r"\s*(?:\(|\[).*$"                      # "IL-6 (19mer, Kd unknown)"
    r"|\s+core\s*\d+\s*[-–]\s*\d+\s*$"   # "IL-6 core16-30"
    r"|\s+\d+\s*mer\s*$"                    # "IL-6 31mer"
    r"|\s*[-–—]\s*(?:parent|aptamer|adaptor).*$",
    re.IGNORECASE)


def canonical(raw: str) -> str:
    """Strip the design annotations the agent appends to a biomarker name."""
    name = _ANNOTATION.sub("", (raw or "").strip()).strip(" -–—,;:/")
    return name or (raw or "").strip()


def use_target(name: str) -> str:
    """Pin this session's target to what the user actually asked for."""
    _target.set(canonical(name))
    return _target.get()


def target_name(fallback: str = "") -> str:
    """The session's target, or the cleaned-up version of what was passed."""
    return _target.get() or canonical(fallback)
