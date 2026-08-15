"""Persist a design run so it survives the browser tab.

Three things were being lost. The memo — the agent's actual reasoning — existed
only in the chat and died with the page. The full scored library was discarded
at the moment of selection, so the 8,481 candidates that did *not* make the plate
left no trace, and with them any way to ask later why a given design was
rejected. And every run wrote to the same filenames, so a second pass at the same
target destroyed the first.

Each run now gets its own timestamped directory holding everything needed to
reconstruct or defend it: what parent was used and where it came from, the
thresholds in force, every candidate with its scores and flags, the plate, the
figures, and the memo.

Written incrementally rather than at the end. A structure prediction can run for
many minutes and a wet-lab afternoon is not the place to discover that a
disconnect threw the results away.
"""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path

RUNS = "runs"


def _git_commit() -> str:
    """Which version of the code produced this. A plate is a lab record."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=Path(__file__).resolve().parent.parent,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def new_run(out_dir: Path, target: str) -> Path:
    """A fresh timestamped directory for this run."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in target)
    path = out_dir / RUNS / f"{safe}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _thresholds() -> dict:
    """The judgement calls that shaped this plate, recorded with it.

    None of these is measured. They are choices, and different choices give
    different plates — the dimer limit alone moves the pass count from 182 to
    1,025. A run that does not record them cannot be reproduced or argued with,
    so they travel in the manifest rather than living only in the source.
    """
    import generate
    import score
    return {
        "_note": "chosen values, not measurements; see module docstrings",
        "switching_window_tail": list(design_window("tail")),
        "switching_window_intrinsic": list(generate.INTRINSIC_WINDOW),
        "specificity_margin_min": score.SPECIFICITY_MARGIN,
        "dimer_margin_min": score.DIMER_MARGIN_LIMIT,
        "intermolecular_init": score.INTERMOLECULAR_INIT,
        "max_g_run_in_designed_region": score.MAX_G_RUN,
        "max_homopolymer": score.MAX_HOMOPOLYMER,
        "gc_percent_band": [25.0, 75.0],
        "well_assignment_seed": SEED_FOR_MANIFEST,
    }


def design_window(kind: str) -> tuple[float, float]:
    import design
    return design.WINDOW if kind == "tail" else __import__("generate").INTRINSIC_WINDOW


SEED_FOR_MANIFEST = 20260815


def save_design(run_dir: Path, result: dict) -> dict:
    """Manifest plus the complete scored library, not merely the survivors."""
    picked = result.get("picked_names", set())

    manifest = {
        "target": result["target"],
        "parent_sequence": result["parent"],
        "core_assumed": result["core"],
        "kd_intrinsic_nM": result["kd_intrinsic_nM"],
        "library_size": result["library_size"],
        "in_switching_window": result["in_window"],
        "passing_all_criteria": result["passing"],
        "selected": result["selected"],
        "failure_reasons": result["failure_reasons"],
        "universal_blockers": result.get("universal_blockers", []),
        "diagnosis": result.get("diagnosis", ""),
        "position_check": result["position_check"],
        "architecture": result.get("architecture"),
        "thresholds": _thresholds(),
        "code_commit": _git_commit(),
        "written_at": datetime.now().isoformat(timespec="seconds"),
        # The core is an assumption unless a paper mapped the epitope. Recording
        # that in the run itself keeps a later reader from mistaking it for data.
        "caveat": ("core_assumed is an assumption unless independently mapped; "
                   "every ddG in candidates.csv is computed against it"),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    rows = result.get("rows", [])
    if rows:
        fields = [k for k in rows[0] if k not in ("flags", "flag_kinds")]
        with (run_dir / "candidates.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([*fields, "on_plate", "flags"])
            for r in rows:
                w.writerow([*[r[k] for k in fields],
                            r["name"] in picked, "; ".join(r.get("flags", []))])
    return manifest


def save_memo(run_dir: Path, target: str, memo: str, trace: list[str]) -> Path:
    """The agent's written analysis and the trace that produced it."""
    path = run_dir / "memo.md"
    lines = [f"# {target} — aptamer switch design", "",
             f"_{datetime.now().isoformat(timespec='seconds')}_", "",
             "## Trace", "", "```"]
    lines += trace or ["(none recorded)"]
    lines += ["```", "", "## Memo", "", memo or "(no memo produced)"]
    path.write_text("\n".join(lines))
    return path


def latest(out_dir: Path, target: str) -> Path | None:
    """Most recent run directory for a target, if any."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in target)
    runs = sorted((out_dir / RUNS).glob(f"{safe}_*")) if (out_dir / RUNS).exists() else []
    return runs[-1] if runs else None


# ---------------------------------------------------------------- GPU cache

CACHE = "cache"


def cache_key(**parts) -> str:
    """Content hash of everything that would change the answer.

    Keyed on inputs, never on the marker name. The same biomarker with a
    different parent aptamer, a different seed or a different checkpoint is a
    different computation, and a name-keyed cache would hand back the wrong
    structure with no way to notice. Conversely the same sequences asked for
    twice under two names should hit once.
    """
    import hashlib
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def cache_get(out_dir: Path, key: str) -> dict | None:
    """A previous result for these exact inputs, or None.

    GPU dispatch is billed per call and a structure prediction is minutes of
    wall clock. Re-running one because a caller asked the same question twice is
    money and time spent to learn nothing.
    """
    path = out_dir / CACHE / f"{key}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None          # a truncated write should re-run, not crash
    payload["_cached"] = True
    payload["_cached_at"] = payload.get("_written_at")
    return payload


def cache_put(out_dir: Path, key: str, payload: dict) -> Path:
    """Record a result. Written atomically so an interrupted run leaves no
    half-file that would later be trusted as a hit."""
    cache_dir = out_dir / CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    tmp = path.with_suffix(".json.tmp")
    body = dict(payload)
    body["_written_at"] = datetime.now().isoformat(timespec="seconds")
    tmp.write_text(json.dumps(body, indent=2, default=str))
    tmp.replace(path)
    return path
