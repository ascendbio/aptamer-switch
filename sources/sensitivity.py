"""Which designs survive not knowing where the binding core is.

Every ddG in the plate is computed against an assumed core — the residues the
target is taken to contact — and for these cytokines no paper maps it. That
assumption sits underneath the entire design, and reporting it in a manifest
footnote is not the same as handling it.

So stop assuming one. Run the whole design under several plausible cores and
keep the candidates that reach the plate under *all* of them. A design selected
regardless of where the core turns out to be does not depend on the guess; one
selected under a single core is an artefact of it, and a wet lab spending money
on ninety-six wells should be told which is which.

This costs a few minutes rather than a few seconds, because the design runs once
per core. That is the correct price for the answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import design  # noqa: E402

# Four is enough to separate robust designs from artefacts and keeps the sweep
# near two minutes; the windows overlap, so a fifth adds little.
N_CORES = 4

# A binding core much shorter than this is not an epitope; much longer and there
# is no sequence left outside it to engineer with.
MIN_CORE = 8


def candidate_cores(length: int, n: int = N_CORES) -> list[tuple[int, int]]:
    """Plausible core spans for a sequence whose epitope is unmapped.

    Half-length windows slid across the molecule. Half because a contact region
    covering the whole aptamer would leave nothing to destabilise, and a very
    short one would make almost any design look switchable.
    """
    span = max(MIN_CORE, length // 2)
    if span >= length:
        return [(1, length)]
    last_start = length - span + 1
    if n == 1:
        return [(1, span)]
    step = max(1, (last_start - 1) // (n - 1))
    starts = sorted({min(1 + i * step, last_start) for i in range(n)})
    return [(s, s + span - 1) for s in starts]


def run(parent: str, target: str = "target", kd_intrinsic_M: float | None = None,
        kd_source: str = "", keep_results: bool = False) -> dict:
    """Design under each plausible core; report what survives all of them.

    `keep_results` returns each core's full design result under "results". The
    hedged plate needs the scores behind every selection, and re-running the
    productive cores to recover them doubled a three-minute sweep.
    """
    parent = parent.upper().replace("U", "T")
    cores = candidate_cores(len(parent))

    per_core, selected_sets, results = [], [], {}
    for core in cores:
        try:
            r = design.run(parent, core, kd_intrinsic_M, target=target,
                           kd_source=kd_source)
        except Exception as exc:                      # a core can be unusable
            per_core.append({"core": list(core), "error": str(exc)[:120],
                             "wells": 0})
            continue

        picked = {w.sequence for w in r["wells"] if w.role == "test"}
        selected_sets.append(picked)
        if keep_results:
            results[tuple(core)] = r
        per_core.append({
            "core": list(core),
            "architecture": r["architecture"],
            "library": r["library_size"],
            "in_window": r["in_window"],
            "passing": r["passing"],
            "wells": r["selected"],
            "blocked_by": r.get("universal_blockers") or [],
        })

    productive = [s for s in selected_sets if s]
    union = set.union(*productive) if productive else set()
    # An intersection over one set is that set. Reporting 88 "robust" designs
    # when only one core produced a plate would claim the assumption had been
    # tested when it had not been, which is the opposite of the point.
    robust = set.intersection(*productive) if len(productive) >= 2 else set()

    return {
        "target": target,
        "parent": parent,
        "cores_tested": [list(c) for c in cores],
        "per_core": per_core,
        "cores_producing_a_plate": len(productive),
        "designs_selected_under_any_core": len(union),
        "designs_selected_under_every_core": (len(robust) if len(productive) >= 2
                                             else None),
        "robust_sequences": sorted(robust),
        # The share of a single plate that would survive learning the true core.
        # Undefined unless at least two cores were productive.
        "robust_fraction": (round(len(robust) / max(len(s) for s in productive), 3)
                            if len(productive) >= 2 else None),
        "verdict": _verdict(len(productive), len(cores), len(robust)),
        **({"results": results} if keep_results else {}),
    }


def _verdict(productive: int, tested: int, robust: int) -> str:
    """The reading, with the sharpest true statement first.

    Disjointness is the finding that matters and it was being hidden behind the
    productive-core count: a run where two cores each yield a plate and the two
    plates share nothing is a run whose plate is an artefact, and saying only
    "2 of 4 cores worked" lets a reader order it anyway.
    """
    if not productive:
        return ("no core assumption yields a plate — the parent, not the core, "
                "is the obstacle")

    coverage = ("" if productive == tested else
                f" Only {productive} of {tested} core assumptions yield a plate "
                f"at all, so designability itself depends on the epitope.")

    if productive == 1:
        return ("only one core assumption yields a plate, so nothing here has "
                "been tested against the assumption. Treat the plate as "
                "conditional on that core being right." + coverage)
    if robust == 0:
        return ("the plates from different core assumptions share NO designs. "
                "Selection is an artefact of the assumed core, not a property of "
                "the sequences: map the epitope before ordering, or order from "
                "more than one core." + coverage)
    return (f"{robust} designs are selected under every core tested, so they do "
            f"not depend on the assumption. Order those first." + coverage)


if __name__ == "__main__":
    import json
    seq = (sys.argv[1] if len(sys.argv) > 1
           else "GGTGGCAGGAGGACTATTTATTTGCTTTTCT")
    out = run(seq, target="IL-6")
    out.pop("robust_sequences", None)          # too long to print
    print(json.dumps(out, indent=2))
