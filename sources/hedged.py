"""One orderable plate that hedges the assumption instead of betting on it.

The sensitivity sweep can show that a plate is an artefact of the assumed binding
core. That is true and worth knowing, but "do not order" is not an answer a wet
lab can act on: they have a budget, a synthesis slot, and a question. Refusing to
choose is not rigour, it is passing the problem back.

So spend the wells on the uncertainty. Allocate the plate across the core
hypotheses that actually produce designs, label every well with the hypothesis it
belongs to, and the single experiment now answers two questions at once: which
switches work, and which core was right. A hypothesis that yields no responsive
well is eliminated, and the surviving wells are interpretable because they were
run on the same plate, the same day, in the same buffer.

Two details make it a real experiment rather than two half-plates in one tray.

Wells are randomised against the core hypothesis as well as against ddG. Split
the plate by hypothesis into rows A-D and E-H and any row gradient - evaporation,
a warm corner, uneven incubation - is perfectly confounded with the thing being
compared, and the result would be indistinguishable from an edge effect.

Controls are shared rather than duplicated per hypothesis. They report on the
assay, not on the core, so paying for them twice buys nothing and costs wells
that could be testing designs.
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import design  # noqa: E402
import generate  # noqa: E402
import plate  # noqa: E402
import score  # noqa: E402
import sensitivity  # noqa: E402
import store  # noqa: E402

SEED = 20260815
WELLS = 96


def run(parent: str, target: str = "target", kd_intrinsic_M: float | None = None,
        kd_source: str = "") -> dict:
    """Design under every plausible core, then spend one plate across them."""
    parent = parent.upper().replace("U", "T")
    already_swept = (parent, target, kd_intrinsic_M) in sensitivity._sweep_cache
    sweep = sensitivity.run(parent, target=target, kd_intrinsic_M=kd_intrinsic_M,
                            kd_source=kd_source, keep_results=True)
    results = sweep.pop("results", {})

    productive = [c for c in sweep["per_core"] if c.get("wells")]
    if not productive:
        return {"target": target, "parent": parent, "wells": [], "sweep": sweep,
                "error": "no core assumption yields any design; the parent is "
                         "the obstacle, and a different parent is the only "
                         "remedy — hedging cannot help here"}

    per_core_rows: list[tuple[tuple[int, int], list[dict]]] = []
    for entry in productive:
        core = tuple(entry["core"])
        r = results.get(core)
        if r is None:
            continue
        rows = [x for x in r["rows"] if x["name"] in r["picked_names"]]
        rows.sort(key=lambda x: -x["rank_score"])
        per_core_rows.append((core, rows))

    # Controls need a library to pick its extremes from - the design that can
    # never fold and the one that never switches. Take the first productive
    # core's, since controls report on the assay rather than on any hypothesis.
    first_core = per_core_rows[0][0]
    lib = (generate.intrinsic_library(parent, first_core)
           if results[first_core]["architecture"] == "intrinsic"
           else generate.library(parent, first_core))
    ctrls = plate.controls(parent, lib)
    budget = WELLS - len(ctrls)
    share = budget // len(per_core_rows)

    chosen: list[tuple[tuple[int, int], dict]] = []
    for core, rows in per_core_rows:
        chosen.extend((core, r) for r in rows[:share])
    # Any remainder goes to the hypothesis with the deepest passing list, which
    # is the one with most still worth testing.
    if len(chosen) < budget:
        core, rows = max(per_core_rows, key=lambda cr: len(cr[1]))
        already = {r["name"] for c, r in chosen if c == core}
        for r in rows:
            if len(chosen) >= budget:
                break
            if r["name"] not in already:
                chosen.append((core, r))

    positions = [f"{r}{c}" for r in plate.ROWS for c in plate.COLS]
    rng = random.Random(SEED)
    rng.shuffle(positions)

    wells: list[plate.Well] = []
    for (kind, seq, note), pos in zip(ctrls, positions):
        wells.append(plate.Well(pos, f"CTRL-{kind}-{pos}", seq, "control", kind,
                                None, note + " · shared across hypotheses"))
    for (core, r), pos in zip(chosen, positions[len(ctrls):]):
        wells.append(plate.Well(
            pos, f"{r['name']}-c{core[0]}", r["sequence"], "test",
            f"core {core[0]}-{core[1]}", r["dd_g"],
            f"core hypothesis {core[0]}-{core[1]} · rank {r['rank']}, "
            f"ddG {r['dd_g']}, margin {r['specificity_margin']}"))
    wells.sort(key=lambda w: (plate.ROWS.index(w.position[0]), int(w.position[1:])))

    return {
        "target": target,
        "parent": parent,
        "hypotheses": [list(c) for c, _ in per_core_rows],
        "wells_per_hypothesis": share,
        "control_wells": len(ctrls),
        "wells": wells,
        "sweep": sweep,
        "position_check": _confounding(wells),
        "reading": _reading([c for c, _ in per_core_rows], share),
        # What this call actually had to do. Allocating 96 wells across
        # hypotheses is arithmetic once the designs exist; the minutes were
        # spent in the sweep that produced them.
        "reused_sweep": already_swept,
        "designs_available": sum(len(r) for _, r in per_core_rows),
        "work_note": (
            f"reused {sum(len(r) for _, r in per_core_rows)} designs already "
            f"computed by the core-sensitivity sweep; this step only allocated "
            f"and laid them out" if already_swept else
            f"designed under {len(sweep['cores_tested'])} cores from scratch"),
    }


def _confounding(wells: list[plate.Well]) -> dict:
    """Is either the design variable or the hypothesis aliased to plate row?"""
    tests = [w for w in wells if w.role == "test"]
    rows = {r: [w for w in tests if w.position[0] == r] for r in plate.ROWS}

    dd_means = [statistics.mean([w.dd_g for w in v]) for v in rows.values() if v]
    # Share of each row belonging to the first hypothesis; near 0.5 everywhere
    # means the hypotheses are mixed through the plate rather than blocked by row.
    first = sorted({w.kind for w in tests})[0]
    shares = [sum(1 for w in v if w.kind == first) / len(v)
              for v in rows.values() if v]

    # Compared against shuffling, not eyeballed. With eleven wells to a row,
    # chance alone moves the share around; "0.27 to 0.67" means nothing until
    # you know what random assignment gives.
    observed = max(shares) - min(shares) if shares else 0.0
    labels = [w.kind for w in tests]
    rng = random.Random(SEED + 7)
    null = []
    for _ in range(2000):
        shuffled = rng.sample(labels, len(labels))
        by_row = [shuffled[i::len(plate.ROWS)] for i in range(len(plate.ROWS))]
        sh = [sum(1 for x in r if x == first) / len(r) for r in by_row if r]
        null.append(max(sh) - min(sh))
    null.sort()
    p95 = null[int(0.95 * len(null))]

    return {
        "ddg_row_spread": round(max(dd_means) - min(dd_means), 3) if dd_means else 0,
        "hypothesis_share_per_row": [round(s, 2) for s in shares],
        "hypothesis_row_spread": round(observed, 3),
        "hypothesis_row_spread_null_p95": round(p95, 3),
        "confounded": bool(observed > p95),
        "note": "hypothesis_row_spread should sit at or below the null 95th "
                "percentile; above it, the hypotheses are blocked by row and "
                "the comparison is confounded with plate position",
    }


def _reading(cores: list[tuple[int, int]], share: int) -> str:
    spans = ", ".join(f"{a}-{b}" for a, b in cores)
    return (
        f"One plate, {len(cores)} core hypotheses ({spans}), {share} wells each, "
        f"controls shared. Read it twice: which wells switch, and which "
        f"hypothesis they belong to. A hypothesis with no responsive well is "
        f"eliminated — that is the epitope answer, obtained from the same "
        f"synthesis run rather than a second one. If both respond, the core "
        f"matters less than assumed and either design route is available."
    )


if __name__ == "__main__":
    seq = sys.argv[1] if len(sys.argv) > 1 else "GGTGGCAGGAGGACTATTTATTTGCTTTTCT"
    out = run(seq, target="IL-6")
    if out.get("error"):
        print(out["error"])
    else:
        print(f"{len(out['wells'])} wells · hypotheses {out['hypotheses']} · "
              f"{out['wells_per_hypothesis']} each + {out['control_wells']} controls")
        print("confounding:", out["position_check"])
        print()
        print(out["reading"])
