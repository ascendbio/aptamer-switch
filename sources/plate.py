"""Choose 96 and lay them out on a plate the lab can actually run.

Two decisions here do most of the work, and both are about experimental design
rather than chemistry.

*Not the top 96.* Ranking by predicted ddG and taking the best 96 gives ninety-six
near-identical sequences and one bit of information. If the model's centre is off
by a kcal/mol — and near zero it can be — every well fails together and the round
teaches nothing. The plate instead tiles the design window, so the experiment
returns the shape of the response and locates the optimum even when the
prediction is wrong.

*Position is randomised against ddG.* Edge wells evaporate faster, and plates
have row and column gradients. Laying the ddG ladder out in plate order aliases
those gradients directly onto the design variable, and the result looks like a
beautiful dose-response that is really just the edge drying out. Assignment is
shuffled under a fixed seed: reproducible, and decorrelated from the thing being
measured.
"""

from __future__ import annotations

import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate  # noqa: E402

ROWS = "ABCDEFGH"
COLS = range(1, 13)
SEED = 20260815          # fixed so the plate is reproducible from the code alone

# The window worth screening, in kcal/mol of tail-versus-fold competition.
WINDOW = (-2.0, 2.0)
N_BINS = 8


@dataclass
class Well:
    position: str
    name: str
    sequence: str
    role: str            # test | control
    kind: str
    dd_g: float | None
    note: str


def _feature(v: generate.Variant) -> tuple:
    return (v.register[0], len(v.tail), v.n_mismatch, v.linker)


def _spread(candidates: list[generate.Variant], k: int) -> list[generate.Variant]:
    """Greedy farthest-point selection on the design levers.

    Within a ddG bin the variants are thermodynamically interchangeable, so the
    thing to maximise is mechanistic variety — different registers, tail lengths
    and mismatch counts reaching the same energy. If the energy model is wrong,
    a bin full of one mechanism fails as a unit; a mixed bin still reports.
    """
    if len(candidates) <= k:
        return candidates
    chosen = [candidates[len(candidates) // 2]]
    while len(chosen) < k:
        best, best_d = None, -1.0
        for c in candidates:
            if c in chosen:
                continue
            d = min(sum(a != b for a, b in zip(_feature(c), _feature(s)))
                    for s in chosen)
            if d > best_d:
                best, best_d = c, d
        if best is None:
            break
        chosen.append(best)
    return chosen


def select(lib: list[generate.Variant], n_test: int) -> list[generate.Variant]:
    """Tile the design window, evenly across ddG and variously within each band."""
    lo, hi = WINDOW
    width = (hi - lo) / N_BINS
    per_bin = n_test // N_BINS

    picked: list[generate.Variant] = []
    for i in range(N_BINS):
        band_lo = lo + i * width
        band = [v for v in lib if band_lo <= v.dd_g < band_lo + width]
        picked.extend(_spread(band, per_bin))

    # Top up from the densest part of the window if some bins were thin.
    if len(picked) < n_test:
        rest = [v for v in lib if lo <= v.dd_g <= hi and v not in picked]
        picked.extend(_spread(rest, n_test - len(picked)))
    return picked[:n_test]


def controls(parent: str, lib: list[generate.Variant]) -> list[tuple[str, str, str]]:
    """(kind, sequence, why it is on the plate).

    Every one of these answers a question a failed plate would otherwise leave
    open: was the aptamer ever any good, does the assay respond at all, and is a
    flat well flat for the reason we think.
    """
    rng = random.Random(SEED)
    scram = list(parent)
    rng.shuffle(scram)

    dead = min(lib, key=lambda v: v.dd_g)        # tail wins outright
    stuck = max(lib, key=lambda v: v.dd_g)       # tail never engages

    return [
        ("parent", parent,
         "unmodified aptamer, no tail — binding without a switch, the affinity reference"),
        ("parent", parent,
         "replicate of the parent, on the far side of the plate — position effects"),
        ("scrambled", "".join(scram),
         "same base composition, no structure — non-specific binding and fouling"),
        ("no-switch-low", dead.sequence,
         f"ddG {dead.dd_g}, tail wins outright — aptamer cannot fold, expect no response"),
        ("no-switch-low", dead.sequence, "replicate"),
        ("no-switch-high", stuck.sequence,
         f"ddG {stuck.dd_g}, tail never engages — no conformational change, expect flat"),
        ("no-switch-high", stuck.sequence, "replicate"),
        ("blank", "",
         "no oligo — electrode background and the baseline everything else is read against"),
    ]


def build(parent: str, core: tuple[int, int]) -> list[Well]:
    lib = generate.library(parent, core)
    ctrls = controls(parent, lib)
    positions = [f"{r}{c}" for r in ROWS for c in COLS]
    # Fill the plate exactly: however many controls the design calls for, the
    # rest of the 96 are test wells. Hardcoding the split silently ships a
    # part-empty plate that still costs a full synthesis run.
    tests = select(lib, n_test=len(positions) - len(ctrls))

    rng = random.Random(SEED)
    rng.shuffle(positions)

    wells: list[Well] = []
    for (kind, seq, note), pos in zip(ctrls, positions):
        wells.append(Well(pos, f"CTRL-{kind}-{pos}", seq, "control", kind, None, note))
    for v, pos in zip(tests, positions[len(ctrls):]):
        wells.append(Well(pos, v.name, v.sequence, "test", v.family, v.dd_g,
                          f"tail {len(v.tail)} nt, {v.n_mismatch} mismatch, "
                          f"linker {v.linker}, register {v.register[0]}"))

    return sorted(wells, key=lambda w: (ROWS.index(w.position[0]), int(w.position[1:])))


def write_order(wells: list[Well], path: Path) -> Path:
    """Vendor-format plate file.

    5' thiol anchors to the gold electrode, 3' methylene blue is the reporter.
    Dual modification is what makes these expensive, so the file is written to be
    checked by a human before it is ever submitted.
    """
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Well Position", "Name", "Sequence", "5' Mod", "3' Mod",
                    "Scale", "Purification", "Role", "ddG (kcal/mol)", "Note"])
        for well in wells:
            if not well.sequence:
                w.writerow([well.position, well.name, "", "", "", "", "",
                            well.role, "", well.note])
                continue
            w.writerow([well.position, well.name, well.sequence,
                        "/5ThioMC6-D/", "/3MeBlN/", "100 nm", "HPLC",
                        well.role, "" if well.dd_g is None else well.dd_g,
                        well.note])
    return path


if __name__ == "__main__":
    IL6 = "GGTGGCAGGAGGACTATTTATTTGCTTTTCT"
    wells = build(IL6, core=(1, 13))

    out = Path(__file__).resolve().parent.parent / "out"
    out.mkdir(exist_ok=True)
    path = write_order(wells, out / "IL6_switch_plate.csv")

    tests = [w for w in wells if w.role == "test"]
    ctrls = [w for w in wells if w.role == "control"]
    print(f"{len(wells)} wells: {len(tests)} test, {len(ctrls)} control")
    print(f"ddG range screened: {min(w.dd_g for w in tests):+.2f} to "
          f"{max(w.dd_g for w in tests):+.2f} kcal/mol")
    print(f"lengths: {min(len(w.sequence) for w in tests)}-"
          f"{max(len(w.sequence) for w in tests)} nt")

    # Is plate position actually decorrelated from ddG? The raw spread means
    # nothing on its own — with ~11 wells per row, chance alone moves row means
    # around. Compare it to the spread from random assignment.
    import statistics

    def row_spread(values: list[float]) -> float:
        rows = [values[i::len(ROWS)] for i in range(len(ROWS))]
        means = [statistics.mean(r) for r in rows if r]
        return max(means) - min(means)

    observed = row_spread([w.dd_g for w in sorted(tests, key=lambda w: w.position)])
    rng = random.Random(SEED + 1)
    vals = [w.dd_g for w in tests]
    null = sorted(row_spread(rng.sample(vals, len(vals))) for _ in range(2000))
    p95 = null[int(0.95 * len(null))]
    print(f"row-mean ddG spread: {observed:.2f} kcal/mol "
          f"(random assignment gives {null[len(null) // 2]:.2f} typical, "
          f"{p95:.2f} at the 95th percentile)")
    print("  -> " + ("position is decorrelated from ddG, as intended"
                     if observed <= p95 else
                     "WARNING: ddG tracks plate position; reshuffle before ordering"))
    print(f"\nwritten: {path}")
