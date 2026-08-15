"""One call: parent sequence in, 96-well plate and its evidence out.

Order matters here, and it is not the obvious one. Candidates are filtered on
specificity *before* being tiled across the switching window, not after. Tiling
first and then filtering leaves holes wherever a band happened to contain only
dimerising designs, and the plate silently stops covering the range it claims to.

Everything the artifacts need comes back in the same structure, so a figure can
never disagree with the CSV: they are drawn from one object.
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate  # noqa: E402
import plate  # noqa: E402
import score  # noqa: E402
import store  # noqa: E402
import thermo  # noqa: E402

WINDOW = (-2.0, 2.0)
N_BINS = 8
SEED = 20260815

# IL-6 and TNF-alpha are both ~20 kDa; used to turn clinical pg/mL into molar.
DEFAULT_MW = 21000.0

# The concentrations that matter clinically, in pg/mL. Cytokine sensing lives or
# dies on whether the sensor reaches these, so they travel with the design.
CLINICAL_PG = [
    ("healthy", 1, 10),
    ("mild inflammation", 10, 100),
    ("sepsis", 100, 2_000),
    ("severe sepsis", 2_000, 10_000),
    ("cytokine storm", 10_000, 100_000),
]


def _tile(rows: list[dict], n: int) -> list[dict]:
    """Spread `n` picks evenly across the switching window, best-ranked first."""
    lo, hi = WINDOW
    width = (hi - lo) / N_BINS
    per_bin = max(n // N_BINS, 1)

    picked, used = [], set()
    for i in range(N_BINS):
        band_lo = lo + i * width
        band = [r for r in rows if band_lo <= r["dd_g"] < band_lo + width]
        for r in band[:per_bin]:
            picked.append(r)
            used.add(r["name"])

    # Top up from the best remaining, wherever they sit, if bands ran thin.
    for r in rows:
        if len(picked) >= n:
            break
        if r["name"] not in used:
            picked.append(r)
            used.add(r["name"])
    return picked[:n]


def run(parent: str, core: tuple[int, int], kd_intrinsic_M: float,
        target: str = "IL-6", mw_da: float = DEFAULT_MW,
        clinical_pg_per_ml: float = 5000.0) -> dict:
    """Design the plate and return everything needed to justify it."""
    clinical_M = thermo.pg_per_ml_to_molar(clinical_pg_per_ml, mw_da)

    lib = generate.library(parent, core)
    in_window = [v for v in lib if WINDOW[0] <= v.dd_g <= WINDOW[1]]
    rows = score.assess_all(in_window, parent, core, kd_intrinsic_M, clinical_M)
    passing = [r for r in rows if r["passes"]]

    ctrls = plate.controls(parent, lib)
    n_test = 96 - len(ctrls)
    selected = _tile(passing, n_test)

    positions = [f"{r}{c}" for r in plate.ROWS for c in plate.COLS]
    rng = random.Random(SEED)
    rng.shuffle(positions)

    wells: list[plate.Well] = []
    for (kind, seq, note), pos in zip(ctrls, positions):
        wells.append(plate.Well(pos, f"CTRL-{kind}-{pos}", seq, "control",
                                kind, None, note))
    for r, pos in zip(selected, positions[len(ctrls):]):
        wells.append(plate.Well(
            pos, r["name"], r["sequence"], "test", r["family"], r["dd_g"],
            f"rank {r['rank']}, tail {r['tail_len']} nt, {r['n_mismatch']} mismatch, "
            f"linker {r['linker']}, register {r['register']}, "
            f"margin {r['specificity_margin']}, Kd_app {r['kd_apparent_nM']} nM"))
    wells.sort(key=lambda w: (plate.ROWS.index(w.position[0]), int(w.position[1:])))

    observed, p95 = _position_check([w for w in wells if w.role == "test"])

    from collections import Counter
    failures = Counter(k for r in rows if not r["passes"] for k in r["flag_kinds"])

    # A criterion that rejects almost everything is describing the parent, not
    # discriminating between designs. Saying which one is far more useful than
    # returning an empty plate, and it is the difference between "this target
    # needs a different architecture" and "the tool broke".
    universal = [k for k, n in failures.items() if rows and n >= 0.9 * len(rows)]

    return {
        "target": target,
        "parent": parent,
        "core": list(core),
        "kd_intrinsic_nM": kd_intrinsic_M * 1e9,
        "library_size": len(lib),
        "in_window": len(in_window),
        "passing": len(passing),
        "selected": len(selected),
        "failure_reasons": dict(failures.most_common()),
        "universal_blockers": universal,
        "diagnosis": _diagnose(universal, len(passing), parent),
        "rows": rows,
        "picked_names": {r["name"] for r in selected},
        "wells": wells,
        "position_check": {"observed": observed, "null_p95": p95},
        "clinical_bands": [(name, thermo.pg_per_ml_to_molar(lo, mw_da),
                            thermo.pg_per_ml_to_molar(hi, mw_da))
                           for name, lo, hi in CLINICAL_PG],
        "kd_apparent_nM": [r["kd_apparent_nM"] for r in selected],
    }


def _diagnose(universal: list[str], n_passing: int, parent: str) -> str:
    """Why the plate is empty, when it is."""
    if n_passing:
        return ""
    if not universal:
        return ("No candidate cleared every criterion, but no single one is "
                "responsible — the constraints are jointly too tight for this parent.")
    reasons = {
        "dimerises": (
            "every candidate is predicted to dimerise. This parent is GC-rich "
            "with self-complementary ends, so the construct pairs with its "
            "neighbour more readily than with itself. That is a property of the "
            "parent, not of any design: switch engineering cannot fix it, and a "
            "different parent or a non-switching sensor architecture is needed."),
        "off-target tail site": (
            "every tail finds a second site in the parent as good as its intended "
            "one, so no design reports a conformational change that can be "
            "attributed to target binding. The parent has too much internal "
            "repetition for this architecture."),
        "quadruplex-prone": (
            "every designed tail carries a G-run long enough to seed a "
            "quadruplex, which binds proteins promiscuously."),
    }
    return " ".join(reasons.get(k, f"all candidates flagged: {k}.") for k in universal)


def _position_check(tests: list[plate.Well]) -> tuple[float, float]:
    """Observed row-mean ddG spread, and the 95th percentile under shuffling."""
    def spread(values: list[float]) -> float:
        rows = [values[i::len(plate.ROWS)] for i in range(len(plate.ROWS))]
        means = [statistics.mean(r) for r in rows if r]
        return max(means) - min(means)

    vals = [w.dd_g for w in sorted(tests, key=lambda w: w.position)]
    if len(vals) < len(plate.ROWS):
        return 0.0, 0.0
    observed = spread(vals)
    rng = random.Random(SEED + 1)
    null = sorted(spread(rng.sample(vals, len(vals))) for _ in range(2000))
    return round(observed, 3), round(null[int(0.95 * len(null))], 3)


def artifacts(result: dict, out_dir: Path) -> dict:
    """Render the four figures and the order file, and archive the run.

    Figures keep their flat `<target>_<name>.png` paths because the UI reads
    those, but every run is also copied into its own timestamped directory. The
    flat files are a view of the newest run; the run directory is the record.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import plots

    out_dir.mkdir(exist_ok=True)
    t = result["target"].replace("/", "_")
    run_dir = store.new_run(out_dir, result["target"])
    store.save_design(run_dir, result)
    result["run_dir"] = str(run_dir)

    stages = [
        ("library", result["library_size"], "every tail length, register, mismatch and linker"),
        ("switching window", result["in_window"], "ddG within +/-2 kcal/mol of balance"),
        ("passes all criteria", result["passing"],
         "  ·  ".join(f"{v:,} lost to {k}" for k, v in result["failure_reasons"].items())),
        ("on the plate", result["selected"], "tiled across the window, 8 wells kept for controls"),
    ]

    pc = result["position_check"]
    out = {
        "funnel": plots.selection_funnel(stages, str(out_dir / f"{t}_funnel.png")),
        "window": plots.design_window(result["rows"], result["picked_names"],
                                      str(out_dir / f"{t}_window.png")),
    }
    # The dose and plate figures need a selection to draw. With none, the funnel
    # and window still carry the whole story of why, so they are rendered anyway.
    if result["selected"]:
        out["dose"] = plots.dose_response(
            result["kd_apparent_nM"], result["clinical_bands"], result["target"],
            str(out_dir / f"{t}_dose.png"), kd_intrinsic_nM=result["kd_intrinsic_nM"])
        out["plate"] = plots.plate_map(
            result["wells"], str(out_dir / f"{t}_plate.png"),
            null_p95=pc["null_p95"], observed=pc["observed"])
        out["csv"] = str(plate.write_order(result["wells"], out_dir / f"{t}_plate.csv"))
    # Copy the run's outputs alongside its manifest so the directory is
    # self-contained: a plate is a lab record and should not depend on files
    # that the next run will overwrite.
    import shutil
    for key, src in list(out.items()):
        if src and Path(src).exists():
            shutil.copy2(src, run_dir / Path(src).name)
    out["run_dir"] = str(run_dir)
    return out


if __name__ == "__main__":
    IL6 = "GGTGGCAGGAGGACTATTTATTTGCTTTTCT"
    res = run(IL6, core=(1, 13), kd_intrinsic_M=8.5e-9, target="IL-6")
    print(f"library {res['library_size']:,} -> window {res['in_window']:,} -> "
          f"pass {res['passing']:,} -> plate {res['selected']}")
    print("failures:", res["failure_reasons"])
    print("position check:", res["position_check"])
    art = artifacts(res, Path(__file__).resolve().parent.parent / "out")
    for k, v in art.items():
        print(f"  {k:7s} {v}")
