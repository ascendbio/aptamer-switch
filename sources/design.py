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

# Mean residue masses, minus one water per peptide bond, so molecular weight is
# computed from the actual sequence instead of assumed. A single default was
# wrong the moment a second target appeared: 21 kDa fits IL-6 but TNF-alpha is a
# 17 kDa monomer that circulates as a ~52 kDa trimer, and pg/mL to molar is a
# division by exactly this number.
_RESIDUE_DA = {
    "A": 71.08, "R": 156.19, "N": 114.10, "D": 115.09, "C": 103.14,
    "E": 129.12, "Q": 128.13, "G": 57.05, "H": 137.14, "I": 113.16,
    "L": 113.16, "K": 128.17, "M": 131.19, "F": 147.18, "P": 97.12,
    "S": 87.08, "T": 101.10, "W": 186.21, "Y": 163.18, "V": 99.13,
}
WATER_DA = 18.02


def molecular_weight(sequence: str) -> float:
    """Monomer mass in daltons from the sequence itself."""
    return sum(_RESIDUE_DA.get(a, 110.0) for a in sequence.upper()) + WATER_DA


# UNCITED, and specific to IL-6. These bands are the author's recollection of
# circulating IL-6 in health, inflammation and sepsis; no source is attached and
# they have not been checked against one. They are used only to shade the
# background of the dose-response figure, never to compute a result, and that
# figure is now drawn only when a measured Kd exists. Do not quote them, and do
# not apply them to another cytokine: TNF-alpha and IL-10 circulate at different
# levels and would need their own, sourced.
CLINICAL_PG_IL6_UNCITED = [
    ("healthy", 1, 10),
    ("mild inflammation", 10, 100),
    ("sepsis", 100, 2_000),
    ("severe sepsis", 2_000, 10_000),
    ("cytokine storm", 10_000, 100_000),
]
CLINICAL_PG = CLINICAL_PG_IL6_UNCITED


def _levers(row: dict) -> tuple:
    """The design choices behind a candidate, for measuring how alike two are."""
    return (row.get("register"), row.get("tail_len"), row.get("n_mismatch"),
            row.get("linker"))


def _diverse(band: list[dict], k: int) -> list[dict]:
    """`k` candidates from one energy band, as mechanistically unalike as possible.

    Within a band every candidate has effectively the same predicted switching
    energy, so ranking them adds nothing: the specificity margin barely varies
    among designs that passed the filter, which makes rank almost exactly
    -|ddG| and picks near-duplicates. Taking the top k that way filled each band
    with one tail length at one register.

    That is the failure the tiling exists to avoid, one level down. If the energy
    model is wrong about a mechanism, a band built from a single mechanism fails
    entirely; a mixed band still reports. Greedy farthest-point on the levers,
    seeded with the best-ranked candidate so the strongest design is always kept.
    """
    if len(band) <= k:
        return band
    chosen = [band[0]]
    while len(chosen) < k:
        best, best_d = None, -1
        for cand in band:
            if cand in chosen:
                continue
            d = min(sum(a != b for a, b in zip(_levers(cand), _levers(c)))
                    for c in chosen)
            if d > best_d:
                best, best_d = cand, d
        if best is None:
            break
        chosen.append(best)
    return chosen


def _tile(rows: list[dict], n: int, window: tuple[float, float] = WINDOW) -> list[dict]:
    """Spread `n` picks evenly across the switching window, diversely within it."""
    lo, hi = window
    width = (hi - lo) / N_BINS
    per_bin = max(n // N_BINS, 1)

    picked, used = [], set()
    for i in range(N_BINS):
        band_lo = lo + i * width
        band = [r for r in rows if band_lo <= r["dd_g"] < band_lo + width]
        for r in _diverse(band, per_bin):
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


# Designing is deterministic for a given parent, core and affinity, so the same
# request inside one process is answered from here. The sensitivity sweep and the
# hedged plate both design over the same cores, and without this a five-minute
# demo spends two of those minutes recomputing what it already knows.
_run_cache: dict[tuple, dict] = {}


def run(parent: str, core: tuple[int, int], kd_intrinsic_M: float | None,
        target: str = "IL-6", mw_da: float | None = None,
        clinical_pg_per_ml: float = 5000.0, kd_source: str = "") -> dict:
    """Design the plate and return everything needed to justify it.

    Cached per (parent, core, affinity) within a process.

    `kd_intrinsic_M` may be None, and often should be. Most published aptamers
    carry no measured affinity, and the honest output then omits every
    affinity-derived number rather than substituting a plausible one. A borrowed
    Kd propagates silently: an 8.5 nM value taken from an anti-IL-6-receptor
    aptamer produced a full column of apparent-Kd figures for an anti-IL-6
    design, all of them wrong and none of them marked as such.

    `kd_source` records where the number came from, so a reader can judge it.
    """
    key = (parent.upper(), tuple(core), kd_intrinsic_M, target, mw_da,
           clinical_pg_per_ml)
    if key in _run_cache:
        # Flagged, not hidden. A step that returns in zero seconds because it
        # reused work looks identical to a step that did nothing, and the second
        # reading is the one a sceptical reader reaches for first.
        return {**_run_cache[key], "from_cache": True}

    # Without a target sequence there is no honest molecular weight, so the
    # clinical conversion is skipped rather than defaulted.
    clinical_M = (thermo.pg_per_ml_to_molar(clinical_pg_per_ml, mw_da)
                  if mw_da else 0.0)
    if kd_intrinsic_M and not kd_source:
        raise ValueError("kd_intrinsic_M requires kd_source naming its origin; "
                         "an unattributed affinity is how a receptor aptamer's "
                         "Kd ends up labelling a design against the ligand")

    # Pick the architecture the parent can actually support. A Watson-Crick
    # parent is destabilised in itself; only a quadruplex, whose opening energy
    # ViennaRNA cannot price, needs a competing tail set against it. Using the
    # tail everywhere was what lost TNF-alpha its entire library: a sequence
    # complementary to the core is a sequence that binds the neighbour's core.
    probe = thermo.fold(parent, core)
    if probe.trustworthy:
        architecture = "intrinsic"
        lib = generate.intrinsic_library(parent, core)
        window = generate.INTRINSIC_WINDOW
    else:
        architecture = "competing-tail"
        lib = generate.library(parent, core)
        window = WINDOW
    in_window = [v for v in lib if window[0] <= v.dd_g <= window[1]]
    rows = score.assess_all(in_window, parent, core, kd_intrinsic_M, clinical_M)
    if kd_intrinsic_M is None:
        for r in rows:
            r["kd_apparent_nM"] = None
            r["occupancy_at_clinical"] = None
    passing = [r for r in rows if r["passes"]]

    ctrls = plate.controls(parent, lib)
    n_test = 96 - len(ctrls)
    selected = _tile(passing, n_test, window)

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

    out = {
        "target": target,
        "architecture": architecture,
        "design_window": list(window),
        "parent": parent,
        "core": list(core),
        "kd_intrinsic_nM": (kd_intrinsic_M * 1e9) if kd_intrinsic_M else None,
        "kd_source": kd_source or "not reported in the literature",
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
        "clinical_bands": ([(name, thermo.pg_per_ml_to_molar(lo, mw_da),
                             thermo.pg_per_ml_to_molar(hi, mw_da))
                            for name, lo, hi in CLINICAL_PG] if mw_da else []),
        "clinical_bands_are_uncited": True,
        "kd_apparent_nM": [r["kd_apparent_nM"] for r in selected
                           if r["kd_apparent_nM"] is not None],
    }
    out["from_cache"] = False
    _run_cache[key] = out
    return out


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
        "window": plots.design_window(
            result["rows"], result["picked_names"],
            str(out_dir / f"{t}_window.png"),
            bands=(*result["design_window"], N_BINS)),
    }
    # The dose and plate figures need a selection to draw. With none, the funnel
    # and window still carry the whole story of why, so they are rendered anyway.
    if result["selected"]:
        # No dose-response without a measured affinity. The curve's whole content
        # is where it sits on the concentration axis, and that position comes
        # from Kd; drawn from a stand-in it is a picture of an assumption.
        if result["kd_apparent_nM"]:
            out["dose"] = plots.dose_response(
                result["kd_apparent_nM"], result["clinical_bands"], result["target"],
                str(out_dir / f"{t}_dose.png"),
                kd_intrinsic_nM=result["kd_intrinsic_nM"])
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
    # No Kd: this parent, the "IL-6 adaptor" of PMC11506342, has no published
    # affinity. The 8.5 nM once used here belongs to AIR-3A, which binds the
    # IL-6 *receptor* — a different protein — and produced a column of confident
    # apparent-Kd values for designs against the ligand.
    res = run(IL6, core=(1, 13), kd_intrinsic_M=None, target="IL-6")
    print(f"library {res['library_size']:,} -> window {res['in_window']:,} -> "
          f"pass {res['passing']:,} -> plate {res['selected']}")
    print("failures:", res["failure_reasons"])
    print("position check:", res["position_check"])
    art = artifacts(res, Path(__file__).resolve().parent.parent / "out")
    for k, v in art.items():
        print(f"  {k:7s} {v}")
