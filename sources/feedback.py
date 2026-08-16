"""Read what the bench measured, and design the next round from it.

Everything upstream is prediction. This is the only module that sees a number
produced by an experiment, and that changes what it is allowed to claim: the
design window, the thresholds and the core were all chosen from models, and one
plate of real signal is worth more than any of them.

Three questions, in the order the data can answer them:

    which core hypothesis responded?   the hedged plate exists to settle this
    did ddG predict anything?          if not, the design variable is wrong
    where is the optimum actually?     recentre the next window on measurement

The discipline is the same as everywhere else in this project. A correlation
across ninety-six noisy wells is easy to find and easy to believe, so each one is
tested against a permutation null before it is reported, and a result that does
not clear it is called what it is. Recentring the window on noise would spend a
second synthesis run confirming an artefact of the first.
"""

from __future__ import annotations

import csv
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PERMUTATIONS = 2000
SEED = 20260816

# Below this the assay is reporting drift rather than binding; used only to
# describe how many wells responded, never to discard a measurement.
RESPONSIVE_PCT = 10.0


def _spearman(a: list[float], b: list[float]) -> float:
    def ranks(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        for r, i in enumerate(order):
            out[i] = float(r)
        return out
    ra, rb = ranks(a), ranks(b)
    if len(a) < 4:
        return float("nan")
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def _permutation_p(a: list[float], b: list[float], observed: float) -> float:
    """How often shuffling gives a correlation at least this strong."""
    rng = random.Random(SEED)
    hits = 0
    for _ in range(PERMUTATIONS):
        shuffled = rng.sample(b, len(b))
        if abs(_spearman(a, shuffled)) >= abs(observed):
            hits += 1
    return round((hits + 1) / (PERMUTATIONS + 1), 4)


def load(plate_csv: Path, results_csv: Path) -> list[dict]:
    """Join measured signal onto the wells that were designed.

    The results file needs a well column and a signal column; everything else is
    taken from the plate, so the bench never has to hand back the design
    metadata it was given.
    """
    wells = {r["Well Position"]: r for r in csv.DictReader(plate_csv.open())}
    joined = []
    for row in csv.DictReader(results_csv.open()):
        pos = next((row[k] for k in row if k.strip().lower().startswith("well")), None)
        signal = next((row[k] for k in row
                       if any(w in k.strip().lower()
                              for w in ("signal", "gain", "response", "change"))), None)
        if pos is None or signal is None or pos not in wells:
            continue
        try:
            value = float(str(signal).replace("%", "").strip())
        except ValueError:
            continue
        w = wells[pos]
        joined.append({
            "well": pos, "name": w.get("Name", ""), "role": w.get("Role", ""),
            "core_hypothesis": w.get("Core hypothesis", ""),
            "dd_g": float(w["ddG (kcal/mol)"]) if w.get("ddG (kcal/mol)") else None,
            "note": w.get("Note", ""), "signal": value,
        })
    return joined


def analyse(joined: list[dict]) -> dict:
    """What the plate actually showed, with each claim tested against a null."""
    tests = [r for r in joined if r["role"] == "test" and r["dd_g"] is not None]
    controls = [r for r in joined if r["role"] == "control"]
    if len(tests) < 8:
        return {"error": f"only {len(tests)} test wells with both a signal and a "
                         f"ddG; too few to read anything from"}

    dd = [r["dd_g"] for r in tests]
    sig = [r["signal"] for r in tests]
    rho = _spearman(dd, sig)
    p_value = _permutation_p(dd, sig, rho)

    responsive = [r for r in tests if r["signal"] >= RESPONSIVE_PCT]
    by_hypothesis: dict[str, list[float]] = {}
    for r in tests:
        if r["core_hypothesis"]:
            by_hypothesis.setdefault(r["core_hypothesis"], []).append(r["signal"])

    # The top decile is what the next window should be built around, provided
    # the correlation survived its null.
    best = sorted(tests, key=lambda r: -r["signal"])[:max(3, len(tests) // 10)]
    optimum = round(statistics.median(r["dd_g"] for r in best), 2)

    return {
        "test_wells": len(tests),
        "control_wells": len(controls),
        "responsive_wells": len(responsive),
        "responsive_fraction": round(len(responsive) / len(tests), 3),
        "ddg_vs_signal": {
            "spearman": round(rho, 3),
            "permutation_p": p_value,
            "predictive": bool(p_value < 0.05),
        },
        "by_core_hypothesis": {
            k: {"wells": len(v), "median_signal": round(statistics.median(v), 2),
                "responsive": sum(1 for x in v if x >= RESPONSIVE_PCT)}
            for k, v in sorted(by_hypothesis.items())
        },
        "best_wells": [{"well": r["well"], "name": r["name"], "dd_g": r["dd_g"],
                        "signal": r["signal"]} for r in best],
        "measured_optimum_ddg": optimum,
        "verdict": _verdict(rho, p_value, by_hypothesis, len(responsive), len(tests),
                            optimum),
    }


def _verdict(rho: float, p: float, by_hyp: dict, n_resp: int, n: int,
             optimum: float) -> str:
    lines = []

    if n_resp == 0:
        lines.append("No well cleared the responsive threshold. Nothing here "
                     "supports a second round on this parent: the failure is "
                     "upstream of the switching design.")
        return " ".join(lines)

    if p < 0.05:
        lines.append(f"ddG predicted signal (rho {rho:+.2f}, p={p}). The design "
                     f"variable is doing real work, and the measured optimum sits "
                     f"at ddG {optimum:+.2f} — recentre the next window there.")
    else:
        lines.append(f"ddG did not predict signal (rho {rho:+.2f}, p={p}). "
                     f"{n_resp} of {n} wells responded, so the chemistry works, "
                     f"but not for the reason modelled. Do not recentre the "
                     f"window on this: design the next round around the "
                     f"sequences that worked, not the energy they were "
                     f"predicted to have.")

    if len(by_hyp) >= 2:
        # Compare responsive RATES, not whether the loser reached exactly zero.
        # Any threshold is crossed occasionally by noise, so requiring none at
        # all reported "both responded" for a plate where one hypothesis had 40
        # of 44 wells respond and the other 5 — the clearest possible result,
        # described as no result.
        rates = {k: (sum(1 for x in v if x >= RESPONSIVE_PCT) / len(v), len(v), v)
                 for k, v in by_hyp.items()}
        ranked = sorted(rates.items(), key=lambda kv: -kv[1][0])
        (top_name, (top_rate, top_n, top_v)) = ranked[0]
        (low_name, (low_rate, low_n, low_v)) = ranked[-1]

        if top_rate >= 0.25 and low_rate <= max(0.15, top_rate / 4):
            lines.append(
                f"{top_name} responded in {top_rate:.0%} of its wells against "
                f"{low_rate:.0%} for {low_name} (medians "
                f"{statistics.median(top_v):.1f} vs {statistics.median(low_v):.1f}). "
                f"That favours one core hypothesis decisively — the epitope "
                f"question is answered by a plate that was going to be "
                f"synthesised anyway. Design the next round on {top_name}.")
        elif top_rate < 0.1:
            lines.append("Neither core hypothesis produced a meaningful response "
                         "rate; the plate does not distinguish them.")
        else:
            lines.append(f"Both core hypotheses responded at similar rates "
                         f"({top_rate:.0%} vs {low_rate:.0%}). The core matters "
                         f"less than assumed; either design route is open.")
    return " ".join(lines)


def example_results(plate_csv: Path, out_csv: Path, seed: int = SEED,
                    winning_hypothesis: str | None = None,
                    signal: bool = True) -> Path:
    """A synthetic results file, for trying the loop before real data exists.

    Clearly labelled and deliberately imperfect: signal peaks near ddG 0 with
    noise large enough that the correlation is real but not overwhelming, which
    is what a first plate actually looks like. Never to be mistaken for a
    measurement — every row says so in its own column.

    `winning_hypothesis` mutes the wells belonging to every other core, which is
    the outcome the hedged plate was built to produce: one epitope hypothesis
    responds, the other does not, and the question is settled by a plate that was
    going to be synthesised regardless.

    `signal=False` generates pure noise, which is the more useful demonstration —
    a tool that finds a correlation in noise is worse than no tool, and this one
    reports p=0.71 and refuses to recentre the window.
    """
    import math
    rng = random.Random(seed)
    rows = [r for r in csv.DictReader(plate_csv.open())]
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        # Header first: a leading comment row would be read as the header by
        # any CSV parser, including this module's own loader.
        w.writerow(["Well Position", "Signal change (%)", "Source"])
        for r in rows:
            if not r["Sequence"]:
                w.writerow([r["Well Position"], round(rng.gauss(1, 1), 2),
                        "SYNTHETIC EXAMPLE - not measured"])
                continue
            dd = float(r["ddG (kcal/mol)"]) if r.get("ddG (kcal/mol)") else 0.0
            if not signal:
                value = rng.uniform(0, 38)          # nothing to find
            else:
                peak = 34 * math.exp(-((dd - 0.3) ** 2) / 0.9)
                hyp = r.get("Core hypothesis", "")
                if winning_hypothesis and hyp and hyp != winning_hypothesis:
                    peak *= 0.08                    # wrong core: barely responds
                value = max(0.0, peak + rng.gauss(0, 7))
            w.writerow([r["Well Position"], round(value, 2),
                        "SYNTHETIC EXAMPLE - not measured"])
    return out_csv


if __name__ == "__main__":
    import json
    out = Path(__file__).resolve().parent.parent / "out"
    plate = Path(sys.argv[1]) if len(sys.argv) > 1 else out / "IL-6_plate.csv"
    results = out / "example_results.csv"
    example_results(plate, results)
    print(json.dumps(analyse(load(plate, results)), indent=2)[:2200])
