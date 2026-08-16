"""Check the plate's thermodynamics against a second, independent engine.

Everything scored so far comes from one source, ViennaRNA with the Mathews 2004
DNA parameters. That is a single point of failure for the whole plate, and this
project has already had to correct its dimer scoring twice: once for an
initiation constant that turned out not to exist, once for comparing a property
of the parent against a property of the design. Both were caught by argument
rather than by measurement.

Primer3 computes the same quantities — hairpin and homodimer free energy, Tm, GC
— from the SantaLucia unified nearest-neighbour parameters, a different
implementation of different published values. Where the two agree, the number is
corroborated. Where they disagree, one of them is wrong and the plate should not
be ordered until it is known which.

The comparison is deliberately of *rankings* as well as values. The absolute
numbers come from different parameter sets and need not match; what must hold is
that both engines put the same designs at the sticky end, because that ordering
is what the dimer criterion acts on.
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score  # noqa: E402
import thermo  # noqa: E402

BODY_TEMP_C = 37.0


def _spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation, which is the thing that matters here."""
    def ranks(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        for rank, i in enumerate(order):
            out[i] = float(rank)
        return out
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    if n < 3:
        return float("nan")
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def run(plate_csv: Path, remote: bool = False) -> dict:
    """Score every well with both engines and compare."""
    rows = [r for r in csv.DictReader(plate_csv.open()) if r["Sequence"]]
    seqs = [r["Sequence"] for r in rows]

    from proto_tools.tools.sequence_scoring.primer3.primer3_thermodynamics import (
        Primer3Oligo,
        Primer3ThermodynamicsConfig,
        Primer3ThermodynamicsInput,
        run_primer3_thermodynamics,
    )

    payload = Primer3ThermodynamicsInput(oligos=[Primer3Oligo(sequence=s) for s in seqs])
    cfg = Primer3ThermodynamicsConfig(temp_c=BODY_TEMP_C)

    if remote:
        from proto_tools.modal import dispatch_to_modal
        result = dispatch_to_modal("primer3-thermodynamics", payload, cfg,
                                   environment="proto-env")
    else:
        # CPU-only, so it runs here for nothing. Modal is for the GPU half.
        result = run_primer3_thermodynamics(payload, cfg)

    # Results come back one per submitted oligo, in order. An earlier version
    # keyed them into a dict by an index parsed out of oligo_id, which silently
    # dropped rows and misaligned the rest — reporting a rank correlation of
    # 0.14 where the correct pairing gives 0.67, i.e. inventing a disagreement
    # between the two engines that was purely a bookkeeping error. Pair by
    # position and assert the lengths match, which is checkable.
    results = list(result.results)
    if len(results) != len(seqs):
        raise RuntimeError(f"primer3 returned {len(results)} results for "
                           f"{len(seqs)} oligos; refusing to guess the pairing")

    ours_dimer, theirs_dimer, ours_fold, theirs_fold, disagree = [], [], [], [], []

    for row, seq, r in zip(rows, seqs, results):
        if int(r.length) != len(seq):
            raise RuntimeError(f"pairing mismatch at {row['Well Position']}: "
                               f"primer3 length {r.length} vs sequence {len(seq)}")
        mine_dimer = score._self_dimer(seq)
        mine_fold = thermo.fold(seq, (1, min(12, len(seq) - 1))).mfe

        ours_dimer.append(mine_dimer)
        theirs_dimer.append(float(r.homodimer_dg))
        ours_fold.append(mine_fold)
        theirs_fold.append(float(r.hairpin_dg))

        # Flag wells where the engines disagree about whether it is sticky at all.
        if (mine_dimer < -10) != (float(r.homodimer_dg) < -10):
            disagree.append((row["Well Position"], row["Name"],
                             round(mine_dimer, 2), round(float(r.homodimer_dg), 2)))

    return {
        "wells_compared": len(ours_dimer),
        "homodimer": {
            "spearman": round(_spearman(ours_dimer, theirs_dimer), 3),
            "vienna_mean": round(statistics.mean(ours_dimer), 2),
            "primer3_mean": round(statistics.mean(theirs_dimer), 2),
        },
        "hairpin_vs_mfe": {
            "spearman": round(_spearman(ours_fold, theirs_fold), 3),
            "vienna_mean": round(statistics.mean(ours_fold), 2),
            "primer3_mean": round(statistics.mean(theirs_fold), 2),
        },
        "wells_where_engines_disagree_on_stickiness": disagree[:10],
        "n_disagreements": len(disagree),
    }


if __name__ == "__main__":
    import json

    default = Path(__file__).resolve().parent.parent / "out" / "IL-6_plate.csv"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else default
    out = run(path, remote="--remote" in sys.argv)
    print(json.dumps(out, indent=2))
