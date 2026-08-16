"""Everything that decides whether a well is worth its place on the plate.

Switching thermodynamics say whether a candidate *can* report. These criteria
say whether it will report the right thing, and whether it can be made at all.
They are separated deliberately: a design that switches beautifully and also
dimerises with itself is not a good design, and averaging those into a single
number hides exactly the failure you needed to see.

    switching       does the tail compete with the fold near enough to flip?
    specificity     does it bind the intended thing, and only that?
    manufacture     will the vendor make it, and will it survive the assay?

Specificity is where a sensor quietly fails. A structure-switching aptamer has
three ways to give a signal that is not the analyte:

  * the tail binds the wrong part of its own construct, so the switch reports a
    conformational change that has nothing to do with target
  * the construct dimerises with its neighbour on the electrode, which at the
    densities used for E-AB is a real and common failure
  * the design is G-rich enough to fold quadruplexes opportunistically, and
    quadruplexes are famously promiscuous protein binders — the reason a
    G-quadruplex hit shows up in so many unrelated SELEX campaigns

All three are computable from sequence, so none of them needs to be discovered
in the lab.

Ranking never collapses to one number alone. Candidates first have to *pass* —
a hard filter, because a dimerising design is not redeemed by a good ddG — and
only survivors are ranked. The plate is then tiled across the ddG window from
the ranked survivors, so position on the plate carries information rather than
ninety-six copies of the same idea.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import RNA

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thermo  # noqa: E402

# Dimerisation is judged against the construct's own fold, not on an absolute
# scale. In this architecture the tail is complementary to the core by design, so
# construct A's tail will always be able to grab construct B's core: every
# candidate self-dimerises, and an absolute cutoff just re-measures ddG. It threw
# out 3,157 of 3,366 candidates for having the property they were built to have.
#
# What actually separates a good design from a bad one is whether the
# intermolecular duplex outcompetes the intramolecular one. Folding back on
# itself is unimolecular and does not pay an entropic search cost, so it wins
# unless the dimer is substantially more stable. Surface density then manages
# what is left, in the lab, where it belongs.
# Chosen, not derived. How much more stable the intermolecular duplex may be
# than the intramolecular fold before the design is rejected.
DIMER_MARGIN_LIMIT = -6.0

# NOT a measured correction. An earlier version added +4.1 kcal/mol here as a
# "duplex initiation" term, justified by memory of the unified nearest-neighbour
# parameters. That justification does not survive checking: ViennaRNA exposes no
# duplexInit parameter, and cofold() and duplexfold() return identical energies
# for the same pair of strands (measured over 60 random duplexes: mean difference
# +0.00, sd 0.00). There is no missing term of that size to restore, and the
# constant silently took one parent's library from 182 passing to 644.
#
# The underlying physics is real - an intermolecular association pays a
# concentration-dependent entropic cost that duplexfold does not model, and on a
# packed electrode surface the effective concentration is not solution-like - but
# ViennaRNA cannot supply that number and neither can I. So no offset is applied,
# and the threshold below is stated for what it is: a judgement, not a
# measurement. Sensitivity is reported by score.py's __main__ so its influence is
# visible rather than buried.
INTERMOLECULAR_INIT = 0.0

# The tail must prefer its intended site over any other site in the construct by
# a clear margin, or the switch reports the wrong conformational change.
SPECIFICITY_MARGIN = 1.0     # kcal/mol

# Four or more consecutive G seeds a quadruplex. Checked on the tail only: the
# parent's own G-runs are not a design choice, and scoring them per-candidate
# flags every variant equally, which is information about the target rather than
# about the design. One TNF-alpha parent contains GGGG and lost its entire
# library of 21,862 candidates to a property none of them could have avoided.
MAX_G_RUN = 3
MAX_HOMOPOLYMER = 4


@dataclass
class Assessment:
    # switching
    dd_g: float
    kd_apparent_nM: float | None
    occupancy_at_clinical: float | None
    # specificity
    specificity_margin: float | None
    self_dimer_dg: float
    dimer_margin: float
    max_g_run: int
    parent_max_g_run: int
    g_fraction: float
    # manufacture
    length: int
    gc_percent: float
    max_homopolymer: int
    # verdict
    passes: bool
    flags: list[str]
    flag_kinds: list[str]
    rank_score: float


def _max_run(seq: str, base: str | None = None) -> int:
    bases = [base] if base else list("ACGT")
    return max((max((len(m.group(0)) for m in re.finditer(f"{b}+", seq)), default=0)
                for b in bases), default=0)


def _self_dimer(seq: str) -> float:
    """Free energy of this construct against a copy of itself.

    Loads the DNA parameters first rather than assuming someone else has.
    ViennaRNA keeps them in global state, so duplexfold silently returns RNA
    energies until something calls the loader: the same 45-mer scores -38.1
    before and -15.0 after, a 23 kcal/mol swing decided by import order. The
    pipeline happened to be safe because thermo.fold() runs first inside
    assess(), which is not a property worth relying on.
    """
    thermo._dna_model()
    RNA.cvar.temperature = thermo.BODY_TEMP_C
    return round(RNA.duplexfold(seq, seq).energy, 2)


def specificity_margin(parent: str, tail: str, window: str) -> float:
    """How much the tail prefers its intended site to its best alternative.

    Positive is good: the intended duplex is more stable than anything else the
    tail could grab in the same molecule. A tail that binds two places switches
    on something that is not the target.
    """
    intended = RNA.duplexfold(window, tail).energy
    n = len(window)
    best_off = 0.0
    for i in range(len(parent) - n + 1):
        alt = parent[i:i + n]
        if alt == window:
            continue
        best_off = min(best_off, RNA.duplexfold(alt, tail).energy)
    return round(best_off - intended, 2)


def assess(sequence: str, parent: str, tail: str, window: str, dd_g: float,
           kd_intrinsic_M: float | None, clinical_M: float,
           core: tuple[int, int]) -> Assessment:
    """Score one candidate on every criterion, then decide whether it qualifies."""
    seq = sequence.upper()
    folded = thermo.fold(seq, core)
    kd_app = folded.kd_apparent(kd_intrinsic_M) if kd_intrinsic_M else None

    # No tail means no off-target tail site to check. The criterion is not
    # "passed" so much as absent, and scoring it as a pass at some arbitrary
    # value would let it silently rank intrinsic designs above tail ones.
    margin = specificity_margin(parent, tail, window) if tail else float("nan")
    dimer = _self_dimer(seq)
    # Positive means the intramolecular fold wins, which is what we want.
    dimer_margin = round((dimer + INTERMOLECULAR_INIT) - folded.mfe, 2)
    # For an intrinsic design the whole molecule is the designed region: there
    # is no tail, and the mutations sit in the parent itself.
    designed = tail if tail else seq
    g_run = _max_run(designed, "G")
    parent_g_run = _max_run(parent, "G")
    g_frac = seq.count("G") / len(seq)
    gc = 100.0 * (seq.count("G") + seq.count("C")) / len(seq)
    homo = _max_run(designed)

    # Each flag carries a stable category alongside its detail, so failures can
    # be counted and charted without parsing numbers back out of prose.
    checks = [
        (tail != "" and margin < SPECIFICITY_MARGIN, "off-target tail site",
         f"tail binds a second site (margin {margin} kcal/mol)"),
        (dimer_margin < DIMER_MARGIN_LIMIT, "dimerises",
         f"dimer beats its own fold by {-dimer_margin:.1f} kcal/mol"),
        (g_run > MAX_G_RUN, "quadruplex-prone",
         f"G{g_run} run in the designed tail — opportunistic quadruplex"),
        (homo > MAX_HOMOPOLYMER, "synthesis risk",
         f"{homo}-base homopolymer"),
        (not 25.0 <= gc <= 75.0, "synthesis risk",
         f"GC {gc:.0f}% outside the synthesisable band"),
    ]
    flags = [detail for bad, _, detail in checks if bad]
    flag_kinds = sorted({kind for bad, kind, _ in checks if bad})

    # Rank the survivors by how close the switch sits to balance, then break ties
    # towards specificity. Deliberately not a weighted sum over all criteria:
    # the criteria that matter are already hard filters, and folding them into a
    # score would let a good ddG buy its way past a dimerising design.
    # Tail designs rank on how close the competition sits to balance; intrinsic
    # designs rank on how close dG_open sits to the middle of its usable band.
    # The two are different quantities and must not be compared to each other.
    if tail:
        rank_score = round(-abs(dd_g) + 0.25 * min(margin, 4.0), 3)
    else:
        rank_score = round(-abs(dd_g - 1.5), 3)

    return Assessment(
        dd_g=dd_g,
        kd_apparent_nM=None if kd_app is None else round(kd_app * 1e9, 2),
        occupancy_at_clinical=(None if kd_app is None
                               else round(thermo.occupancy(clinical_M, kd_app), 4)),
        specificity_margin=None if tail == "" else margin,
        self_dimer_dg=dimer,
        dimer_margin=dimer_margin,
        max_g_run=g_run,
        parent_max_g_run=parent_g_run,
        g_fraction=round(g_frac, 3),
        length=len(seq),
        gc_percent=round(gc, 1),
        max_homopolymer=homo,
        passes=not flags,
        flags=flags,
        flag_kinds=flag_kinds,
        rank_score=rank_score,
    )


def assess_all(variants, parent: str, core: tuple[int, int],
               kd_intrinsic_M: float, clinical_M: float) -> list[dict]:
    """Score a whole library, ranked best-first among those that pass."""
    out = []
    for v in variants:
        window = parent[v.register[0] - 1:v.register[1]]
        a = assess(v.sequence, parent, v.tail, window, v.dd_g,
                   kd_intrinsic_M, clinical_M, core)
        row = {"name": v.name, "sequence": v.sequence, "family": v.family,
               "tail": v.tail, "tail_len": len(v.tail), "linker": v.linker,
               "n_mismatch": v.n_mismatch, "register": v.register[0], **asdict(a)}
        out.append(row)

    out.sort(key=lambda r: (not r["passes"], -r["rank_score"]))
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return out


if __name__ == "__main__":
    import generate

    IL6 = "GGTGGCAGGAGGACTATTTATTTGCTTTTCT"
    CORE = (1, 13)
    # No Kd: AIR-3A's 8.5 nM belongs to an anti-IL-6-RECEPTOR aptamer, a
    # different protein, and this parent has no published affinity.
    KD = None
    CLINICAL = 0.0

    lib = generate.library(IL6, CORE)
    window = [v for v in lib if -2 <= v.dd_g <= 2]
    rows = assess_all(window, IL6, CORE, KD, CLINICAL)

    n_pass = sum(r["passes"] for r in rows)
    print(f"{len(rows)} candidates in the ddG window, {n_pass} pass all criteria "
          f"({n_pass / len(rows):.0%})")

    from collections import Counter
    reasons = Counter(k for r in rows if not r["passes"] for k in r["flag_kinds"])
    print("\nwhy candidates fail:")
    for reason, n in reasons.most_common():
        print(f"  {n:5d}  {reason}")

    print("\nsensitivity of the dimer threshold (a chosen value, not a measured one):")
    for limit in (-2.0, -4.0, -6.0, -8.0, -10.0):
        n = sum(1 for r in rows
                if r["dimer_margin"] >= limit and r["specificity_margin"] is not None
                and r["specificity_margin"] >= SPECIFICITY_MARGIN)
        mark = "  <- in use" if limit == DIMER_MARGIN_LIMIT else ""
        print(f"  DIMER_MARGIN_LIMIT {limit:+5.1f} -> {n:5d} pass{mark}")

    print("\ntop 5 by rank:")
    for r in rows[:5]:
        print(f"  {r['rank']:3d}  {r['name']}  ddG {r['dd_g']:+5.2f}  "
              f"margin {r['specificity_margin']:+5.2f}  "
              f"dimer margin {r['dimer_margin']:+6.2f}")
