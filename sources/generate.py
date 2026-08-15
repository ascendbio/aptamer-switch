"""Build the variant library the plate is chosen from.

The switch architecture, for a quadruplex parent on a gold electrode:

    5'-thiol-[ aptamer ]-[ linker ]-[ tail ]-methylene blue-3'

The tail is complementary to part of the aptamer. With no target present it
zips up against that region and holds the folded quadruplex open, parking the
reporter at one distance from the electrode. Target binding pays to displace the
tail, the quadruplex reforms, the reporter moves, and that movement is the
signal.

Everything therefore hinges on one competition:

    dd_G  =  dG(tail:core duplex)  -  dG(quadruplex)

Strongly negative and the tail wins permanently — the aptamer can never fold and
the sensor is dead. Strongly positive and the tail never binds — no switch, no
signal. The useful designs sit near zero, and near zero is exactly where a
~1 kcal/mol model error decides the answer. Hence a plate rather than a
prediction.

Four levers, deliberately overlapping in the energies they reach so the plate is
not one long extrapolation:

    register    which stretch of the aptamer the tail is aimed at
    length      coarse tuning, roughly 2.5 kcal/mol per added base
    mismatch    fine tuning, roughly 1 kcal/mol, finer than length can reach
    linker      loop entropy between aptamer and tail
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thermo  # noqa: E402

COMPLEMENT = str.maketrans("ACGT", "TGCA")

# Long runs of one base misprime and synthesise badly; vendors flag them and
# they are a common cause of a well that never works for reasons unrelated to
# the science.
MAX_HOMOPOLYMER = 4


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


@dataclass
class Variant:
    name: str
    sequence: str
    family: str             # which lever produced it
    register: tuple[int, int]
    tail: str
    linker: int
    n_mismatch: int
    dg_duplex: float = 0.0   # tail against its target window
    dg_fold: float = 0.0     # the folded parent it must beat
    dd_g: float = 0.0        # the competition; near zero is the design window
    trustworthy: bool = True
    notes: list[str] = field(default_factory=list)


def _homopolymer_ok(seq: str) -> bool:
    run, prev = 1, ""
    for b in seq:
        run = run + 1 if b == prev else 1
        if run > MAX_HOMOPOLYMER:
            return False
        prev = b
    return True


def _mismatched_tails(tail: str, n_mismatch: int) -> list[tuple[str, int]]:
    """Tails carrying deliberate mismatches.

    Mismatch position matters as much as count: central mismatches destabilise a
    duplex far more than terminal ones, which is what makes them the fine lever.
    A mismatch three bases in costs roughly a kcal/mol where removing a base
    costs two and a half, and walking the position sweeps that range
    continuously instead of in steps.
    """
    if n_mismatch == 0:
        return [(tail, 0)]

    singles = []
    for pos in range(1, len(tail) - 1):
        for sub in "ACGT":
            if sub != tail[pos]:
                singles.append((tail[:pos] + sub + tail[pos + 1:], 1))
    if n_mismatch == 1:
        return singles

    # Two mismatches, kept apart so they act independently rather than merging
    # into one large internal loop.
    doubles = []
    for i in range(1, len(tail) - 4):
        for j in range(i + 3, len(tail) - 1):
            for si in "ACGT":
                if si == tail[i]:
                    continue
                for sj in "ACGT":
                    if sj == tail[j]:
                        continue
                    t = list(tail)
                    t[i], t[j] = si, sj
                    doubles.append(("".join(t), 2))
    return doubles


def library(parent: str, core: tuple[int, int],
            tail_lengths: range = range(4, 15),
            linkers: tuple[int, ...] = (0, 1, 2, 3),
            registers: int = 8,
            max_mismatch: int = 2) -> list[Variant]:
    """Enumerate switch candidates around one parent aptamer."""
    parent = parent.upper().replace("U", "T")
    folded = thermo.fold(parent, core)
    dg_fold = folded.mfe

    core_start, core_end = core
    core_len = core_end - core_start + 1

    seen: set[str] = set()
    out: list[Variant] = []

    # Slide the targeted window across the core, so the plate is not betting
    # everything on one guess at where the tail should aim.
    step = max(1, core_len // max(registers - 1, 1))
    starts = sorted({min(core_start + i * step, core_end - 3)
                     for i in range(registers)})

    for start in starts:
        for tail_len in tail_lengths:
            # The window must slide with `start`, not merely shrink towards a
            # fixed end. Anchoring it at core_end made every register produce the
            # same tail — the reverse complement was always read from the same
            # 3' bases — so dedup collapsed all eight registers into one.
            stop = start - 1 + tail_len
            if stop > core_end:
                continue
            window = parent[start - 1:stop]
            base_tail = revcomp(window)
            for n_mm in range(max_mismatch + 1):
                for tail, mm in _mismatched_tails(base_tail, n_mm):
                    for linker in linkers:
                        seq = parent + ("T" * linker) + tail
                        if seq in seen or not _homopolymer_ok(seq):
                            continue
                        seen.add(seq)

                        dg_dup = thermo.displacement_dg(window, tail)
                        family = ("length" if mm == 0 and linker == 0
                                  else "mismatch" if mm else "linker")
                        out.append(Variant(
                            name="", sequence=seq, family=family,
                            register=(start, core_end), tail=tail, linker=linker,
                            n_mismatch=mm, dg_duplex=dg_dup, dg_fold=dg_fold,
                            dd_g=round(dg_dup - dg_fold, 2),
                            trustworthy=folded.trustworthy,
                        ))

    for i, v in enumerate(sorted(out, key=lambda x: x.dd_g), 1):
        v.name = f"SW{i:04d}"
    return sorted(out, key=lambda x: x.dd_g)


if __name__ == "__main__":
    IL6 = "GGTGGCAGGAGGACTATTTATTTGCTTTTCT"
    lib = library(IL6, core=(1, 13))

    print(f"{len(lib)} candidate switches from one 31 nt parent")
    print(f"parent fold: {lib[0].dg_fold} kcal/mol — the energy a tail must beat\n")

    in_window = [v for v in lib if -2.0 <= v.dd_g <= 2.0]
    print(f"in the design window (ddG within +/-2 kcal/mol): {len(in_window)}")
    print(f"tail always wins (ddG < -2):  "
          f"{sum(1 for v in lib if v.dd_g < -2)}  -> aptamer cannot fold")
    print(f"tail never binds (ddG > +2):  "
          f"{sum(1 for v in lib if v.dd_g > 2)}  -> no switch\n")

    print("spread across the window:")
    for lo in (-2.0, -1.0, 0.0, 1.0):
        band = [v for v in in_window if lo <= v.dd_g < lo + 1.0]
        lens = sorted({len(v.tail) for v in band})
        print(f"  ddG {lo:+.0f} to {lo + 1:+.0f}  {len(band):4d} variants  "
              f"tail lengths {lens}")
