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
    length      -1.6 kcal/mol per added base (measured over this library)
    mismatch    +3.0 kcal/mol per central mismatch (measured, n=447)
    linker      loop entropy between aptamer and tail

Those two figures were once documented the other way round - length as the coarse
lever at 2.5 and mismatch as the fine one at 1.0 - which is backwards. Measured
on the library itself, a mismatch moves the energy roughly twice as far as adding
a base does. Length is the fine adjustment here, not the coarse one. The levers
are kept for the coverage they give together, but neither number was checked
before it was written down.
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
    duplex far more than terminal ones, so walking the position sweeps a range
    rather than stepping. Measured on this library a central mismatch costs about
    +3.0 kcal/mol against about -1.6 for each base of length, so mismatches are
    the coarser of the two levers, not the finer.
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


# --------------------------------------------------- intrinsic destabilisation

"""A second architecture, for parents that do not need a tail.

The competing tail exists because a G-quadruplex cannot be priced directly:
ViennaRNA ignores hard constraints inside one, so the only computable lever is a
Watson-Crick duplex set against it. A parent that folds by ordinary base pairing
has no such problem — dG_open is exact for it — and the tail becomes not merely
unnecessary but harmful, since a sequence complementary to the core is by
construction a sequence that binds the neighbouring construct's core.

That is what killed TNF-alpha: 21,862 candidates, every one flagged as
dimerising, all of them for a feature the architecture forced on them.

Here the aptamer is destabilised in itself instead. Shorten a terminal stem, or
break a pair inside it, until the fold sits close enough to open that binding can
finish the job. No added sequence, so no tail to cross-hybridise, and the design
variable is dG_open directly rather than a difference of two energies.
"""

# The useful band for dG_open, in kcal/mol. Below this the core is already open
# and there is nothing to switch; above it the target cannot pay the opening
# cost, and every kcal/mol multiplies apparent Kd by about five.
INTRINSIC_WINDOW = (0.3, 3.0)


def intrinsic_library(parent: str, core: tuple[int, int]) -> list[Variant]:
    """Truncation and point-mutation variants of the parent itself.

    Both levers are confined to sequence outside the binding core. Trimming into
    the core, or mutating a residue the target contacts, does not produce a
    weaker switch — it produces a molecule that no longer binds, which the
    thermodynamics will happily score as an excellent switch because dG_open of
    a core that was deleted is zero.
    """
    parent = parent.upper().replace("U", "T")
    core_lo, core_hi = core
    max_left = core_lo - 1                    # 5' residues free to remove
    max_right = len(parent) - core_hi         # 3' residues free to remove

    seen: set[str] = set()
    out: list[Variant] = []

    def add(seq: str, left: int, family: str, n_mm: int) -> None:
        if seq in seen or len(seq) < 12 or not _homopolymer_ok(seq):
            return
        seen.add(seq)
        lo, hi = core_lo - left, core_hi - left
        if not 1 <= lo < hi <= len(seq):
            return
        f = thermo.fold(seq, (lo, hi))
        if not f.trustworthy:
            return
        out.append(Variant(
            name="", sequence=seq, family=family, register=(lo, hi),
            tail="", linker=0, n_mismatch=n_mm,
            dg_duplex=0.0, dg_fold=f.mfe, dd_g=round(f.dg_open, 2),
            trustworthy=True,
            notes=[f"architecture=intrinsic, dG_open {f.dg_open}"]))

    # Mutable positions are those outside the core, 0-based.
    free = [i for i in range(len(parent)) if not (core_lo - 1 <= i <= core_hi - 1)]

    for left in range(max_left + 1):
        for right in range(max_right + 1):
            trimmed = parent[left:len(parent) - right] if right else parent[left:]
            add(trimmed, left, "truncation", 0)

            # Point mutations on top of each truncation. Truncation is the coarse
            # lever and mutation the fine one; combining them reaches energies
            # neither gets to alone, which matters because the usable band is
            # narrow and a single lever lands in it only by luck.
            for i in free:
                j = i - left
                if not 0 <= j < len(trimmed):
                    continue
                for sub in "ACGT":
                    if sub == trimmed[j]:
                        continue
                    add(trimmed[:j] + sub + trimmed[j + 1:], left,
                        "truncation+mismatch" if (left or right) else "mismatch", 1)

    for i, v in enumerate(sorted(out, key=lambda x: x.dd_g), 1):
        v.name = f"IN{i:04d}"
    return sorted(out, key=lambda x: x.dd_g)
