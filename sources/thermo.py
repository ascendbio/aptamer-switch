"""The thermodynamics that decide whether an aptamer works as a sensor.

An electrochemical aptamer-based (E-AB) sensor does not report binding. It
reports a *conformational change* on binding: the redox reporter's distance to
the electrode changes, the electron-transfer rate changes, and that is the
signal. A high-affinity aptamer that does not change shape gives no signal at
all.

So the design variable is not affinity — which nobody can predict — but the free
energy cost of opening the binding core, which ViennaRNA computes directly:

    dG_open  =  G(ensemble, core forced unpaired)  -  G(ensemble, unconstrained)

That single number drives both halves of the trade-off this module exists to
quantify:

    gain     rises with dG_open — a core that is already open cannot switch
    affinity FALLS with dG_open — the target must pay the opening cost itself,
                                  so  Kd_app = Kd_intrinsic * (1 + e^(dG_open/RT))

For a small-molecule sensor at micromolar analyte you can spend 3-4 kcal/mol on
gain and not care what it costs in affinity. For a cytokine at picomolar you
cannot: every kcal/mol of opening cost multiplies apparent Kd by ~5x, and the
starting Kd is already only ~1 nM against an analyte that circulates 10^3-10^4
times below that. The usable window is roughly 0.5-2.5 kcal/mol wide, which is
inside ViennaRNA's own error bar — which is precisely why the answer is a
96-variant plate and not a single predicted sequence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import RNA

# Gas constant in kcal/(mol.K), and the temperature the assay actually runs at.
R_KCAL = 1.98720425e-3
BODY_TEMP_C = 37.0
RT = R_KCAL * (273.15 + BODY_TEMP_C)

_DNA_PARAMS_LOADED = False


def _dna_model() -> RNA.md:
    """Model details for DNA at body temperature, with G-quadruplexes on.

    The DNA parameter set is global state in ViennaRNA rather than a property of
    the model details object, so it is loaded once and never unloaded — every
    fold in this module is a DNA fold.

    G-quadruplex modelling matters here: a large fraction of DNA aptamers are
    quadruplexes, and with gquad off they fold to nothing at all. It is not a
    complete fix — ViennaRNA models 3-layer parallel quadruplexes, so the
    2-quartet antiparallel chair of the 15-mer thrombin aptamer is still missed.
    Treat quadruplex-rich designs as scored with a wider error bar.
    """
    global _DNA_PARAMS_LOADED
    if not _DNA_PARAMS_LOADED:
        RNA.params_load_DNA_Mathews2004()
        _DNA_PARAMS_LOADED = True
    md = RNA.md()
    md.temperature = BODY_TEMP_C
    md.gquad = 1
    return md


@dataclass
class Fold:
    """What one candidate sequence does thermodynamically."""

    sequence: str
    structure: str          # MFE secondary structure, dot-bracket
    mfe: float              # kcal/mol
    ensemble_dg: float      # kcal/mol, free energy of the whole ensemble
    dg_open: float          # kcal/mol, cost of vacating the binding core
    p_core_open: float      # fraction of unbound molecules with the core free
    kd_penalty: float       # multiplier on intrinsic Kd, = 1 + e^(dG_open/RT)
    core_is_quadruplex: bool  # core overlaps a predicted G-quadruplex
    dg_open_wc: float       # same quantity with quadruplexes switched off
    trustworthy: bool       # False when the two models disagree materially

    def kd_apparent(self, kd_intrinsic_M: float) -> float:
        """Effective Kd of the switch, in molar.

        The target has to pay the core-opening cost out of its own binding
        energy, so any construct that switches binds more weakly than the
        aptamer it was built from. This is the tax that makes cytokine sensing
        hard, and it is worth reporting next to every gain estimate.
        """
        return kd_intrinsic_M * self.kd_penalty


def _open_cost(seq: str, core: tuple[int, int], gquad: int) -> tuple[float, float]:
    """(ensemble dG, dG of the same ensemble with the core forced unpaired)."""
    md = _dna_model()
    md.gquad = gquad

    fc = RNA.fold_compound(seq, md)
    _, mfe = fc.mfe()
    fc.exp_params_rescale(mfe)          # keeps the partition function in range
    _, ensemble_dg = fc.pf()

    fc_open = RNA.fold_compound(seq, md)
    for i in range(core[0], core[1] + 1):
        fc_open.hc_add_up(i)
    fc_open.exp_params_rescale(mfe)
    _, open_dg = fc_open.pf()

    return ensemble_dg, max(open_dg - ensemble_dg, 0.0)


def fold(sequence: str, core: tuple[int, int]) -> Fold:
    """Fold `sequence` and price the opening of `core`.

    `core` is a 1-based inclusive (start, end) span naming the residues the
    target must contact — the part that has to be free for binding to happen.

    Computed twice, with G-quadruplexes on and off, because ViennaRNA's hard
    constraints are silently ignored inside a quadruplex: forcing positions 1-13
    of the IL-6 aptamer unpaired still returns a structure that pairs 1 with 12,
    and prices the opening at 0.04 kcal/mol instead of the ~1.5 the same
    constraint costs without quadruplex modelling. The Watson-Crick machinery
    itself is exact — a clean 10 bp stem costs 11.53 kcal/mol to open under both
    models — so the failure is specific to quadruplexes rather than general.

    Where the two models disagree, `trustworthy` is False and the number should
    not be ranked on. For quadruplex aptamers, engineer the switch with a
    competing tail instead and score it with `displacement_dg`, which is
    Watson-Crick throughout and therefore inside what this package can compute.
    """
    seq = sequence.upper().replace("U", "T")
    start, end = core
    if not (1 <= start <= end <= len(seq)):
        raise ValueError(f"core {core} outside sequence of length {len(seq)}")

    md = _dna_model()
    fc = RNA.fold_compound(seq, md)
    structure, mfe = fc.mfe()

    ensemble_dg, dg_open = _open_cost(seq, core, gquad=1)
    _, dg_open_wc = _open_cost(seq, core, gquad=0)

    in_core = structure[start - 1:end]
    is_quad = "+" in in_core or "~" in in_core
    trustworthy = not (is_quad and abs(dg_open - dg_open_wc) > 1.0)

    return Fold(
        sequence=seq,
        structure=structure,
        mfe=round(mfe, 2),
        ensemble_dg=round(ensemble_dg, 2),
        dg_open=round(dg_open, 3),
        p_core_open=math.exp(-dg_open / RT),
        kd_penalty=1.0 + math.exp(dg_open / RT),
        core_is_quadruplex=is_quad,
        dg_open_wc=round(dg_open_wc, 3),
        trustworthy=trustworthy,
    )


def displacement_dg(core_seq: str, tail_seq: str) -> float:
    """Stability of the tail:core duplex, in kcal/mol (negative = stable).

    The design lever for a quadruplex aptamer. A complementary tail appended to
    the construct competes with the folded aptamer; the target must displace it
    to bind. Tail length and deliberate mismatches tune this number continuously,
    and unlike the quadruplex itself it is ordinary Watson-Crick duplex formation
    — the regime ViennaRNA is reliable in.
    """
    _dna_model()                        # ensure DNA parameters are loaded
    RNA.cvar.temperature = BODY_TEMP_C
    return round(RNA.duplexfold(core_seq.upper(), tail_seq.upper()).energy, 3)


def occupancy(analyte_M: float, kd_M: float) -> float:
    """Fraction of sensors bound — the ceiling on any signal at that analyte level."""
    return analyte_M / (analyte_M + kd_M)


def pg_per_ml_to_molar(pg_per_ml: float, mw_da: float) -> float:
    """Clinical cytokine units are pg/mL; binding maths needs molar."""
    return (pg_per_ml * 1e-12 / 1e-3) / mw_da


if __name__ == "__main__":
    # The published IL-6 DNA aptamer used in a continuous E-AB sensor
    # (PMC12384986). No Kd reported there; SPRi-selected IL-6 aptamers in
    # PMC12848694-adjacent work sit at ~1.2 nM, used here as the intrinsic value.
    IL6 = "GGTGGCAGGAGGACTATTTATTTGCTTTTCT"
    KD_INTRINSIC = 1.2e-9
    IL6_MW = 21000.0

    f = fold(IL6, core=(1, 13))
    print(f"parent  {f.sequence}")
    print(f"        {f.structure}  MFE {f.mfe} kcal/mol")
    print(f"  dG_open {f.dg_open} (gquad) vs {f.dg_open_wc} (WC only) kcal/mol")
    print(f"  core is a quadruplex: {f.core_is_quadruplex}   "
          f"score trustworthy: {f.trustworthy}")
    print(f"  Kd penalty x{f.kd_penalty:.1f}  ->  Kd_app "
          f"{f.kd_apparent(KD_INTRINSIC) * 1e9:.2f} nM")
    print()
    print("Competing-tail lever (the computable one for a quadruplex parent):")
    core_seq = IL6[:13]
    revcomp = core_seq[::-1].translate(str.maketrans("ACGT", "TGCA"))
    print(f"  quadruplex it must beat: {f.mfe} kcal/mol")
    for n in (5, 7, 9, 11, 13):
        print(f"  {n:2d} nt tail  dG_duplex {displacement_dg(core_seq, revcomp[:n]):6.2f} kcal/mol")
    print()
    print("IL-6 in blood, and what this sensor would report:")
    for label, pg in (("healthy baseline", 3), ("mild inflammation", 50),
                      ("sepsis", 1_000), ("severe sepsis", 5_000),
                      ("severe CRS", 20_000)):
        molar = pg_per_ml_to_molar(pg, IL6_MW)
        print(f"  {label:20s} {pg:7,d} pg/mL = {molar * 1e12:8.1f} pM   "
              f"occupancy {occupancy(molar, f.kd_apparent(KD_INTRINSIC)):6.2%}")
