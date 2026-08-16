"""What is measured, what is assumed, and what was tested and rejected.

The uncertainty handling in this project is real but scattered: a trustworthy
flag in one module, a threshold sensitivity printed by another's __main__, a
caveat in a manifest, a negative result living only in a git commit message. A
reader deciding whether to spend twenty thousand dollars on ninety-six oligos
cannot assemble that picture from the source.

So assemble it here, in three columns that mean different things:

    measured    computed from sequence or read from a paper, reproducible
    assumed     a choice, with the value stated, that changes the answer
    rejected    tried, tested against a control, and found not to work

The third column is the one usually missing. A tool that only reports what it
used looks more certain than it is; the models that were tried and discarded are
what show the remaining numbers were earned.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Results of experiments run in this project, each against a control. Values are
# recorded rather than described, so a reader can disagree with the conclusion.
REJECTED = [
    {
        "model": "OpenDDE (AlphaFold3-class complex prediction)",
        "asked": "predict the IL-6:aptamer complex and read the binding core "
                 "off the interface",
        "control": "the same protein against a scrambled aptamer of identical "
                   "length and base composition",
        "result": "Scrambled scored HIGHER: interface pTM 0.676 vs 0.580, and "
                  "higher on pLDDT, pTM and ranking score too",
        "conclusion": "no sequence-specific signal — the contacts cannot define "
                      "a core. Not used.",
    },
    {
        "model": "Evo2-7B (genome language model)",
        "asked": "score designs by how plausible the sequence is",
        "control": "six published aptamers, each against its own scramble",
        "result": "Real scored lower in 4 of 6, mean perplexity difference "
                  "-0.22; the effect is carried almost entirely by AS1411, a "
                  "GGTGGTGGT repeat, and both clear losses are the "
                  "non-repetitive sequences",
        "conclusion": "a signal consistent with preferring repetitive DNA rather "
                      "than recognising aptamers, and 4/6 is what a coin flip "
                      "gives a third of the time. Not used for ranking.",
    },
    {
        "model": "surface electrostatics from an ESMFold2 structure",
        "asked": "rank targets by whether they present a basic patch for a "
                 "polyanionic aptamer to bind",
        "control": "proteins whose aptamers are textbook — nucleolin, thrombin, "
                   "lysozyme",
        "result": "IL-6 (+1.78) outranked lysozyme (+1.39), and lysozyme — net "
                  "charge +8, pI about 11 — reports zero exposed basic residues",
        "conclusion": "the burial proxy scales with how compact a protein is, so "
                      "it is not measuring surface. Kept as a measurement, no "
                      "verdict issued, not wired in.",
    },
]

CORROBORATED = [
    {
        "claim": "the dimer and fold energies the plate is filtered on",
        "against": "Primer3 with the SantaLucia parameters, an independent "
                   "implementation",
        "result": "homodimer rank correlation 0.667 across 95 wells, 0.808 on "
                  "random DNA. Absolute values differ by about 6 kcal/mol "
                  "because the parameter sets differ; the ordering, which is "
                  "what the criterion acts on, holds.",
    },
]


def build(manifest_path: Path | None = None) -> str:
    """Markdown ledger for a run, or the standing one if no run is given."""
    lines = ["### What this plate rests on", ""]

    if manifest_path and manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        th = m.get("thresholds", {})
        kd = m.get("kd_intrinsic_nM")

        lines += [
            "**Measured**", "",
            f"- folding and opening energies: ViennaRNA 2.7.2, DNA parameters, "
            f"37 °C — deterministic, identical between runs",
            f"- library: {m.get('library_size', 0):,} variants enumerated, not "
            f"sampled",
            f"- affinity: " + (f"{kd:.2f} nM, from {m.get('kd_source')}" if kd
                               else "**none published for this parent** — every "
                                    "affinity-derived figure is omitted rather "
                                    "than estimated"),
            f"- well positions randomised against ddG: row spread "
            f"{m.get('position_check', {}).get('observed')} vs "
            f"{m.get('position_check', {}).get('null_p95')} at the 95th "
            f"percentile of random assignment",
            "",
            "**Assumed**", "",
            f"- **binding core {m.get('core_assumed')}** — no paper maps where "
            f"this cytokine contacts its aptamer. Every ddG is computed against "
            f"this span.",
            f"- switching window {th.get('switching_window_tail')} kcal/mol "
            f"(tail) / {th.get('switching_window_intrinsic')} (intrinsic)",
            f"- specificity margin ≥ {th.get('specificity_margin_min')} kcal/mol, "
            f"dimer margin ≥ {th.get('dimer_margin_min')} kcal/mol — chosen, not "
            f"measured; the dimer limit alone moves the pass count from 182 to "
            f"1,025",
            f"- synthesis limits: max G-run {th.get('max_g_run_in_designed_region')}, "
            f"max homopolymer {th.get('max_homopolymer')}, GC "
            f"{th.get('gc_percent_band')}%",
            f"- produced by commit `{m.get('code_commit')}`",
            "",
        ]

    lines += ["**Tested and rejected**", ""]
    for r in REJECTED:
        lines += [f"- **{r['model']}** — asked to {r['asked']}. "
                  f"Control: {r['control']}. {r['result']}. "
                  f"→ {r['conclusion']}"]
    lines += ["", "**Independently corroborated**", ""]
    for c in CORROBORATED:
        lines += [f"- {c['claim']}, against {c['against']}: {c['result']}"]

    return "\n".join(lines)


if __name__ == "__main__":
    runs = Path(__file__).resolve().parent.parent / "out" / "runs"
    latest = max(runs.iterdir(), default=None) if runs.exists() else None
    print(build(latest / "manifest.json" if latest else None))
