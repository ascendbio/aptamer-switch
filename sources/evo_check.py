"""Does Evo2 know anything about aptamers? Test before trusting, not after.

Evo2 is a genome language model: its own documentation says perplexity measures
"how well a DNA sequence matches the patterns the model learned during training
across all domains of life". An aptamer is a synthetic sequence selected in vitro
for a shape, never subject to genomic evolution, so there is no prior reason its
likelihood under a genomic distribution should track binding or switching.

There is also no prior reason it should not. Evo2 has seen riboswitches,
ribozymes and structured non-coding elements — natural aptamers, in effect — so
it may carry some notion of a foldable functional oligo. That is an empirical
question, and answering it costs one small experiment.

The test is the one that settled OpenDDE: score real published aptamers against
scrambles of identical length and base composition. If the model separates them,
its scores carry information about this molecule class and can be used as a
ranking signal. If it does not, its numbers are decoration, and 96 confident
perplexities would be worse than none at all.

Deliberately paired: each scramble is generated from its own aptamer, so length
and composition are held constant and only the arrangement differs.
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

# Published aptamers with independent literature support, spanning the classes
# this project cares about: a G-quadruplex, Watson-Crick folds, and the two
# cytokine parents actually in use.
REFERENCE_APTAMERS = {
    "TBA (thrombin, G-quadruplex)": "GGTTGGTGTGGTTGG",
    "AS1411 (nucleolin, G-quadruplex)": "GGTGGTGGTGGTTGTGGTGGTGGTGG",
    "VR11 (TNF-alpha)": "TGGTGGATGGCGCAGTCGGCGACAA",
    "IL-6 adaptor (PMC11506342)": "GGTGGCAGGAGGACTATTTATTTGCTTTTCT",
    "AIR-3A (IL-6 receptor)": "GGGGAGGCTGTGGTGAGGG",
    "aptTNF-alpha": "GCGCCACTACAGGGGAGCTGCCATTCGAATAGGTGGGCCGC",
}

SCRAMBLE_SEED = 20260816


def scramble(seq: str, seed: int) -> str:
    bases = list(seq)
    random.Random(seed).shuffle(bases)
    return "".join(bases)


def run(checkpoint: str = "evo2_7b", remote: bool = True) -> dict:
    """Score each reference aptamer against its own scramble."""
    # Input comes from the shared causal-model schema, not an evo2-specific one.
    from proto_tools.tools.causal_models.evo2.evo2_score import Evo2ScoringConfig
    from proto_tools.tools.causal_models.shared_data_models import (
        CausalModelScoringInput,
    )

    labels, seqs, kinds = [], [], []
    for i, (name, seq) in enumerate(sorted(REFERENCE_APTAMERS.items())):
        labels += [name, name]
        seqs += [seq, scramble(seq, SCRAMBLE_SEED + i)]
        kinds += ["real", "scrambled"]

    payload = CausalModelScoringInput(sequences=seqs)
    cfg = Evo2ScoringConfig(model_checkpoint=checkpoint)

    if remote:
        from proto_tools.modal import dispatch_to_modal
        result = dispatch_to_modal("evo2-score", payload, cfg, environment="proto-env")
    else:
        from proto_tools.tools.causal_models.evo2.evo2_score import run_evo2_score
        result = run_evo2_score(payload, cfg)

    scores = list(result.scores)
    if len(scores) != len(seqs):
        raise RuntimeError(f"got {len(scores)} scores for {len(seqs)} sequences")

    pairs, rows = [], []
    for i in range(0, len(scores), 2):
        real, scram = scores[i], scores[i + 1]
        r_ppl, s_ppl = float(real.perplexity), float(scram.perplexity)
        pairs.append(r_ppl - s_ppl)
        rows.append({"aptamer": labels[i], "length": len(seqs[i]),
                     "real_perplexity": round(r_ppl, 3),
                     "scrambled_perplexity": round(s_ppl, 3),
                     "real_lower": r_ppl < s_ppl})

    wins = sum(1 for r in rows if r["real_lower"])
    # A model that carried no information about this class would win about half
    # by chance, so a handful of pairs can only ever be suggestive. It is enough
    # to rule the signal out, which is what the decision needs.
    verdict = (
        "no usable signal — Evo2 does not separate real aptamers from scrambles "
        "of the same composition, so its perplexity should not be used to rank "
        "aptamer designs"
        if wins <= len(rows) * 0.6 else
        "possible signal — real aptamers score lower than their scrambles more "
        "often than chance would suggest. Worth a larger test before ranking on "
        "it; this is far too few pairs to rely on"
    )

    return {
        "checkpoint": checkpoint,
        "pairs_tested": len(rows),
        "real_scored_lower": f"{wins}/{len(rows)}",
        "mean_perplexity_difference": round(statistics.mean(pairs), 4),
        "verdict": verdict,
        "detail": rows,
    }


if __name__ == "__main__":
    import json
    out = run(remote="--local" not in sys.argv)
    print(json.dumps(out, indent=2))
