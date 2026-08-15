"""Predict an aptamer:protein complex with OpenDDE, and check whether to believe it.

Run in the proto-sandbox venv, which has proto-tools and dispatches the GPU work
to Modal. Kept out of the design environment for the usual reason: proto-tools
drags in torch, rdkit, jupyter and a pinned numba, and mixing that into the
venv the plate pipeline runs on is how the MCP server broke.

The point of this is the binding core. Every ddG in the plate is computed against
an assumed core span, because no paper maps where the cytokine actually contacts
its aptamer. A predicted complex would replace that assumption with a model.

Whether it can is an open question, and this asks it properly. AF3-class models
learn protein-DNA overwhelmingly from transcription factors bound to duplex DNA,
not from single-stranded aptamers folded into tertiary structures, so the prior
is poor. Every real complex is therefore run alongside a scrambled-aptamer
control of identical length and base composition. If the model scores the two
alike, its interface is not sequence-specific and the contacts mean nothing, no
matter how confident the pLDDT looks. That comparison is the result; ipTM on its
own is not.

Requires the tool to be deployed to your Modal workspace first, once:

    proto-tools deploy --create-env --env proto-env
    proto-tools deploy --apps opendde --env proto-env

Results are cached on a hash of the inputs. Asking the same question twice costs
nothing the second time; pass --force to override and pay for a fresh run.

Usage:
    <proto-sandbox>/venv/bin/python complex_cli.py <PROTEIN_SEQ> <APTAMER_SEQ> [SEED] [--force]
Output:
    JSON with per-complex metrics for the real and scrambled pairings.
"""

from __future__ import annotations

import json
import pathlib
import random
import sys

# Same seed every run so the control is reproducible; a control that changes
# between runs cannot be compared against anything.
SCRAMBLE_SEED = 20260816


def scramble(seq: str, seed: int = SCRAMBLE_SEED) -> str:
    """Same length, same base composition, no sequence. The negative control."""
    bases = list(seq)
    random.Random(seed).shuffle(bases)
    return "".join(bases)


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit("usage: complex_cli.py <PROTEIN_SEQ> <APTAMER_SEQ> [SEED]")
    protein = sys.argv[1].strip().upper()
    aptamer = sys.argv[2].strip().upper()
    seed = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 0

    control = scramble(aptamer)
    pairings = [("real", aptamer), ("scrambled", control)]

    # Cache before dispatch. Every call to this tool is billed GPU time, so a
    # repeated question must never become a repeated job.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import store

    out_dir = pathlib.Path(__file__).resolve().parent.parent / "out"
    key = store.cache_key(tool="opendde-prediction", protein=protein,
                          aptamer=aptamer, control=control, seed=seed,
                          num_samples=1)
    force = "--force" in sys.argv
    if not force:
        hit = store.cache_get(out_dir, key)
        if hit is not None:
            print(json.dumps(hit, indent=2, default=str))
            return

    from proto_tools.modal import dispatch_to_modal
    from proto_tools.tools.structure_prediction.opendde.opendde import (
        OpenDDEConfig,
        OpenDDEInput,
    )

    # Both complexes go in one call: same weights, same MSA settings, same GPU
    # allocation, so the only difference between them is the aptamer sequence.
    complexes = [
        {"chains": [{"sequence": protein, "entity_type": "protein"},
                    {"sequence": dna, "entity_type": "dna"}]}
        for _, dna in pairings
    ]

    # dispatch_to_modal, not run_opendde. The run_*() functions execute LOCALLY:
    # they build a standalone env under ~/.proto and import torch in-process. On
    # an Intel Mac that cannot even install (no macOS x86_64 torch wheels past
    # 2.2), and on any machine without CUDA a diffusion structure model is not
    # going to finish. The GPU half of proto-tools is deploy-then-dispatch only.
    result = dispatch_to_modal(
        "opendde-prediction",
        OpenDDEInput(complexes=complexes),
        OpenDDEConfig(seed=seed, num_samples=1),
        environment="proto-env",
    )

    out = {
        "success": getattr(result, "success", None),
        "errors": getattr(result, "errors", None),
        "aptamer": aptamer,
        "scrambled_control": control,
        "per_complex": {},
    }

    # Metrics live on each Structure, not on the Output. OpenDDEOutput carries
    # only `structures`; reading result.iptm returns None for every field and
    # looks exactly like a model that declined to answer.
    structures = getattr(result, "structures", []) or []
    for (label, dna), struct in zip(pairings, structures):
        metrics = dict(getattr(struct, "metrics", {}) or {})
        out["per_complex"][label] = {"aptamer": dna, "metrics": metrics}

        # Keep the coordinates. Contact extraction needs them, and a five-minute
        # GPU run should not have to be repeated to answer a follow-up question.
        text = getattr(struct, "structure", None)
        if text:
            ext = getattr(struct, "structure_format", "cif") or "cif"
            pathlib.Path(f"out/complex_{label}.{ext}").write_text(str(text))
            out["per_complex"][label]["file"] = f"out/complex_{label}.{ext}"

    # The comparison is the result. A model that scores a shuffled aptamer as
    # confidently as the real one has learned no sequence specificity, and its
    # contacts cannot be used to define a binding core.
    real = out["per_complex"].get("real", {}).get("metrics", {})
    ctrl = out["per_complex"].get("scrambled", {}).get("metrics", {})
    if real and ctrl:
        out["discrimination"] = {
            k: (None if real.get(k) is None or ctrl.get(k) is None
                else round(float(real[k]) - float(ctrl[k]), 4))
            for k in ("iptm", "ptm", "avg_plddt", "ranking_score")
        }
    out["_cache_key"] = key
    store.cache_put(out_dir, key, out)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
