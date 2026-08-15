"""Is this protein an aptamer target at all? Ask its surface.

An aptamer is a polyanion: one negative charge per phosphate, so a 30-mer carries
about thirty. What it binds, overwhelmingly, is a patch of positive charge. That
is why thrombin, nucleolin and lysozyme — all strongly basic — gave up good
aptamers early and easily, and why acidic targets have resisted decades of SELEX.

So before designing anything, fold the target and look for a basic patch. This
does not predict binding; nothing does. It predicts whether binding is *available
to be found*, which is the question that decides whether a SELEX campaign or a
literature search is worth starting.

Structure comes from ESMFold2 on Biohub (no GPU, ~20 s). Patches are scored on
the CA trace: for each lysine or arginine, count the other basic residues within
one aptamer-loop's reach, and subtract the acidic ones, which repel.

STATUS: MEASUREMENT ONLY - THIS METRIC DOES NOT DISCRIMINATE. Calibrated against
proteins whose aptamers are textbook, it fails to rank them above cytokines:

    nucleolin  +3.11        thrombin  +2.28        lysozyme  +1.39
    IL-6       +1.78        TNF-alpha +1.33        IL-10     +0.61

IL-6 outscores lysozyme, and lysozyme - net charge +8, pI about 11 - is reported
as having zero exposed basic residues. The fault is the burial proxy: CA
neighbour density scales with how compact a protein is, so a small globular
domain reads as buried everywhere. It is size-dependent, and therefore not a
measure of surface at all.

The number is kept because the measurement is real and the reference panel makes
it interpretable, but no verdict is issued from it and nothing downstream should
branch on it. The fix is a proper solvent-accessible surface: ESMFold2 already
returns all 37 atoms per residue and fold_cli.py discards everything but the CA,
so full-atom coordinates plus Shrake-Rupley would replace the proxy with the
standard calculation. That is the next step, not a tuning exercise.

Signal peptides are stripped first. They are cleaved before the protein ever
circulates, they do not fold, and leaving them in drags a disordered 20-residue
tail through the structure and into the patch statistics.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

BASIC = {"K", "R"}
ACIDIC = {"D", "E"}

# Reach of a patch, in angstroms on the CA trace. An aptamer stem-loop presents a
# binding face roughly this wide, so charges further apart than this are not
# contacted by the same aptamer even though both are on the surface.
PATCH_RADIUS = 12.0

# Burial proxy. Counting CA neighbours within this radius separates surface from
# core when only backbone coordinates are available; at or above the neighbour
# count below, a residue is buried and its charge is unavailable to a ligand.
BURIAL_RADIUS = 10.0
BURIED_NEIGHBOURS = 18.0

# Sibling projects, overridable by environment so this is not tied to one
# machine's directory layout. ESM_PY needs the esm SDK; ORTHOLOG_DIR holds the
# UniProt helpers.
_BASE = Path(os.environ.get("ASCEND_ROOT", Path.home() / "Documents" / "AscendBiosci"))
FOLD_PY = Path(os.environ.get("ESM_PY", _BASE / "esm-sandbox" / "venv" / "bin" / "python"))
FOLD_CLI = Path(os.environ.get("FOLD_CLI", _BASE / "target-feasibility" / "sources" / "fold_cli.py"))
HACKATHON = Path(os.environ.get("ORTHOLOG_DIR", _BASE / "hackathon-mcp"))


def _sequence(gene: str) -> tuple[str, str, int]:
    """(accession, mature sequence, residues trimmed) for a human gene."""
    sys.path.insert(0, str(HACKATHON))
    from ortholog_server import SPECIES, _fetch

    rec = _fetch(gene, SPECIES["human"][1])
    if not rec:
        raise LookupError(f"no human UniProt entry for {gene}")

    sys.path.insert(0, str(FOLD_CLI.parent))
    import accessibility

    seq = rec["sequence"]
    entry = accessibility._query(gene)
    trimmed = 0
    if entry:
        for f in accessibility._features(entry, "Signal"):
            end = f.get("end")
            if end and 0 < end < len(seq):
                seq, trimmed = seq[end:], end
                break
    return rec["accession"], seq, trimmed


def assess(gene: str) -> dict:
    """Fold the target and score its best basic patch."""
    import numpy as np

    accession, seq, trimmed = _sequence(gene)
    proc = subprocess.run([str(FOLD_PY), str(FOLD_CLI), seq],
                          capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        return {"gene": gene, "error": f"fold failed: {proc.stderr.strip()[-200:]}"}
    fold = json.loads(proc.stdout.strip().splitlines()[-1])

    ca = np.asarray(fold["ca"], dtype=float)
    charge = np.array([1.0 if a in BASIC else -1.0 if a in ACIDIC else 0.0
                       for a in seq])

    # Pairwise CA distances, then a charge sum inside the patch radius. Acidic
    # residues count against the patch rather than merely not counting for it:
    # a lysine flanked by glutamates presents no usable positive surface.
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)

    # Weight each charge by how exposed it is. Counting buried residues was the
    # first version's mistake and it made the metric meaningless: a basic cluster
    # packed into the hydrophobic core scores the same as one presented on the
    # surface, and an aptamer can only touch the second. CA neighbour density is
    # the standard cheap proxy for burial when only backbone coordinates exist.
    neighbours = (d <= BURIAL_RADIUS).sum(axis=1) - 1
    exposure = np.clip((BURIED_NEIGHBOURS - neighbours) / BURIED_NEIGHBOURS, 0.0, 1.0)

    within = d <= PATCH_RADIUS
    patch = within @ (charge * exposure)

    basic_idx = [i for i, a in enumerate(seq) if a in BASIC]
    exposed_basic = int(sum(1 for i in basic_idx if exposure[i] > 0.5))
    best = int(np.argmax(patch))

    return {
        "gene": gene,
        "accession": accession,
        "length": len(seq),
        "signal_peptide_trimmed": trimmed,
        "ptm": fold["ptm"],
        "net_charge": float(charge.sum()),
        "n_basic": len(basic_idx),
        "n_basic_exposed": exposed_basic,
        "best_patch_score": round(float(patch[best]), 2),
        "best_patch_residue": f"{seq[best]}{best + 1 + trimmed}",
        "patch_scores": [round(float(v), 2) for v in patch],
        "reference_panel": REFERENCE,
        "caveat": ("Uncalibrated. This score does not rank known aptamer targets "
                   "above known-hard ones - see reference_panel and the module "
                   "docstring. Report it as a measurement, never as a verdict, "
                   "and do not use it to accept or reject a target."),
    }


# Measured on this machine, same code path, so a new target's score can be read
# against something. Nucleolin, thrombin and lysozyme all have well-known
# aptamers; the spread between them and the cytokines is the point, and it is
# too small to act on.
REFERENCE = {"nucleolin": 3.11, "thrombin": 2.28, "lysozyme": 1.39,
             "IL-6": 1.78, "TNF-alpha": 1.33, "IL-10": 0.61}


if __name__ == "__main__":
    genes = sys.argv[1:] or ["IL6", "TNF", "IL10"]
    for g in genes:
        d = assess(g)
        if "error" in d:
            print(f"{g}: {d['error']}")
            continue
        print(f"{d['gene']:5s} {d['accession']}  {d['length']:3d} aa  pTM {d['ptm']}  "
              f"net charge {d['net_charge']:+.0f}  "
              f"best patch {d['best_patch_score']:+.1f} at {d['best_patch_residue']}")
        print(f"      reference: " + "  ".join(f"{k} {v:+.2f}"
                                                for k, v in REFERENCE.items()))
