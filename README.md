# Aptamer switch design for continuous biosensors

Name a biomarker. The agent finds published aptamers for it in the literature,
folds them, builds a library of structure-switching variants, scores every
candidate on switching thermodynamics, specificity and manufacturability, and
returns 96 synthesis-ready wells with a vendor order file.

It does not predict binding affinity. For aptamer-protein pairs there is no
validated predictor the author is aware of: the field establishes affinity by
measurement, and published aptamers frequently carry no reported Kd at all. This
is a claim about aptamers specifically, not about affinity prediction generally -
free-energy perturbation and models such as Boltz-2 do make useful predictions
for protein-small-molecule binding, and Boltz-2 is one command away in the same
Proto toolkit used here. It is trained on small-molecule ligands, so it is not
applicable to a 40-mer of DNA.

The plate is therefore designed to *find* the optimum in one wet-lab round rather
than to guess it.

## Why a plate and not a prediction

The sensor is an electrochemical aptamer-based (E-AB) device: aptamer with a
5' thiol on gold and a 3' methylene blue reporter. Signal comes from a
binding-induced **conformational change**, so a high-affinity aptamer that does
not switch reports nothing.

That makes folding free energy the design variable — which ViennaRNA computes —
rather than affinity, which for this class of molecule has to be measured. But
switching costs affinity: the target pays the core-opening energy itself, so

    Kd_app = Kd_intrinsic x (1 + e^(dG_open / RT))

Every kcal/mol of gain multiplies apparent Kd by about five. For a cytokine
circulating at pg/mL against a nanomolar aptamer there is almost nothing to
spend, so the usable window is a couple of kcal/mol wide — narrower than
ViennaRNA's own error bar. Hence 96 wells tiling the window instead of one
predicted sequence.

## Running it

Python 3.12 (ViennaRNA lacks a 3.13 wheel on some platforms).

```bash
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt

./run              # web UI on http://localhost:7860
./run "IL-6"       # same agent, terminal output
```

`run` finds an interpreter automatically; set `APTAMER_PY` to override.

The agent authenticates through your local Claude Code session, so no
`ANTHROPIC_API_KEY` is needed and nothing is billed to the API. Literature search
needs the `paperclip` CLI on PATH.

To skip retrieval and design from a known parent:

```bash
./venv/bin/python sources/design.py
```

## What each piece does

| Module | Role |
|---|---|
| `literature.py` | Full-text **grep** for sequences, then deterministic regex extraction requiring the target named in the same paragraph. Topic search returns SELEX reviews; the thing wanted is a literal ACGT string. The LLM reader is an optional fallback. |
| `thermo.py` | Folding, opening energy, apparent Kd. DNA parameters at 37 C. |
| `generate.py` | Two architectures: a competing tail (for quadruplex parents) and intrinsic destabilisation (for Watson-Crick ones). |
| `score.py` | Switching, specificity, manufacturability. Hard filters first, then rank. |
| `design.py` | Orchestrates, tiles the window, lays out the plate. |
| `plate.py` | 96-well layout, controls, vendor CSV. |
| `store.py` | Per-run archive and a GPU result cache keyed on input hash. |
| `complex_cli.py` | Optional: aptamer:protein complex prediction via Proto/Modal. |

## Known limitations — read before trusting output

**The binding core is an assumption.** No paper maps where these cytokines
contact their aptamers. Every ddG is computed against an assumed span, recorded
as `core_assumed` in each run's `manifest.json`.

**Retrieval attributes by proximity, not by reading.** A sequence is accepted
when the target is named in the same paragraph. That is far stricter than the
document-level co-occurrence used earlier — which returned anti-HIV-integrase
aptamers for an IL-6 query, and an anti-IL-6-*receptor* aptamer as though it
bound IL-6 — but it is still not comprehension. A regex over fifty papers also
matches primers and linkers, so candidates are ranked by how many independent
papers print the same sequence and only the best ten are returned. Check that a
candidate's paper is really about your target before designing from it.

**Most parents have no published Kd.** Affinity-derived output is then omitted
rather than estimated. Supplying a Kd requires naming its source.

**Several thresholds are judgements, not measurements** — the dimer margin, the
specificity margin, the G-run limit. `score.py`'s `__main__` prints the dimer
threshold's sensitivity: it moves the pass count from 182 to 1,025.

**Structure prediction does not work for this.** OpenDDE scored a *scrambled*
aptamer higher than the real one against IL-6 (ipTM 0.676 vs 0.580), so its
contacts cannot define a binding core. AF3-class models learn protein-DNA from
transcription factors on duplex DNA, which does not transfer to folded
single-stranded aptamers.

**`target_surface.py` fails calibration** and is deliberately not wired in: it
ranks IL-6 above lysozyme, a textbook aptamer target.

## Optional: GPU tools via Proto and Modal

```bash
proto-tools deploy --create-env --env proto-env
proto-tools deploy --apps opendde --env proto-env
```

On an Intel Mac, `proto-tools` needs `numba==0.62.*` (later versions pull an
`llvmlite` with no macOS x86_64 wheel), and nothing in its GPU half runs
locally — deploy-then-dispatch is the only path. Results are cached on an input
hash; each call is billed.
