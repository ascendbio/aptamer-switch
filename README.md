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

Verified from a clean clone on macOS with Python 3.12. ViennaRNA has no 3.13
wheel on some platforms, so 3.12 is not optional.

```bash
git clone https://github.com/ascendbio/aptamer-switch
cd aptamer-switch
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt

./run --help       # usage
./run              # web UI on http://localhost:7860
./run "IL-6"       # same agent, terminal output
```

`run` locates a suitable interpreter itself; set `APTAMER_PY` to choose one.

### Letting teammates try it

```bash
./run --share          # password-protected public link, prints the credentials
```

The tunnel points at **your** machine. The agent still runs locally, on your
Claude Code session, your Paperclip account and your Modal credits, so every
teammate's run spends your quota rather than theirs. The link is therefore
always password protected — `gradio.live` URLs are open to anyone holding them
and live for 72 hours. `APTAMER_PASSWORD` sets the password.

Up to four people can run at once. Each browser session gets its own output
directory under `out/sessions/`, so concurrent runs cannot overwrite each
other's figures — without that, the newest-file lookup hands one user another
user's plate. The limit is deliberately small: every run is a Claude session and
a literature search on the host's account, and Paperclip rate-limits well before
the laptop runs out of CPU.

For a teammate who wants their own quota, cloning the repo and running locally
is better: the design pipeline needs nothing but `requirements.txt`, and the
agent uses whatever Claude session is on their machine.

### What needs what

| | needs | if missing |
|---|---|---|
| design pipeline | nothing beyond `requirements.txt` | — |
| the agent | a local Claude Code session | agent will not start |
| literature search | the `paperclip` CLI on `PATH` | agent reports the search failed, and says so rather than reporting an absence |
| `validate_plate` | proto-tools in a separate venv, `PROTO_PY` pointing at it | reported as unavailable; the plate is unaffected |

No `ANTHROPIC_API_KEY` is required — the agent authenticates through your Claude
Code session and nothing is billed to the API. No GPU is used, and no part of the
design pipeline calls a paid service.

### Running without the agent

The whole design path works offline from a known parent, which is the fastest way
to see what the tool produces:

```bash
./venv/bin/python sources/design.py
```

That writes a 96-well plate, its figures and a run manifest to `out/` in about
30 seconds.

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

## Reproducibility

Run the same query twice and the **computation** is identical. Verified across
separate processes and across `PYTHONHASHSEED` values:

| layer | reproducible | why |
|---|---|---|
| corpus grep | yes | deterministic regex over a fixed corpus |
| sequence extraction | yes | regex, no model in the loop |
| folding and scoring | yes | ViennaRNA is deterministic |
| variant library | yes | enumerated, not sampled |
| plate selection and layout | yes | seeded shuffle, fixed seed |
| **the agent's own choices** | **no** | it is a language model |

The plate is bit-identical between runs: same library size, same pass count, same
96 sequences in the same wells. What varies is which parent the agent picks when
several are returned, the order it calls tools in, and the wording of its memo.

Each run records what it used — parent, core, thresholds, architecture, git
commit — in `out/runs/<target>_<timestamp>/manifest.json`, so any plate can be
regenerated exactly by feeding those values back to `design.run()`, without the
agent in the loop.

## Closing the loop with wet-lab results

The plate is one round of a design-build-test-learn cycle, and the learn step is
where prediction stops mattering. Hand the bench's results back:

```bash
./run "IL-6 — here are the results from out/my_results.csv"
```

The results file needs a well column and a signal column; any header wording
works, and everything else is taken from the plate that was designed.

What it reports, and what it refuses to report:

* **whether ddG actually predicted signal** — tested against a 2,000-permutation
  null. On pure noise it says so rather than finding a story: recentring the next
  window on an artefact would spend a second synthesis run confirming the first
  one's mistake.
* **which core hypothesis responded** — the hedged plate exists to settle this,
  and a hypothesis with no responsive well is eliminated. That is the epitope
  answer, from a plate that was going to be synthesised anyway.
* **where the optimum actually sits**, so the next window is centred on a
  measurement rather than a model.

To try the loop before real data exists, a matching simulated file is kept in
`out/demo/`, refreshed silently whenever a plate is written. Upload it through
the feedback panel like any other results file.

It is written there rather than beside the design output, and never offered in
the interface: the app designs plates, and a file of invented measurements
presented among its outputs would say it produces bench data. `./run
--demo-results` regenerates it explicitly, plus a pure-noise variant.

`DEMO_1_signal.csv` has signal peaking near the predicted optimum with one core
hypothesis responding; `DEMO_2_no_signal.csv` is noise. **Demonstrate the second
one** — a tool that finds a story in noise is worse than no tool, and this one
reports p=0.26 and refuses to move the window. Every row of both files carries
`SYNTHETIC EXAMPLE - not measured` in its own column.

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
