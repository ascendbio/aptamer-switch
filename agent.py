"""Aptamer biosensor design agent.

Answers "give me 96 candidates I can synthesise for a continuous sensor against
this biomarker" — starting from nothing but the biomarker's name.

Three tools, and the agent chooses the order:

    find_parents    what aptamers has anyone published for this target?
    assess_parent   what does the fold look like, and can we model it?
    design_plate    generate, score, rank, tile, lay out, export

The middle step is not decoration. Whether the parent is a G-quadruplex decides
which design lever is even available, and it is the one thing that cannot be
assumed from the sequence by eye.

Built on the Claude Agent SDK, so it authenticates through the local Claude Code
session — no API key, nothing billed. Claude Code's own tools are switched off;
this agent gets exactly these three.

CLI:  python agent.py "IL-6"
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import AsyncIterator

import anyio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
    query,
    tool,
)

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "sources"))
import design  # noqa: E402
import hedged  # noqa: E402
import ledger  # noqa: E402
import plate  # noqa: E402
import sensitivity  # noqa: E402
import store  # noqa: E402

REJECTED_COUNT = ledger.REJECTED
import literature  # noqa: E402
import thermo  # noqa: E402

MODEL = "claude-opus-5"
SERVER = "aptamer"
OUT = Path(__file__).parent / "out"
# Interpreter that has proto-tools; overridable so this is not tied to one machine.
PROTO_PY = Path(os.environ.get(
    "PROTO_PY", Path.home() / "Documents" / "AscendBiosci" / "proto-sandbox"
    / "venv" / "bin" / "python"))

SYSTEM = """You design aptamer switches for continuous electrochemical biosensors \
(E-AB). Your reader runs the wet lab and will synthesise what you recommend.

Use the tools before judging. Find a published parent aptamer first — de novo \
aptamer invention has no validated computational method, so without a parent \
there is nothing honest to build on, and saying so is the right answer.

Required sequence once a plate exists. Do all four before writing anything:

  1. validate_plate          — independent engine
  2. test_core_sensitivity   — does the plate survive the core assumption?
  3. build_hedged_plate      — call this whenever designs_selected_under_every_core
                               is 0, null, or small. That is the usual outcome and
                               it is the whole point: it converts "do not order"
                               into a plate the lab can actually synthesise.
  4. read_ledger             — what the design rests on

Skipping step 3 after a failed sensitivity test leaves the reader with a
diagnosis and no deliverable, which is the one outcome this tool exists to avoid.

Then write a short design memo:

- One line up front: what was designed, from which parent, and whether it is \
worth synthesising.
- The parent you used, its citation and reported Kd, and how many independent \
papers report the same sequence.
- The funnel: library size, how many survived each criterion, what killed the rest.
- The single biggest risk, named plainly.

After building a plate: validate it with the independent engine, test it against
the core assumption, and read the ledger. Report all three. Name at least one external model that was tried and rejected, and why — a design
presented without its failures reads as more certain than it is.

Recommend the hedged plate by name and file path when you built one, say what
reading it will settle, and end with the single caveat that most affects how it
should be read. Never end without a plate the lab can order. A plate filtered on a dimer criterion that only one
engine supports is not ready to synthesise.

Three things to be precise about, because they are where this analysis goes wrong.

Concentration first: cytokines circulate at pg/mL, and aptamer Kd values are \
nanomolar. Always state which part of the clinical range the sensor can actually \
reach, and say plainly when baseline monitoring is out of reach — it usually is.

Second, switching costs affinity. The target pays the core-opening energy itself, \
so an apparent Kd is always worse than the parent's. Quote both.

Third, if the parent is a G-quadruplex, ViennaRNA's hard constraints are silently \
ignored inside it and the opening energy is not trustworthy. The tool reports \
this. When it does, say the design rests on the competing-tail calculation instead.

Keep it brief. No preamble."""


def _ok(payload: dict) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(payload, indent=2, default=str)}]}


@tool(
    "find_parents",
    "Find published aptamer sequences for a biomarker. Call this first for any "
    "new target. Greps the full text of the literature for sequences rather than "
    "ranking on topic, then reads the hits. Returns each sequence with its "
    "chemistry, source papers, how many independent papers report it "
    "identically, and any reported Kd parsed to nanomolar as kd_nM with the "
    "paper it came from as kd_source. When a parent has kd_nM, pass both "
    "straight through to design_plate so apparent affinity and clinical reach "
    "are computed. affinities_reported_in_papers lists values stated elsewhere "
    "in the same papers; those are not attributed to the sequence, so use one "
    "only after checking the paper says it is that sequence's. Pass the biomarker as it would be written in a paper, "
    "e.g. 'IL-6' or 'TNF-alpha'. If the result carries search_failed, the "
    "extraction service failed and an empty parent list is NOT evidence that no "
    "aptamer exists - say so and retry rather than concluding absence.",
    {"target": str},
)
async def find_parents(args: dict) -> dict:
    return _ok(literature.find_parents(args["target"]))


@tool(
    "assess_parent",
    "Fold a candidate parent aptamer and report whether it can be modelled. "
    "Returns the secondary structure, folding energy, whether the binding core "
    "forms a G-quadruplex, and whether the opening-energy calculation is "
    "trustworthy for it. Pass the core as 1-based 'start-end' naming the residues "
    "the target must contact; if unknown, use the first half of the sequence and "
    "say that it is an assumption.",
    {"sequence": str, "core": str},
)
async def assess_parent(args: dict) -> dict:
    seq = args["sequence"].strip().upper()
    try:
        start, end = (int(x) for x in args.get("core", "").replace(",", "-").split("-")[:2])
    except (ValueError, TypeError):
        start, end = 1, max(4, len(seq) // 2)
    f = thermo.fold(seq, (start, end))
    return _ok({
        "sequence": seq, "length": len(seq), "core": [start, end],
        "structure": f.structure, "mfe": f.mfe,
        "core_is_quadruplex": f.core_is_quadruplex,
        "dg_open": f.dg_open, "dg_open_watson_crick_only": f.dg_open_wc,
        "opening_energy_trustworthy": f.trustworthy,
        "note": ("Core forms a G-quadruplex. ViennaRNA ignores hard constraints "
                 "inside quadruplexes, so dg_open is unreliable; design must use "
                 "the competing-tail route." if not f.trustworthy else
                 "Watson-Crick fold — opening energy is reliable."),
    })


@tool(
    "design_plate",
    "Design the full 96-well plate from a parent aptamer: build the variant "
    "library, score every candidate on switching, specificity and "
    "manufacturability, rank them, tile the survivors across the switching "
    "window, randomise well positions against the design variable, and export a "
    "vendor-ready order file plus four figures. Takes about 30 seconds. Pass the "
    "parent's reported Kd in nM ONLY if a paper reports it for this exact "
    "sequence and this exact target, together with kd_source naming that paper. "
    "Leave kd_nM empty otherwise: affinity-derived output is then omitted rather "
    "than invented. Do not borrow a Kd from a related aptamer - a receptor "
    "binder's Kd is not the ligand binder's.",
    {"target": str, "sequence": str, "core": str, "kd_nM": float,
     "kd_source": str},
)
async def design_plate(args: dict) -> dict:
    seq = args["sequence"].strip().upper()
    try:
        start, end = (int(x) for x in args.get("core", "").replace(",", "-").split("-")[:2])
    except (ValueError, TypeError):
        start, end = 1, max(4, len(seq) // 2)

    # A missing Kd is a normal, common state and must stay missing. Defaulting
    # it to a plausible number silently manufactures affinity claims for a
    # parent that has none.
    raw_kd = args.get("kd_nM")
    kd_M = float(raw_kd) * 1e-9 if raw_kd else None
    kd_source = (args.get("kd_source") or "").strip()
    # Do not manufacture the provenance. An earlier version filled a missing
    # kd_source with the string "reported for this exact sequence", which made
    # the attribution check pass by inventing the very attribution it existed to
    # demand — the same failure as the borrowed Kd, one level up.
    if kd_M and not kd_source:
        return _ok({"error": "kd_nM was given without kd_source. Supply the paper "
                             "reporting that Kd for this exact sequence and target, "
                             "or omit kd_nM and affinity-derived output will be "
                             "omitted rather than estimated."})
    result = design.run(seq, (start, end), kd_M, target=args["target"],
                        kd_source=kd_source)
    art = design.artifacts(result, OUT)

    if not result["selected"]:
        return _ok({
            "target": result["target"], "wells": 0,
            "library_size": result["library_size"],
            "in_switching_window": result["in_window"],
            "passing_all_criteria": 0,
            "failure_reasons": result["failure_reasons"],
            "universal_blockers": result["universal_blockers"],
            "diagnosis": result["diagnosis"],
            "figures": {k: v for k, v in art.items()},
            "run_dir": art.get("run_dir"),
            "advice": "Do not retry with a different core - the blocker is a "
                      "property of the parent sequence. Report it and recommend "
                      "a different parent or architecture.",
        })

    top = [{k: r[k] for k in ("rank", "name", "sequence", "dd_g",
                              "specificity_margin", "kd_apparent_nM")}
           for r in result["rows"] if r["name"] in result["picked_names"]][:8]
    return _ok({
        "target": result["target"],
        "library_size": result["library_size"],
        "in_switching_window": result["in_window"],
        "passing_all_criteria": result["passing"],
        "wells": 96,
        "test_wells": result["selected"],
        "failure_reasons": result["failure_reasons"],
        "kd_intrinsic_nM": (round(result["kd_intrinsic_nM"], 2)
                            if result["kd_intrinsic_nM"] else None),
        "kd_source": result["kd_source"],
        "kd_apparent_nM_range": ([min(result["kd_apparent_nM"]),
                                  max(result["kd_apparent_nM"])]
                                 if result["kd_apparent_nM"] else None),
        "position_check": result["position_check"],
        "order_file": art["csv"],
        "run_dir": art.get("run_dir"),
        "top_candidates": top,
    })


@tool(
    "validate_plate",
    "Independently re-score the finished plate with Primer3, which computes "
    "hairpin and homodimer free energy from the SantaLucia parameters rather "
    "than ViennaRNA's. Reports rank correlation between the two engines and any "
    "wells where they disagree about whether a design is sticky. Call after "
    "design_plate. Runs locally on CPU and costs nothing. Agreement corroborates "
    "the dimer scoring the plate was filtered on; disagreement means one engine "
    "is wrong and the plate should not be ordered until it is known which.",
    {"plate_csv": str},
)
async def validate_plate(args: dict) -> dict:
    path = (args.get("plate_csv") or "").strip()
    csv_path = Path(path) if path else (OUT / "IL-6_plate.csv")
    if not csv_path.exists():
        return _ok({"error": f"no plate at {csv_path}; run design_plate first"})

    # Optional dependency, and its absence is a normal state on a fresh clone.
    # Say what is missing and that the plate is unaffected, rather than surfacing
    # a subprocess error that reads like the validation found something wrong.
    if not PROTO_PY.exists():
        return _ok({
            "validation_unavailable": True,
            "reason": f"proto-tools interpreter not found at {PROTO_PY}",
            "how_to_enable": "install proto-tools in a separate venv and point "
                             "PROTO_PY at its interpreter; it is CPU-only and "
                             "costs nothing to run",
            "note": "the plate itself is unaffected — this is an independent "
                    "second opinion on the thermodynamics, not a step in "
                    "building it",
        })

    # Primer3 lives in the proto-tools environment, which carries torch, rdkit
    # and a pinned numba. Shelling out keeps that out of the agent's venv - the
    # collision that broke the MCP server earlier came from merging exactly this
    # kind of dependency tree.
    proc = await anyio.to_thread.run_sync(
        lambda: subprocess.run([str(PROTO_PY), str(Path(__file__).parent / "sources"
                                                  / "crosscheck.py"), str(csv_path)],
                               capture_output=True, text=True, timeout=900))
    if proc.returncode != 0:
        return _ok({"error": f"cross-check failed: {proc.stderr.strip()[-300:]}"})
    try:
        payload = json.loads(proc.stdout[proc.stdout.index("{"):])
    except (ValueError, json.JSONDecodeError):
        return _ok({"error": "cross-check produced no parsable result"})
    payload["engines"] = "ViennaRNA (Mathews 2004) vs Primer3 (SantaLucia)"
    return _ok(payload)


@tool(
    "test_core_sensitivity",
    "Re-run the whole design under several plausible binding cores and report "
    "which candidates reach the plate under all of them. The core is an "
    "assumption for any parent whose epitope is unmapped, and every ddG rests on "
    "it. Call after design_plate. Takes about two minutes because the design "
    "runs once per core. Designs selected under every core do not depend on the "
    "guess; if the cores share none, the plate is an artefact of the assumption "
    "and should not be ordered on it.",
    {"sequence": str, "target": str},
)
async def test_core_sensitivity(args: dict) -> dict:
    seq = (args.get("sequence") or "").strip().upper()
    if len(seq) < 16:
        return _ok({"error": "need the parent sequence, at least 16 nt"})
    result = await anyio.to_thread.run_sync(
        lambda: sensitivity.run(seq, target=args.get("target") or "target"))
    result.pop("robust_sequences", None)          # too long for a tool result
    return _ok(result)


@tool(
    "build_hedged_plate",
    "Build one orderable 96-well plate that splits its wells across the binding "
    "core hypotheses which actually produce designs, instead of betting all of "
    "them on one. Call this when test_core_sensitivity shows the cores share few "
    "or no designs. The lab still gets a plate they can synthesise, and reading "
    "it answers two questions at once: which switches work, and which core was "
    "right. Takes about two minutes. Returns the order file path.",
    {"sequence": str, "target": str},
)
async def build_hedged_plate(args: dict) -> dict:
    seq = (args.get("sequence") or "").strip().upper()
    if len(seq) < 16:
        return _ok({"error": "need the parent sequence, at least 16 nt"})

    result = await anyio.to_thread.run_sync(
        lambda: hedged.run(seq, target=args.get("target") or "target"))
    if result.get("error"):
        return _ok({"error": result["error"]})

    OUT.mkdir(exist_ok=True)
    safe = (args.get("target") or "target").replace("/", "_")
    csv_path = plate.write_order(result["wells"], OUT / f"{safe}_hedged_plate.csv")
    return _ok({
        "target": result["target"],
        "wells": len(result["wells"]),
        "core_hypotheses": result["hypotheses"],
        "wells_per_hypothesis": result["wells_per_hypothesis"],
        "control_wells": result["control_wells"],
        "position_check": result["position_check"],
        "how_to_read_it": result["reading"],
        "work_note": result["work_note"],
        "designs_available": result["designs_available"],
        "order_file": str(csv_path),
    })


@tool(
    "read_ledger",
    "Return what the most recent plate rests on: which numbers were measured, "
    "which are chosen thresholds with their values, and which external models "
    "were tried and rejected against a control. Call last, before writing the "
    "memo, and use it to state the plate's assumptions in your own words rather "
    "than presenting the design as settled.",
    {},
)
async def read_ledger(args: dict) -> dict:
    latest = store.latest_run(OUT)
    if latest is None:
        return _ok({"error": "no run archived yet; design a plate first"})
    return _ok({
        "run": latest.name,
        "ledger_markdown": ledger.build(latest / "manifest.json"),
    })


TOOLS = [find_parents, assess_parent, design_plate, validate_plate,
         test_core_sensitivity, build_hedged_plate, read_ledger]
TOOL_NAMES = [f"mcp__{SERVER}__{t.name}" for t in TOOLS]

LABELS = {
    "find_parents": "Searching the literature for published aptamers",
    "assess_parent": "Folding the parent aptamer",
    "design_plate": "Designing, scoring and laying out the plate",
    "validate_plate": "Re-scoring the plate with an independent engine",
    "test_core_sensitivity": "Testing the plate against the core assumption",
    "build_hedged_plate": "Splitting the plate across core hypotheses",
    "read_ledger": "Reading what the plate rests on",
}


def _summarise(payload: str) -> str:
    """One line saying what a tool actually returned.

    The trace is the demo, and a trace that only records what was *called* is
    half a trace: the interesting content is in the answers. Design decisions
    the agent makes downstream — which parent to take, whether to build a plate
    at all — are only legible if the reader can see what came back.
    """
    try:
        d = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(d, dict):
        return ""

    # An error is an outcome. Without this the trace showed the call and then
    # nothing, so seven failed attempts rendered as seven identical lines with
    # no explanation - indistinguishable from the agent spinning.
    if d.get("error"):
        return f"FAILED — {str(d['error'])[:150]}"

    if "parents" in d:
        n = len(d["parents"])
        if d.get("search_failed"):
            # Name the stage. "0 papers unread" was printed when the corpus query
            # itself never ran, which reads as though the papers were checked and
            # found wanting.
            if d.get("search_stage") == "corpus query":
                return ("SEARCH FAILED — the corpus query did not run "
                        f"(retried {d.get('attempts', 1)}x). Nothing was checked; "
                        "this says nothing about whether an aptamer exists")
            return (f"SEARCH FAILED — {d.get('papers_failed', 0)} of "
                    f"{d.get('n_papers', 0)} matched papers could not be read; "
                    f"this is not a negative result")
        if not n:
            failed = d.get("papers_failed", 0)
            suffix = f" ({failed} unread)" if failed else ""
            return (f"no sequence in {d.get('papers_read', d.get('n_papers', 0))} "
                    f"papers read{suffix}")
        best = d["parents"][0]
        line = (f"{n} parent{'s' if n != 1 else ''} from {d.get('n_papers', 0)} "
                f"papers · best {best['length']} nt {best['chemistry']}, "
                f"{best['corroborating_papers']} paper(s)")
        if best.get("kd_nM"):
            line += f", Kd {best['kd_as_written']} ({best['kd_source']})"
        with_kd = sum(1 for p in d["parents"] if p.get("kd_nM"))
        if with_kd:
            line += f" · {with_kd}/{n} with an attributed affinity"
        elif best.get("affinities_reported_in_papers"):
            line += (f" · affinities in its papers: "
                     f"{', '.join(best['affinities_reported_in_papers'][:3])} "
                     f"(not attributed to this sequence)")
        return line

    if "opening_energy_trustworthy" in d:
        kind = "G-quadruplex" if d["core_is_quadruplex"] else "Watson-Crick"
        trust = "modellable" if d["opening_energy_trustworthy"] else "opening energy NOT reliable"
        return f"{kind} fold, MFE {d['mfe']} kcal/mol · {trust}"

    if "core_hypotheses" in d:
        spans = ", ".join(f"{a}-{b}" for a, b in d["core_hypotheses"])
        return (f"{d['wells']} wells split across {len(d['core_hypotheses'])} core "
                f"hypotheses ({spans}), {d['wells_per_hypothesis']} each · "
                f"confounded: {d['position_check']['confounded']} · "
                f"{d.get('work_note', '')}")

    if "ledger_markdown" in d:
        text = d["ledger_markdown"]
        return (f"{text.count(chr(10) + '- ')} entries: measured, assumed, and "
                f"{len(REJECTED_COUNT)} models tested and rejected")

    if "cores_tested" in d:
        every = d.get("designs_selected_under_every_core")
        made = d.get("designs_computed")
        used = d.get("designs_reused_from_cache")
        work = (f" · {made} designs computed"
                + (f", {used} reused" if used else "")) if made is not None else ""
        return (f"{d['cores_producing_a_plate']}/{len(d['cores_tested'])} cores "
                f"yield a plate{work} · {d['designs_selected_under_any_core']} "
                f"designs across them · "
                + (f"{every} survive every core" if every is not None
                   else "too few productive cores to test the assumption"))

    if d.get("validation_unavailable"):
        return "independent validation unavailable (proto-tools not installed)"

    if "homodimer" in d and "wells_compared" in d:
        hd = d["homodimer"]
        return (f"{d['wells_compared']} wells re-scored · homodimer rank "
                f"correlation {hd['spearman']} between ViennaRNA and Primer3 · "
                f"{d.get('n_disagreements', 0)} disagree on stickiness")

    if "library_size" in d:
        if not d.get("wells"):
            blockers = ", ".join(d.get("universal_blockers", [])) or "no candidate passed"
            return f"{d['library_size']:,} candidates, 0 usable — {blockers}"
        line = (f"{d['library_size']:,} → {d['in_switching_window']:,} in window → "
                f"{d['passing_all_criteria']:,} pass → {d['test_wells']} wells")
        # Kd is absent whenever no paper reports one for this exact sequence,
        # which is the usual case. The summary says so rather than unpacking a
        # range that is not there.
        rng = d.get("kd_apparent_nM_range")
        if rng:
            line += f" · Kd_app {rng[0]:.0f}-{rng[1]:.0f} nM"
        else:
            line += " · no published Kd, affinity output omitted"
        return line
    return ""


def _options() -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        model=MODEL,
        system_prompt=SYSTEM,
        mcp_servers={SERVER: create_sdk_mcp_server(SERVER, tools=TOOLS)},
        tools=[],                       # switch off Claude Code's built-ins
        allowed_tools=TOOL_NAMES,
        permission_mode="bypassPermissions",
        effort="high",
        # Seven tools, and the last two matter most: a run that exhausts its
        # turns retrying a failed design never reaches the hedged plate, and
        # stops at a diagnosis the lab cannot act on.
        max_turns=22,
    )


async def analyse(target: str) -> AsyncIterator[tuple[str, str]]:
    """Yield ("tool"|"result"|"text", content) as the agent works."""
    prompt = (f"Design a 96-well plate of aptamer switch candidates for a "
              f"continuous E-AB biosensor against {target}.")
    started: dict[str, float] = {}

    async for message in query(prompt=prompt, options=_options()):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    short = block.name.split("__")[-1]
                    started[block.id] = time.monotonic()
                    arg = ""
                    if isinstance(block.input, dict):
                        arg = (block.input.get("target")
                               or block.input.get("sequence", "")[:24])
                    label = LABELS.get(short, short)
                    yield "tool", f"{label} — {arg}" if arg else label
                elif isinstance(block, TextBlock) and block.text.strip():
                    yield "text", block.text

        # Tool results arrive as a UserMessage carrying ToolResultBlocks. Without
        # reading these the trace shows only intent, never outcome.
        elif isinstance(message, UserMessage):
            for block in getattr(message, "content", []) or []:
                if not isinstance(block, ToolResultBlock):
                    continue
                payload = block.content
                if isinstance(payload, list):
                    payload = "".join(p.get("text", "") for p in payload
                                      if isinstance(p, dict))
                summary = _summarise(payload or "")
                elapsed = time.monotonic() - started.get(block.tool_use_id, 0)
                if summary:
                    took = f"  [{elapsed:.0f}s]" if elapsed and elapsed < 3600 else ""
                    yield "result", summary + took


async def _main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "IL-6"
    async for kind, content in analyse(target):
        if kind == "tool":
            print(f"\n  ▶ {content}", flush=True)
        elif kind == "result":
            print(f"      └ {content}", flush=True)
        else:
            print(f"\n{content}", flush=True)


if __name__ == "__main__":
    anyio.run(_main)
