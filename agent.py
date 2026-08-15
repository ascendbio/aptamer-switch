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
import literature  # noqa: E402
import thermo  # noqa: E402

MODEL = "claude-opus-5"
SERVER = "aptamer"
OUT = Path(__file__).parent / "out"

SYSTEM = """You design aptamer switches for continuous electrochemical biosensors \
(E-AB). Your reader runs the wet lab and will synthesise what you recommend.

Use the tools before judging. Find a published parent aptamer first — de novo \
aptamer invention has no validated computational method, so without a parent \
there is nothing honest to build on, and saying so is the right answer.

Then write a short design memo:

- One line up front: what was designed, from which parent, and whether it is \
worth synthesising.
- The parent you used, its citation and reported Kd, and how many independent \
papers report the same sequence.
- The funnel: library size, how many survived each criterion, what killed the rest.
- The single biggest risk, named plainly.

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
    "chemistry, reported Kd, source papers, and how many independent papers "
    "report it identically. Pass the biomarker as it would be written in a paper, "
    "e.g. 'IL-6' or 'TNF-alpha'.",
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
    "parent's reported Kd in nM (from find_parents) so apparent Kd and clinical "
    "reach are computed from real data rather than a guess.",
    {"target": str, "sequence": str, "core": str, "kd_nM": float},
)
async def design_plate(args: dict) -> dict:
    seq = args["sequence"].strip().upper()
    try:
        start, end = (int(x) for x in args.get("core", "").replace(",", "-").split("-")[:2])
    except (ValueError, TypeError):
        start, end = 1, max(4, len(seq) // 2)

    kd_nM = float(args.get("kd_nM") or 10.0)
    result = design.run(seq, (start, end), kd_nM * 1e-9, target=args["target"])
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
        "kd_intrinsic_nM": round(result["kd_intrinsic_nM"], 2),
        "kd_apparent_nM_range": [min(result["kd_apparent_nM"]),
                                 max(result["kd_apparent_nM"])],
        "position_check": result["position_check"],
        "order_file": art["csv"],
        "run_dir": art.get("run_dir"),
        "top_candidates": top,
    })


TOOLS = [find_parents, assess_parent, design_plate]
TOOL_NAMES = [f"mcp__{SERVER}__{t.name}" for t in TOOLS]

LABELS = {
    "find_parents": "Searching the literature for published aptamers",
    "assess_parent": "Folding the parent aptamer",
    "design_plate": "Designing, scoring and laying out the plate",
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

    if "parents" in d:
        n = len(d["parents"])
        if not n:
            return f"no published sequence found across {d.get('n_papers', 0)} papers"
        best = d["parents"][0]
        return (f"{n} parent{'s' if n != 1 else ''} from {d.get('n_papers', 0)} papers · "
                f"best {best['length']} nt {best['chemistry']}"
                + (f", Kd {best['reported_kd'][0]}" if best.get("reported_kd") else "")
                + f", {best['corroborating_papers']} paper(s)")

    if "opening_energy_trustworthy" in d:
        kind = "G-quadruplex" if d["core_is_quadruplex"] else "Watson-Crick"
        trust = "modellable" if d["opening_energy_trustworthy"] else "opening energy NOT reliable"
        return f"{kind} fold, MFE {d['mfe']} kcal/mol · {trust}"

    if "library_size" in d:
        if not d.get("wells"):
            blockers = ", ".join(d.get("universal_blockers", [])) or "no candidate passed"
            return f"{d['library_size']:,} candidates, 0 usable — {blockers}"
        lo, hi = d["kd_apparent_nM_range"]
        return (f"{d['library_size']:,} → {d['in_switching_window']:,} in window → "
                f"{d['passing_all_criteria']:,} pass → {d['test_wells']} wells · "
                f"Kd_app {lo:.0f}-{hi:.0f} nM")
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
        max_turns=14,
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
