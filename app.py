"""Chat UI for the aptamer switch design agent.

The agent takes a couple of minutes: a literature grep, a fold, then thirty
seconds of scoring nine thousand candidates. A blank screen for that long reads
as broken, so every tool call streams into the transcript as it happens. The
visible trace of the agent working is the demo, not the memo at the end.

Run:  ./run          then open http://localhost:7860

Authenticates through the local Claude Code session, so no ANTHROPIC_API_KEY is
needed and nothing is billed to the API.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import gradio as gr

from agent import analyse

sys.path.insert(0, str(Path(__file__).parent / "sources"))
import store  # noqa: E402

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

EXAMPLES = ["IL-6", "TNF-alpha", "IL-10", "thrombin", "cortisol"]

INTRO = """### Aptamer switch design for continuous biosensors

Name a biomarker. I search the literature for published aptamers against it,
fold them, and build a library of structure-switching designs — thousands of
variants across tail length, register, mismatch and truncation. Each is scored on
switching thermodynamics, on specificity against its own off-target sites and
self-dimers, and on whether a vendor will synthesise it.

Where a paper reports an affinity I carry it through, and compute the apparent Kd
the switch will have: destabilising an aptamer so it reports costs binding
energy, and that cost is quantifiable.

You get 96 wells tiling the usable design window, with controls, randomised
positions, and a vendor order file — a plate built to locate the optimum in one
wet-lab round."""

NO_DOSE = """**What the sensor can actually see** — not shown.

No paper reports a binding affinity for this parent, and the whole content of a
dose-response curve is *where it sits* on the concentration axis, which comes
from Kd. Drawn from a placeholder it would be a picture of an assumption, so it
is left out rather than filled in.

Everything else on the plate is unaffected: ddG, specificity and dimer margins
are computed from sequence and never touch affinity."""

EXPLAIN = {
    "funnel": """
**Every candidate considered, and what removed each one.**

The interesting number is not 96 — it is the library it came from and the two
criteria that account for nearly all of the loss.

* **library** — every combination of tail length, register, mismatch and linker
* **switching window** — designs where the tail and the fold are close enough in
  energy that target binding can flip the balance. Outside it, the switch is
  stuck open or stuck shut and reports nothing.
* **passes all criteria** — survivors of the specificity and manufacturability
  filters. These are hard filters, not weights: a design that dimerises is not
  redeemed by a good switching score.
* **on the plate** — tiled across the window, with 8 wells kept for controls.

A stage that removes almost everything is describing the *parent*, not choosing
between designs.
""",
    "dose": """
**What fraction of sensors are bound at a given analyte concentration.**

Occupancy is the ceiling on any signal: a sensor cannot report on a molecule that
never binds it. The shaded bands are clinical ranges; the dashed line is a 10%
occupancy floor, which is an **assumed** working limit, not a measured one.

The dashed curve is the parent aptamer; the solid curve is the switch built from
it. The gap between them is the price of switching — the target has to pay the
core-opening energy itself, so a switch always binds more weakly than its parent.

**This panel is blank when no paper reports an affinity for the parent**, which
is the common case. The curve's whole content is where it sits on the
concentration axis, and that position comes from Kd. Drawn from a placeholder it
would be a picture of an assumption.
""",
    "window": """
**The two axes the plate is actually selected on.**

*Horizontal* — ddG, the competition between the tail and the fold. Near zero is
where a sensor can switch. Strongly negative, the tail wins and the aptamer can
never fold; strongly positive, the tail never engages and nothing moves.

*Vertical* — specificity margin: how much the tail prefers its intended site over
the best alternative site in the same molecule. A tail that binds two places
reports a shape change that has nothing to do with the target.

Grey points were filtered out and are left visible on purpose, so the shape of
what was rejected stays legible instead of being cropped away. Orange points are
the wells on the plate.
""",
    "plate": """
**The physical 96-well plate, coloured by ddG.**

Blue wells are controls: the parent with no switch, a scrambled sequence, both
no-switch extremes in duplicate, and a blank. Each answers a question a failed
plate would otherwise leave open — was the aptamer ever any good, does the assay
respond at all, is a flat well flat for the reason we think.

**Position is randomised against ddG on purpose.** Plates have row and column
gradients and edge wells evaporate faster; laying the energy ladder out in plate
order would alias those effects onto the design variable and produce a beautiful
dose-response that is really the edge drying out. The caption reports the check:
observed row spread against what random assignment gives.
""",
}

FOOTER = ("<sub>Evidence: published aptamer sequences via Paperclip full-text grep · "
          "ViennaRNA 2.7.2 folding with DNA parameters at 37 °C · "
          "designed for electrochemical aptamer-based (E-AB) sensors</sub>")


def _figures(target: str, since: float = 0.0) -> tuple:
    """This run's figures, in reading order.

    Found by modification time, not by filename. The agent names its own target
    argument, and it does not use the string the user typed: a query for "IL-6"
    produced files called "IL-6 (19mer, Kd unknown - placeholder)_funnel.png" and
    "IL-6R (AIR-3A parent)_funnel.png". Matching on the typed name therefore
    showed nothing at all — or worse, silently showed figures left over from an
    earlier run, which is a wrong answer rather than a missing one.

    `since` is the moment this run began, so only files it actually wrote qualify.
    """
    def newest(pattern: str) -> str | None:
        hits = [p for p in OUT.glob(pattern) if p.stat().st_mtime >= since]
        return str(max(hits, key=lambda p: p.stat().st_mtime)) if hits else None

    return (newest("*_funnel.png"), newest("*_dose.png"), newest("*_window.png"),
            newest("*_plate.png"), newest("*_plate.csv"), _gallery(since),
            _comparison(since))


BLANK = (None, None, None, None, None, [], None)


def _panels(figs: tuple):
    """Figure outputs plus the visibility of the box each one lives in."""
    funnel, dose, window, plate = figs[0], figs[1], figs[2], figs[3]
    csv, gallery_items, table = figs[4], figs[5], figs[6]
    return (*figs,
            gr.update(visible=bool(funnel)),
            gr.update(visible=bool(dose)),
            gr.update(visible=not dose and bool(plate)),   # explain the absence
            gr.update(visible=bool(window)),
            gr.update(visible=bool(plate)),
            gr.update(visible=bool(csv)),
            # The comparison table earns its place only once there is something
            # to compare; the gallery only once a second parent exists, since
            # with one it merely repeats the panels above.
            gr.update(visible=bool(table)),
            gr.update(visible=len(gallery_items or []) > 4))

COMPARE_COLUMNS = ["parent", "architecture", "library", "in window", "passing",
                   "wells", "blocked by", "best |ddG|"]


def _comparison(since: float) -> list[list]:
    """One row per parent assessed, from the manifests each run already writes.

    Sixteen figures across four candidates is not something a customer can hold
    in their head. The decision they are actually making is which parent to back,
    and that comes down to a handful of numbers per candidate, side by side.
    Read from the archived manifests rather than recomputed, so the table cannot
    drift from the plate it describes.
    """
    runs = OUT / "runs"
    if not runs.exists():
        return []

    rows = []
    for d in sorted(runs.iterdir()):
        manifest = d / "manifest.json"
        if not manifest.exists() or manifest.stat().st_mtime < since:
            continue
        try:
            m = json.loads(manifest.read_text())
        except json.JSONDecodeError:
            continue

        wells = m.get("selected") or 0
        if wells:
            blocked = "-"
        else:
            blockers = m.get("universal_blockers") or []
            reasons = m.get("failure_reasons") or {}
            blocked = ", ".join(blockers) or (
                max(reasons, key=reasons.get) if reasons else "nothing passed")

        rows.append([
            f"{m.get('target', '?')} ({m.get('parent_length', '?')} nt)",
            m.get("architecture", "-"),
            f"{m.get('library_size', 0):,}",
            f"{m.get('in_switching_window', 0):,}",
            f"{m.get('passing_all_criteria', 0):,}",
            wells or "none",
            blocked,
            m.get("best_abs_ddg_selected") if wells else "-",
        ])
    return rows


KIND_LABEL = {"funnel": "library → plate", "dose": "what the sensor can see",
              "window": "switching vs specificity", "plate": "the 96-well plate"}
KIND_ORDER = ["funnel", "window", "dose", "plate"]


def _gallery(since: float) -> list[tuple[str, str]]:
    """Every figure this run produced, grouped by the parent it describes.

    The agent routinely evaluates several parents before recommending one, and
    each attempt used to overwrite the last panel — so the page showed whichever
    candidate happened to be assessed most recently, not the one being
    recommended, and the rejected candidates left no evidence at all. Comparing
    them is the entire point of running more than one.
    """
    found: dict[str, dict[str, str]] = {}
    for path in OUT.glob("*_*.png"):
        if path.stat().st_mtime < since:
            continue
        stem, _, kind = path.stem.rpartition("_")
        if kind not in KIND_LABEL or not stem:
            continue
        found.setdefault(stem, {})[kind] = str(path)

    items: list[tuple[str, str]] = []
    for parent in sorted(found):
        for kind in KIND_ORDER:
            if kind in found[parent]:
                items.append((found[parent][kind], f"{parent} · {KIND_LABEL[kind]}"))
    return items

# How often the page refreshes while a tool is still running.
TICK_SECONDS = 1.0


async def respond(target: str, history: list):
    target = (target or "").strip()
    if not target:
        yield history, "", *_panels(BLANK)
        return

    history = history + [{"role": "user", "content": target}]
    trace: list[str] = []
    memo = ""
    history.append({"role": "assistant", "content": "_Working…_"})
    yield history, "", *_panels(BLANK)

    # The agent is consumed through a queue rather than iterated directly, so the
    # page can keep updating while a tool is still running. A literature search
    # takes ~30s and a design pass ~32s; iterating the stream directly means the
    # transcript freezes for that whole time on the line that started the call,
    # which is indistinguishable from a hang.
    queue: asyncio.Queue = asyncio.Queue()

    async def pump() -> None:
        try:
            async for item in analyse(target):
                await queue.put(item)
        except Exception as exc:               # forwarded, not swallowed
            await queue.put(("error", str(exc)))
        finally:
            await queue.put(None)

    task = asyncio.create_task(pump())
    started = time.monotonic()
    run_began = time.time()          # wall clock, to match file mtimes
    figures = BLANK

    def render(tick: str = "") -> str:
        body = "\n".join(f"`{line}`" for line in trace)
        if tick:
            body += f"\n`{tick}`" if body else f"`{tick}`"
        if memo:
            body += "\n\n---\n\n" + memo
        return body or "_Working…_"

    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=TICK_SECONDS)
            except asyncio.TimeoutError:
                # Nothing new from the agent: refresh the elapsed counter so the
                # page visibly keeps time while the tool works.
                waited = time.monotonic() - started
                history[-1] = {"role": "assistant",
                               "content": render(f"   … working, {waited:.0f}s")}
                yield history, "", *_panels(figures)
                continue

            if item is None:
                break
            kind, content = item
            if kind == "error":
                raise RuntimeError(content)
            if kind == "tool":
                trace.append(f"▸ {content}")
                started = time.monotonic()
            elif kind == "result":
                # Indented under the call it answers, so the transcript reads as
                # action then outcome rather than a flat list of intentions.
                trace.append(f"\u2003 └ {content}")
                started = time.monotonic()
            else:
                memo += content

            history[-1] = {"role": "assistant", "content": render()}
            # Figures appear as soon as the design tool has written them, which
            # is mid-run — the reader gets the plate while the memo is still
            # being written.
            figures = await asyncio.to_thread(_figures, target, run_began)
            yield history, "", *_panels(figures)
    except Exception as exc:
        task.cancel()
        history[-1] = {"role": "assistant",
                       "content": f"{chr(10).join(trace)}\n\n**Failed:** `{exc}`"}
        yield history, "", *_panels(BLANK)
        return

    # Persist the memo and trace. design_plate already archived the numbers, but
    # the agent's written reasoning lived only in this browser tab until now —
    # and the reasoning is the part that explains why the numbers were accepted.
    await asyncio.to_thread(_archive, target, memo, trace)
    yield history, "", *_panels(await asyncio.to_thread(_figures, target, run_began))


def _archive(target: str, memo: str, trace: list[str]) -> None:
    """Write the memo into this target's run directory, or a new one if the
    design tool never got far enough to create one."""
    try:
        run_dir = store.latest(OUT, target) or store.new_run(OUT, target)
        store.save_memo(run_dir, target, memo, trace)
    except Exception:
        pass          # a failed archive must never lose the user their answer


with gr.Blocks(title="Aptamer switch design") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(height=560, show_label=False)
            with gr.Row():
                box = gr.Textbox(placeholder="Biomarker, e.g. IL-6", show_label=False,
                                 scale=8, autofocus=True)
                send = gr.Button("Design plate", variant="primary", scale=1)
            gr.Examples(EXAMPLES, inputs=box, label="Try one")
            order = gr.File(label="Vendor order file (96 wells)", height=90,
                            visible=False)
        with gr.Column(scale=2):
            # Each figure carries its own reading guide, collapsed. The plots
            # answer questions that are not obvious from the axes — what a
            # negative ddG means, why a well can pass on rank and still be
            # rejected — and a reader who has to ask is a reader who will not
            # trust the plate.
            # Each panel is hidden until its figure exists. An empty image frame
            # reads as a broken tool; the reason a figure is absent is
            # information, and belongs where the figure would have been rather
            # than inside a collapsed accordion nobody opens.
            with gr.Column(visible=False) as funnel_box:
                funnel_img = gr.Image(label="From library to plate", height=200)
                with gr.Accordion("What am I looking at?", open=False):
                    gr.Markdown(EXPLAIN["funnel"])

            with gr.Column(visible=False) as dose_box:
                dose_img = gr.Image(label="What the sensor can actually see",
                                    height=250)
                with gr.Accordion("What am I looking at?", open=False):
                    gr.Markdown(EXPLAIN["dose"])
            dose_absent = gr.Markdown(NO_DOSE, visible=False)

            with gr.Column(visible=False) as window_box:
                window_img = gr.Image(label="Switching vs specificity", height=260)
                with gr.Accordion("What am I looking at?", open=False):
                    gr.Markdown(EXPLAIN["window"])

            with gr.Column(visible=False) as plate_box:
                plate_img = gr.Image(label="The plate", height=250)
                with gr.Accordion("What am I looking at?", open=False):
                    gr.Markdown(EXPLAIN["plate"])

    # Every candidate the agent assessed, not only the one it settled on. A
    # rejected parent's funnel is often the more informative figure: it shows
    # which criterion removed the whole library and therefore why that candidate
    # could not work, which is what makes the recommendation checkable.
    with gr.Column(visible=False) as compare_box:
        gr.Markdown("#### Parents assessed this run")
        gr.Markdown(
        "<sub>One row per published parent the agent evaluated. **wells = none** "
        "means that candidate produced no usable plate, and **blocked by** names "
        "the criterion that eliminated its entire library — a property of the "
        "parent sequence, which no design choice can work around. **best |ddG|** "
            "is how close the best selected design sits to switching balance; "
            "far from zero means it was never going to switch.</sub>")
        compare = gr.Dataframe(headers=COMPARE_COLUMNS, wrap=True,
                               interactive=False)

    with gr.Column(visible=False) as gallery_box:
        with gr.Accordion("Every parent evaluated in this run — click any figure "
                          "to enlarge", open=False):
            gr.Markdown(
                "The agent usually assesses several published parents before "
                "recommending one. Captions name the parent; each has up to four "
                "figures. A parent showing only a funnel and a window plot "
                "produced **no usable plate** — read its funnel to see which "
                "criterion eliminated the whole library.")
            gallery = gr.Gallery(label=None, columns=2, height=520,
                                 allow_preview=True, object_fit="contain")
    gr.Markdown(FOOTER)

    for trigger in (box.submit, send.click):
        trigger(respond, [box, chat],
                [chat, box, funnel_img, dose_img, window_img, plate_img, order,
                 gallery, compare,
                 funnel_box, dose_box, dose_absent, window_box, plate_box,
                 order, compare_box, gallery_box])


if __name__ == "__main__":
    # System fonts only: the Soft theme's display face renders a capital E as a
    # curved epsilon, and it avoids a Google Fonts fetch on conference wifi.
    demo.launch(theme=gr.themes.Soft(
        font=["system-ui", "-apple-system", "Helvetica Neue", "Arial", "sans-serif"],
        font_mono=["ui-monospace", "SF Mono", "Menlo", "Consolas", "monospace"],
    ))
