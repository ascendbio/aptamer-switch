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
import sys
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

FOOTER = ("<sub>Evidence: published aptamer sequences via Paperclip full-text grep · "
          "ViennaRNA 2.7.2 folding with DNA parameters at 37 °C · "
          "designed for electrochemical aptamer-based (E-AB) sensors</sub>")


def _figures(target: str) -> tuple:
    """Whatever the agent rendered for this target, in reading order."""
    stem = target.strip().replace("/", "_")
    names = ("funnel", "dose", "window", "plate")
    paths = []
    for n in names:
        p = OUT / f"{stem}_{n}.png"
        paths.append(str(p) if p.exists() else None)
    csv = OUT / f"{stem}_plate.csv"
    paths.append(str(csv) if csv.exists() else None)
    return tuple(paths)


BLANK = (None, None, None, None, None)


async def respond(target: str, history: list):
    target = (target or "").strip()
    if not target:
        yield history, "", *BLANK
        return

    history = history + [{"role": "user", "content": target}]
    trace: list[str] = []
    memo = ""
    history.append({"role": "assistant", "content": "_Working…_"})
    yield history, "", *BLANK

    try:
        async for kind, content in analyse(target):
            if kind == "tool":
                trace.append(f"▸ {content}")
            elif kind == "result":
                # Indented under the call it answers, so the transcript reads as
                # action then outcome rather than a flat list of intentions.
                trace.append(f"\u2003 └ {content}")
            else:
                memo += content
            body = "\n".join(f"`{line}`" for line in trace)
            if memo:
                body += "\n\n---\n\n" + memo
            history[-1] = {"role": "assistant", "content": body or "_Working…_"}
            # Figures appear as soon as the design tool has written them, which
            # is mid-run — the reader gets the plate while the memo is still
            # being written.
            yield history, "", *await asyncio.to_thread(_figures, target)
    except Exception as exc:
        history[-1] = {"role": "assistant",
                       "content": f"{chr(10).join(trace)}\n\n**Failed:** `{exc}`"}
        yield history, "", *BLANK
        return

    # Persist the memo and trace. design_plate already archived the numbers, but
    # the agent's written reasoning lived only in this browser tab until now —
    # and the reasoning is the part that explains why the numbers were accepted.
    await asyncio.to_thread(_archive, target, memo, trace)
    yield history, "", *await asyncio.to_thread(_figures, target)


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
            order = gr.File(label="Vendor order file (96 wells)", height=90)
        with gr.Column(scale=2):
            funnel_img = gr.Image(label="From library to plate", height=200)
            dose_img = gr.Image(label="What the sensor can actually see", height=250)
            window_img = gr.Image(label="Switching vs specificity", height=260)
            plate_img = gr.Image(label="The plate", height=250)
    gr.Markdown(FOOTER)

    for trigger in (box.submit, send.click):
        trigger(respond, [box, chat],
                [chat, box, funnel_img, dose_img, window_img, plate_img, order])


if __name__ == "__main__":
    # System fonts only: the Soft theme's display face renders a capital E as a
    # curved epsilon, and it avoids a Google Fonts fetch on conference wifi.
    demo.launch(theme=gr.themes.Soft(
        font=["system-ui", "-apple-system", "Helvetica Neue", "Arial", "sans-serif"],
        font_mono=["ui-monospace", "SF Mono", "Menlo", "Consolas", "monospace"],
    ))
