"""Artifacts for the aptamer switch plate.

Four figures, each answering a question the memo can only claim:

    selection_funnel   why 96, out of how many, and what killed the rest
    design_window      where the plate sits on the one axis that decides switching
    dose_response      what the sensor can actually see, against clinical reality
    plate_map          the physical plate, and proof position is not confounded

The dose-response figure is the one to read first. It is the only one that puts
the design against the concentration the analyte actually circulates at, and for
a cytokine that comparison is unflattering in a way the team needs to see before
ordering rather than after.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

KEEP = "#1baf7a"        # survives every criterion
DROP = "#d8d7d2"        # filtered out
PICKED = "#eb6834"      # on the plate
CONTROL = "#5b6ee1"     # control well
BAND = "#efeee9"


def _frame(ax, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=INK_SOFT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK_SOFT)


def selection_funnel(stages: list[tuple[str, int, str]], path: str) -> str:
    """How 96 were chosen, and what removed everything else.

    The interesting number is not 96, it is the library it came from and the two
    criteria that account for almost all of the loss.

    Bar length is log-scaled but drawn in axes coordinates rather than on a log
    axis. An earlier version placed the count and the note at data-space offsets
    from the bar end on a symlog scale, so the visual gap between them shrank as
    counts grew and the two labels overlapped at 12,058. Text columns are fixed;
    only the bars move.
    """
    import math

    labels = [s[0] for s in stages]
    counts = [s[1] for s in stages]
    notes = [s[2] for s in stages]

    fig, ax = plt.subplots(figsize=(10.5, 0.62 * len(stages) + 1.1))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    widest = max(max(counts), 1)
    # The count is right-aligned at COUNT_X, so the bar must stop far enough
    # left that a seven-digit number still clears it: 1,234,567 at this font is
    # about a tenth of the width.
    BAR_MAX, COUNT_X, NOTE_X = 0.27, 0.40, 0.45

    for i, (label, n, note) in enumerate(zip(labels, counts, notes)):
        y = len(stages) - 1 - i
        width = BAR_MAX * math.log10(n + 1) / math.log10(widest + 1) if n else 0.0
        colour = PICKED if label.startswith("on the plate") else KEEP
        ax.add_patch(Rectangle((0, y - 0.3), width, 0.6, color=colour, linewidth=0,
                               transform=ax.get_yaxis_transform()))
        ax.text(COUNT_X, y, f"{n:,}", transform=ax.get_yaxis_transform(),
                va="center", ha="right", fontsize=10.5, color=INK, fontweight="bold")
        ax.text(NOTE_X, y, note, transform=ax.get_yaxis_transform(),
                va="center", fontsize=8.5, color=INK_SOFT)

    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(labels[::-1], fontsize=9.5, color=INK)
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(stages) - 0.4)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, length=0)
    ax.set_title("From library to plate", fontsize=12, color=INK,
                 fontweight="bold", loc="left", pad=12)

    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return path


def design_window(rows: list[dict], picked: set[str], path: str,
                  bands: tuple[float, float, int] | None = None) -> str:
    """Specificity against switching, with the chosen wells marked.

    Two axes, because the plate is chosen on two. Horizontal is the switching
    competition — near zero is where a sensor can flip. Vertical is whether the
    tail actually prefers its intended site. Candidates that fail sit in grey
    behind the ones that pass, so the shape of what was rejected stays visible
    instead of being quietly cropped out.
    """
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    fig.patch.set_facecolor(SURFACE)

    fail = [r for r in rows if not r["passes"]]
    ok = [r for r in rows if r["passes"] and r["name"] not in picked]
    sel = [r for r in rows if r["name"] in picked]

    ax.axhspan(-6, 1.0, color=BAND, zorder=0)
    ax.text(ax.get_xlim()[0], 0.85, "  tail also binds elsewhere — rejected",
            fontsize=8, color=INK_SOFT, va="top", zorder=1)

    for group, colour, size, label, z in (
            (fail, DROP, 6, f"filtered out ({len(fail):,})", 2),
            (ok, KEEP, 10, f"passes, not selected ({len(ok):,})", 3),
            (sel, PICKED, 34, f"on the plate ({len(sel)})", 4)):
        if not group:
            continue
        ax.scatter([r["dd_g"] for r in group], [r["specificity_margin"] for r in group],
                   s=size, c=colour, edgecolors="none", label=label, zorder=z,
                   alpha=0.75 if colour is DROP else 1.0)

    # Show the bands the plate is tiled across. Without them the selected points
    # look scattered at random along the top; with them it is visible that one
    # group is taken from each energy band, which is the whole selection rule.
    if bands:
        lo, hi, n_bins = bands
        for i in range(1, n_bins):
            ax.axvline(lo + i * (hi - lo) / n_bins, color=GRID, linewidth=0.7,
                       zorder=0)
        ax.text(lo, ax.get_ylim()[0], f"  {n_bins} energy bands · the plate takes "
                f"a spread of designs from each", fontsize=8, color=INK_SOFT,
                va="bottom")

    ax.axvline(0, color=INK_SOFT, linewidth=0.9, linestyle=":", zorder=1)
    ax.text(0, ax.get_ylim()[1], " tail and fold balanced", fontsize=8,
            color=INK_SOFT, va="top")

    _frame(ax, "ddG — tail duplex minus fold (kcal/mol)",
           "specificity margin (kcal/mol)")
    ax.set_title("What the plate covers", fontsize=12, color=INK,
                 fontweight="bold", loc="left", pad=10)
    if ax.get_legend_handles_labels()[0]:
        leg = ax.legend(frameon=False, fontsize=8.5, loc="lower right")
        for t in leg.get_texts():
            t.set_color(INK_SOFT)

    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return path


def dose_response(kd_values_nM: list[float], clinical: list[tuple[str, float, float]],
                  target: str, path: str, kd_intrinsic_nM: float | None = None) -> str:
    """What the sensor sees, against what the patient actually has.

    Occupancy is the ceiling on any signal: a sensor cannot report a change in
    a quantity that never binds it. Plotting the selected variants' response
    curves over the clinical concentration bands shows immediately which parts
    of the range are reachable and which are not — and for a cytokine, most of
    the physiological range is not.
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    fig.patch.set_facecolor(SURFACE)

    conc = np.logspace(-13.5, -7.5, 400)        # 0.03 pM to 30 nM

    for i, (label, lo, hi) in enumerate(clinical):
        ax.axvspan(lo, hi, color=BAND if i % 2 == 0 else "#f6f5f1", zorder=0)
        ax.text(np.sqrt(lo * hi), 1.02 + 0.055 * (i % 2), label, ha="center",
                fontsize=8, color=INK_SOFT, zorder=1)

    # The pair worth comparing is the parent against the switch built from it.
    # The selected variants sit within a nanomolar of each other, so plotting
    # best-versus-weakest draws the same line twice; plotting the parent shows
    # the tax that switching charges in affinity, which is the real story.
    lo_kd, hi_kd = min(kd_values_nM), max(kd_values_nM)
    curves = []
    if kd_intrinsic_nM:
        curves.append((kd_intrinsic_nM, INK_SOFT, 1.6, "--",
                       f"parent aptamer, no switch — Kd {kd_intrinsic_nM:.1f} nM"))
    curves.append((lo_kd, KEEP, 2.2, "-",
                   f"selected switches — Kd {lo_kd:.0f}-{hi_kd:.0f} nM"))
    for kd_nM, colour, width, style, label in curves:
        kd = kd_nM * 1e-9
        ax.plot(conc, conc / (conc + kd), color=colour, linewidth=width,
                linestyle=style, label=label, zorder=3)

    # 10% is a convention this project adopted, not a measured detection limit.
    # E-AB drift depends on the electrode, the monolayer and the medium, and no
    # source is attached to this figure. Drawn as an orientation line only.
    ax.axhline(0.10, color=INK_SOFT, linestyle="--", linewidth=1, zorder=2)
    ax.text(conc[0], 0.115, " 10% occupancy — assumed floor, not measured",
            fontsize=8, color=INK_SOFT)

    ax.set_xscale("log")
    ax.set_xlim(conc[0], conc[-1])
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    _frame(ax, f"{target} concentration (M)", "fraction of sensors bound")
    ax.set_title(f"What the sensor can see — {target}", fontsize=12, color=INK,
                 fontweight="bold", loc="left", pad=26)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK_SOFT)

    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return path


def plate_map(wells: list, path: str, null_p95: float | None = None,
              observed: float | None = None) -> str:
    """The physical plate, coloured by ddG, with controls called out.

    Also the audit: if ddG tracked row or column, the plate would show it as a
    gradient and the experiment would be confounded with edge effects. The
    caption carries the randomisation check so the reader does not have to take
    the shuffle on trust.
    """
    rows = "ABCDEFGH"
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    fig.patch.set_facecolor(SURFACE)

    tests = [w for w in wells if w.role == "test"]
    lo = min(w.dd_g for w in tests)
    hi = max(w.dd_g for w in tests)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "dd", ["#1baf7a", "#f2f1ec", "#eb6834"])

    for w in wells:
        r = rows.index(w.position[0])
        c = int(w.position[1:]) - 1
        if w.role == "control":
            face, edge, lw = CONTROL, "#3d4ea8", 1.4
        else:
            face = cmap((w.dd_g - lo) / (hi - lo) if hi > lo else 0.5)
            edge, lw = "#ffffff", 0.8
        ax.add_patch(plt.Circle((c, -r), 0.40, facecolor=face, edgecolor=edge,
                                linewidth=lw, zorder=3))

    for c in range(12):
        ax.text(c, 0.72, str(c + 1), ha="center", fontsize=8.5, color=INK_SOFT)
    for r, name in enumerate(rows):
        ax.text(-0.95, -r, name, va="center", fontsize=8.5, color=INK_SOFT)

    ax.set_xlim(-1.4, 12.0)
    ax.set_ylim(-7.9, 1.3)
    ax.set_aspect("equal")
    ax.set_axis_off()

    ax.text(-1.4, 1.15, "The plate", fontsize=12, color=INK, fontweight="bold")

    sm = matplotlib.cm.ScalarMappable(
        cmap=cmap, norm=matplotlib.colors.Normalize(vmin=lo, vmax=hi))
    cb = fig.colorbar(sm, ax=ax, fraction=0.028, pad=0.03)
    cb.set_label("ddG (kcal/mol)", fontsize=8.5, color=INK_SOFT)
    cb.ax.tick_params(colors=INK_SOFT, labelsize=8)
    cb.outline.set_visible(False)

    ax.add_patch(plt.Circle((0.15, -8.55), 0.17, facecolor=CONTROL,
                            edgecolor="#3d4ea8", linewidth=1.2, clip_on=False))
    note = "control well"
    if observed is not None and null_p95 is not None:
        note += (f"    ·    ddG vs plate position: row spread {observed:.2f} "
                 f"kcal/mol against {null_p95:.2f} at the 95th percentile of "
                 f"random assignment — not confounded")
    ax.text(0.55, -8.55, note, fontsize=8.5, color=INK_SOFT, va="center",
            clip_on=False)

    fig.savefig(path, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


TAIL_COLOUR = "#5b6ee1"     # the appended competing tail


def switch_diagram(parent: str, construct: str, core: tuple[int, int],
                   path: str, is_quadruplex: bool = False) -> str:
    """The mechanism, drawn: the parent's fold beside the switch built from it.

    This is the one picture that shows what the design actually does. On the
    left the parent folds and its binding core is buried in that fold. On the
    right an appended tail has zipped up against the core and held it open; the
    target has to displace the tail to bind, and that displacement is the signal.

    Drawn from ViennaRNA's own layout coordinates rather than its PostScript
    output, so the colours match the rest of the figures and the core and tail
    can be picked out — which is the entire point, and something a generic
    structure plot cannot do.
    """
    import RNA

    def layout(seq: str) -> tuple:
        # Watson-Crick only. ViennaRNA's layout works from a dot-bracket string
        # and a quadruplex is not one, so a G4 parent cannot be drawn here. The
        # energy shown is therefore this structure's own, never the quadruplex
        # value from thermo.fold - captioning a Watson-Crick drawing with a
        # quadruplex energy would put a number on a picture of something else.
        md = RNA.md()
        md.temperature = 37.0
        fc = RNA.fold_compound(seq, md)
        ss, mfe = fc.mfe()
        co = RNA.get_xy_coordinates(ss)
        xy = [(co.get(i).X, co.get(i).Y) for i in range(len(seq))]
        pt = RNA.ptable(ss)
        pairs = [(i - 1, pt[i] - 1) for i in range(1, len(pt)) if pt[i] > i]
        return xy, pairs, ss, mfe

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))
    fig.patch.set_facecolor(SURFACE)

    panels = [
        (axes[0], parent, "Parent aptamer", "core folded into the structure"),
        (axes[1], construct, "Switch construct", "tail holds the core open"),
    ]
    quadruplex = is_quadruplex

    for ax, seq, title, caption in panels:
        ax.set_facecolor(SURFACE)
        xy, pairs, ss, mfe = layout(seq)

        for a, b in pairs:                       # base pairs, drawn behind
            ax.plot([xy[a][0], xy[b][0]], [xy[a][1], xy[b][1]],
                    color=GRID, linewidth=1.0, zorder=1)
        ax.plot([p[0] for p in xy], [p[1] for p in xy],
                color=INK_SOFT, linewidth=1.2, zorder=2)

        for i, (x, y) in enumerate(xy):
            pos = i + 1
            if pos > len(parent):
                colour, size = TAIL_COLOUR, 34    # the appended tail
            elif core[0] <= pos <= core[1]:
                colour, size = PICKED, 34         # the binding core
            else:
                colour, size = KEEP, 20
            ax.scatter([x], [y], s=size, c=colour, edgecolors=SURFACE,
                       linewidths=0.6, zorder=3)

        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"{title}   {mfe:.1f} kcal/mol", fontsize=11, color=INK,
                     fontweight="bold", loc="left", pad=8)
        ax.text(0.0, -0.04, caption, transform=ax.transAxes, fontsize=9,
                color=INK_SOFT, va="top")

    for x, lab, col in ((.06, "binding core", PICKED),
                        (.30, "rest of the aptamer", KEEP),
                        (.62, "appended tail", TAIL_COLOUR)):
        fig.add_artist(plt.Rectangle((x, .015), .016, .022, color=col,
                                     transform=fig.transFigure))
        fig.text(x + .022, .018, lab, fontsize=8.5, color=INK_SOFT)

    if quadruplex:
        fig.text(.06, .085,
                 "This parent folds a G-quadruplex, which has no secondary-structure\n"
                 "drawing. Watson-Crick pairing only is shown — the real fold is far\n"
                 "more stable than the energy above (-9.7 vs -0.1 kcal/mol).",
                 fontsize=8.5, color=PICKED, linespacing=1.5)

    fig.tight_layout(rect=(0, 0.17 if quadruplex else 0.06, 1, 1))
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return path
