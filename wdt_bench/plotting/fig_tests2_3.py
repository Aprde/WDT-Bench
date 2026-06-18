"""Paper figure: Tests 2-3 combined, two panels.

* Panel a - Test 2 constituent preference: per-replicate (node-rule rate,
  parent-rule rate) scatter per model, with mean diamonds and the y=x
  reference;
* Panel b - Test 3 localization accuracy: per-(prompt, replicate) accuracy
  clouds with a gradient mean line across models (prompt variants D/E/F only,
  for a fair comparison).

Output: ``figures/paper/fig_tests2_3.svg``.  Renders with whichever models
are present in ``results/processed/general_tests``.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from . import style
from .paper_common import (
    MEAN_DIAMOND_CORE_SIZE,
    MEAN_DIAMOND_EDGE_WIDTH,
    MEAN_DIAMOND_HALO_SIZE,
    MODEL_COLORS,
    MODEL_COLORS_DARK,
    MODEL_COLORS_SCATTER,
    MODEL_LABELS,
    MODEL_ORDER,
    apply_font,
    load_subtask,
    out_path,
)

logger = logging.getLogger(__name__)


FS_PANEL = 12
FS_TITLE = 12
FS_AXIS = 10
FS_TICK = 10
FS_LEGEND = 10

_FALLBACK_SCATTER = "#666666"
_FALLBACK_DARK = "#444444"
_FALLBACK_FACE = "#bbbbbb"


def _apply_times_font() -> None:
    apply_font()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    })


def _place_panel_label(fig, ax, letter: str) -> None:
    """Bold panel letter, vertically centred on the axes title.

    Measured after a draw, so the letter tracks the title's true position
    regardless of font size.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tb = ax.title.get_window_extent(renderer=renderer)
    _, title_yc = fig.transFigure.inverted().transform(
        (tb.x0, (tb.y0 + tb.y1) / 2)
    )
    pos = ax.get_position()
    x = pos.x0 - 0.12 * (pos.x1 - pos.x0)
    fig.text(x, title_yc, letter, fontsize=FS_PANEL, fontweight="bold",
             va="center", ha="left")


def _hex_to_rgb01(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _blend_hex(c0: str, c1: str, t: float):
    r0, g0, b0 = _hex_to_rgb01(c0)
    r1, g1, b1 = _hex_to_rgb01(c1)
    return (r0 + (r1 - r0) * t, g0 + (g1 - g0) * t, b0 + (b1 - b0) * t)


def _draw_gradient_line(ax, xs, ys, colors, lw=1.8, n_steps=26):
    """Draw a piecewise gradient line across adjacent points."""
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        y0, y1 = ys[i], ys[i + 1]
        c0, c1 = colors[i], colors[i + 1]
        for s in range(n_steps):
            t0 = s / n_steps
            t1 = (s + 1) / n_steps
            xa0 = x0 + (x1 - x0) * t0
            xa1 = x0 + (x1 - x0) * t1
            ya0 = y0 + (y1 - y0) * t0
            ya1 = y0 + (y1 - y0) * t1
            ax.plot([xa0, xa1], [ya0, ya1],
                    color=_blend_hex(c0, c1, (t0 + t1) / 2.0),
                    lw=lw, zorder=4, solid_capstyle="round")


def _per_replicate_rates(rows):
    """Return (node_rate_arr, parent_rate_arr) over replicates.

    node = NP deletions, parent = PP deletions; the panel plots node on x
    and parent on y.
    """
    buckets = defaultdict(lambda: [0, 0, 0])  # rep -> [n_node, n_parent, n_total]
    for r in rows:
        rep = r.get("replicate_index")
        lab = r.get("label")
        b = buckets[rep]
        b[2] += 1
        if lab == "node_rule":
            b[0] += 1
        elif lab == "parent_rule":
            b[1] += 1
    node, parent = [], []
    for _, (n_node, n_par, n_tot) in sorted(buckets.items()):
        if n_tot == 0:
            continue
        node.append(n_node / n_tot)
        parent.append(n_par / n_tot)
    return np.array(node, float), np.array(parent, float)


def _cell_accuracies(rows):
    """Per-(prompt, replicate) accuracies + overall mean, D/E/F prompts only."""
    buckets = defaultdict(lambda: [0, 0])  # (prompt, rep) -> [n_correct, n_total]
    for r in rows:
        key = (r.get("prompt_variant"), r.get("replicate_index"))
        b = buckets[key]
        b[1] += 1
        if r.get("correct_vs_gold") is True:
            b[0] += 1
    accs = np.array([c / t for c, t in buckets.values() if t > 0], float)
    overall = float(np.mean([r.get("correct_vs_gold") is True for r in rows])) if rows else float("nan")
    return accs, overall


def _ordered_models(data) -> list[str]:
    known = [m for m in MODEL_ORDER if m in data]
    extra = sorted(m for m in data if m not in MODEL_ORDER)
    return known + extra


def _draw_panel_a(ax, data12, models):
    """Test 2: per-replicate scatter -- node rate (x) vs parent rate (y)."""
    node_arrs, par_arrs = [], []
    for m in models:
        nd, pr = _per_replicate_rates(data12[m])
        node_arrs.append(nd)
        par_arrs.append(pr)

    dark = [MODEL_COLORS_DARK.get(m, _FALLBACK_DARK) for m in models]
    scatter_cols = [MODEL_COLORS_SCATTER.get(m, _FALLBACK_SCATTER) for m in models]
    n = len(models)
    rng_a = np.random.default_rng()  # unseeded - unique jitter each render

    for i in range(n - 1, -1, -1):
        # Rates are discrete k/n -> coincident points stack exactly;
        # Gaussian jitter on both axes spreads the cloud.
        xj = rng_a.normal(0, 0.018, len(node_arrs[i]))
        yj = rng_a.normal(0, 0.018, len(par_arrs[i]))
        xs = np.clip(node_arrs[i] + xj, 0.0, 1.0)
        ys = np.clip(par_arrs[i] + yj, 0.0, 1.0)
        cs = scatter_cols[i]
        # dual-layer: fill + subtle outline for visibility
        ax.scatter(xs, ys, color=cs, alpha=0.22, s=16, linewidths=0, zorder=n - i)
        ax.scatter(xs, ys, facecolors="none", edgecolors=cs,
                   linewidths=0.7, alpha=0.55, s=18, zorder=n - i + 0.5)
        # Mean diamond - sized and edged exactly like panel b so the two
        # panels visually match.
        ax.scatter([np.mean(node_arrs[i])], [np.mean(par_arrs[i])],
                   color="white", s=MEAN_DIAMOND_HALO_SIZE, marker="D", zorder=2 * n - i - 0.5)
        ax.scatter([np.mean(node_arrs[i])], [np.mean(par_arrs[i])],
                   color=dark[i], s=MEAN_DIAMOND_CORE_SIZE, marker="D",
                   edgecolors="white", linewidths=MEAN_DIAMOND_EDGE_WIDTH, zorder=2 * n - i)

    ax.plot([0, 1], [0, 1], color="#d0d0d0", lw=0.8, ls="--", zorder=0)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("Explained ratio of node-category rule", fontsize=FS_AXIS, fontweight="normal")
    ax.set_ylabel("Explained ratio of parent-category rule", fontsize=FS_AXIS, fontweight="normal")
    for t in (0.25, 0.5, 0.75):
        ax.axhline(t, color="#f0f0f0", lw=0.5, zorder=0)
        ax.axvline(t, color="#f0f0f0", lw=0.5, zorder=0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"], fontsize=FS_TICK)
    ax.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"], fontsize=FS_TICK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Test 2: constituent preference", fontsize=FS_TITLE,
                 pad=12, fontweight="bold")
    # No legend here - a single shared legend serves both panels (see main()).


def _draw_panel_b(ax, data13, models):
    cell_acc, means = {}, {}
    for m in models:
        a, mu = _cell_accuracies(data13[m])
        cell_acc[m] = a
        means[m] = mu

    x_pos = list(range(len(models)))
    rng = np.random.default_rng()  # unseeded - unique jitter each render
    for xi, m in zip(x_pos, models):
        a = cell_acc[m]
        # Accuracies are discrete k/n -> identical y values form rigid rows;
        # jitter x (wide) AND y (Gaussian) to break the grid look.
        xj = rng.uniform(-0.17, 0.17, size=len(a))
        yj = rng.normal(0.0, 0.013, size=len(a))
        xv = np.full(len(a), xi) + xj
        av = np.clip(a + yj, 0.0, 1.0)
        cs = MODEL_COLORS_SCATTER.get(m, _FALLBACK_SCATTER)
        ax.scatter(xv, av, s=16, color=cs, alpha=0.22, linewidths=0, zorder=2)
        ax.scatter(xv, av, s=16, facecolors="none", edgecolors=cs,
                   linewidths=0.7, alpha=0.55, zorder=3)

    ys = [means[m] for m in models]
    line_cols = [MODEL_COLORS_DARK.get(m, _FALLBACK_DARK) for m in models]
    if len(models) > 1:
        _draw_gradient_line(ax, x_pos, ys, line_cols, lw=1.8, n_steps=30)

    for xi, m in zip(x_pos, models):
        # Same sizes and edge linewidth as panel a - keeps the mean diamonds
        # visually identical across both panels.
        ax.scatter([xi], [means[m]], color="white", s=MEAN_DIAMOND_HALO_SIZE, marker="D", zorder=5)
        ax.scatter([xi], [means[m]], color=MODEL_COLORS_DARK.get(m, _FALLBACK_DARK),
                   s=MEAN_DIAMOND_CORE_SIZE, marker="D",
                   edgecolors="white", linewidths=MEAN_DIAMOND_EDGE_WIDTH, zorder=6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=FS_TICK)
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"], fontsize=FS_TICK)
    ax.set_ylabel("Proportion of corresponding constituent",
                  fontsize=FS_AXIS, fontweight="normal")
    ax.set_xlabel("Model variant", fontsize=FS_AXIS, fontweight="normal")
    ax.set_title("Test 3: constituent localization", fontsize=FS_TITLE,
                 pad=12, fontweight="bold")

    for t in (0.25, 0.5, 0.75):
        ax.axhline(t, color="#f2f2f2", lw=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # No per-panel legend; the shared vertical legend lives inside panel a.


def main() -> None:
    style.apply_style()
    _apply_times_font()

    data12 = load_subtask(2)
    data13 = load_subtask(3)
    models12 = _ordered_models(data12)
    models13 = _ordered_models(data13)
    if not models12:
        raise RuntimeError(
            "No Test 2 classified data found under results/processed/general_tests; "
            "run the Test 2 analysis first."
        )
    if not models13:
        raise RuntimeError(
            "No Test 3 classified data found under results/processed/general_tests; "
            "run the Test 3 analysis first."
        )

    fig_w = 16.0 / 2.54
    fig_h = fig_w * (6.6 / 13.2)
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    # Match the coordinate-plot boxes in the Tests 4-5 figure: same physical
    # width and height, while keeping the local two-column layout.
    plot_left = 0.110
    plot_w = 0.38
    plot_gap = 0.48 - plot_w
    plot_h = (0.25 * 7.4) / fig_h
    plot_bottom = 0.16

    ax_a = fig.add_axes([plot_left, plot_bottom, plot_w, plot_h])
    _draw_panel_a(ax_a, data12, models12)

    ax_b = fig.add_axes([plot_left + plot_w + plot_gap, plot_bottom, plot_w, plot_h])
    _draw_panel_b(ax_b, data13, models13)

    # Shared vertical legend inside panel a's open right side.
    legend_models = models12 if len(models12) >= len(models13) else models13
    patches = [
        mpatches.Patch(facecolor=MODEL_COLORS.get(m, _FALLBACK_FACE),
                       edgecolor=MODEL_COLORS_DARK.get(m, _FALLBACK_DARK),
                       linewidth=0.5, alpha=0.85, label=MODEL_LABELS.get(m, m))
        for m in legend_models
    ]
    diamond_h = mlines.Line2D([], [], marker="D", color="w",
                              markerfacecolor="#888888", markeredgecolor="white",
                              markersize=5.0, label="Mean", linewidth=0)

    ax_a.legend(handles=patches + [diamond_h],
                loc="center",
                bbox_to_anchor=(0.72, 0.25),
                bbox_transform=ax_a.transAxes,
                ncol=1,
                fontsize=FS_LEGEND, frameon=False,
                handlelength=0.8, handletextpad=0.35,
                labelspacing=0.55, borderaxespad=0.0)

    _place_panel_label(fig, ax_a, "a")
    _place_panel_label(fig, ax_b, "b")

    p = out_path("fig_tests2_3.svg")
    fig.savefig(p, facecolor="white", dpi=180)
    plt.close(fig)
    logger.info("Saved %s", p)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
