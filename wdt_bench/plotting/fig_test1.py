"""Paper figure: Test 1 (constituent recognition), three panels.

* Panel a - per-model stacked horizontal bars of the five tree-span
  categories, with the heterogeneous random-span baseline as a dashed tick;
* Panel b - effect of prompt design (per-replicate single-constituent rate
  for the six prompt variants of the focal model);
* Panel c - effect of the number of demonstrations.

Output: ``figures/paper/fig_test1.svg``.

The focal model defaults to ``qwen-max`` but falls back to the first
available model, and panel c uses whichever demonstration counts are present,
so the figure renders even when only part of the runs is shipped.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from . import style
from .paper_common import (
    CAT11_COLORS,
    CAT11_ORDER,
    MEAN_DIAMOND_CORE_SIZE,
    MEAN_DIAMOND_EDGE_WIDTH,
    MEAN_DIAMOND_HALO_SIZE,
    MODEL_LABELS,
    MODEL_ORDER,
    apply_font,
    load_subtask,
    load_test1_by_demos,
    out_path,
)

logger = logging.getLogger(__name__)

# Variant letters: A/B/C narrative, D/E/F erase-only editorial.
_PROMPTS = ["A", "B", "C", "D", "E", "F"]
_DEMOS = [1, 2, 5, 10]
_FOCAL_MODEL = "qwen-max"
_CAT = "single_constituent"
# Panel a plots the 5-way tree-span taxonomy (``tree_span_category``); the
# category names match the summary fields (``p_single_constituent``, ...)
# and the legend labels below.
_CAT_LABELS = {
    "single_constituent": "Single constituent",
    "multiple_constituents": "Multiple constituents",
    "partial_constituent": "Partial constituent",
    "constituent_plus_partial": "Constituent + partial",
    "other": "Other",
}

# Unified typography scale (target journal style: 9pt text)
FS_PANEL_LABEL = 12
FS_TITLE = 12
FS_AXIS_LABEL = 10
FS_TICK = 10
FS_LEGEND = 10
FS_BAR_TEXT = 10
LOWER_PANEL_TITLE_PAD = 10


def _fmt_decimal(v: float, ndigits: int = 2) -> str:
    return f"{v:.{ndigits}f}"


def _mean_single_constituent_baseline(rows) -> float | None:
    vals = []
    for r in rows:
        v = r.get("baseline_constituent_prob_trial")
        if v is None:
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    return float(np.mean(vals)) if vals else None


def _apply_times_font() -> None:
    apply_font()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    })


def _panel_label_fig(
    fig,
    ax,
    letter: str,
    x_override: float | None = None,
    *,
    title_gap: float | None = None,
) -> None:
    """Bold panel letter, vertically centred on the axes title.

    ``title_gap``: when set, the letter is right-aligned this far (figure
    fraction) to the LEFT of the title's own left edge - used when a title is
    wider than its axes and overhangs the inter-panel gap.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_bbox = ax.title.get_window_extent(renderer=renderer)
    title_x0, title_y = fig.transFigure.inverted().transform(
        (title_bbox.x0, (title_bbox.y0 + title_bbox.y1) / 2)
    )
    if title_gap is not None:
        fig.text(title_x0 - title_gap, title_y, letter,
                 fontsize=FS_PANEL_LABEL, fontweight="bold",
                 va="center", ha="right")
        return
    pos = ax.get_position()
    x = x_override if x_override is not None else (pos.x0 - 0.034)
    fig.text(x, title_y, letter, fontsize=FS_PANEL_LABEL,
             fontweight="bold", va="center", ha="left")


def _style_share_yaxis(ax) -> None:
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"], fontsize=FS_TICK)
    ax.set_ylabel("Proportion", fontsize=FS_AXIS_LABEL)
    for t in (0.25, 0.5, 0.75):
        ax.axhline(t, color="#f2f2f2", lw=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _per_replicate_one_node(rows) -> Dict[int, float]:
    """rep -> rate of single_constituent; prompts pooled within replicate."""
    by_rep = defaultdict(Counter)
    tot = Counter()
    for r in rows:
        rep = r.get("replicate_index")
        by_rep[rep][r.get("tree_span_category")] += 1
        tot[rep] += 1
    return {rep: c.get(_CAT, 0) / (tot[rep] or 1) for rep, c in by_rep.items()}


def _per_prompt_replicate_rates(rows) -> Dict[str, Dict[int, float]]:
    """prompt -> {rep -> rate}."""
    by_pv = defaultdict(list)
    for r in rows:
        by_pv[r.get("prompt_variant")].append(r)
    return {pv: _per_replicate_one_node(rs) for pv, rs in by_pv.items()}


def _per_prompt_replicate_points(rows) -> Dict[tuple, float]:
    """(prompt, rep) -> rate; keeps the c-panel scatter at the same grain as b."""
    by_cell = defaultdict(Counter)
    tot = Counter()
    for r in rows:
        key = (r.get("prompt_variant"), r.get("replicate_index"))
        by_cell[key][r.get("tree_span_category")] += 1
        tot[key] += 1
    return {key: c.get(_CAT, 0) / (tot[key] or 1) for key, c in by_cell.items()}


def _draw_one_node_line(
    ax,
    x_keys: Sequence,
    x_labels: Sequence[str],
    rep_rates: Dict,
    *,
    xlabel: str,
    title: str | None = None,
    show_legend: bool = False,
) -> None:
    color = CAT11_COLORS[_CAT]
    x_base = list(range(len(x_keys)))
    rng = np.random.default_rng()  # unseeded - unique jitter each render
    means = []

    for xi, key in enumerate(x_keys):
        vals = np.array(list(rep_rates[key].values()), float)
        means.append(float(np.mean(vals)) if len(vals) else float("nan"))
        # Data are discrete k/n rates -> many land on identical y, forming
        # rigid rows.  Jitter BOTH axes (wide x + Gaussian y) to break it.
        xj = rng.uniform(-0.17, 0.17, size=len(vals))
        yj = rng.normal(0.0, 0.013, size=len(vals))
        xv = np.full(len(vals), x_base[xi]) + xj
        yv = np.clip(vals + yj, 0.0, 1.0)
        # dual-layer: fill + subtle outline for visibility
        ax.scatter(xv, yv, s=9, color=color, alpha=0.18, linewidths=0, zorder=2)
        ax.scatter(xv, yv, s=9, facecolors="none", edgecolors=color,
                   linewidths=0.7, alpha=0.55, zorder=3)

    ax.plot(x_base, means, color=color, lw=1.5, zorder=4, alpha=0.9)
    ax.scatter(x_base, means, color="white", s=MEAN_DIAMOND_HALO_SIZE, marker="D", zorder=5)
    ax.scatter(x_base, means, color=color, s=MEAN_DIAMOND_CORE_SIZE, marker="D",
               edgecolors="white", linewidths=MEAN_DIAMOND_EDGE_WIDTH, zorder=6)

    ax.set_xticks(x_base)
    ax.set_xticklabels(list(x_labels), fontsize=FS_TICK)
    ax.set_xlim(-0.5, len(x_keys) - 0.5)
    ax.set_xlabel(xlabel, fontsize=FS_AXIS_LABEL)
    if title:
        ax.set_title(title, fontsize=FS_TITLE, pad=LOWER_PANEL_TITLE_PAD, fontweight="bold")
    _style_share_yaxis(ax)

    if show_legend:
        cat_h = mlines.Line2D([], [], marker="D", color=color,
                              markerfacecolor=color, markeredgecolor="white",
                              markersize=7.2, lw=1.6, label="Single constituent")
        dot_h = mlines.Line2D([], [], marker=" ", color="w",
                              markerfacecolor="#bbbbbb", markersize=5.4,
                              label=" ",
                              linewidth=0, alpha=0.7)
        ax.legend(handles=[cat_h, dot_h], loc="lower right",
                  fontsize=FS_LEGEND, frameon=False, handlelength=1.5,
                  handletextpad=0.5, labelspacing=0.6)


def _draw_panel_a(ax, data, *, title: str | None = None) -> None:
    models = [m for m in MODEL_ORDER if m in data]
    rates = {}
    baselines = {}
    for m in models:
        c = Counter(r.get("tree_span_category") for r in data[m])
        n = sum(c.values()) or 1
        rates[m] = {cat: c.get(cat, 0) / n for cat in CAT11_ORDER}
        baselines[m] = _mean_single_constituent_baseline(data[m])

    bar_h = 0.88
    y_gap = 1.12
    y_pos = [i * y_gap for i in range(len(models))][::-1]

    for yi, m in zip(y_pos, models):
        left = 0.0
        for cat in CAT11_ORDER:
            w = rates[m][cat]
            ax.barh(yi, w, left=left, height=bar_h,
                    color=CAT11_COLORS[cat], edgecolor="white",
                    linewidth=0.6, zorder=3)
            if w >= 0.05:
                ax.text(left + w / 2, yi, _fmt_decimal(w, ndigits=2),
                        ha="center", va="center", fontsize=FS_BAR_TEXT,
                        color="#3A3A3A", zorder=4)
            left += w
        base = baselines.get(m)
        if base is not None:
            ax.vlines(base, yi - bar_h / 2, yi + bar_h / 2,
                      color="#555555", linewidth=1.1,
                      linestyles=(0, (2.2, 1.6)), zorder=6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=FS_AXIS_LABEL)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"], fontsize=FS_TICK)
    ax.set_xlabel("Proportion", fontsize=FS_AXIS_LABEL, fontweight="normal", labelpad=6)
    ax.set_ylim(-0.72, (len(models) - 1) * y_gap + 0.72)
    if title:
        ax.set_title(title, fontsize=FS_TITLE, pad=10, fontweight="bold")

    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=1)

    handles = [
        mpatches.Patch(facecolor=CAT11_COLORS[c], edgecolor="white",
                       linewidth=0.6, label=_CAT_LABELS[c])
        for c in CAT11_ORDER
    ]
    # Push the legend clear of the x-axis tick labels.
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.48, -0.42), ncol=5,
              fontsize=FS_LEGEND, frameon=False,
              handlelength=0.8, handletextpad=0.25, columnspacing=0.55)


def main() -> None:
    style.apply_style()
    _apply_times_font()

    data = load_subtask(1)
    if not data:
        raise RuntimeError(
            "No Test 1 classified data found under results/processed/general_tests; "
            "run the Test 1 analysis first."
        )

    focal = _FOCAL_MODEL if _FOCAL_MODEL in data else next(
        (m for m in MODEL_ORDER if m in data), sorted(data)[0]
    )
    if focal != _FOCAL_MODEL:
        logger.warning("Focal model %s not found; using %s for panels b/c.", _FOCAL_MODEL, focal)
    focal_rows = data[focal]

    by_demo = load_test1_by_demos(focal)
    demos = [d for d in _DEMOS if d in by_demo] or sorted(by_demo)
    if not demos:
        raise RuntimeError(f"No Test 1 data found for model {focal!r} (panel c).")

    prompt_by_pv = _per_prompt_replicate_rates(focal_rows)
    prompt_rates = {pv: prompt_by_pv.get(pv, {}) for pv in _PROMPTS}
    prompt_rates = {pv: r for pv, r in prompt_rates.items() if r}
    prompts = [pv for pv in _PROMPTS if pv in prompt_rates]
    demo_rates = {d: _per_prompt_replicate_points(by_demo[d]) for d in demos}

    fig_w = 16.0 / 2.54
    fig_h = fig_w * (10.5 / 13.2)
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # Match the coordinate-plot boxes in the Tests 4-5 figure: same physical
    # width and height, while keeping the local two-column layout.
    plot_left = 0.110
    plot_w = 0.36
    plot_gap = 0.48 - plot_w
    plot_h = (0.25 * 7.4) / fig_h
    plot_bottom = 0.162

    ax_a_right_tick = 0.145 + 0.805
    ax_a = fig.add_axes([plot_left, 0.740, ax_a_right_tick - plot_left, 0.180])
    ax_b = fig.add_axes([plot_left, plot_bottom, plot_w, plot_h])
    ax_c = fig.add_axes([plot_left + plot_w + plot_gap, plot_bottom, plot_w, plot_h])

    _draw_panel_a(ax_a, data, title="Test 1: constituent recognition (averaged across prompts)")
    ax_a.title.set_x((0.145 + 0.805 / 2 - plot_left) / (ax_a_right_tick - plot_left))

    _draw_one_node_line(
        ax_b, prompts, list(prompts), prompt_rates,
        xlabel="Prompt variant",
        title="Effect of prompt design",
    )

    _draw_one_node_line(
        ax_c, demos, [str(d) for d in demos], demo_rates,
        xlabel="Number of demonstrations",
        title="Effect of number of demonstrations",
        show_legend=True,
    )

    fig.add_artist(
        mlines.Line2D(
            [ax_a_right_tick, ax_c.get_position().x1],
            [ax_a.get_position().y0, ax_a.get_position().y0],
            transform=fig.transFigure,
            color="black",
            lw=0.8,
            solid_capstyle="butt",
            clip_on=False,
        )
    )

    left_col_x = ax_b.get_position().x0 - 0.045
    _panel_label_fig(fig, ax_a, "a", x_override=left_col_x)
    _panel_label_fig(fig, ax_b, "b", x_override=left_col_x)
    # Panel c's title is wider than its axes; anchor the letter to the
    # title's own left edge so it cannot collide with the overhanging text.
    _panel_label_fig(fig, ax_c, "c", title_gap=0.05)

    p = out_path("fig_test1.svg")
    fig.savefig(p, facecolor="white", dpi=180)
    plt.close(fig)
    logger.info("Saved %s", p)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
