"""Paper figure: Test 6 (attachment ambiguity), two bar panels.

* Panel a - PP attachment ambiguity: per-model rate of deleting the target
  string under structure 1 (plausible) vs structure 2 (implausible);
* Panel b - the same for adjunct (relative clause) attachment ambiguity.

Wilson CIs on each bar; per-model permutation tests between the two
structures (seeds 6100 for PP and 7100 for adjunct keep the significance
labels deterministic across re-runs).

Output: ``figures/paper/fig_test6.svg``.
"""
from __future__ import annotations

import logging
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

from .. import paths
from ..io_utils import load_json
from ..stats import permutation_p, stars_from_p, wilson_interval
from . import style

logger = logging.getLogger(__name__)

_MODEL_ORDER = [
    "qwen-max",
    "deepseek-v4-pro",
    "claude-opus-4-7",
    "gpt-5.5",
    "gemini-3.1-pro-preview",
    "grok-4.20-0309-non-reasoning",
]
_MODEL_LABELS = {
    "claude-opus-4-7": "Claude 4.7",
    "deepseek-v4-pro": "DeepSeek V4",
    "gemini-3.1-pro-preview": "Gemini 3.1",
    "gpt-5.5": "GPT-5.5",
    "grok-4.20-0309-non-reasoning": "Grok 4.20",
    "qwen-max": "Qwen Max",
}

_COL_S1 = "#B6B3D6"
_COL_S2 = "#E9687A"
_COL_S1_EDGE = "#8C88B8"
_COL_S2_EDGE = "#D44A60"

FS_TITLE = 12
FS_AXIS = 10
FS_TICK = 10
FS_LEGEND = 10
FS_SIG = 7.5

# Match the Tests 4-5 figure panel-a axes exactly:
# - 16 cm total SVG width
# - each bar-panel axis width = 0.34 of figure width
# - each bar-panel axis height = 0.25 * 7.4 in
_FIG_W_IN = 16.0 / 2.54
_FIG_H_IN = 3.45
_AX_W = 0.34
_AX_H = (0.25 * 7.4) / _FIG_H_IN
_LM = 0.135
_GAP_X = 0.48 - _AX_W
_RIGHT_BODY_LEFT_SHIFT = 0.025


def _apply_font() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    })


def _extract(doc: Dict, condition: str) -> Dict[str, Dict]:
    """``{model: {"s1": rates, "s2": rates}}`` for one ambiguity condition."""
    key = f"summary_{condition}_by_model"
    out: Dict[str, Dict] = {}
    for r in doc.get(key, []):
        m = r.get("model")
        if m not in _MODEL_ORDER:
            continue
        bs = r.get("by_structure", {})
        s1 = np.array(bs.get("structure_1", {}).get("per_trial_rates", []), float)
        s2 = np.array(bs.get("structure_2", {}).get("per_trial_rates", []), float)
        if len(s1) and len(s2):
            out[m] = {"s1": s1, "s2": s2}
    return out


def _wilson_ci(p: float, n: int):
    """Wilson interval around an observed rate (mean of per-trial rates)."""
    if n == 0:
        return p, p
    return wilson_interval(p * n, n)


def _draw_centered_star(ax: plt.Axes, x: float, y: float) -> None:
    prop = FontProperties(family="Times New Roman", weight="bold")
    path = TextPath((0, 0), "*", size=FS_SIG, prop=prop)
    bbox = path.get_extents()
    cx = (bbox.x0 + bbox.x1) / 2
    cy = (bbox.y0 + bbox.y1) / 2
    disp_x, disp_y = ax.transData.transform((x, y))
    inv = ax.transData.inverted()
    px_per_pt = ax.figure.dpi / 72.0
    dx = inv.transform((disp_x + px_per_pt, disp_y))[0] - x
    dy = inv.transform((disp_x, disp_y + px_per_pt))[1] - y
    trans = (
        Affine2D()
        .translate(-cx, -cy)
        .scale(dx, dy)
        .translate(x, y)
        + ax.transData
    )
    ax.add_patch(mpatches.PathPatch(
        path, transform=trans, facecolor="#222222", edgecolor="none",
        zorder=6, clip_on=False,
    ))


def _draw_sig_label(ax: plt.Axes, x: float, y: float, label: str) -> None:
    if label.startswith("*"):
        offsets = np.linspace(0.125 * (len(label) - 1) / 2,
                              -0.125 * (len(label) - 1) / 2,
                              len(label))
        for dy in offsets:
            _draw_centered_star(ax, x, y + dy)
        return
    ax.text(x, y, label, fontsize=FS_SIG, fontweight="normal",
            ha="center", va="center", color="#222222",
            zorder=6, clip_on=False)


def _draw_condition_panel(ax: plt.Axes, cond_data: Dict[str, Dict], models: list[str],
                          title: str, show_ylabels: bool, *, seed_base: int) -> None:
    n = len(models)
    # Bar width and gap match the Tests 4-5 figure's panel a exactly so the
    # two bar-chart figures look the same scale side by side.
    bar_h = 0.36
    gap = 0.22
    ax.set_xlim(0, 1.10)
    ax.set_ylim(-0.55, n - 0.45)
    for idx, m in enumerate(models):
        y_base = n - 1 - idx
        y1 = y_base + gap
        y2 = y_base - gap
        d = cond_data[m]
        s1 = d["s1"]
        s2 = d["s2"]

        m1 = float(np.mean(s1))
        lo1, hi1 = _wilson_ci(m1, len(s1))
        ax.barh(y1, m1, height=bar_h, color=_COL_S1, edgecolor=_COL_S1_EDGE,
                linewidth=0.6, alpha=0.88, zorder=3)
        ax.errorbar(m1, y1, xerr=[[m1 - lo1], [hi1 - m1]], fmt="none",
                    ecolor=_COL_S1_EDGE, elinewidth=0.9, capsize=2.0, capthick=0.8, zorder=4)
        m2 = float(np.mean(s2))
        lo2, hi2 = _wilson_ci(m2, len(s2))
        ax.barh(y2, m2, height=bar_h, color=_COL_S2, edgecolor=_COL_S2_EDGE,
                linewidth=0.6, alpha=0.88, zorder=3)
        ax.errorbar(m2, y2, xerr=[[m2 - lo2], [hi2 - m2]], fmt="none",
                    ecolor=_COL_S2_EDGE, elinewidth=0.9, capsize=2.0, capthick=0.8, zorder=4)

        label = stars_from_p(permutation_p(s1, s2, seed=seed_base + idx))
        x_star = min(1.085, max(hi1, hi2, m1, m2) + 0.055)
        _draw_sig_label(ax, x_star, y_base, label)
    y_ticks = [n - 1 - i for i in range(n)]
    ax.set_yticks(y_ticks)
    if show_ylabels:
        ax.set_yticklabels([_MODEL_LABELS.get(m, m) for m in models], fontsize=FS_AXIS)
    else:
        ax.set_yticklabels([])

    ax.set_xlim(0, 1.10)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"], fontsize=FS_TICK)
    ax.set_xlabel("Proportion of target string", fontsize=FS_AXIS, fontweight="normal", labelpad=7)
    ax.set_ylim(-0.55, n - 0.45)
    ax.set_title(title, fontsize=FS_TITLE, pad=8, fontweight="bold")

    for i in range(n - 1):
        ax.axhline(n - 1 - i - 0.5, color="#ededed", lw=0.5, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)


def _recenter_titles_and_labels(fig, ax_pp, ax_adj, *, adj_title_reference_dx: float = 0.0) -> None:
    """Centre each panel title on its own visual span, then place the bold
    panel letters just left of (and vertically centred on) those titles.

    A title centred on its axes box alone sits visibly right of the panel:
    panel a additionally carries the y-tick-label strip on the figure's left
    edge.  Each panel is given the half of the inter-panel gap nearest it,
    and panel a also the label strip; the title is centred on that span.
    The letters are anchored to the title's measured left edge so they keep
    a fixed clearance and can never collide with it, whatever the font size.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    bb_pp = ax_pp.get_tightbbox(renderer)
    bb_adj = ax_adj.get_tightbbox(renderer)
    content_left = inv.transform((bb_pp.x0, 0.0))[0]
    content_right = inv.transform((bb_adj.x1, 0.0))[0] + adj_title_reference_dx
    gap_mid = (ax_pp.get_position().x1 + ax_adj.get_position().x0 + adj_title_reference_dx) / 2

    for ax, lo, hi in ((ax_pp, content_left, gap_mid),
                       (ax_adj, gap_mid, content_right)):
        target = (lo + hi) / 2
        pos = ax.get_position()
        ax.title.set_x((target - pos.x0) / (pos.x1 - pos.x0))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, letter in ((ax_pp, "a"), (ax_adj, "b")):
        tb = ax.title.get_window_extent(renderer=renderer)
        tx0, tyc = inv.transform((tb.x0, (tb.y0 + tb.y1) / 2))
        fig.text(tx0 - 0.03, tyc, letter, fontsize=FS_TITLE,
                 fontweight="bold", va="center", ha="right")


def main() -> None:
    style.apply_style()
    _apply_font()

    p6 = paths.diagnostic_classified(6)
    if not p6.is_file():
        raise RuntimeError(
            f"Test 6 classified results not found: {p6}; "
            "run `analyze_results.py --test 6` first."
        )
    doc = load_json(p6)
    pp_data = _extract(doc, "pp")
    adj_data = _extract(doc, "adjunct")
    models = [m for m in _MODEL_ORDER if m in pp_data and m in adj_data]
    if not models:
        raise RuntimeError("No model has data for both PP and adjunct conditions.")

    # Total width matches the Tests 4-5 figure; each inner bar chart also has
    # the same physical axis width and height as its panel-a bar charts.
    fig = plt.figure(figsize=(_FIG_W_IN, _FIG_H_IN), facecolor="white")

    bottom = 0.29

    ax_pp = fig.add_axes([_LM, bottom, _AX_W, _AX_H])
    ax_adj = fig.add_axes([_LM + _AX_W + _GAP_X - _RIGHT_BODY_LEFT_SHIFT, bottom, _AX_W, _AX_H])

    _draw_condition_panel(ax_pp, pp_data, models, "PP attachment ambiguity",
                          show_ylabels=True, seed_base=6100)
    _draw_condition_panel(ax_adj, adj_data, models, "Adjunct attachment ambiguity",
                          show_ylabels=False, seed_base=7100)

    _recenter_titles_and_labels(fig, ax_pp, ax_adj,
                                adj_title_reference_dx=_RIGHT_BODY_LEFT_SHIFT)

    h1 = mpatches.Patch(facecolor=_COL_S1, edgecolor=_COL_S1_EDGE,
                        linewidth=0.6, alpha=0.88, label="Structure 1 (plausible)")
    h2 = mpatches.Patch(facecolor=_COL_S2, edgecolor=_COL_S2_EDGE,
                        linewidth=0.6, alpha=0.88, label="Structure 2 (implausible)")
    # Wide-spaced legend - matches the Tests 4-5 figure's panel-a legend.
    fig.legend(handles=[h1, h2], loc="lower center", ncol=2, fontsize=FS_LEGEND,
               frameon=False, bbox_to_anchor=(0.5, 0.085),
               handlelength=1.1, handletextpad=0.45, columnspacing=2.4,
               borderaxespad=0.0)

    paths.FIGURES_PAPER.mkdir(parents=True, exist_ok=True)
    out = paths.FIGURES_PAPER / "fig_test6.svg"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
