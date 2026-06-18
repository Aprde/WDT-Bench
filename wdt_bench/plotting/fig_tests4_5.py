"""Paper figure: Tests 4-5 combined (two panels, EN/ZH sub-columns).

* Panel a - per-model constituent rate in Test 4 (meaningful sentences) vs
  Test 5 (nonsense sentences), with Wilson CIs, the random-span baseline as
  a dashed tick, and per-model permutation tests (Test 4 vs Test 5);
* Panel b - per-model means of the node- vs parent-category rule, with each
  model's Test-4 and Test-5 means connected by a coloured line.

Output: ``figures/paper/fig_tests4_5.svg``.

The permutation seeds (4100 for English, 5100 for Chinese) keep the
significance labels deterministic across re-runs.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D, Bbox

from .. import paths
from ..io_utils import load_json
from ..stats import permutation_p, stars_from_p, wilson_interval
from ..trees import load_tree_map
from . import style

logger = logging.getLogger(__name__)

# ============================================================================
#  FONT-SIZE KNOB - edit this one number and re-run.
#
#  * 1.0  = the default 9-pt look.
#  * >1   = bigger text, useful when the figure will be SHRUNK in the
#           manuscript (so labels stay readable post-shrink).
#  * <1   = smaller text (rarely needed).
#
#  Every text element (titles, axis labels, ticks, legends, panel letters)
#  scales from this single number; vertical spacing between rows of text is
#  keyed off it too, so the layout constants below normally need no edits.
# ============================================================================
FONT_SCALE = 10.0 / 9.0
# ============================================================================

_BASE_FS = 9.0 * FONT_SCALE
FS_PANEL = 12.0
FS_TITLE = 12.0
FS_SUBTITLE = FS_AXIS = FS_TICK = FS_LEGEND = _BASE_FS
FS_SIG = 7.5

# -- Models ------------------------------------------------------------------
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

# One hue family, but saturated and spaced so all six markers stay visually
# distinct; the three warm entries (GPT / Gemini / Grok) are deliberately
# pulled apart in both hue and lightness.
_MODEL_COLORS = {
    "qwen-max": "#6E5FAE",                      # deep violet
    "deepseek-v4-pro": "#8A8AA0",               # cool slate (purple-gray)
    "claude-opus-4-7": "#D49A7A",               # warm cream-tan
    "gpt-5.5": "#E68A4A",                       # bright peach-orange
    "gemini-3.1-pro-preview": "#B22830",        # clear deep red
    "grok-4.20-0309-non-reasoning": "#5C1A48",  # very deep wine
}

# Test 4 = purple, Test 5 = pink-red.
_COL_T4 = "#B6B3D6"
_COL_T4_EDGE = "#8C88B8"
_COL_T5 = "#E9687A"
_COL_T5_EDGE = "#D44A60"

# Marker shapes distinguishing the two tests in panel b (same area).
_MK_T4 = "o"  # Test 4 = circle
_MK_T5 = "s"  # Test 5 = square
_MK_SIZE = 48

# -- Layout (figure fraction) ------------------------------------------------
W_IN, H_IN = 16.0 / 2.54, 7.4  # ~6.30 x 7.40 in
TOP_CROP_IN = 0.55

# Left margin large enough that the longest model name ("DeepSeek V4") clears
# the panel-a y-axis without clipping.
LM, RM = 0.135, 0.955

SC_W = 0.34                          # subpanel width
GAP_X = 0.48 - SC_W                  # shared two-column y-axis spacing
RIGHT_COLUMN_LEFT_SHIFT = 0.025
X_EN = LM
X_ZH = LM + SC_W + GAP_X - RIGHT_COLUMN_LEFT_SHIFT
X_CENTER = (X_EN + X_ZH + SC_W) / 2
X_LIM = (0.0, 1.10)
X_TICKS = [0, 0.25, 0.5, 0.75, 1.0]
X_TICK_LABELS = ["0.00", "0.25", "0.50", "0.75", "1.00"]

SC_H = SC_W * W_IN / H_IN            # square reference height
A_H = 0.25                           # panel a - taller for wider bars
B_SC_H = A_H                         # same axis box height as panel a

# Vertical spacing in figure fractions, keyed off font size.
_LINE_F = (_BASE_FS * 1.6) / 72 / H_IN

# Bottom legend lives in TWO rows (8 entries x 4 cols), centred at B_LEG_Y.
B_LEG_Y = 0.020 + _LINE_F * 1.2
B_LEG_X_SHIFT = 0.018
B_BOT = B_LEG_Y + _LINE_F * 3.6      # clear xlabel + 2-row legend
B_TOP = B_BOT + B_SC_H
B_SUB_Y = B_TOP + _LINE_F * 0.7
B_TITLE_Y = B_SUB_Y + _LINE_F * 1.15

A_LEG_Y = B_TITLE_Y + _LINE_F * 1.35  # between panel-a xlabel and panel-b title
A_BOT = B_TITLE_Y + _LINE_F * 4.05    # clear panel a xlabel + moved legend
A_TOP = A_BOT + A_H
A_SUB_Y = A_TOP + _LINE_F * 0.75
A_TITLE_Y = A_SUB_Y + _LINE_F * 0.9


def _apply_font() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    })


def _wilson_ci(p: float, n: int):
    """Wilson interval around an observed rate (mean of per-trial rates)."""
    if n == 0:
        return p, p
    k = p * n
    return wilson_interval(k, n)


def _draw_centered_star(ax, x: float, y: float) -> None:
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


def _draw_sig_label(ax, x: float, y: float, label: str) -> None:
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


def _lookup_tree(sentence: str, tree_map: dict[str, str]) -> str:
    sent = (sentence or "").strip()
    return tree_map.get(sent) or tree_map.get(sent.lower()) or ""


@lru_cache(maxsize=4096)
def _random_span_baselines(tree_str: str) -> tuple[float, float]:
    """Random-span baselines: (task target constituent, any single constituent)."""
    if not tree_str:
        return 0.0, 0.0
    try:
        from nltk.tree import Tree

        tree = Tree.fromstring(tree_str)
    except Exception:
        return 0.0, 0.0

    leaves = tree.leaves()
    n = len(leaves)
    # Match the random-deletion baseline used elsewhere in the project:
    # exclude trivial one-word spans and the whole sentence, which otherwise
    # inflate the baseline because nearly every single word is a subtree.
    eligible_spans = {
        (start, start + length)
        for length in range(2, n)
        for start in range(0, n - length + 1)
    }
    total = len(eligible_spans)
    if total <= 0:
        return 0.0, 0.0

    span_by_id: dict[int, tuple[int, int]] = {}

    def collect(subtree, pos: int) -> int:
        if not hasattr(subtree, "label"):
            return pos + 1
        start = pos
        for child in subtree:
            pos = collect(child, pos)
        span_by_id[id(subtree)] = (start, pos)
        return pos

    collect(tree, 0)
    single_spans = set(span_by_id.values()) & eligible_spans
    target_spans: set[tuple[int, int]] = set()

    for subtree in tree.subtrees():
        if not hasattr(subtree, "label") or subtree.label() != "VP":
            continue
        for child in subtree:
            if not hasattr(child, "label") or not child.label().startswith("PP"):
                continue
            pp_span = span_by_id.get(id(child))
            if pp_span:
                target_spans.add(pp_span)
            for pp_child in child:
                if hasattr(pp_child, "label") and pp_child.label().startswith("NP"):
                    np_span = span_by_id.get(id(pp_child))
                    if np_span:
                        target_spans.add(np_span)
                    break
            if target_spans:
                break
        if target_spans:
            break

    target_spans &= eligible_spans
    return len(target_spans) / total, len(single_spans) / total


def _attach_baselines(doc: dict, tree_map: dict[str, str]) -> None:
    """Add random-span baseline rates to each ``per_trial_rates`` entry.

    The conditional single-constituent rates are already computed by the
    Test 4/5 analysis; only the tree-derived baselines are figure-specific.
    """
    grouped: dict[tuple[str, str, int], list[dict]] = {}
    for row in doc.get("results", []):
        key = (row.get("model"), row.get("language"), row.get("trial"))
        if all(k is not None for k in key):
            grouped.setdefault(key, []).append(row)

    for entry in doc.get("summary_by_model", []):
        model = entry.get("model")
        language = entry.get("language")
        if not model or not language:
            continue
        for pf in entry.get("per_trial_rates", []):
            rows = grouped.get((model, language, pf.get("trial")), [])
            if not rows:
                continue
            baseline_pairs = [
                _random_span_baselines(_lookup_tree(row.get("sentence", ""), tree_map))
                for row in rows
            ]
            pf["baseline_target_constituent_rate"] = float(
                np.mean([x[0] for x in baseline_pairs])
            )
            pf["baseline_single_constituent_rate"] = float(
                np.mean([x[1] for x in baseline_pairs])
            )


def _constituent_rates(doc, language: str):
    out = {}
    for entry in doc.get("summary_by_model", []):
        if entry.get("language") != language:
            continue
        model = entry.get("model")
        pf = entry.get("per_trial_rates", [])
        if not pf:
            continue
        vals = [r["node_rule_rate"] + r["parent_rule_rate"] for r in pf]
        mean = float(np.mean(vals))
        lo, hi = _wilson_ci(mean, len(vals))
        baselines = [
            float(r.get("baseline_single_constituent_rate", 0.0) or 0.0)
            for r in pf
        ]
        out[model] = {
            "mean": mean,
            "lo": lo,
            "hi": hi,
            "vals": np.array(vals, float),
            "baseline": float(np.mean(baselines)) if baselines else 0.0,
        }
    return out


def _single_constituent_rule_rate(row: dict, rule: str) -> float:
    """Conditional rate used by panel b: rule outcomes / single constituents."""
    key = f"{rule}_rule_given_single_constituent_rate"
    if key in row:
        return float(row.get(key) or 0.0)

    node_raw = float(row.get("node_rule_rate", 0.0) or 0.0)
    parent_raw = float(row.get("parent_rule_rate", 0.0) or 0.0)
    denom = node_raw + parent_raw
    if denom <= 0:
        return 0.0
    if rule == "node":
        return node_raw / denom
    if rule == "parent":
        return parent_raw / denom
    raise ValueError(f"Unknown rule: {rule}")


def _rule_data(doc, language: str):
    """Per-model dict with per-trial arrays and the mean (node_x, parent_y)."""
    out = {}
    for model in _MODEL_ORDER:
        entry = next(
            (e for e in doc.get("summary_by_model", [])
             if e.get("model") == model and e.get("language") == language),
            None,
        )
        if not entry:
            continue
        pf = entry.get("per_trial_rates", [])
        if not pf:
            continue
        # node (NP deletions) on x, parent (PP deletions) on y.
        xs = np.array([_single_constituent_rule_rate(r, "node") for r in pf], float)
        ys = np.array([_single_constituent_rule_rate(r, "parent") for r in pf], float)
        out[model] = {
            "x": xs,
            "y": ys,
            "mean": (float(np.mean(xs)), float(np.mean(ys))),
        }
    return out


def _draw_constituent_lang_panel(ax, t4_data, t5_data, models, show_ylabels, *, seed_base: int):
    n = len(models)
    bar_h = 0.36
    gap = 0.22
    ax.set_xlim(*X_LIM)
    ax.set_ylim(-0.55, n - 0.45)

    for idx, m in enumerate(models):
        y_base = n - 1 - idx
        y4, y5 = y_base + gap, y_base - gap

        for (d, y_pos, c_fill, c_edge) in (
            (t4_data[m], y4, _COL_T4, _COL_T4_EDGE),
            (t5_data[m], y5, _COL_T5, _COL_T5_EDGE),
        ):
            w = d["mean"]
            ax.barh(y_pos, w, height=bar_h, color=c_fill,
                    edgecolor=c_edge, linewidth=0.6, alpha=0.88, zorder=3)
            base = d.get("baseline", 0.0)
            ax.vlines(base, y_pos - bar_h / 2, y_pos + bar_h / 2,
                      color="#555555", linewidth=1.1,
                      linestyles=(0, (2.2, 1.6)), zorder=6)
            ax.errorbar(w, y_pos,
                        xerr=[[w - d["lo"]], [d["hi"] - w]], fmt="none",
                        ecolor=c_edge, elinewidth=0.9, capsize=2.0,
                        capthick=0.8, zorder=4)

        p = permutation_p(t4_data[m]["vals"], t5_data[m]["vals"],
                          seed=seed_base + idx)
        label = stars_from_p(p)
        x_star = min(
            1.085,
            max(t4_data[m]["hi"], t5_data[m]["hi"],
                t4_data[m]["mean"], t5_data[m]["mean"]) + 0.055,
        )
        _draw_sig_label(ax, x_star, y_base, label)

    y_ticks = [n - 1 - i for i in range(n)]
    ax.set_yticks(y_ticks)
    if show_ylabels:
        ax.set_yticklabels([_MODEL_LABELS.get(m, m) for m in models],
                           fontsize=FS_AXIS)
    else:
        ax.set_yticklabels([])

    ax.set_xlim(*X_LIM)
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels(X_TICK_LABELS, fontsize=FS_TICK)
    ax.set_xlabel("Constituent rate",
                  fontsize=FS_AXIS, fontweight="normal", labelpad=7)
    ax.set_ylim(-0.55, n - 0.45)
    for i in range(n - 1):
        ax.axhline(n - 1 - i - 0.5, color="#ededed", lw=0.5, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)


def _draw_rule_lang_panel(ax, t4_data, t5_data, models, show_ylabel: bool):
    """Per-model rule means in one language."""
    for m in models:
        if m not in t4_data or m not in t5_data:
            continue
        col = _MODEL_COLORS.get(m, "#555555")

        # Line between the two means + the mean markers on top.
        x4m, y4m = t4_data[m]["mean"]
        x5m, y5m = t5_data[m]["mean"]
        ax.plot([x4m, x5m], [y4m, y5m], color=col, lw=1.2,
                alpha=0.85, zorder=4)
        ax.scatter([x4m], [y4m], marker=_MK_T4, s=_MK_SIZE, color=col,
                   edgecolors="white", linewidths=0.9, zorder=5)
        ax.scatter([x5m], [y5m], marker=_MK_T5, s=_MK_SIZE, color=col,
                   edgecolors="white", linewidths=0.9, zorder=5)

    ax.plot([0, 1], [0, 1], color="#d2d2d2", lw=0.8, ls="--", zorder=0)
    ax.set_xlim(*X_LIM)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(X_TICKS)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(X_TICK_LABELS, fontsize=FS_TICK)
    ax.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"], fontsize=FS_TICK)
    ax.set_xlabel("Explained ratio of node-category rule",
                  fontsize=FS_AXIS, fontweight="normal", labelpad=6)
    if show_ylabel:
        ax.set_ylabel("Explained ratio of parent-category rule",
                      fontsize=FS_AXIS, fontweight="normal", labelpad=6)
    for t in (0.25, 0.5, 0.75):
        ax.axhline(t, color="#f0f0f0", lw=0.5, zorder=0)
        ax.axvline(t, color="#f0f0f0", lw=0.5, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main() -> None:
    style.apply_style()
    _apply_font()

    p4 = paths.diagnostic_classified(4)
    p5 = paths.diagnostic_classified(5)
    for p, k in ((p4, 4), (p5, 5)):
        if not p.is_file():
            raise RuntimeError(
                f"Test {k} classified results not found: {p}; "
                f"run `analyze_results.py --test {k}` first."
            )
    doc4 = load_json(p4)
    doc5 = load_json(p5)
    _attach_baselines(doc4, load_tree_map(paths.diagnostic_trees("parallel")))
    _attach_baselines(doc5, load_tree_map(paths.diagnostic_trees("nonsense")))

    t4_en = _constituent_rates(doc4, "english")
    t4_zh = _constituent_rates(doc4, "chinese")
    t5_en = _constituent_rates(doc5, "english")
    t5_zh = _constituent_rates(doc5, "chinese")

    r4_en = _rule_data(doc4, "english")
    r4_zh = _rule_data(doc4, "chinese")
    r5_en = _rule_data(doc5, "english")
    r5_zh = _rule_data(doc5, "chinese")

    models = [m for m in _MODEL_ORDER
              if m in t4_en and m in t4_zh and m in t5_en and m in t5_zh]
    if not models:
        raise RuntimeError("No model has complete Test 4 + Test 5 data in both languages.")

    fig = plt.figure(figsize=(W_IN, H_IN), facecolor="white")

    # -- Panel a - constituent proportion -------------------------------------
    ax_a_en = fig.add_axes([X_EN, A_BOT, SC_W, A_H])
    ax_a_zh = fig.add_axes([X_ZH, A_BOT, SC_W, A_H])
    _draw_constituent_lang_panel(ax_a_en, t4_en, t5_en, models, True, seed_base=4100)
    _draw_constituent_lang_panel(ax_a_zh, t4_zh, t5_zh, models, False, seed_base=5100)

    title_a = fig.text(X_CENTER, A_TITLE_Y,
                       "Constituent rate in Test 4 and Test 5",
                       ha="center", va="center", fontsize=FS_TITLE,
                       fontweight="bold")
    leg_a = fig.legend(
        handles=[
            mpatches.Patch(facecolor=_COL_T4, edgecolor=_COL_T4_EDGE,
                           linewidth=0.6, alpha=0.88,
                           label="Test 4 (meaningful sentences)"),
            mpatches.Patch(facecolor=_COL_T5, edgecolor=_COL_T5_EDGE,
                           linewidth=0.6, alpha=0.88,
                           label="Test 5 (nonsense sentences)"),
        ],
        loc="center", ncol=2, fontsize=FS_LEGEND,
        frameon=False, bbox_to_anchor=(X_CENTER, A_LEG_Y),
        handlelength=1.1, handletextpad=0.45, columnspacing=2.4)
    fig.text(X_EN + SC_W / 2, A_SUB_Y, "English",
             ha="center", va="center", fontsize=FS_SUBTITLE)
    fig.text(X_ZH + SC_W / 2, A_SUB_Y, "Chinese",
             ha="center", va="center", fontsize=FS_SUBTITLE)

    # -- Panel b - per-model means, Test 4 x Test 5 linked --------------------
    ax_b_en = fig.add_axes([X_EN, B_BOT, SC_W, B_SC_H])
    ax_b_zh = fig.add_axes([X_ZH, B_BOT, SC_W, B_SC_H])
    _draw_rule_lang_panel(ax_b_en, r4_en, r5_en, models, show_ylabel=True)
    _draw_rule_lang_panel(ax_b_zh, r4_zh, r5_zh, models, show_ylabel=False)
    title_b = fig.text(X_CENTER, B_TITLE_Y,
                       "Explained ratio of each rule in Test 4 and Test 5",
                       ha="center", va="center", fontsize=FS_TITLE,
                       fontweight="bold")
    fig.text(X_EN + SC_W / 2, B_SUB_Y, "English",
             ha="center", va="center", fontsize=FS_SUBTITLE)
    fig.text(X_ZH + SC_W / 2, B_SUB_Y, "Chinese",
             ha="center", va="center", fontsize=FS_SUBTITLE)

    # Shared bottom legend: model colours + test-marker shapes.
    leg_handles = [
        mlines.Line2D([], [], marker="o", color="w",
                      markerfacecolor=_MODEL_COLORS.get(m, "#555555"),
                      markeredgecolor="white",
                      markersize=6.2, label=_MODEL_LABELS.get(m, m), linewidth=0)
        for m in models
    ] + [
        mlines.Line2D([], [], marker=_MK_T4, color="#555",
                      markerfacecolor="#555", markeredgecolor="white",
                      markersize=6.2, label="Test 4 (meaningful sentences)", linewidth=0),
        mlines.Line2D([], [], marker=_MK_T5, color="#555",
                      markerfacecolor="#555", markeredgecolor="white",
                      markersize=6.2, label="Test 5 (nonsense sentences)", linewidth=0),
    ]
    # 8 entries laid out in 2 rows of 4 - a single row was too wide to fit.
    leg_b = fig.legend(handles=leg_handles, loc="center",
                       ncol=4, fontsize=FS_LEGEND,
                       frameon=False, bbox_to_anchor=(X_CENTER, B_LEG_Y),
                       handlelength=1.0, handletextpad=0.4,
                       columnspacing=1.1, labelspacing=0.7)

    # -- Panel letters ---------------------------------------------------------
    for letter, y in (("a", A_TITLE_Y), ("b", B_TITLE_Y)):
        fig.text(X_EN - 0.055, y, letter,
                 fontsize=FS_PANEL, fontweight="bold",
                 va="center", ha="left")

    # Centre the two titles and both shared legends on the figure's visual
    # mid-line.  The EN sub-panels carry y-axis tick labels, so the inked
    # content reaches further left than the axes block - keying these
    # full-width elements to the axes-block centre (X_CENTER) left them
    # noticeably shifted right.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    xs = []
    for ax in (ax_a_en, ax_a_zh, ax_b_en, ax_b_zh):
        bb = ax.get_tightbbox(renderer)
        xs.append(inv.transform((bb.x0, bb.y0))[0])
        xs.append(inv.transform((bb.x1, bb.y1))[0])
    content_cx = (min(xs) + max(xs)) / 2
    title_a.set_x(content_cx)
    title_b.set_x(content_cx)
    leg_a.set_bbox_to_anchor((content_cx, A_LEG_Y), transform=fig.transFigure)
    leg_b.set_bbox_to_anchor((content_cx + B_LEG_X_SHIFT, B_LEG_Y),
                             transform=fig.transFigure)

    paths.FIGURES_PAPER.mkdir(parents=True, exist_ok=True)
    out = paths.FIGURES_PAPER / "fig_tests4_5.svg"
    fig.savefig(
        out,
        facecolor="white",
        bbox_inches=Bbox.from_bounds(0, 0, W_IN, H_IN - TOP_CROP_IN),
        pad_inches=0,
    )
    plt.close(fig)
    logger.info("Saved %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
