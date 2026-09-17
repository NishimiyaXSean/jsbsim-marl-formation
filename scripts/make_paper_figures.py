"""Figure 1 (framework sketch) and Figure 3 (difficulty robustness) for the small paper.

Why this script exists: both figures must agree with the numbers quoted in the
text, so every value plotted here is read from the live eval artifacts under
results/shoot_eval/ instead of being typed in. If an artifact is missing (e.g.
the fresh-env expert CLR is still being measured) the script renders an honest
placeholder and says so on stderr -- it never substitutes a number from a
different measurement basis.

Usage (WSL Ubuntu shell, repo root):
  /home/sean/miniconda3/envs/marl_env/bin/python scripts/make_paper_figures.py
  /home/sean/miniconda3/envs/marl_env/bin/python scripts/make_paper_figures.py --outdir results/shoot_eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(REPO_ROOT, "results", "shoot_eval")

# Artifacts used. All three are n=400, seeds 20000-20399, geometry U(0,60),
# deterministic masked argmax, fresh env per episode.
E7_BC = "E7_bc_round1_d0_n400_s20000.json"
E7_SPC = "E7_spc_d0_n400_s20000.json"
E7_PAIRED = "E7_paired_bc_vs_spc_d0_n400.json"
E5_BC = "E5_bc_round1_d03_n400_s20000.json"
E5_SPC = "E5_spc_d03_n400_s20000.json"
E5_PAIRED = "E5_paired_bc_vs_spc_d03_n400.json"
E3_D0 = "E3_paired_bc_vs_expert_d0_n400_s20000_v2.json"
E3_D03 = "E3_paired_bc_vs_expert_d03_n400_s20000.json"
# Written by the fresh-env expert-CLR run; optional at render time.
E3_D0_CLR = "E3_paired_bc_vs_expert_d0_n400_s20000_v3_clr.json"

INK = "#222222"
MUTED = "#6b7280"
C_EXPERT = "#9aa5b1"
C_BC = "#dd8452"
C_SPC = "#4c9f70"
C_BOX = "#eef2f7"


# --------------------------------------------------------------------------- io


def load(name):
    path = os.path.join(EVAL_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, "r") as handle:
        return json.load(handle)


def require(name):
    data = load(name)
    if data is None:
        sys.exit("missing required artifact: %s" % name)
    return data


def single(name):
    """Kill counts from one eval_bc_1v1 artifact."""
    d = require(name)
    return {
        "n": int(d["episodes"]),
        "kills": int(d["termination_reasons"].get("target_killed", 0)),
        "kill": 100.0 * float(d["kill_rate"]),
        "clr": 100.0 * float(d["clr"]),
        "clr_num": int(d["clr_fire_commands"]),
        "clr_den": int(d["clr_allowed_steps"]),
    }


def expert_arm(name):
    """Kill counts for the rule-expert arm of an E3 paired artifact."""
    d = require(name)
    arm = d["expert"]
    n = int(arm["n"])
    return {"n": n, "kills": int(round(float(arm["kill_rate"]) * n)), "kill": 100.0 * float(arm["kill_rate"])}


def wilson_95(k, n):
    """Wilson score interval (no scipy), returned in percent."""
    if n == 0:
        return 0.0, 0.0
    z = 1.959963985
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return 100.0 * max(0.0, centre - half), 100.0 * min(1.0, centre + half)


# ------------------------------------------------------------------- figure 1


def _box(ax, x, w, title, body, facecolor, edgecolor):
    y_top, y_bot = 0.74, 0.14
    patch = FancyBboxPatch(
        (x, y_bot),
        w,
        y_top - y_bot,
        boxstyle="round,pad=0.010",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y_top - 0.10,
        title,
        ha="center",
        va="center",
        fontsize=11.0,
        fontweight="bold",
        color=INK,
        zorder=3,
    )
    ax.text(
        x + w / 2,
        (y_top + y_bot) / 2 - 0.07,
        body,
        ha="center",
        va="center",
        fontsize=8.4,
        linespacing=1.65,
        color="#3d3d3d",
        zorder=3,
    )
    return x + w / 2, y_top, y_bot


def make_figure_1(outdir, bc_d0, spc_d0, expert_clr):
    fig, ax = plt.subplots(figsize=(10.6, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    w = 0.20
    xs = [0.015, 0.272, 0.529, 0.786]

    _box(
        ax,
        xs[0],
        w,
        "Rule-based expert",
        "flies the engagement, but\nlaunches only when its\nown quality gate passes",
        "#ffffff",
        C_EXPERT,
    )
    _box(
        ax,
        xs[1],
        w,
        "Behaviour cloning",
        "imitates the demonstration;\nreproduces the gate\nfaithfully, refusals included",
        "#ffffff",
        C_BC,
    )

    if expert_clr is None:
        expert_line = "expert:  pending"
    else:
        expert_line = "expert:  %.2f%%" % expert_clr
    _box(
        ax,
        xs[2],
        w,
        "C1:  diagnosis (CLR)",
        "CLR = P(a_fire = 1\n| m_fire = 1)\n" + expert_line + "\nBC:       %.2f%%" % bc_d0["clr"],
        C_BOX,
        "#4c72b0",
    )
    _box(
        ax,
        xs[3],
        w,
        "C2:  correction",
        "re-train the launch head\nonly; maneuver heads stay\nfrozen (logit diff < 1e-9)",
        C_BOX,
        C_SPC,
    )

    # Arrows between the four stages.
    y = 0.44
    for i in range(3):
        x0 = xs[i] + w
        x1 = xs[i + 1]
        ax.add_patch(
            FancyArrowPatch(
                (x0, y),
                (x1, y),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.3,
                color="#404040",
                shrinkA=2,
                shrinkB=2,
                zorder=4,
            )
        )

    ax.text(
        0.5,
        0.045,
        "the maneuver is bit-identical before and after the correction, so the "
        "outcome change is attributable to the launch decision alone",
        ha="center",
        fontsize=9.2,
        style="italic",
        color=MUTED,
    )

    out_png = os.path.join(outdir, "framework_fig1.png")
    out_pdf = os.path.join(outdir, "framework_fig1.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png, out_pdf


# ------------------------------------------------------------------- figure 3


def make_figure_3(outdir, rows, gaps):
    """rows: list of (difficulty label, [(name, kills, n, color), ...]).

    gaps: list of (gap_pp, detail) parallel to rows -- the paired SPC - BC
    difference and its discordant / p-value detail, read from the paired
    artifacts, so the annotation cannot drift from the reported test.
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    n_groups = len(rows)
    n_bars = len(rows[0][1])
    group_w = 0.74
    bar_w = group_w / n_bars

    labels = []
    for gi, (label, bars) in enumerate(rows):
        base = gi + (1 - group_w) / 2
        labels.append(label)
        tops = {}
        for bi, (name, kills, n, color) in enumerate(bars):
            x = base + bi * bar_w
            pct = 100.0 * kills / n
            lo, hi = wilson_95(kills, n)
            tops[name] = (x, pct)
            ax.bar(
                x,
                pct,
                width=bar_w * 0.86,
                color=color,
                edgecolor="#404040",
                linewidth=0.7,
                zorder=3,
            )
            ax.errorbar(
                x,
                pct,
                yerr=[[pct - lo], [hi - pct]],
                fmt="none",
                ecolor="#2f2f2f",
                elinewidth=0.9,
                capsize=3,
                zorder=4,
            )
            ax.text(
                x,
                hi + 3.0,
                "%.2f%%\n%d/%d" % (pct, kills, n),
                ha="center",
                va="bottom",
                fontsize=8.3,
                linespacing=1.35,
                color=INK,
            )

        # Paired gap annotation: a horizontal span above the two bars with
        # dotted leaders down to each bar top. The discordant counts and exact
        # p-value live in Table 4 and the caption, not in the plot, to keep the
        # figure legible.
        gap_pp, _detail = gaps[gi]
        x_bc, y_bc = tops["bc"]
        x_spc, y_spc = tops["spc"]
        y_arrow = 106.0
        for x_bar, y_bar in ((x_bc, y_bc), (x_spc, y_spc)):
            ax.plot(
                [x_bar, x_bar],
                [y_bar + 11.0, y_arrow],
                linestyle=":",
                linewidth=0.9,
                color="#8c3b3b",
                zorder=4,
            )
        ax.add_patch(
            FancyArrowPatch(
                (x_bc, y_arrow),
                (x_spc, y_arrow),
                arrowstyle="<|-|>",
                mutation_scale=9,
                linewidth=1.1,
                color="#8c3b3b",
                zorder=5,
            )
        )
        ax.text(
            (x_bc + x_spc) / 2.0,
            y_arrow + 2.0,
            "+%.2f pp" % gap_pp,
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#8c3b3b",
            zorder=5,
        )

    ax.set_xticks([i + 0.5 for i in range(n_groups)])
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("Kill rate", fontsize=11)
    ax.set_ylim(0, 116)
    ax.set_xlim(0, n_groups)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, color="#b9b9b9", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="#404040", linewidth=0.7)
        for c in (C_EXPERT, C_BC, C_SPC)
    ]
    ax.legend(
        handles,
        ["rule expert", "BC (inherited launch policy)", "SPC (corrected launch head)"],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9.2,
        bbox_to_anchor=(0.5, 1.005),
    )

    out_png = os.path.join(outdir, "robustness_fig3.png")
    out_pdf = os.path.join(outdir, "robustness_fig3.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png, out_pdf


# ----------------------------------------------------------------------- main


def paired_gap(name):
    """Paired SPC - BC difference plus its test detail, from a paired artifact.

    Reads kill_rate.paired_diff_pp / contingency / mcnemar by name, so the
    annotation always matches the test the text reports.
    """
    d = require(name)
    pp = float(d["kill_rate"]["paired_diff_pp"])
    cont = d["contingency"]
    p = float(d["mcnemar"]["exact_two_sided_p"])
    detail = "paired SPC-BC\n%d:%d discordant\np = %.2e" % (
        cont["discordant_favouring_b"],
        cont["discordant_total"] - cont["discordant_favouring_b"],
        p,
    )
    return pp, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=EVAL_DIR)
    args = ap.parse_args()
    if not os.path.isdir(args.outdir):
        sys.exit("outdir does not exist: %s" % args.outdir)

    bc_d0 = single(E7_BC)
    spc_d0 = single(E7_SPC)
    bc_d03 = single(E5_BC)
    spc_d03 = single(E5_SPC)
    exp_d0 = expert_arm(E3_D0)
    exp_d03 = expert_arm(E3_D03)

    clr_run = load(E3_D0_CLR)
    expert_clr = None
    if clr_run is not None and not clr_run.get("run_meta", {}).get("complete", False):
        print("note: %s exists but is not complete yet -- expert CLR left blank" % E3_D0_CLR, file=sys.stderr)
    elif clr_run is not None:
        expert_clr = 100.0 * float(clr_run["expert"]["clr"])
    else:
        print("note: %s not found -- expert CLR left blank" % E3_D0_CLR, file=sys.stderr)

    f1 = make_figure_1(args.outdir, bc_d0, spc_d0, expert_clr)

    rows = [
        ("straight-and-level target\n(difficulty 0)",
         [("expert", exp_d0["kills"], exp_d0["n"], C_EXPERT),
          ("bc", bc_d0["kills"], bc_d0["n"], C_BC),
          ("spc", spc_d0["kills"], spc_d0["n"], C_SPC)]),
        ("evading target\n(difficulty 0.3)",
         [("expert", exp_d03["kills"], exp_d03["n"], C_EXPERT),
          ("bc", bc_d03["kills"], bc_d03["n"], C_BC),
          ("spc", spc_d03["kills"], spc_d03["n"], C_SPC)]),
    ]
    gaps = [paired_gap(E7_PAIRED), paired_gap(E5_PAIRED)]
    f3 = make_figure_3(args.outdir, rows, gaps)

    print("wrote %s (+.pdf)" % f1[0])
    print("wrote %s (+.pdf)" % f3[0])
    print(
        "source: d=0 expert %.2f%% / BC %.2f%% / SPC %.2f%% ; "
        "d=0.3 expert %.2f%% / BC %.2f%% / SPC %.2f%%"
        % (exp_d0["kill"], bc_d0["kill"], spc_d0["kill"], exp_d03["kill"], bc_d03["kill"], spc_d03["kill"])
    )


if __name__ == "__main__":
    main()
