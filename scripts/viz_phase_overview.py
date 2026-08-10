"""Phase 1 overview chart: kill-rate progression + final-holdout bars."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    outdir = "results/ctrl_viz"
    os.makedirs(outdir, exist_ok=True)

    # version progression (ID distribution)
    versions = ["v19 (PPO)", "BC round1", "ASAP distill", "v2 speed-distill"]
    kills = [10.0, 43.2, 91.2, 99.9]
    lost = [32.0, 0.0, 0.0, 0.0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    bars = ax.bar(versions, kills, color=["#7f7f7f", "#1f77b4", "#2ca02c",
                                          "#d62728"], alpha=0.9)
    for b, k in zip(bars, kills):
        ax.text(b.get_x() + b.get_width() / 2, k + 1.2, f"{k:.1f}%",
                ha="center", fontsize=10)
    ax.set_ylabel("ID kill rate (%)")
    ax.set_title("Phase 1 kill-rate progression (ID 500 seeds)")
    ax.set_ylim(0, 112)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(versions, lost, color=["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"],
           alpha=0.9)
    for i, l in enumerate(lost):
        ax.text(i, l + 0.7, f"{l:.0f}%", ha="center", fontsize=10)
    ax.set_ylabel("ID lost-target rate (%)")
    ax.set_title("lost-target (v19 32% -> 0%)")
    ax.set_ylim(0, 36)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(outdir, "phase1_kill_progression.png")
    fig.savefig(p1, dpi=140, facecolor="white")
    plt.close(fig)

    # final holdout summary
    h = json.load(open("results/shoot_eval/final_holdout.json"))
    groups = h["groups"]
    labels = ["ID", "dist2_3k", "OOD cells", "stress"]
    idg, d2g = groups["id"], groups["dist2_3k"]
    cell_k = [g["kill"] for k, g in groups.items() if k.startswith("cell_")]
    st_k = [g["kill"] for k, g in groups.items() if k.startswith("stress_")]
    values = [idg["kill"] * 100, d2g["kill"] * 100,
              sum(cell_k) / len(cell_k) * 100, sum(st_k) / len(st_k) * 100]
    losts = [idg["lost"], d2g["lost"],
             sum(g["lost"] for g in groups.values() if g["lost"] > 0), 0]
    fig2, ax2 = plt.subplots(figsize=(9, 4.8))
    bars = ax2.bar(labels, values, color=["#1f77b4", "#ff7f0e", "#2ca02c",
                                          "#d62728"], alpha=0.9)
    for b, v in zip(bars, values):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}%",
                 ha="center", fontsize=10)
    ax2.set_ylabel("kill rate (%)")
    ax2.set_title("v2 final holdout (preregistered) — observed lost = 0, bad = 0")
    ax2.set_ylim(0, 108)
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    p2 = os.path.join(outdir, "phase1_final_holdout.png")
    fig2.savefig(p2, dpi=140, facecolor="white")
    plt.close(fig2)
    print(f"saved {p1}")
    print(f"saved {p2}")


if __name__ == "__main__":
    main()
