"""Figures for the IBM hardware run.

The honest story has four parts, one per panel:

  A  rho vs phi -- hardware sits just above the uniform FLOOR, far below the
     noiseless ceiling. Plotting hardware alone would be misleading: rho is
     not zero-based, so a fully decohered device still returns ~rho_uniform.
  B  signal retained, measured vs what the device noise model predicted.
     The model was optimistic by ~3.6x, which is a methodological result in
     its own right.
  C  the output distribution, sorted. Shows *how* the signal is lost: the
     ideal distribution concentrates on a few designs, the hardware one is
     nearly flat.
  D  the depth wall -- two-qubit gate count for the configuration that was
     run against the ones that were not, which is why p=1 on the smallest
     ladder instance is the only viable option.

Usage:
    python experiments/hardware_figures.py results/hardware_Q13.json \\
        --dry-run-json <predicted.json> --out figures/hardware_Q13.png
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from esdqaoa.problems import problem_registry

IDEAL_C = "#7b1d1d"     # noiseless simulation (the ceiling)
HW_C = "#1f5fd0"        # device (CVD deltaE 27.3 vs the maroon)
PRED_C = "#8a94a6"      # noise-model prediction
INK, MUTED = "#22272e", "#6b7684"
GRID = dict(ls="--", alpha=0.30, lw=0.8)

# Two-qubit gate counts after transpilation to heavy-hex connectivity,
# measured with experiments/resource_counts.py. Listed to show why deeper
# configurations were never submitted.
DEPTH_WALL = [("Q13\np=1", 663, True), ("Q13\np=2", 1293, False),
              ("Q17\np=2", 2498, False), ("Q20\np=2", 4159, False)]


def panel_rho(ax, sweep):
    phis = [r["phi"] for r in sweep]
    ax.plot(phis, [r["ideal"]["rho"] for r in sweep], "-D", color=IDEAL_C,
            ms=6, lw=1.8, label="noiseless simulation")
    ax.plot(phis, [r["hardware"]["rho"] for r in sweep], "-o", color=HW_C,
            ms=7, lw=1.8, label="ibm_marrakesh")
    ax.plot(phis, [r["uniform"]["rho"] for r in sweep], "--", color=MUTED,
            lw=1.5, label="uniform superposition (floor)")
    ax.fill_between(phis, [r["uniform"]["rho"] for r in sweep],
                    [r["hardware"]["rho"] for r in sweep], color=HW_C,
                    alpha=.12)
    ax.set_xlabel(r"$\varphi$", fontsize=9, color=INK)
    ax.set_ylabel(r"approximation ratio $\rho$", fontsize=9, color=INK)
    ax.set_title("A  Hardware sits just above the floor, far below the ceiling",
                 fontsize=9.5, color=INK, loc="left")
    ax.set_ylim(0, 0.95)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.grid(**GRID)


def panel_retained(ax, sweep, pred):
    phis = [r["phi"] for r in sweep]
    x = np.arange(len(phis))
    meas = [100 * r["signal_retained"] for r in sweep]
    ax.bar(x - 0.19, meas, 0.38, color=HW_C, label="measured on device")
    # the ratio goes in the TITLE, not the plot area: an annotation here
    # collides with the tallest predicted bar
    ratio = None
    if pred:
        ax.bar(x + 0.19, [100 * v for v in pred], 0.38, facecolor="none",
               edgecolor=PRED_C, lw=1.6, hatch="//",
               label="predicted by noise model")
        ratio = np.mean(pred) / max(np.mean([r["signal_retained"]
                                             for r in sweep]), 1e-9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.2f}" for p in phis], fontsize=8)
    ax.set_xlabel(r"$\varphi$", fontsize=9, color=INK)
    ax.set_ylabel("% of ideal signal retained\nabove uniform", fontsize=9,
                  color=INK)
    ax.set_title("B  Device noise models overestimate what survives"
                 + (f" (by {ratio:.1f}$\\times$)" if ratio else ""),
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(axis="y", **GRID)


def panel_distribution(ax, pb, rec):
    """Sorted probability over designs: ideal vs hardware vs uniform."""
    hw = np.array(sorted(rec["counts"].values(), reverse=True))
    # ideal distribution is not stored; recompute from the same angles
    from esdqaoa.fastsim import FastESDRunner
    from esdqaoa.simulate import default_alpha, default_beta_dep
    r = FastESDRunner(pb, rec.get("p", 1), rec["phi"],
                      default_alpha(pb), default_beta_dep(pb))
    ideal = np.sort(r.probabilities(rec["angles"]))[::-1]
    k = min(40, len(ideal))
    ax.plot(range(1, k + 1), ideal[:k], "-D", color=IDEAL_C, ms=4, lw=1.6,
            label="noiseless simulation")
    ax.plot(range(1, min(k, len(hw)) + 1), hw[:k], "-o", color=HW_C, ms=4,
            lw=1.6, label="ibm_marrakesh")
    ax.axhline(1 / 2 ** pb.N, color=MUTED, ls="--", lw=1.4,
               label=f"uniform ($1/2^{{{pb.N}}}$)")
    ax.set_yscale("log")
    ax.set_xlabel("design, ranked by probability", fontsize=9, color=INK)
    ax.set_ylabel("probability", fontsize=9, color=INK)
    ax.set_title(f"C  How the signal is lost ($\\varphi$={rec['phi']:.2f}): "
                 f"the device output is nearly flat",
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(**GRID)


def panel_wall(ax, measured_gates):
    labels = [d[0] for d in DEPTH_WALL]
    vals = [d[1] for d in DEPTH_WALL]
    cols = [HW_C if d[2] else PRED_C for d in DEPTH_WALL]
    x = np.arange(len(labels))
    ax.bar(x, vals, 0.6, color=cols)
    ax.axhline(measured_gates, color=INK, lw=1.4, ls="--")
    ax.text(len(labels) - 0.5, measured_gates, "  only configuration\n"
            "  that returned signal", va="bottom", ha="right", fontsize=8,
            color=INK)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8,
                color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("two-qubit gates (heavy-hex)", fontsize=9, color=INK)
    ax.set_title("D  The depth wall: deeper configurations were not submitted",
                 fontsize=9.5, color=INK, loc="left")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(axis="y", **GRID)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hardware_json", nargs="?",
                    default="results/hardware_Q13.json")
    ap.add_argument("--dry-run-json", default=None,
                    help="matching --dry-run output, for the panel-B overlay")
    ap.add_argument("--problem", default=None)
    ap.add_argument("--out", default="figures/hardware_Q13.png")
    args = ap.parse_args()

    data = json.load(open(args.hardware_json))
    name = args.problem or next(iter(data["runs"]))
    rec = data["runs"][name]
    sweep = rec["sweep"]
    pb = problem_registry()[name]

    pred = None
    if args.dry_run_json and os.path.exists(args.dry_run_json):
        dj = json.load(open(args.dry_run_json))["runs"][name]["sweep"]
        by_phi = {round(r["phi"], 4): r["signal_retained"] for r in dj}
        pred = [by_phi.get(round(r["phi"], 4)) for r in sweep]
        if any(v is None for v in pred):
            pred = None

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0))
    panel_rho(axes[0][0], sweep)
    panel_retained(axes[0][1], sweep, pred)
    mid = sweep[len(sweep) // 2]
    panel_distribution(axes[1][0], pb, {**mid, "p": rec.get("p", 1)})
    panel_wall(axes[1][1], sweep[0]["transpiled"]["two_qubit_gates"])
    for ax in axes.ravel():
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
        ax.set_axisbelow(True)

    tgt = data.get("target", "hardware")
    fig.suptitle(f"ESD-QAOA on {tgt}: {name}, p={rec.get('p', 1)}, "
                 f"{data['settings']['shots']} shots — feasibility, "
                 f"not performance",
                 fontsize=12.5, fontweight="bold", color=INK, y=0.985)
    fig.text(0.5, 0.005,
             f"Mean {100 * rec['mean_signal_retained']:.1f}% of the ideal "
             "signal survives. Output is statistically above uniform at 4 of "
             "5 $\\varphi$, so the circuit runs and retains measurable "
             "structure — but hardware does NOT approach simulation.",
             ha="center", fontsize=9, color="#33415c")
    fig.tight_layout(rect=[0, 0.022, 1, 0.955])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
