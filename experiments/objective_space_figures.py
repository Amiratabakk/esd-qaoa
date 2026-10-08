"""Brute-force objective spaces and true Pareto frontiers.

Problem CHARACTERIZATION, not method comparison: this figure contains no
quantum results at all. Everything is exact brute force over the feasible
set, so it is fully deterministic -- no optimizer, no seed, nothing to
average over. It exists to justify the instances: it shows that each front
really curves (diminishing returns) and really has unsupported points sitting
below the convex envelope.

Layers per panel:
  feasible vectors     every feasible (cost, performance) pair, faint
  exact Pareto front   the non-dominated set, connected
  supported envelope   upper convex hull -- what a weighted sum can reach
  unsupported points   Pareto-optimal but BELOW the envelope, so no weighted
                       sum recovers them at any phi

Usage:
    python experiments/objective_space_figures.py --out figures/objective_space.png
    python experiments/objective_space_figures.py --problems I II III IV V \
        --unsupported-marker square --out figures/objective_space_IV.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from esdqaoa.problems import problem_registry

FEAS_C = "#9aa5b1"
FRONT_C = "#1f5fd0"
HULL_C = "#1a7f37"
UNSUP_C = "#b3261e"

MARKERS = {                    # (marker, filled?)
    "ring": ("o", False),
    "square": ("s", False),
    "diamond": ("D", False),
    "triangle": ("^", False),
    "dot": ("o", True),
}


def panel(ax, name, pb, unsup_marker="ring", show_feasible=True):
    cost, perf, _ = pb._tables()
    front = pb.pareto_front()
    hull = pb.supported_pareto_points()
    unsup = [(c, p) for c, p, _ in front if (c, p) not in set(hull)]

    if show_feasible:
        # jitter-free; many designs share objective vectors, so alpha stacks
        # into a visible density and doubles as a picture of the search space
        ax.scatter(cost, perf, s=14, c=FEAS_C, alpha=0.28, linewidths=0,
                   zorder=1)

    fx = [c for c, _, _ in front]
    fy = [p for _, p, _ in front]
    ax.plot(fx, fy, "-o", color=FRONT_C, ms=4.5, lw=1.5, zorder=3)

    hx = [c for c, _ in hull]
    hy = [p for _, p in hull]
    ax.plot(hx, hy, "-o", color=HULL_C, ms=6.5, lw=2.0, mfc="white",
            mew=1.8, zorder=2)

    if unsup:
        mk, filled = MARKERS[unsup_marker]
        ux = [c for c, _ in unsup]
        uy = [p for _, p in unsup]
        if filled:
            ax.plot(ux, uy, mk, color=UNSUP_C, ms=7, zorder=5)
        else:
            ax.plot(ux, uy, mk, mfc="none", mec=UNSUP_C, mew=2.0, ms=11,
                    zorder=5)

    ax.set_title(f"{pb.n_qubits} qubits  |  Pareto {len(front)}, "
                 f"unsupported {len(unsup)}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Total cost", fontsize=9)
    ax.grid(ls="--", alpha=0.30)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="+",
                    default=["Q13", "Q14", "Q15", "Q16",
                             "Q17", "Q18", "Q19", "Q20"])
    ap.add_argument("--instances", default="results/instances.json")
    ap.add_argument("--unsupported-marker", default="ring",
                    choices=sorted(MARKERS),
                    help="marker for unsupported Pareto points")
    ap.add_argument("--no-feasible", action="store_true",
                    help="omit the feasible-vector cloud")
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--out", default="figures/objective_space.png")
    args = ap.parse_args()

    registry = problem_registry(args.instances)
    missing = [n for n in args.problems if n not in registry]
    if missing:
        ap.error(f"unknown problem(s) {missing}; available: {sorted(registry)}")

    names = args.problems
    ncols = min(args.ncols, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.6 * ncols, 3.9 * nrows),
                             squeeze=False)
    for idx, name in enumerate(names):
        ax = axes[idx // ncols][idx % ncols]
        panel(ax, name, registry[name], args.unsupported_marker,
              not args.no_feasible)
        if idx % ncols == 0:
            ax.set_ylabel("Total performance", fontsize=9)
    for idx in range(len(names), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    mk, filled = MARKERS[args.unsupported_marker]
    handles = [
        Line2D([], [], ls="none", marker="o", ms=6, color=FEAS_C, alpha=.6,
               label="Feasible vectors"),
        Line2D([], [], marker="o", ms=5, color=FRONT_C, lw=1.5,
               label="Exact Pareto frontier"),
        Line2D([], [], marker="o", ms=6.5, color=HULL_C, lw=2.0, mfc="white",
               mew=1.8, label="Supported convex envelope"),
        Line2D([], [], ls="none", marker=mk, ms=10,
               mfc=UNSUP_C if filled else "none", mec=UNSUP_C, mew=2.0,
               label="Unsupported Pareto point"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.975))
    fig.suptitle("Brute-force objective spaces and true Pareto frontiers",
                 fontsize=15, fontweight="bold", y=1.0)
    fig.text(0.5, 0.005,
             "Every front rises monotonically, but the marked points lie "
             "strictly below the supported convex envelope: no weighted-sum "
             "scalarization reaches them at any $\\varphi$.",
             ha="center", fontsize=9.5, color="#33415c")
    fig.tight_layout(rect=[0, 0.022, 1, 0.94])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("wrote", args.out)

    print(f"\n{'prob':>5} {'qubits':>7} {'feasible':>9} {'Pareto':>7} "
          f"{'unsupported':>12}")
    for name in names:
        pb = registry[name]
        front = pb.pareto_front()
        hull = set(pb.supported_pareto_points())
        print(f"{name:>5} {pb.n_qubits:>7} {len(pb._tables()[0]):>9} "
              f"{len(front):>7} "
              f"{sum(1 for c, p, _ in front if (c, p) not in hull):>12}")


if __name__ == "__main__":
    main()
