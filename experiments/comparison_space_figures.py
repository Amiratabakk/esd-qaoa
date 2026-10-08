"""Objective-space comparison: both encodings under both scalarizations.

Same visual language as objective_space_figures.py (feasible cloud, exact
front, supported envelope, unsupported points) with the recovered designs
overlaid.

Layout is rows = scalarization, columns = problem, which lets one figure
carry both comparisons without overplotting:

    within a panel   ESD vs Pauli          (which ENCODING is better)
    top vs bottom    weighted sum vs eps   (which SCALARIZATION is better)

Putting all four method series in a single panel needs eight overlapping
layers and becomes unreadable -- the failure mode this project already hit
once with 47 highlighted points on a flat front.

MODAL fronts are used for both scalarizations, not harvested ones. That is
deliberate: a weighted-sum run's modal design is always a convex-hull vertex,
so the modal view is what exposes the structural blind spot. Harvesting
flatters both methods and blurs exactly the point the figure exists to make.

Usage:
    python experiments/comparison_space_figures.py \\
        --phi-sweep results/sweep_fixed.json --eps-sweep results/eps_sweep_I_V.json \\
        --problems II III IV V --out figures/comparison_space_I_V.png
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
from matplotlib.lines import Line2D

from esdqaoa.problems import problem_registry
from esdqaoa.pareto import recovered_front

FEAS_C = "#9aa5b1"
FRONT_C = "#1f5fd0"
HULL_C = "#1a7f37"
UNSUP_C = "#b3261e"
ESD_C = "#7b1d1d"
PAULI_C = "#5b7c1f"


def phi_modal(pb, mrec):
    sweep = [(s["phi"], {tuple(int(b) for b in k): v
                         for k, v in s["probs"].items()})
             for s in mrec["sweep"]]
    return {(c, p) for c, p, _, _ in recovered_front(pb, sweep)}


def eps_modal(mrec):
    return {(c, p) for c, p, _ in mrec["modal_front"]}


def design_mass(pb, mrec):
    """(cost, perf) -> the largest probability this method ever put there.

    Recovery is presence/absence, and under eps-constraint both encodings
    recover largely the SAME designs -- so a plain marker plot shows them on
    top of each other and cannot express the difference. The difference is in
    concentration: how much probability actually sits on a good design. Using
    this as marker area turns the overlap from uninformative into the point.
    """
    best = {}
    for s in mrec["sweep"]:
        for bits, pr in s["probs"].items():
            x = np.array([int(b) for b in bits])
            key = (pb.cost(x), pb.perf(x))
            if pr > best.get(key, 0.0):
                best[key] = pr
    return best


def _sizes(pts, mass, uniform, scale=900, floor=18):
    """Marker areas: uniform by default, or proportional to mass if given.

    The uniform sizes differ per method on purpose -- Pauli large and open,
    ESD small and filled -- so a design found by both reads as a diamond
    inside a triangle instead of one marker hiding the other.
    """
    if mass is None:
        return [uniform] * len(pts)
    return [floor + scale * mass.get(q, 0.0) for q in pts]


def panel(ax, pb, esd_pts, pauli_pts, show_feasible=True,
          esd_mass=None, pauli_mass=None):
    cost, perf, _ = pb._tables()
    front = pb.pareto_front()
    hull = pb.supported_pareto_points()
    unsup = [(c, p) for c, p, _ in front if (c, p) not in set(hull)]

    if show_feasible:
        ax.scatter(cost, perf, s=12, c=FEAS_C, alpha=0.22, linewidths=0,
                   zorder=1)
    ax.plot([c for c, _, _ in front], [p for _, p, _ in front], "-",
            color=FRONT_C, lw=1.2, alpha=0.85, zorder=2)
    # envelope kept light: it is reference geometry, not a result, and at
    # full weight it competes with the recovered-design markers
    ax.plot([c for c, _ in hull], [p for _, p in hull], "-", color=HULL_C,
            lw=1.1, alpha=0.45, zorder=3)
    if unsup:
        ax.plot([c for c, _ in unsup], [p for _, p in unsup], "o", mfc="none",
                mec=UNSUP_C, mew=1.8, ms=11, zorder=4)

    pauli_pts, esd_pts = sorted(pauli_pts), sorted(esd_pts)
    if pauli_pts:
        ax.scatter([c for c, _ in pauli_pts], [p for _, p in pauli_pts],
                   s=_sizes(pauli_pts, pauli_mass, 130), marker="^",
                   facecolors="none", edgecolors=PAULI_C, linewidths=1.6,
                   zorder=5)
    if esd_pts:
        ax.scatter([c for c, _ in esd_pts], [p for _, p in esd_pts],
                   s=_sizes(esd_pts, esd_mass, 34), marker="D",
                   facecolors=ESD_C, edgecolors="none", zorder=6)
    ax.grid(ls="--", alpha=0.30)
    ax.set_axisbelow(True)
    return len(front), len(unsup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi-sweep", default="results/sweep_fixed.json")
    ap.add_argument("--eps-sweep", default="results/eps_sweep_I_V.json")
    ap.add_argument("--problems", nargs="+", default=["II", "III", "IV", "V"])
    ap.add_argument("--no-feasible", action="store_true")
    ap.add_argument("--size-by-mass", action="store_true",
                    help="scale marker area by the probability mass each "
                         "method puts on that design. Off by default: it "
                         "makes the panels heavy. The same information is "
                         "carried more cleanly by the rho / Pareto-mass "
                         "figures, which is where the encoding difference "
                         "belongs -- this plot is about WHICH designs are "
                         "recovered, not how much mass sits on them.")
    ap.add_argument("--out", default="figures/comparison_space.png")
    args = ap.parse_args()

    phi_data = json.load(open(args.phi_sweep))["runs"]
    eps_data = json.load(open(args.eps_sweep))["runs"]
    registry = problem_registry()

    names = [n for n in args.problems if n in phi_data and n in eps_data]
    if not names:
        raise SystemExit("no requested problem is present in both sweeps")
    skipped = [n for n in args.problems if n not in names]
    if skipped:
        print(f"note: {skipped} not present in both sweeps; skipped")

    rows = [("Weighted sum  ($\\varphi$ sweep)", phi_data, "phi"),
            ("$\\varepsilon$-constraint  ($\\varepsilon$ sweep)",
             eps_data, "eps")]
    fig, axes = plt.subplots(2, len(names),
                             figsize=(4.3 * len(names), 7.6), squeeze=False)

    missing_pauli = []
    summary = []
    for r, (row_label, data, kind) in enumerate(rows):
        for c, name in enumerate(names):
            ax = axes[r][c]
            pb = registry[name]
            rec = data[name]["methods"]

            def pts(method):
                if method not in rec:
                    return None
                return (phi_modal(pb, rec[method]) if kind == "phi"
                        else eps_modal(rec[method]))

            esd_pts, pauli_pts = pts("esd"), pts("pauli")
            have_pauli = pauli_pts is not None
            if not have_pauli:
                missing_pauli.append((row_label, name))
                pauli_pts = set()
            em = design_mass(pb, rec["esd"]) if args.size_by_mass else None
            pm = (design_mass(pb, rec["pauli"])
                  if (have_pauli and args.size_by_mass) else None)
            n_front, n_unsup = panel(ax, pb, esd_pts, pauli_pts,
                                     not args.no_feasible, em, pm)
            tf = {(c_, p_) for c_, p_, _ in pb.pareto_front()}
            nc = {(c_, p_) for c_, p_ in data[name]["nonconvex"]}
            ax.set_title(
                f"{name} ({pb.n_qubits}q)   ESD {len(esd_pts & tf)}/{n_front}"
                + (f",  Pauli {len(pauli_pts & tf)}/{n_front}"
                   if have_pauli else ",  Pauli: no data"),
                fontsize=9)
            if r == 1:
                ax.set_xlabel("Total cost", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{row_label}\n\nTotal performance", fontsize=9)
            # None, not 0, when the run does not exist -- printing "0/13"
            # for absent data invites it being read as a real measurement.
            summary.append((row_label.split("  ")[0], name,
                            len(esd_pts & tf),
                            len(pauli_pts & tf) if have_pauli else None,
                            n_front, len(esd_pts & nc),
                            len(pauli_pts & nc) if have_pauli else None,
                            len(nc)))

    handles = [
        Line2D([], [], ls="none", marker="o", ms=6, color=FEAS_C, alpha=.6,
               label="Feasible vectors"),
        Line2D([], [], color=FRONT_C, lw=1.5, label="Exact Pareto frontier"),
        Line2D([], [], color=HULL_C, lw=2.0, label="Supported convex envelope"),
        Line2D([], [], ls="none", marker="o", ms=10, mfc="none", mec=UNSUP_C,
               mew=1.8, label="Unsupported Pareto point"),
        Line2D([], [], ls="none", marker="^", ms=11, mfc="none", mec=PAULI_C,
               mew=1.7, label="Pauli QAOA recovered"),
        Line2D([], [], ls="none", marker="D", ms=6, color=ESD_C,
               label="ESD-QAOA recovered"),
    ]
    if args.size_by_mass:
        handles.append(Line2D([], [], ls="none", marker="o", ms=11,
                              mfc="#8a8a8a", mec="none", alpha=.55,
                              label="marker area $\propto$ probability mass"))
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Recovered designs by encoding (within panel) and "
                 "scalarization (top vs bottom row)",
                 fontsize=13, fontweight="bold", y=1.0)
    caption = ("Modal design per run. A weighted-sum run's modal design is "
               "always a convex-hull vertex, so the top row cannot reach the "
               "circled points at any $\\varphi$.")
    if args.size_by_mass:
        caption += ("   Marker area is the probability mass each method puts "
                    "on that design: the two encodings recover similar design "
                    "SETS and differ in concentration.")
    fig.text(0.5, 0.005, caption, ha="center", fontsize=9, color="#33415c")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("wrote", args.out)

    if missing_pauli:
        print("\nWARNING: no Pauli data for:")
        for row_label, name in missing_pauli:
            print(f"  {row_label} / {name}")
        print("  -> run run_epsilon_sweep.py without '--methods esd'")

    print(f"\n{'scalarization':>16} {'prob':>5} {'ESD':>9} {'Pauli':>9} "
          f"{'ESD unsup':>10} {'Pauli unsup':>12}")
    for s, n, e, q, t, en, qn, nt in summary:
        q_s = f"{q:>5}/{t:<3}" if q is not None else f"{'no data':>9}"
        qn_s = f"{qn:>8}/{nt:<3}" if qn is not None else f"{'no data':>12}"
        print(f"{s:>16} {n:>5} {e:>5}/{t:<3} {q_s} {en:>6}/{nt:<3} {qn_s}")


if __name__ == "__main__":
    main()
