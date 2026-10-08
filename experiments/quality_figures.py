"""Quality figures for the ESD-QAOA vs Pauli QAOA comparison.

The Pareto-coverage scatter answers "can each method reach the front?"
(both can). These figures answer "how efficiently does each method
concentrate probability on the front?" -- which is where ESD-QAOA's
advantage lives and what the approximation ratio already reports.

Generates, from an existing sweep JSON (no re-simulation):

  figures/quality_rho_vs_phi.png     rho(phi), one panel per problem
  figures/quality_mass_bars.png      mean feasible-mass & Pareto-mass bars
  figures/quality_expected_front.png expected (cost,perf) overlay per phi
                                     (the "old figure" estimator, labelled
                                     honestly as an expectation)
  figures/quality_*.tex              PGFPlots snippets for the paper

Usage (from repo root):
    python experiments/quality_figures.py results/sweep.json
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

# includes the generated Q13..Q20 ladder when
# results/instances.json is present
PROBLEMS = problem_registry()
from esdqaoa.pareto_v2 import feasible_mass, pareto_mass

ESD_C, PAULI_C = "maroon", "olivedrab"
ORDER = ["I", "II", "III", "IV", "V"]


def ordered_names(data):
    """Problems present in this sweep, I-V first then any Q<width> ladder.

    Taking the order from the JSON rather than a fixed list lets the same
    figures serve the I-V campaign and the generated ladder.
    """
    runs = list(data["runs"])
    known = [n for n in ORDER if n in runs]
    extra = sorted((n for n in runs if n not in ORDER),
                   key=lambda n: (len(n), n))
    return known + extra


def parse_sweep(mrec):
    out = []
    for s in mrec["sweep"]:
        probs = {tuple(int(b) for b in k): v for k, v in s["probs"].items()}
        out.append((s["phi"], probs, s["rho"]))
    return out


def expected_point(pb, probs):
    """Probability-weighted expected (cost, perf) over feasible mass.

    This is the estimator used in the earlier draft figure. It is NOT an
    attainable design (a convex mixture of designs), so it is reported only
    as an expectation, never as a recovered Pareto point.
    """
    c = p = m = 0.0
    for x, pr in probs.items():
        xa = np.array(x)
        if pb.is_feasible(xa):
            c += pr * pb.cost(xa)
            p += pr * pb.perf(xa)
            m += pr
    if m == 0:
        return None
    return c / m, p / m


# ---------------------------------------------------------------------------
# Figure 1: rho(phi) panels
# ---------------------------------------------------------------------------
def fig_rho_vs_phi(data, outdir):
    names = ordered_names(data)
    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 3.0),
                             sharey=True)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        rec = data["runs"][name]
        for method, color, lbl in [("esd", ESD_C, "ESD-QAOA"),
                                   ("pauli", PAULI_C, "Pauli QAOA")]:
            if method not in rec["methods"]:
                continue
            sw = parse_sweep(rec["methods"][method])
            phis = [s[0] for s in sw]
            rhos = [s[2] for s in sw]
            ax.plot(phis, rhos, "-o", color=color, ms=3, lw=1.4, label=lbl)
        pb = PROBLEMS[name]
        ax.set_title(f"Problem {name} ({pb.n_qubits}q)", fontsize=9)
        ax.set_xlabel(r"$\varphi$", fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.grid(ls="--", alpha=0.35)
    axes[0].set_ylabel(r"Approximation ratio $\rho$", fontsize=9)
    axes[-1].legend(fontsize=7, loc="lower center")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "quality_rho_vs_phi.png"), dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: mean-mass bars
# ---------------------------------------------------------------------------
def fig_mass_bars(data, outdir):
    names = ordered_names(data)
    esd_pm, pauli_pm, esd_fm, pauli_fm = [], [], [], []
    for name in names:
        pb = PROBLEMS[name]
        tf = {(c, p) for c, p, _ in pb.pareto_front()}
        for method, pmlist, fmlist in [("esd", esd_pm, esd_fm),
                                       ("pauli", pauli_pm, pauli_fm)]:
            sw = parse_sweep(data["runs"][name]["methods"][method])
            fmlist.append(np.mean([feasible_mass(pb, p) for _, p, _ in sw]))
            pmlist.append(np.mean([pareto_mass(pb, p, tf)
                                   for _, p, _ in sw]))
    x = np.arange(len(names))
    w = 0.35
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    a1.bar(x - w / 2, esd_pm, w, color=ESD_C, label="ESD-QAOA")
    a1.bar(x + w / 2, pauli_pm, w, color=PAULI_C, label="Pauli QAOA")
    a1.set_title("Mean Pareto mass", fontsize=10)
    a2.bar(x - w / 2, esd_fm, w, color=ESD_C, label="ESD-QAOA")
    a2.bar(x + w / 2, pauli_fm, w, color=PAULI_C, label="Pauli QAOA")
    a2.set_title("Mean feasible mass", fontsize=10)
    for a in (a1, a2):
        a.set_xticks(x)
        a.set_xticklabels([n if n.startswith("Q") else f"P{n}" for n in names])
        a.set_ylim(0, 1.02)
        a.grid(axis="y", ls="--", alpha=0.35)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "quality_mass_bars.png"), dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: expected-front overlay (the old-style estimator, labelled)
# ---------------------------------------------------------------------------
def fig_expected_front(data, outdir):
    names = ordered_names(data)
    ncol = len(names)
    fig, axes = plt.subplots(1, ncol, figsize=(3.0 * ncol, 3.0))
    if ncol == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        pb = PROBLEMS[name]
        tf = sorted((c, p) for c, p, _ in pb.pareto_front())
        xs, ys = zip(*tf)
        ax.plot(xs, ys, drawstyle="steps-post", color="black", lw=1.5,
                zorder=2, label="True Pareto")
        ax.plot(xs, ys, "ko", ms=3, zorder=3)
        for method, color, lbl in [("esd", ESD_C, "ESD-QAOA"),
                                   ("pauli", PAULI_C, "Pauli QAOA")]:
            sw = parse_sweep(data["runs"][name]["methods"][method])
            pts = [expected_point(pb, p) for _, p, _ in sw]
            pts = [q for q in pts if q]
            if pts:
                mk = "D" if method == "esd" else "^"
                ax.scatter(*zip(*pts), marker=mk, s=26, c=color,
                           alpha=0.8, zorder=5, label=lbl)
        ax.set_title(f"Problem {name}", fontsize=9)
        ax.set_xlabel("Total Cost", fontsize=9)
        ax.grid(ls="--", alpha=0.35)
    axes[0].set_ylabel("Total Performance", fontsize=9)
    axes[-1].legend(fontsize=6.5, loc="lower right")
    fig.suptitle(r"Probability-weighted expected design per $\varphi$ "
                 r"(expectation, not an attainable configuration)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "quality_expected_front.png"), dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_json")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    data = json.load(open(args.sweep_json))
    os.makedirs(args.outdir, exist_ok=True)
    fig_rho_vs_phi(data, args.outdir)
    fig_mass_bars(data, args.outdir)
    fig_expected_front(data, args.outdir)

    # summary table to stdout
    print(f"\n{'Prob':<5}{'meanRho_esd':>12}{'meanRho_pau':>12}"
          f"{'Mpar_esd':>10}{'Mpar_pau':>10}")
    for name in ordered_names(data):
        pb = PROBLEMS[name]
        tf = {(c, p) for c, p, _ in pb.pareto_front()}
        row = [name]
        rho, pm = {}, {}
        for method in ("esd", "pauli"):
            sw = parse_sweep(data["runs"][name]["methods"][method])
            rho[method] = np.mean([s[2] for s in sw])
            pm[method] = np.mean([pareto_mass(pb, p, tf) for _, p, _ in sw])
        print(f"{name:<5}{rho['esd']:>12.3f}{rho['pauli']:>12.3f}"
              f"{pm['esd']:>10.3f}{pm['pauli']:>10.3f}")
    print(f"\nWrote quality figures to {args.outdir}/")


if __name__ == "__main__":
    main()
