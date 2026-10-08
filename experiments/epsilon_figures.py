"""Scalarization comparison figure: weighted sum vs epsilon-constraint.

Both sweeps run the SAME Hamiltonian encoding at the SAME number of quantum
runs, so the only variable is the scalarization. The point of the figure is
structural, not incremental: the modal design of a weighted-sum run is always
a convex-hull vertex, so a phi sweep can never land in a dent of the front no
matter how good the optimizer is. An eps-constraint sweep has no such blind
spot and fills those dents in.

Panels show the true Pareto front, its unsupported ("non-convex") points, and
what each sweep's modal designs recovered.

Usage:
    python experiments/epsilon_figures.py results/eps_sweep_I_V.json \
        --phi-sweep results/sweep_fixed.json --outdir figures/epsilon
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from esdqaoa.problems import problem_registry
from esdqaoa.pareto import recovered_front

TRUE_C, PHI_C, EPS_C = "0.25", "maroon", "seagreen"


def phi_modal_points(pb, mrec):
    sweep = [(s["phi"], {tuple(int(b) for b in k): v
                         for k, v in s["probs"].items()})
             for s in mrec["sweep"]]
    return {(c, p) for c, p, _, _ in recovered_front(pb, sweep)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eps_json")
    ap.add_argument("--phi-sweep", default="results/sweep_fixed.json")
    ap.add_argument("--method", default="esd", choices=["esd", "pauli"])
    ap.add_argument("--outdir", default="figures/epsilon")
    args = ap.parse_args()

    eps_data = json.load(open(args.eps_json))["runs"]
    phi_data = json.load(open(args.phi_sweep))["runs"]
    registry = problem_registry()
    os.makedirs(args.outdir, exist_ok=True)

    names = [n for n in eps_data if n in phi_data]
    if not names:
        raise SystemExit("no problems common to both sweeps")

    fig, axes = plt.subplots(1, len(names), figsize=(3.5 * len(names), 3.6),
                             squeeze=False)
    summary = []
    for ax, name in zip(axes[0], names):
        pb = registry[name]
        rec = eps_data[name]
        front = [(c, p) for c, p, _ in pb.pareto_front()]
        nonconvex = {(c, p) for c, p in rec["nonconvex"]}

        xs = [c for c, _ in front]
        ys = [p for _, p in front]
        ax.plot(xs, ys, "-", color=TRUE_C, lw=1.2, alpha=.7, zorder=1)
        sup = [(c, p) for c, p in front if (c, p) not in nonconvex]
        if sup:
            ax.plot(*zip(*sup), "o", color=TRUE_C, ms=4.5, zorder=2,
                    label="supported")
        if nonconvex:
            ax.plot(*zip(*sorted(nonconvex)), "o", mfc="white", mec=TRUE_C,
                    mew=1.5, ms=8, zorder=3, label="non-convex")

        phi_pts = phi_modal_points(pb, phi_data[name]["methods"][args.method])
        eps_pts = {(c, p) for c, p, _ in
                   rec["methods"][args.method]["modal_front"]}
        # Sizes differ so overlapping recoveries stay legible: a large hollow
        # diamond with a cross inside means both sweeps found that design.
        if phi_pts:
            ax.plot(*zip(*sorted(phi_pts)), "D", mfc="none", mec=PHI_C,
                    mew=1.8, ms=12, zorder=4, label="weighted sum")
        if eps_pts:
            ax.plot(*zip(*sorted(eps_pts)), "x", color=EPS_C, ms=6.5, mew=2,
                    zorder=5, label=r"$\varepsilon$-constraint")

        # eps values are de-duplicated integers, so a small budget yields
        # fewer eps runs than phi runs -- report both rather than claiming
        # parity that does not hold on the small instances.
        n_eps = len(rec["eps_values"])
        n_phi = len(phi_data[name]["methods"][args.method]["sweep"])
        ax.set_title(
            f"Problem {name}  ({pb.n_qubits} qubits)\n"
            f"weighted sum {len(phi_pts & set(front))}/{len(front)} "
            f"in {n_phi} runs   vs   "
            r"$\varepsilon$ " + f"{len(eps_pts & set(front))}/{len(front)} "
            f"in {n_eps} runs", fontsize=8.5)
        ax.set_xlabel("Total Cost", fontsize=9)
        ax.grid(ls="--", alpha=.3)
        summary.append((name, len(phi_pts & set(front)),
                        len(eps_pts & set(front)), len(front),
                        len(phi_pts & nonconvex), len(eps_pts & nonconvex),
                        len(nonconvex)))
    axes[0][0].set_ylabel("Total Performance", fontsize=9)
    axes[0][-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Same Hamiltonian encoding, same optimizer: a weighted-sum "
                 "run's modal design is always a hull vertex, so it cannot "
                 "land in a dent of the front", fontsize=9.5)
    fig.tight_layout()
    path = os.path.join(args.outdir, f"scalarization_compare_{args.method}.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)

    print(f"{'prob':>5} {'weighted sum':>14} {'eps-constraint':>15} "
          f"{'ws non-cvx':>11} {'eps non-cvx':>12}")
    for n, a, b, t, na, nb, nt in summary:
        print(f"{n:>5} {a:>8}/{t:<5} {b:>9}/{t:<5} {na:>7}/{nt:<3} {nb:>8}/{nt:<3}")
    print("\nwrote", path)


if __name__ == "__main__":
    main()
