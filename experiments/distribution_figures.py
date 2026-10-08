"""Output-distribution figures: the clearest visual of ESD-QAOA's advantage.

All metrics in this paper (rho, Pareto mass, feasible mass) are summaries of
the raw output distribution. These figures show the distribution itself,
where the encoding difference is most direct: ESD-QAOA concentrates
probability on optimal feasible designs; the Pauli baseline spreads it and
leaks mass onto infeasible states.

Generates, from an existing sweep JSON (no re-simulation):

  figures/dist_bars_<prob>_phi<k>.png
        Paired bar chart of the two output distributions at one (problem,
        phi), configurations sorted by scalarized objective, colored by
        {Pareto-optimal / feasible / infeasible}. Shows concentration.

  figures/dist_feasmass_<prob>.png
        Stacked feasible/infeasible probability mass vs phi, both methods.
        Shows the Pauli baseline bleeding mass onto infeasible designs.

  figures/dist_cumulative_<prob>_phi<k>.png
        Cumulative probability vs objective rank (best-first), both methods.
        A steep curve = mass on top designs; shallow = spread. One line per
        method -- the paper-friendly summary of concentration.

Usage:
    python experiments/distribution_figures.py results/sweep.json \\
        --problem IV --phi-index 4
    python experiments/distribution_figures.py results/sweep.json --all
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

ESD_C, PAULI_C = "maroon", "olivedrab"


def probs_at(data, name, method, phi_index):
    s = data["runs"][name]["methods"][method]["sweep"][phi_index]
    probs = {tuple(int(b) for b in k): v for k, v in s["probs"].items()}
    return s["phi"], probs


def classify(pb, x, pareto_set):
    xa = np.array(x)
    if not pb.is_feasible(xa):
        return "infeasible"
    if (pb.cost(xa), pb.perf(xa)) in pareto_set:
        return "pareto"
    return "feasible"


# ---------------------------------------------------------------------------
# Figure 1: paired distribution bars at one (problem, phi)
# ---------------------------------------------------------------------------
def fig_dist_bars(data, name, phi_index, outdir):
    pb = PROBLEMS[name]
    pareto_set = {(c, p) for c, p, _ in pb.pareto_front()}
    phi, esd = probs_at(data, name, "esd", phi_index)
    _, pauli = probs_at(data, name, "pauli", phi_index)

    # order configurations by scalarized objective at this phi (best first)
    all_cfg = sorted(pb.all_configs(),
                     key=lambda x: -pb.scalarized(x, phi))
    labels = [f"{''.join(map(str, x))}" for x in all_cfg]
    color_map = {"pareto": "crimson", "feasible": ESD_C,
                 "infeasible": "lightgray"}
    colors = [color_map[classify(pb, tuple(x), pareto_set)] for x in all_cfg]

    e = [esd.get(tuple(x), 0.0) for x in all_cfg]
    p = [pauli.get(tuple(x), 0.0) for x in all_cfg]
    idx = np.arange(len(all_cfg))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(max(6, 0.28 * len(idx)), 5),
                                 sharex=True)
    a1.bar(idx, e, color=colors)
    a1.set_ylabel("ESD-QAOA\nprobability")
    a1.set_title(f"Problem {name}, $\\varphi$={phi:.2f} "
                 f"(configs sorted best objective first)", fontsize=10)
    a2.bar(idx, p, color=colors)
    a2.set_ylabel("Pauli QAOA\nprobability")
    a2.set_xticks(idx)
    a2.set_xticklabels(labels, rotation=90, fontsize=6)
    # legend
    from matplotlib.patches import Patch
    handles = [Patch(color="crimson", label="Pareto-optimal"),
               Patch(color=ESD_C, label="feasible (non-Pareto)"),
               Patch(color="lightgray", label="infeasible")]
    a1.legend(handles=handles, fontsize=7, loc="upper right")
    fig.tight_layout()
    out = os.path.join(outdir, f"dist_bars_{name}_phi{phi_index}.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: feasible/infeasible mass vs phi
# ---------------------------------------------------------------------------
def fig_feasmass(data, name, outdir):
    pb = PROBLEMS[name]
    phis = [s["phi"] for s in data["runs"][name]["methods"]["esd"]["sweep"]]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    width = (phis[1] - phis[0]) * 0.4 if len(phis) > 1 else 0.03
    for k, (method, color, off) in enumerate(
            [("esd", ESD_C, -width / 2), ("pauli", PAULI_C, width / 2)]):
        feas, infeas = [], []
        for i in range(len(phis)):
            _, probs = probs_at(data, name, method, i)
            fm = sum(pr for x, pr in probs.items()
                     if pb.is_feasible(np.array(x)))
            feas.append(fm)
            infeas.append(1 - fm)
        xpos = np.array(phis) + off
        ax.bar(xpos, feas, width, color=color, label=f"{method} feasible")
        ax.bar(xpos, infeas, width, bottom=feas, color=color, alpha=0.3,
               hatch="//", label=f"{method} infeasible")
    ax.set_xlabel(r"$\varphi$")
    ax.set_ylabel("Probability mass")
    ax.set_title(f"Problem {name}: feasible vs infeasible mass", fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = os.path.join(outdir, f"dist_feasmass_{name}.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3: cumulative probability vs objective rank
# ---------------------------------------------------------------------------
def fig_cumulative(data, name, phi_index, outdir):
    pb = PROBLEMS[name]
    phi, esd = probs_at(data, name, "esd", phi_index)
    _, pauli = probs_at(data, name, "pauli", phi_index)
    all_cfg = sorted(pb.all_configs(),
                     key=lambda x: -pb.scalarized(x, phi))
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for method, dist, color, lbl in [("esd", esd, ESD_C, "ESD-QAOA"),
                                     ("pauli", pauli, PAULI_C, "Pauli QAOA")]:
        vals = [dist.get(tuple(x), 0.0) for x in all_cfg]
        cum = np.cumsum(vals)
        ax.plot(np.arange(1, len(cum) + 1), cum, "-o", color=color, ms=3,
                lw=1.6, label=lbl)
    ax.set_xlabel("Configuration rank (best objective first)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title(f"Problem {name}, $\\varphi$={phi:.2f}: "
                 f"concentration on top designs", fontsize=9.5)
    ax.set_ylim(0, 1.02)
    ax.grid(ls="--", alpha=0.35)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = os.path.join(outdir, f"dist_cumulative_{name}_phi{phi_index}.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_json")
    ap.add_argument("--problem", default=None,
                    help="default: first problem in the sweep")
    ap.add_argument("--phi-index", type=int, default=4,
                    help="index into the phi sweep (default 4 ~ phi=1/3)")
    ap.add_argument("--all", action="store_true",
                    help="feasmass + cumulative for every problem")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    data = json.load(open(args.sweep_json))
    os.makedirs(args.outdir, exist_ok=True)
    if args.problem is None:
        args.problem = next(iter(data["runs"]))
    elif args.problem not in data["runs"]:
        ap.error(f"{args.problem!r} not in this sweep; available: "
                 f"{list(data['runs'])}")

    if args.all:
        for name in data["runs"]:
            print(fig_feasmass(data, name, args.outdir))
            print(fig_cumulative(data, name, args.phi_index, args.outdir))
        # bar chart only for small problems (readable x-axis)
        for name in list(data["runs"])[:3]:
            if name in data["runs"]:
                print(fig_dist_bars(data, name, args.phi_index, args.outdir))
    else:
        print(fig_dist_bars(data, args.problem, args.phi_index, args.outdir))
        print(fig_feasmass(data, args.problem, args.outdir))
        print(fig_cumulative(data, args.problem, args.phi_index, args.outdir))
    print(f"Wrote distribution figures to {args.outdir}/")


if __name__ == "__main__":
    main()
