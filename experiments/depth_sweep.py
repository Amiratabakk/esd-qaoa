"""Circuit-depth sweep: approximation ratio vs QAOA layers p.

Answers the reviewer question "is the ESD-QAOA advantage an artifact of
shallow depth?" by running both methods at p = 1..P_MAX on selected
problems and plotting mean rho(p).

Key design choices for a FAIR sweep:
  * Warm start: each p is initialized from the optimized angles of p-1
    (padding the new layer with a small value), so higher-p points are not
    penalized by a harder cold-start optimization. Without this, both
    methods look artificially bad at high p and the comparison is muddied.
  * Same optimizer, seed, penalty strengths, and phi grid for both methods.
  * Multiple restarts on top of the warm start guard against local minima.

Runtime note: cost scales with p and problem size. Defaults (Problems
II, IV, V; p = 1..4; 7 phi points) are a few CPU-hours. Reduce --problems,
--p-max, or --n-phi to shorten. Writes results/depth_sweep.json.

Usage:
    python experiments/depth_sweep.py --problems II IV V --p-max 4 \\
        --n-phi 7 --out results/depth_sweep.json
    python experiments/depth_sweep.py --plot results/depth_sweep.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from esdqaoa.problems import PROBLEMS, load_instances, problem_registry
from esdqaoa.simulate import (default_alpha, default_beta_dep,
                              approximation_ratio)
from esdqaoa.fastsim import FastESDRunner
from esdqaoa.fastbaseline import run_pauli_qaoa_fast


def run_esd_depth(pb, p, phi, x0=None, restarts=2, maxiter=60):
    """Optimize ESD-QAOA at depth p, optionally warm-started from x0.

    Returns (probs, angles, mean approximation ratio contribution).
    """
    from scipy.optimize import differential_evolution
    import math
    alpha, beta_dep = default_alpha(pb), default_beta_dep(pb)
    runner = FastESDRunner(pb, p, phi, alpha, beta_dep)
    bounds = [(0, 4 * math.pi), (0, math.pi)] * p

    def neg(a):
        return -runner.gibbs(a, eta=5.0)

    best = None
    inits = []
    if x0 is not None:
        # pad previous (p-1) angles with a fresh small layer
        pad = list(x0) + [0.1, 0.1]
        inits.append(np.array(pad[:2 * p]))
    for r in range(restarts):
        seed = 7 + r
        res = differential_evolution(
            neg, bounds, seed=seed, maxiter=maxiter, popsize=15,
            tol=1e-8, polish=True,
            x0=inits[0] if (r == 0 and inits) else None)
        if best is None or res.fun < best.fun:
            best = res
    probs = runner.decision_probs(best.x)
    return probs, best.x


def sweep(args):
    phis = np.linspace(0, 1, args.n_phi)
    out = {"settings": vars(args), "phis": phis.tolist(), "runs": {}}
    # PROBLEMS (I-V) + the pinned LADDER (Q13-Q20); --instances
    # only needed to evaluate a freshly generated ladder
    registry = problem_registry()
    if args.instances:
        registry.update(load_instances(args.instances))
    for name in args.problems:
        pb = registry[name]
        rec = {"n_qubits": pb.n_qubits, "esd": {}, "pauli": {}}
        # ESD with warm start across p, per phi
        for method in ("esd", "pauli"):
            # `prev` MUST live outside the p loop: it carries each phi's
            # optimized angles from depth p-1 into depth p. Declaring it
            # inside the loop wipes it every depth, so x0 is always None and
            # the warm start silently never happens -- which is exactly the
            # cold-start failure this script exists to avoid (rho then falls
            # with p because differential evolution cannot converge in 2p
            # dimensions on a fixed iteration budget).
            prev = {}
            for p in range(1, args.p_max + 1):
                rhos = []
                for j, phi in enumerate(phis):
                    if method == "esd":
                        x0 = prev.get(j)  # angles from p-1 at this phi
                        probs, ang = run_esd_depth(
                            pb, p, float(phi), x0=x0,
                            maxiter=args.maxiter)
                        prev[j] = ang
                    else:
                        # Warm start the baseline too. Doing this for ESD
                        # only makes the gap grow with p purely because the
                        # baseline's optimizer is degrading, which would be
                        # an artifact of the harness, not the encoding.
                        probs, ang, _ = run_pauli_qaoa_fast(
                            pb, p, float(phi), seed=7,
                            maxiter=args.maxiter, x0=prev.get(j))
                        prev[j] = ang
                    rhos.append(approximation_ratio(pb, probs, float(phi)))
                rec[method][str(p)] = {
                    "mean_rho": float(np.mean(rhos)),
                    "min_rho": float(np.min(rhos)),
                    "rhos": list(map(float, rhos)),
                }
                print(f"[{name}/{method}] p={p} "
                      f"mean_rho={np.mean(rhos):.3f}", flush=True)
        out["runs"][name] = rec
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)


def plot(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = json.load(open(path))
    names = list(data["runs"])
    fig, axes = plt.subplots(1, len(names),
                             figsize=(3.2 * len(names), 3.2), sharey=True)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        rec = data["runs"][name]
        for method, color, lbl in [("esd", "maroon", "ESD-QAOA"),
                                   ("pauli", "olivedrab", "Pauli QAOA")]:
            ps = sorted(int(k) for k in rec[method])
            ys = [rec[method][str(p)]["mean_rho"] for p in ps]
            ax.plot(ps, ys, "-o", color=color, ms=5, lw=1.6, label=lbl)
        ax.set_title(f"Problem {name} ({rec['n_qubits']}q)", fontsize=9)
        ax.set_xlabel("QAOA layers $p$")
        ax.set_xticks(ps)
        ax.set_ylim(0, 1.02)
        ax.grid(ls="--", alpha=0.35)
    axes[0].set_ylabel(r"mean approximation ratio $\rho$")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    # figure name TRACKS the json name. A fixed "depth_vs_rho.png" meant a
    # run on a subset of problems silently overwrote the full figure with a
    # partial one -- no error, no visible marker, just a stale plot next to
    # current data. Only the canonical depth_all.json owns the bare name.
    stem = os.path.splitext(os.path.basename(path))[0]
    suffix = "" if stem == "depth_all" else f"_{stem}"
    outpng = os.path.join(os.path.dirname(path) or ".",
                          "..", "figures", f"depth_vs_rho{suffix}.png")
    outpng = os.path.normpath(outpng)
    os.makedirs(os.path.dirname(outpng), exist_ok=True)
    fig.savefig(outpng, dpi=170)
    print("wrote", outpng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="+", default=["II", "IV", "V"])
    ap.add_argument("--instances", default=None,
                    help="instance ladder JSON; adds Q<width> problems")
    ap.add_argument("--p-max", type=int, default=4)
    ap.add_argument("--n-phi", type=int, default=7)
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--out", default="results/depth_sweep.json")
    ap.add_argument("--plot", metavar="JSON",
                    help="skip sweep, just plot an existing depth JSON")
    args = ap.parse_args()
    if args.plot:
        plot(args.plot)
    else:
        sweep(args)
        plot(args.out)


if __name__ == "__main__":
    main()
