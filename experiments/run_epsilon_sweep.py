"""Epsilon-constraint campaign: max P(x) s.t. C(x) <= eps, swept over eps.

Companion to run_sweep.py, which sweeps the weighted-sum weight phi. Both
scalarizations run inside the SAME Hamiltonian encoding -- eps-constraint is
just phi=1 with the budget register threshold set to eps -- so this tests the
framework claim rather than the algorithm.

Why it matters. A weighted-sum sweep is structurally blind to unsupported
("non-convex") Pareto points: the modal design at any phi is a convex-hull
vertex, so those points cannot be recovered at any phi. The eps-constraint
sweep has no such blind spot. Running both at the SAME number of quantum
runs (default 13, matching the phi sweep) asks the paper's real question:
which scalarization recovers more of the front per run?

Register widths are pinned across the sweep (circuits.pinned_encoding for
ESD, slack_bits for the baseline) so the qubit count does not drift with eps
and the comparison stays like-for-like.

Usage:
    python experiments/run_epsilon_sweep.py --problems V --out results/eps_V.json
    python experiments/run_epsilon_sweep.py --instances results/instances.json \
        --problems Q20 --n-eps 13 --out results/eps_ladder.json
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from esdqaoa.problems import PROBLEMS, load_instances, problem_registry
from esdqaoa.circuits import pinned_encoding
from esdqaoa.fastsim import FastESDRunner
from esdqaoa.fastbaseline import run_pauli_qaoa_fast
from esdqaoa.simulate import (optimize_angles, epsilon_approximation_ratio,
                              epsilon_prob_of_optimum, default_alpha,
                              default_beta_dep)


def eps_grid(problem, n_eps):
    """n_eps integer budgets spanning [0, B], de-duplicated."""
    raw = np.linspace(0, problem.budget, n_eps)
    return sorted({int(round(e)) for e in raw})


def run_esd_epsilon(problem, p, eps, alpha, beta_dep, k, seed, maxiter):
    """ESD-QAOA in eps-constraint mode: phi=1 with the threshold at eps."""
    runner = FastESDRunner(problem, p, 1.0, alpha, beta_dep, budget=eps)
    angles, val = optimize_angles(problem, p, 1.0, alpha, beta_dep,
                                  runner=runner, seed=seed, maxiter=maxiter)
    return runner.decision_probs(angles), angles, val


def modal_eps_design(problem, probs, eps):
    """Highest-probability design satisfying deps and C(x) <= eps."""
    best, best_p = None, -1.0
    for x, pr in probs.items():
        xa = np.array(x)
        if problem.is_eps_feasible(xa, eps) and pr > best_p:
            best, best_p = xa, pr
    return best, best_p


def non_dominated(points):
    """Filter (cost, perf, config) triples to the non-dominated set."""
    out = []
    for c, p, x in points:
        if not any((c2 <= c and p2 >= p) and (c2 < c or p2 > p)
                   for c2, p2, _ in points):
            out.append((c, p, x))
    seen, uniq = set(), []
    for c, p, x in sorted(out):
        if (c, p) not in seen:
            seen.add((c, p))
            uniq.append((c, p, x))
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="+", default=["V"])
    ap.add_argument("--instances", default=None)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--n-eps", type=int, default=13,
                    help="number of eps values; keep equal to the phi sweep's "
                         "n-phi so both scalarizations get the same run budget")
    ap.add_argument("--methods", nargs="+", default=["esd", "pauli"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--out", default="results/eps_sweep.json")
    args = ap.parse_args()

    # PROBLEMS (I-V) + the pinned LADDER (Q13-Q20); --instances
    # only needed to evaluate a freshly generated ladder
    registry = problem_registry()
    if args.instances:
        registry.update(load_instances(args.instances))
    unknown = [n for n in args.problems if n not in registry]
    if unknown:
        ap.error(f"unknown problem(s) {unknown}; available: {sorted(registry)}")

    out = {"settings": vars(args), "runs": {}}
    for name in args.problems:
        pb = registry[name]
        alpha, beta_dep = default_alpha(pb), default_beta_dep(pb)
        grid = eps_grid(pb, args.n_eps)
        _, k = pinned_encoding(pb, grid)              # constant ESD width
        slack = max(math.ceil(math.log2(e + 1)) if e > 0 else 1 for e in grid)

        true_front = [(int(c), int(p)) for c, p, _ in pb.pareto_front()]
        rec = {"n_qubits": pb.n_qubits, "eps_values": grid,
               "true_front": true_front,
               "nonconvex": [(int(c), int(p))
                             for c, p in pb.nonconvex_pareto_points()],
               "spec": {"performances": [int(v) for v in pb.performances],
                        "costs": [int(v) for v in pb.costs],
                        "budget": int(pb.budget),
                        "dependencies": [list(map(int, d))
                                         for d in pb.dependencies]},
               "classical_eps_front": sorted(
                   (int(c), int(p))
                   for c, p in pb.epsilon_constraint_reachable(grid)),
               "methods": {}}

        for method in args.methods:
            sweep, t0 = [], time.time()
            for eps in grid:
                if method == "esd":
                    probs, angles, val = run_esd_epsilon(
                        pb, args.p, eps, alpha, beta_dep, k,
                        args.seed, args.maxiter)
                else:
                    probs, angles, val = run_pauli_qaoa_fast(
                        pb, args.p, 1.0, seed=args.seed, maxiter=args.maxiter,
                        budget=eps, slack_bits=slack)
                rho = epsilon_approximation_ratio(pb, probs, eps)
                po = epsilon_prob_of_optimum(pb, probs, eps)
                x, _ = modal_eps_design(pb, probs, eps)
                sweep.append({
                    "eps": int(eps), "rho_eps": rho, "p_opt": po,
                    "modal": None if x is None else
                             [int(pb.cost(x)), int(pb.perf(x)),
                              list(map(int, x))],
                    "angles": list(map(float, angles)),
                    "probs": {"".join(map(str, xx)): pr
                              for xx, pr in probs.items() if pr > 1e-4}})
                print(f"[{name}/{method}] eps={eps:>3} rho_eps={rho:.3f} "
                      f"P(opt)={po:.3f}", flush=True)

            modal_pts = [(s["modal"][0], s["modal"][1], tuple(s["modal"][2]))
                         for s in sweep if s["modal"]]
            front = non_dominated(modal_pts)
            found = {(c, p) for c, p, _ in front}
            tf = set(true_front)
            nc = {(c, p) for c, p in rec["nonconvex"]}
            rec["methods"][method] = {
                "sweep": sweep,
                "modal_front": [(int(c), int(p), list(map(int, x)))
                                for c, p, x in front],
                "metrics": {
                    "mean_rho_eps": float(np.mean([s["rho_eps"] for s in sweep])),
                    "min_rho_eps": float(np.min([s["rho_eps"] for s in sweep])),
                    "front_pts_recovered": len(found & tf),
                    "front_pts_total": len(tf),
                    "nonconvex_recovered": len(found & nc),
                    "nonconvex_total": len(nc),
                },
                "wall_time_s": time.time() - t0,
            }
            m = rec["methods"][method]["metrics"]
            print(f"[{name}/{method}] modal front {m['front_pts_recovered']}"
                  f"/{m['front_pts_total']} pts, non-convex "
                  f"{m['nonconvex_recovered']}/{m['nonconvex_total']}, "
                  f"mean rho_eps {m['mean_rho_eps']:.3f}", flush=True)
        out["runs"][name] = rec

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
