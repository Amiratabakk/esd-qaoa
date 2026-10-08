"""Full experimental campaign: phi sweep for ESD-QAOA and Pauli QAOA.

Usage:
    python experiments/run_sweep.py --problems I II III IV V --p 2 \
        --n-phi 13 --methods esd pauli --out results/sweep.json

Every run is seeded and all settings are recorded in the output JSON, so
results are exactly reproducible. Figures and LaTeX tables are generated
from the JSON by experiments/make_figures.py.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from esdqaoa.problems import PROBLEMS, load_instances, problem_registry
from esdqaoa.simulate import (run_esd_qaoa, approximation_ratio,
                              prob_of_optimum, default_alpha,
                              default_beta_dep)
from esdqaoa.baseline import run_pauli_qaoa
from esdqaoa.fastbaseline import run_pauli_qaoa_fast
from esdqaoa.pareto import recovered_front, adpf, pcr, hypervolume_pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="+", default=None,
                    help="names from PROBLEMS (I..V) or the pinned ladder "
                         "(Q13..Q20); default is I..V")
    ap.add_argument("--instances", default=None,
                    help="optional instance JSON to OVERRIDE the pinned "
                         "ladder, for evaluating a freshly generated one")
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--n-phi", type=int, default=13)
    ap.add_argument("--methods", nargs="+", default=["esd", "pauli"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--backend", choices=["fast", "full"], default="fast",
                    help="'fast' uses the exact ancilla-free/diagonal "
                         "simulators (esdqaoa.fastsim, esdqaoa.fastbaseline), "
                         "verified equal to the Qiskit circuits to ~1e-12; "
                         "'full' builds the real circuits. Applied to BOTH "
                         "methods so the comparison stays like-for-like.")
    ap.add_argument("--out", default="results/sweep.json")
    args = ap.parse_args()

    # PROBLEMS (I-V) + the pinned LADDER (Q13-Q20); --instances
    # only needed to evaluate a freshly generated ladder
    registry = problem_registry()
    if args.instances:
        registry.update(load_instances(args.instances))
    names = args.problems if args.problems else list(PROBLEMS)
    unknown = [n for n in names if n not in registry]
    if unknown:
        ap.error(f"unknown problem(s) {unknown}; "
                 f"available: {sorted(registry)}")

    phis = np.linspace(0, 1, args.n_phi)
    out = {"settings": vars(args) | {"phis": phis.tolist()}, "runs": {}}

    for name in names:
        pb = registry[name]
        true_front = [(c, p) for c, p, _ in pb.pareto_front()]
        rec = {"true_front": [(int(c), int(p)) for c, p in true_front],
               "nonconvex": [(int(c), int(p)) for c, p in pb.nonconvex_pareto_points()],
               "n_qubits": pb.n_qubits,
               # Store the full instance definition so this sweep can always
               # be re-analysed against the problem it actually ran on. The
               # generated ladder is regenerable, and a Q<width> name alone
               # would silently bind to whatever instances.json holds later.
               "spec": {"performances": [int(v) for v in pb.performances],
                        "costs": [int(v) for v in pb.costs],
                        "budget": int(pb.budget),
                        "dependencies": [list(map(int, d))
                                         for d in pb.dependencies]},
               "methods": {}}
        for method in args.methods:
            sweep, t0 = [], time.time()
            for phi in phis:
                if method == "esd":
                    probs, angles, val = run_esd_qaoa(
                        pb, args.p, float(phi), seed=args.seed,
                        maxiter=args.maxiter, backend=args.backend)
                elif args.backend == "fast":
                    probs, angles, val = run_pauli_qaoa_fast(
                        pb, args.p, float(phi), seed=args.seed,
                        maxiter=args.maxiter)
                else:
                    probs, angles, val = run_pauli_qaoa(
                        pb, args.p, float(phi))
                rho = approximation_ratio(pb, probs, float(phi))
                po = prob_of_optimum(pb, probs, float(phi))
                sweep.append({"phi": float(phi), "rho": rho, "p_opt": po,
                              "angles": list(map(float, angles)),
                              "probs": {"".join(map(str, x)): pr
                                        for x, pr in probs.items()
                                        if pr > 1e-4}})
                print(f"[{name}/{method}] phi={phi:.3f} rho={rho:.3f} "
                      f"P(opt)={po:.3f}", flush=True)
            sweep_probs = [(s["phi"],
                            {tuple(map(int, k)): v
                             for k, v in s["probs"].items()})
                           for s in sweep]
            front = recovered_front(pb, sweep_probs)
            found = [(c, p) for c, p, _, _ in front]
            rec["methods"][method] = {
                "sweep": sweep,
                "recovered_front": [(int(c), int(p), list(map(int, x)), float(phi))
                                    for c, p, x, phi in front],
                "metrics": {
                    "ADPF": adpf(found, true_front),
                    "PCR": pcr(found, true_front),
                    "HV_pct": hypervolume_pct(found, true_front),
                    "mean_rho": float(np.mean([s["rho"] for s in sweep])),
                    "min_rho": float(np.min([s["rho"] for s in sweep])),
                },
                "wall_time_s": time.time() - t0,
            }
            print(f"[{name}/{method}] metrics:",
                  rec["methods"][method]["metrics"], flush=True)
        out["runs"][name] = rec

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
