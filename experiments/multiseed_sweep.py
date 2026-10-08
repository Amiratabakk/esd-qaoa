"""Multi-seed phi sweep: error bars on the ESD-vs-Pauli approximation ratio.

A single seed cannot distinguish a real advantage from optimizer luck. On the
curved instance ladder some per-instance gaps are small enough (a few
hundredths) that the sign could plausibly flip between seeds, so any claim
that the gap grows with N needs a spread, not a point estimate.

Only per-phi rho and the aggregates are stored -- keeping the full output
distributions for every seed would bloat the JSON by an order of magnitude
and nothing downstream needs them.

Cost note: the ESD side of the ladder is ~150 s per seed, the Pauli side
~2 h, so --methods esd is a cheap first pass and Pauli is the long pole.

Usage:
    python experiments/multiseed_sweep.py --instances results/instances.json \
        --problems Q13 Q14 Q15 Q16 Q17 Q18 Q19 Q20 \
        --seeds 7 11 23 42 101 --methods esd --out results/multiseed_esd.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from esdqaoa.problems import PROBLEMS, load_instances, problem_registry
from esdqaoa.simulate import run_esd_qaoa, approximation_ratio
from esdqaoa.fastbaseline import run_pauli_qaoa_fast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="+", default=None)
    ap.add_argument("--instances", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 23, 42, 101])
    ap.add_argument("--methods", nargs="+", default=["esd", "pauli"])
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--n-phi", type=int, default=13)
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--out", default="results/multiseed.json")
    args = ap.parse_args()

    # PROBLEMS (I-V) + the pinned LADDER (Q13-Q20); --instances
    # only needed to evaluate a freshly generated ladder
    registry = problem_registry()
    if args.instances:
        registry.update(load_instances(args.instances))
    names = args.problems if args.problems else list(PROBLEMS)
    unknown = [n for n in names if n not in registry]
    if unknown:
        ap.error(f"unknown problem(s) {unknown}; available: {sorted(registry)}")

    phis = np.linspace(0, 1, args.n_phi)
    out = {"settings": vars(args), "phis": phis.tolist(), "runs": {}}

    for name in names:
        pb = registry[name]
        rec = {"n_qubits": pb.n_qubits, "methods": {}}
        for method in args.methods:
            per_seed = {}
            for seed in args.seeds:
                t0, rhos = time.time(), []
                for phi in phis:
                    if method == "esd":
                        probs, _, _ = run_esd_qaoa(
                            pb, args.p, float(phi), seed=seed,
                            maxiter=args.maxiter)
                    else:
                        probs, _, _ = run_pauli_qaoa_fast(
                            pb, args.p, float(phi), seed=seed,
                            maxiter=args.maxiter)
                    rhos.append(approximation_ratio(pb, probs, float(phi)))
                per_seed[str(seed)] = {
                    "rhos": [float(r) for r in rhos],
                    "mean_rho": float(np.mean(rhos)),
                    "min_rho": float(np.min(rhos)),
                    "wall_time_s": round(time.time() - t0, 1),
                }
                print(f"[{name}/{method}] seed={seed} "
                      f"mean_rho={np.mean(rhos):.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            means = [per_seed[str(s)]["mean_rho"] for s in args.seeds]
            rec["methods"][method] = {
                "per_seed": per_seed,
                "mean_of_means": float(np.mean(means)),
                "std_of_means": float(np.std(means)),
            }
        # gap statistics, seed-paired where both methods ran
        if len(args.methods) == 2:
            a, b = args.methods
            gaps = [rec["methods"][a]["per_seed"][str(s)]["mean_rho"]
                    - rec["methods"][b]["per_seed"][str(s)]["mean_rho"]
                    for s in args.seeds]
            rec["gap"] = {"per_seed": gaps, "mean": float(np.mean(gaps)),
                          "std": float(np.std(gaps)),
                          "min": float(np.min(gaps)),
                          "max": float(np.max(gaps)),
                          # sign is consistent only if every seed agrees
                          "sign_consistent": bool(all(g > 0 for g in gaps)
                                                  or all(g < 0 for g in gaps))}
            g = rec["gap"]
            print(f"[{name}] gap {g['mean']:+.3f} +/- {g['std']:.3f} "
                  f"(range {g['min']:+.3f}..{g['max']:+.3f}) "
                  f"{'CONSISTENT' if g['sign_consistent'] else 'SIGN FLIPS'}",
                  flush=True)
        out["runs"][name] = rec

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
