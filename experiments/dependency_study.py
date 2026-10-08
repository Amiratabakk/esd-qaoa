"""Does the ESD-QAOA advantage depend on how many dependencies there are?

Controlled experiment on one instance: costs, performances and budget are
held FIXED at the Q20 values and ONLY the dependency set varies. Comparing
freshly generated instances at each dependency count would confound the
constraint density with ordinary instance-to-instance difficulty.

Why this is worth measuring. The two encodings treat dependencies very
differently:

  ESD-QAOA  one controlled-phase gate per pair, applied directly on the
            decision register -- cost grows linearly in the number of pairs.
  Pauli     each pair becomes a quadratic penalty term in the QUBO, i.e. an
            extra ZZ coupling in the Ising Hamiltonian, on top of the budget
            penalty that already couples every variable pair.

So the hypothesis is that ESD's margin should widen as dependencies are
added. This script tests that rather than assuming it -- the margin could
equally shrink, since more dependencies also shrink the feasible set and
make the problem harder for both methods.

Dependency sets are re-searched here with a fixed seed (not hardcoded) so the
whole study is reproducible from this file alone, and are filtered on the
same front-quality criteria as the instance generator: real dent depth,
bounded unsupported fraction, genuine curvature.

Usage:
    python experiments/dependency_study.py --deps 2 4 6 --n-phi 7 \
        --out results/dependency_study.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from esdqaoa.problems import SDProblem, load_instances
from esdqaoa.simulate import run_esd_qaoa, approximation_ratio
from esdqaoa.fastbaseline import run_pauli_qaoa_fast

MIN_DENT, MAX_UNSUP_FRAC, MIN_CURV, MIN_FEAS = 3.0, 0.40, 2.0, 40


def front_metrics(pb):
    front = pb.pareto_front()
    hull = pb.supported_pareto_points()
    if len(front) < 6 or len(hull) < 2:
        return None
    sup = set(hull)
    hx = [h[0] for h in hull]
    hy = [h[1] for h in hull]
    span = max(p for _, p, _ in front) - min(p for _, p, _ in front)
    if span <= 0:
        return None
    dents = [np.interp(c, hx, hy) - p
             for c, p, _ in front if (c, p) not in sup]
    if not dents:
        return None
    s0 = (hy[1] - hy[0]) / max(hx[1] - hx[0], 1e-9)
    s1 = (hy[-1] - hy[-2]) / max(hx[-1] - hx[-2], 1e-9)
    return {"front": len(front), "nonconvex": len(dents),
            "mean_dent_pct": 100 * float(np.mean(dents)) / span,
            "curvature": float(s0 / max(s1, 1e-9)),
            "feasible": int(len(pb._tables()[0]))}


def search_deps(perfs, costs, budget, N, n_dep, rng, trials=8000):
    """Best dependency set of the requested size, by front quality."""
    best = None
    for _ in range(trials):
        deps = set()
        while len(deps) < n_dep:
            i = int(rng.integers(0, N - 1))
            j = int(rng.integers(i + 1, N))
            deps.add((i, j))
        pb = SDProblem("x", perfs, costs, budget, sorted(deps))
        m = front_metrics(pb)
        if m is None or m["feasible"] < MIN_FEAS:
            continue
        if (m["mean_dent_pct"] < MIN_DENT or m["curvature"] < MIN_CURV
                or m["nonconvex"] > MAX_UNSUP_FRAC * m["front"]):
            continue
        score = (m["front"], round(m["mean_dent_pct"], 2))
        if best is None or score > best[0]:
            best = (score, sorted(deps), m)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Q20")
    ap.add_argument("--instances", default="results/instances.json")
    ap.add_argument("--deps", nargs="+", type=int, default=[2, 4, 6])
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--n-phi", type=int, default=7)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dep-seed", type=int, default=3)
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--out", default="results/dependency_study.json")
    args = ap.parse_args()

    base = load_instances(args.instances)[args.base]
    perfs = [int(v) for v in base.performances]
    costs = [int(v) for v in base.costs]
    budget, N = int(base.budget), base.N
    phis = np.linspace(0, 1, args.n_phi)
    rng = np.random.default_rng(args.dep_seed)

    out = {"settings": vars(args),
           "base": {"performances": perfs, "costs": costs, "budget": budget,
                    "N": N, "n_qubits": base.n_qubits},
           "phis": phis.tolist(), "variants": {}}

    print(f"base {args.base}: N={N}, B={budget}, {base.n_qubits} qubits; "
          f"costs/perfs/budget fixed, only dependencies vary\n")

    for n_dep in args.deps:
        found = search_deps(perfs, costs, budget, N, n_dep, rng)
        if found is None:
            print(f"[deps={n_dep}] no set passes the quality filters; skipped")
            continue
        _, deps, m = found
        pb = SDProblem(f"{args.base}_d{n_dep}", perfs, costs, budget, deps)
        print(f"[deps={n_dep}] {deps}  feasible={m['feasible']} "
              f"({m['feasible']/2**N:.3f})  front={m['front']} "
              f"non-convex={m['nonconvex']}")

        rec = {"dependencies": [list(d) for d in deps], "front_metrics": m,
               "methods": {}}
        for method in ("esd", "pauli"):
            t0, rhos = time.time(), []
            for phi in phis:
                if method == "esd":
                    probs, _, _ = run_esd_qaoa(pb, args.p, float(phi),
                                               seed=args.seed,
                                               maxiter=args.maxiter)
                else:
                    probs, _, _ = run_pauli_qaoa_fast(pb, args.p, float(phi),
                                                      seed=args.seed,
                                                      maxiter=args.maxiter)
                rhos.append(approximation_ratio(pb, probs, float(phi)))
            rec["methods"][method] = {
                "rhos": [float(r) for r in rhos],
                "mean_rho": float(np.mean(rhos)),
                "min_rho": float(np.min(rhos)),
                "wall_time_s": round(time.time() - t0, 1)}
            print(f"    {method:>5}: mean_rho={np.mean(rhos):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        rec["gap"] = (rec["methods"]["esd"]["mean_rho"]
                      - rec["methods"]["pauli"]["mean_rho"])
        print(f"    gap = {rec['gap']:+.3f}", flush=True)
        out["variants"][str(n_dep)] = rec

    print(f"\n{'deps':>5} {'feasible':>9} {'ESD':>7} {'Pauli':>7} {'gap':>8}")
    print("-" * 40)
    for k, v in out["variants"].items():
        print(f"{k:>5} {v['front_metrics']['feasible']:>9} "
              f"{v['methods']['esd']['mean_rho']:>7.3f} "
              f"{v['methods']['pauli']['mean_rho']:>7.3f} {v['gap']:>+8.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
