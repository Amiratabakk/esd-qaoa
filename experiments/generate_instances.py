"""Search for design instances that maximise Pareto-front non-convexity.

Extends the I-V ladder up to a target total circuit width (default 20 qubits
= N + k + 1). Selection criterion is the number of UNSUPPORTED Pareto points:
those a weighted-sum sweep can never reach at any phi, because they sit in
dents of the front rather than on its convex hull. Those points are the
paper's motivation, so an instance ladder is only interesting if they are
plentiful and grow with N.

Non-convexity is created by breaking the monotone performance/cost ratio:
performances are drawn as cost * ratio with per-item noise, so some
subsystems are unusually good or bad value and the front develops dents.

Dependencies are sampled with i < j only, which makes the requirement graph
a DAG by construction -- a cycle would force those subsystems permanently
off and silently shrink the problem.

Usage:
    python experiments/generate_instances.py --max-qubits 20 --trials 40000
    python experiments/generate_instances.py --max-qubits 20 --out results/instances.json
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from esdqaoa.problems import SDProblem


def random_instance(rng, N, shape="curved", min_deps=None):
    """Sample one candidate instance.

    shape="linear" (the original) draws perf = cost * ratio * noise, which
    makes performance a near-straight multiple of cost. That produces a
    FLAT front with no visible trade-off, and -- because almost every point
    then sits a hair below the chord -- a huge count of unsupported points
    whose dents are numerically negligible (<1% of the performance range at
    N=12). Kept only for reproducing the first ladder.

    shape="curved" gives DIMINISHING RETURNS: cheap subsystems have a high
    performance-per-cost ratio and expensive ones progressively less. The
    resulting front is visibly concave -- the classic engineering trade-off
    curve -- and its unsupported points are real dents rather than rounding.
    """
    costs = rng.integers(1, max(3, N), size=N)
    if shape == "linear":
        perfs = np.maximum(1, np.round(
            costs * rng.uniform(1.6, 2.4) * rng.uniform(0.45, 1.7, size=N)))
    else:
        # efficiency falls with cost -> concave front
        order = np.argsort(costs)
        eff = np.empty(N, dtype=float)
        hi, decay = rng.uniform(3.0, 4.5), rng.uniform(0.45, 0.75)
        for rank, idx in enumerate(order):
            frac = rank / max(N - 1, 1)
            eff[idx] = hi * (1.0 - decay * frac)
        # a few anomalous items create genuine, deep dents in the front
        n_odd = int(rng.integers(1, 4))
        for idx in rng.choice(N, size=n_odd, replace=False):
            eff[idx] *= rng.uniform(1.25, 1.6)
        eff *= rng.uniform(0.94, 1.06, size=N)          # mild jitter
        perfs = np.maximum(1, np.round(costs * eff))
    perfs = perfs.astype(int)

    budget = int(round(costs.sum() * rng.uniform(0.35, 0.65)))
    budget = max(int(costs.min()), budget)

    # Dependency count needs a FLOOR. Sampling it freely lets the search
    # drift to one pair on a 12-subsystem instance, because the selection
    # criteria (front length, dent depth) do not reward dependencies -- and
    # an interdependency paper whose largest instance has a single
    # dependency invites the obvious objection. ceil(N/3) roughly matches
    # the density of the hand-built Problems I-V.
    # Exactly the floor, not a range: dependency count is the knob being
    # controlled, and the dependency study showed rho falls steeply with it
    # (N=12: 0.269 at 4 pairs, 0.203 at 6) while the ESD-vs-Pauli gap stays
    # flat. Letting the count drift upward just costs reportable rho.
    n_dep = math.ceil(N / 3) if min_deps is None else int(min_deps)
    n_dep = max(1, min(n_dep, N - 1))
    deps = set()
    guard = 0
    while len(deps) < n_dep and guard < 500:
        guard += 1
        i = int(rng.integers(0, N - 1))
        j = int(rng.integers(i + 1, N))
        deps.add((i, j))          # i < j keeps the requirement graph acyclic
    return SDProblem(name="?", performances=perfs.tolist(),
                     costs=costs.tolist(), budget=budget,
                     dependencies=sorted(deps))


def evaluate(pb):
    """Front size, dent depth and curvature.

    Selecting on the COUNT of unsupported points is a trap: it is maximised
    by near-linear fronts where every point is a hair off the hull. What
    makes an instance interesting is dent DEPTH (how far a Pareto point sits
    below the hull, relative to the performance range) and front CURVATURE
    (how much the performance-per-cost slope decays).
    """
    front = pb.pareto_front()
    hull = pb.supported_pareto_points()
    if len(front) < 2 or len(hull) < 2:
        return None
    supported = set(hull)
    hx = [h[0] for h in hull]
    hy = [h[1] for h in hull]
    p_range = max(p for _, p, _ in front) - min(p for _, p, _ in front)
    if p_range <= 0:
        return None

    dents = [np.interp(c, hx, hy) - p
             for c, p, _ in front if (c, p) not in supported]
    # curvature: first hull segment slope vs last; >1 means returns diminish
    def slope(a, b):
        return (hy[b] - hy[a]) / max(hx[b] - hx[a], 1e-9)
    s0, s1 = slope(0, 1), slope(len(hull) - 2, len(hull) - 1)
    curvature = s0 / max(s1, 1e-9)

    return {
        "n_pareto": len(front),
        "n_unsupported": len(dents),
        "max_dent_pct": 100 * max(dents) / p_range if dents else 0.0,
        "mean_dent_pct": 100 * float(np.mean(dents)) / p_range if dents else 0.0,
        "curvature": float(curvature),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-qubits", type=int, default=20)
    ap.add_argument("--min-qubits", type=int, default=13)
    ap.add_argument("--trials", type=int, default=40000)
    ap.add_argument("--min-pareto", type=int, default=6,
                    help="reject instances with a trivially short front")
    ap.add_argument("--shape", choices=["curved", "linear"], default="curved",
                    help="'curved' = diminishing returns (concave front); "
                         "'linear' reproduces the original flat-front ladder")
    ap.add_argument("--min-curvature", type=float, default=2.0,
                    help="min ratio of first to last hull slope; >1 means "
                         "performance-per-cost actually decays")
    ap.add_argument("--min-dent", type=float, default=2.0,
                    help="min mean dent depth, %% of performance range")
    ap.add_argument("--max-unsup-frac", type=float, default=0.40,
                    help="reject fronts that are mostly dents -- the front "
                         "should read as a curve with a few real notches")
    ap.add_argument("--min-deps", type=int, default=None,
                    help="minimum dependency pairs per instance "
                         "(default ceil(N/3), matching Problems I-V density)")
    ap.add_argument("--min-feas-frac", type=float, default=0.10,
                    help="reject instances whose feasible set is too thin; "
                         "dependencies cut feasibility fast and rho collapses")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    best = {}                       # total qubits -> (score, record)
    t0, kept = time.time(), 0

    for t in range(args.trials):
        N = int(rng.integers(6, args.max_qubits - 4))
        pb = random_instance(rng, N, args.shape, args.min_deps)
        q = pb.n_qubits
        if not (args.min_qubits <= q <= args.max_qubits):
            continue
        m = evaluate(pb)
        if m is None or m["n_pareto"] < args.min_pareto:
            continue
        if len(pb._tables()[0]) < args.min_feas_frac * 2 ** pb.N:
            continue
        if (m["curvature"] < args.min_curvature
                or m["mean_dent_pct"] < args.min_dent
                or m["n_unsupported"] < 2
                or m["n_unsupported"] > args.max_unsup_frac * m["n_pareto"]):
            continue
        kept += 1
        # Prefer the LONGEST front, then deepest dents. The filters above
        # already guarantee curvature and dent depth, so maximising length
        # here buys resolution: a front needs enough points to read as a
        # curve rather than a handful of scattered markers.
        score = (m["n_pareto"], round(m["mean_dent_pct"], 2))
        if q not in best or score > best[q][0]:
            best[q] = (score, {
                "n_qubits": q, "N": pb.N, "k": pb.k,
                "performances": [int(v) for v in pb.performances],
                "costs": [int(v) for v in pb.costs],
                "budget": int(pb.budget),
                "dependencies": [list(map(int, d)) for d in pb.dependencies],
                "n_feasible": int(len(pb._tables()[0])),
                **{kk: (round(vv, 2) if isinstance(vv, float) else vv)
                   for kk, vv in m.items()},
            })

    print(f"{args.trials} trials, {kept} passed filters, "
          f"{time.time() - t0:.0f}s (shape={args.shape})\n")
    print(f"{'Q':>3} {'N':>3} {'k':>2} {'feas':>6} {'|Pareto|':>9} "
          f"{'unsup':>6} {'dent%':>7} {'curv':>6} {'deps':>5} {'eps-cov':>9}")
    print("-" * 74)

    records = []
    for q in sorted(best):
        r = best[q][1]
        pb = SDProblem(name=f"Q{q}", performances=r["performances"],
                       costs=r["costs"], budget=r["budget"],
                       dependencies=[tuple(d) for d in r["dependencies"]])
        front = {(c, p) for c, p, _ in pb.pareto_front()}
        eps_cov = len(pb.epsilon_constraint_reachable() & front)
        r["eps_constraint_covered"] = eps_cov
        records.append(r)
        print(f"{q:>3} {r['N']:>3} {r['k']:>2} {r['n_feasible']:>6} "
              f"{r['n_pareto']:>9} {r['n_unsupported']:>6} "
              f"{r['mean_dent_pct']:>7.2f} {r['curvature']:>6.2f} "
              f"{len(r['dependencies']):>5} "
              f"{eps_cov}/{len(front)}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"settings": vars(args), "instances": records}, f,
                      indent=1)
        print("\nwrote", args.out)


if __name__ == "__main__":
    main()
