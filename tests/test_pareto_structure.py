"""Brute-force verification of the Pareto structure of instances I-V.

Convention (matches paper + memory): dep pair (i, j) means x[j] requires x[i],
i.e. x_j <= x_i  (selecting j forces i to be selected).

Feasible = budget respected AND all dependencies respected.
Pareto: maximize performance, minimize cost over feasible set.
Non-convex Pareto points = Pareto points NOT recoverable by any weighted-sum
phi*P - (1-phi)*C over phi in [0,1] (i.e., not on the upper-left convex hull).
"""
import itertools, math
import numpy as np

def analyze(name, perf, cost, budget, deps):
    N = len(perf)
    feas = []
    for bits in itertools.product([0, 1], repeat=N):
        x = np.array(bits)
        if x @ cost > budget:
            continue
        if any(x[j] == 1 and x[i] == 0 for (i, j) in deps):
            continue
        feas.append((x.copy(), int(x @ cost), int(x @ perf)))
    # Pareto set (max perf, min cost)
    pareto = []
    for x, c, p in feas:
        dominated = any((c2 <= c and p2 >= p) and (c2 < c or p2 > p)
                        for _, c2, p2 in feas)
        if not dominated:
            pareto.append((c, p, tuple(x)))
    pareto = sorted(set(pareto))
    # Weighted-sum reachable points (classical scalarization sweep)
    reachable = set()
    for phi in np.linspace(0, 1, 2001):
        best, arg = -1e18, None
        for x, c, p in feas:
            v = phi * p - (1 - phi) * c
            if v > best + 1e-12:
                best, arg = v, (c, p)
        reachable.add(arg)
    nonconvex = [pt for pt in pareto if (pt[0], pt[1]) not in reachable]
    Cmax = int(sum(cost))
    k = math.ceil(math.log2(Cmax + 1))
    print(f"\n=== {name}: N={N}, perf={list(perf)}, cost={list(cost)}, "
          f"B={budget}, deps={deps}")
    print(f"    feasible={len(feas)}/{2**N}, Cmax={Cmax}, k={k}, "
          f"Q_total={N + k + 1}")
    print(f"    Pareto front ({len(pareto)} pts): "
          f"{[(c, p) for c, p, _ in pareto]}")
    print(f"    non-convex pts ({len(nonconvex)}): "
          f"{[(c, p) for c, p, _ in nonconvex]}")
    print(f"    Pareto configs: {[x for _, _, x in pareto]}")
    return len(pareto), len(nonconvex), N + k + 1

perf = np.array; cost = np.array

# --- Candidate designs ---------------------------------------------------
# I: keep a tiny hardware-friendly instance (N=2)
analyze("I  (N=2)", perf([10, 2]), cost([12, 1]), 13, [(1, 0)])

# II: N=3, richer than old II
analyze("II (N=3)", perf([1, 2, 4]), cost([1, 2, 3]), 3, [(2, 1)])

# III: N=4, chain dependency, aim for 4-5 Pareto pts w/ 1 non-convex
analyze("III(N=4) v1", perf([3, 5, 8, 13]), cost([1, 2, 4, 7]), 10,
        [(0, 1), (2, 3)])
analyze("III(N=4) v2", perf([2, 6, 7, 12]), cost([1, 3, 4, 6]), 9,
        [(0, 1), (1, 3)])

# IV: N=5
analyze("IV (N=5) v1", perf([4, 7, 9, 12, 16]), cost([2, 3, 5, 7, 9]), 15,
        [(0, 1), (2, 4)])
analyze("IV (N=5) v2", perf([3, 5, 9, 11, 14]), cost([1, 2, 5, 6, 8]), 13,
        [(0, 2), (1, 3)])

# V: N=6, want many Pareto points, several non-convex
analyze("V  (N=6) v1", perf([3, 5, 8, 10, 13, 17]),
        cost([1, 2, 4, 5, 7, 9]), 16, [(0, 1), (2, 3), (4, 5)])
analyze("V  (N=6) v2", perf([2, 4, 7, 9, 12, 15]),
        cost([1, 2, 3, 5, 6, 8]), 14, [(0, 2), (1, 4), (3, 5)])
