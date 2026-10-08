"""Pareto-front recovery and quantitative coverage metrics.

Two estimators are provided:

1. Modal tracking (legacy): the highest-probability feasible design at each
   phi. NOTE: because the sweep optimizes a weighted-sum objective, the
   modal design tracks the scalarization optimum, which is by construction
   a convex-hull vertex. This estimator therefore CANNOT systematically
   recover non-convex Pareto points and yields at most a handful of
   distinct designs per sweep. Retained for comparison only.

2. Distribution harvesting (default for the paper): every feasible design
   observed with probability >= tau at any phi in the sweep, filtered to
   the non-dominated set. This treats each run as returning a sampled
   portfolio of candidate designs -- the quantum-native estimator -- and
   recovers non-convex points from secondary probability mass. tau = 0.02
   corresponds to designs reliably observed within a few hundred shots.

Coverage metrics (ADPF / PCR / HV%) apply to either recovered set.
Portfolio-quality metrics (feasible mass, Pareto mass) quantify sampling
efficiency and are where encoding differences show up even when raw
coverage saturates.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Estimator 1: modal tracking (legacy)
# ---------------------------------------------------------------------------
def modal_feasible(problem, probs):
    """Highest-probability feasible configuration, or (None, -1)."""
    best, best_p = None, -1.0
    for x, pr in probs.items():
        xa = np.array(x)
        if problem.is_feasible(xa) and pr > best_p:
            best, best_p = xa, pr
    return best, best_p


def recovered_front(problem, sweep_probs):
    """Non-dominated set of modal designs over a phi sweep (legacy).

    sweep_probs: list of (phi, probs_dict) with tuple keys, index i =
    subsystem i. Returns sorted list of (cost, perf, config, phi_first).
    """
    pts = {}
    for phi, probs in sweep_probs:
        x, _ = modal_feasible(problem, probs)
        if x is None:
            continue
        key = tuple(x)
        if key not in pts:
            pts[key] = (problem.cost(x), problem.perf(x), key, phi)
    vals = list(pts.values())
    front = []
    for c, p, x, phi in vals:
        dominated = any((c2 <= c and p2 >= p) and (c2 < c or p2 > p)
                        for c2, p2, _, _ in vals)
        if not dominated:
            front.append((c, p, x, phi))
    return sorted(front)


# ---------------------------------------------------------------------------
# Estimator 2: distribution harvesting
# ---------------------------------------------------------------------------
def harvested_front(problem, sweep_probs, tau=0.02):
    """Non-dominated set of all feasible designs with prob >= tau anywhere
    in the sweep.

    Returns sorted list of (cost, perf, config, p_max) where p_max is the
    largest probability with which the design was observed at any phi --
    reported so the paper can state the sampling cost of each recovered
    point.
    """
    pts = {}
    for phi, probs in sweep_probs:
        for x, pr in probs.items():
            if pr < tau:
                continue
            xa = np.array(x)
            if not problem.is_feasible(xa):
                continue
            key = tuple(x)
            if key not in pts or pr > pts[key][3]:
                pts[key] = (problem.cost(xa), problem.perf(xa), key, pr)
    vals = list(pts.values())
    front = []
    for c, p, x, pm in vals:
        dominated = any((c2 <= c and p2 >= p) and (c2 < c or p2 > p)
                        for c2, p2, _, _ in vals)
        if not dominated:
            front.append((c, p, x, pm))
    return sorted(front)


# ---------------------------------------------------------------------------
# Portfolio-quality metrics
# ---------------------------------------------------------------------------
def feasible_mass(problem, probs):
    """Fraction of probability mass on feasible designs."""
    return sum(pr for x, pr in probs.items()
               if problem.is_feasible(np.array(x)))


def pareto_mass(problem, probs, true_front_set=None):
    """Fraction of probability mass on Pareto-optimal designs."""
    if true_front_set is None:
        true_front_set = {(c, p) for c, p, _ in problem.pareto_front()}
    total = 0.0
    for x, pr in probs.items():
        xa = np.array(x)
        if problem.is_feasible(xa) and \
                (problem.cost(xa), problem.perf(xa)) in true_front_set:
            total += pr
    return total


def sweep_mass_metrics(problem, sweep_probs):
    """Mean feasible mass and mean Pareto mass across the phi sweep."""
    tf = {(c, p) for c, p, _ in problem.pareto_front()}
    fm = [feasible_mass(problem, probs) for _, probs in sweep_probs]
    pm = [pareto_mass(problem, probs, tf) for _, probs in sweep_probs]
    return float(np.mean(fm)), float(np.mean(pm))


# ---------------------------------------------------------------------------
# Coverage metrics vs the true front
# ---------------------------------------------------------------------------
def _normalize(points, ref_front):
    """Scale (cost, perf) to [0,1]^2 using the true front's ranges."""
    cs = [c for c, p in ref_front]
    ps = [p for c, p in ref_front]
    c_rng = max(cs) - min(cs) or 1.0
    p_rng = max(ps) - min(ps) or 1.0
    return [((c - min(cs)) / c_rng, (p - min(ps)) / p_rng)
            for c, p in points], \
           [((c - min(cs)) / c_rng, (p - min(ps)) / p_rng)
            for c, p in ref_front]


def adpf(found, true_front):
    """Average distance from true-front points to nearest found point
    (coverage-oriented IGD). 0 = every true Pareto point matched."""
    if not found:
        return float("inf")
    f_n, t_n = _normalize(found, true_front)
    dists = [min(np.hypot(tp[0] - fp[0], tp[1] - fp[1]) for fp in f_n)
             for tp in t_n]
    return float(np.mean(dists))


def pcr(found, true_front, eps=0.15):
    """Fraction of true Pareto points with a found point within
    normalized distance eps."""
    if not found:
        return 0.0
    f_n, t_n = _normalize(found, true_front)
    hits = sum(1 for tp in t_n
               if min(np.hypot(tp[0] - fp[0], tp[1] - fp[1])
                      for fp in f_n) <= eps)
    return hits / len(t_n)


def hypervolume_pct(found, true_front):
    """2-D hypervolume of the found set as % of the true front's HV.
    Reference point (1, 0) in normalized space; maximize perf, min cost."""
    def hv(points):
        pts = sorted(set(points))
        nd = [q for q in pts
              if not any((c2 <= q[0] and p2 >= q[1]) and (c2, p2) != q
                         for c2, p2 in pts)]
        nd = sorted(nd)
        area = 0.0
        for idx, (c, p) in enumerate(nd):
            c_next = nd[idx + 1][0] if idx + 1 < len(nd) else 1.0
            area += max(0.0, c_next - c) * max(0.0, p)
        return area
    if not found:
        return 0.0
    f_n, t_n = _normalize(found, true_front)
    hv_true = hv(t_n)
    if hv_true == 0:
        return 100.0
    return 100.0 * hv(f_n) / hv_true


def nonconvex_coverage(found, nonconvex_pts):
    """(#captured, #total) exact matches of non-convex Pareto points."""
    fs = {(c, p) for c, p in found}
    hit = sum(1 for pt in nonconvex_pts if tuple(pt) in fs)
    return hit, len(nonconvex_pts)
