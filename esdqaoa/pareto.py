"""Pareto-front tracing and quantitative coverage metrics.

Design decision (important for the paper): the quantum trace is built from
ATTAINABLE designs, not probability-weighted expectations. For each phi in
the sweep, the modal feasible configuration (highest-probability feasible
bitstring) is recorded; the union over the sweep, filtered to non-dominated
points, forms the method's recovered front. Expected-value points are convex
mixtures of designs and are generally not realizable configurations, which
reviewers can (rightly) object to.
"""
import numpy as np


def modal_feasible(problem, probs):
    """Highest-probability feasible configuration, or None."""
    best, best_p = None, -1.0
    for x, pr in probs.items():
        xa = np.array(x)
        if problem.is_feasible(xa) and pr > best_p:
            best, best_p = xa, pr
    return best, best_p


def recovered_front(problem, sweep_probs):
    """Non-dominated set of modal designs over a phi sweep.

    sweep_probs: list of (phi, probs_dict).
    Returns sorted list of (cost, perf, config, phi_first_seen).
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
# Metrics vs the true front
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
    """Average distance from true-front points to nearest found point.

    (Coverage-oriented IGD.) 0 = every true Pareto point matched.
    """
    if not found:
        return float("inf")
    f_n, t_n = _normalize(found, true_front)
    dists = []
    for tp in t_n:
        dists.append(min(np.hypot(tp[0] - fp[0], tp[1] - fp[1])
                         for fp in f_n))
    return float(np.mean(dists))


def pcr(found, true_front, eps=0.15):
    """Pareto coverage rate: fraction of true points with a found point
    within normalized distance eps."""
    if not found:
        return 0.0
    f_n, t_n = _normalize(found, true_front)
    hits = sum(1 for tp in t_n
               if min(np.hypot(tp[0] - fp[0], tp[1] - fp[1])
                      for fp in f_n) <= eps)
    return hits / len(t_n)


def hypervolume_pct(found, true_front):
    """2-D hypervolume of the found set as % of the true front's HV.

    Reference point: (max cost, min perf) of the true front, normalized to
    (1, 0); maximize perf, minimize cost.
    """
    def hv(points):
        # points normalized; sort by cost asc, sweep dominated area vs (1,0)
        pts = sorted(set(points))
        area, prev_c, best_p = 0.0, None, 0.0
        # keep only non-dominated in normalized space
        nd = [q for q in pts
              if not any((c2 <= q[0] and p2 >= q[1]) and (c2, p2) != q
                         for c2, p2 in pts)]
        nd = sorted(nd)
        area = 0.0
        for idx, (c, p) in enumerate(nd):
            c_next = nd[idx + 1][0] if idx + 1 < len(nd) else 1.0
            area += max(0.0, (c_next - c)) * max(0.0, p)
        # region to the right of the last point up to ref c=1 already counted
        return area
    if not found:
        return 0.0
    f_n, t_n = _normalize(found, true_front)
    hv_true = hv(t_n)
    if hv_true == 0:
        return 100.0
    return 100.0 * hv(f_n) / hv_true
