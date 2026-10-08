"""P0.1 — exact supported / unsupported classification of Pareto points.

The manuscript defines a Pareto point as *supported* when it maximises
f_phi(x) = phi*P - (1-phi)*C for some phi in [0,1], maximiser not required to
be unique. Tied optima on a supported face are therefore supported.

The shipped classifier does not implement that definition:

  * ``scalarization_reachable`` takes ``np.argmax`` over a phi grid, which
    returns only the FIRST maximiser, so every other point tied with it on the
    same face is recorded as unreachable;
  * ``supported_pareto_points`` pops the hull on ``cross >= 0``, which discards
    collinear points for the same reason.

Both therefore report face-tied points as unsupported.

This module decides the question exactly, with integer arithmetic and no phi
grid. Point i is supported iff there exists phi in [0,1] with

    phi*P_i - (1-phi)*C_i  >=  phi*P_j - (1-phi)*C_j   for every feasible j.

Rearranged, each j gives a linear constraint in phi:

    phi * A_j >= B_j,    A_j = (P_i - P_j) + (C_i - C_j),   B_j = C_i - C_j

so the admissible phi set is an intersection of half-lines with [0,1]. The
point is supported iff that intersection is non-empty. Using Fraction keeps
the comparison exact.

A supported point is additionally labelled ``extreme`` when its admissible phi
set has non-empty interior (it is the unique maximiser somewhere, i.e. a hull
vertex) and ``face`` when the set is a single phi value (it is only ever a
tied maximiser).

Usage:
    python experiments/classify_audit.py
"""
import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esdqaoa.problems import problem_registry  # noqa: E402

PAPER = ["III", "IV", "V", "Q13", "Q14", "Q15", "Q16", "Q17", "Q18",
         "Q19", "Q20"]
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"]
DISPLAY = dict(zip(PAPER, ROMAN))


def admissible_phi(point, feasible):
    """Exact [lo, hi] interval of phi for which `point` is a maximiser.

    Returns None when no such phi exists in [0,1]."""
    ci, pi = point
    lo, hi = Fraction(0), Fraction(1)
    for cj, pj in feasible:
        if (cj, pj) == (ci, pi):
            continue
        a = (pi - pj) + (ci - cj)
        b = ci - cj
        if a == 0:
            if b > 0:                      # 0 >= b>0 impossible
                return None
            continue                       # constraint vacuous
        bound = Fraction(b, a)
        if a > 0:
            lo = max(lo, bound)
        else:
            hi = min(hi, bound)
        if lo > hi:
            return None
    return (lo, hi)


def classify(pb):
    """Return (front, labels) with labels in extreme/face/unsupported."""
    cost, perf, _ = pb._tables()
    feasible = sorted({(int(c), int(p)) for c, p in zip(cost, perf)})
    front = [(c, p) for c, p, _ in pb.pareto_front()]
    labels = {}
    for pt in front:
        iv = admissible_phi(pt, feasible)
        if iv is None:
            labels[pt] = "unsupported"
        elif iv[0] == iv[1]:
            labels[pt] = "face"
        else:
            labels[pt] = "extreme"
    return front, labels


def main():
    reg = problem_registry()
    print("P0.1 exact classification vs the shipped classifier\n")
    print(" paper  |PF|  extreme  face  unsup | shipped_unsup  delta")
    tot_new = tot_old = 0
    rows = []
    for key in PAPER:
        pb = reg[key]
        front, labels = classify(pb)
        n_ext = sum(1 for v in labels.values() if v == "extreme")
        n_face = sum(1 for v in labels.values() if v == "face")
        n_uns = sum(1 for v in labels.values() if v == "unsupported")
        shipped = len(pb.nonconvex_pareto_points())
        tot_new += n_uns
        tot_old += shipped
        rows.append((DISPLAY[key], len(front), n_ext, n_face, n_uns, shipped))
        print("  %-5s %4d %7d %6d %6d | %12d  %+d"
              % (DISPLAY[key], len(front), n_ext, n_face, n_uns,
                 shipped, n_uns - shipped))
    print("\n  TOTAL unsupported: exact %d, shipped %d  (delta %+d)"
          % (tot_new, tot_old, tot_new - tot_old))

    # regression case named in the plan: problem I, the point (1, 2)
    pb = reg["III"]
    front, labels = classify(pb)
    print("\nregression check, paper problem I (internal 'III'):")
    for pt in front:
        iv = admissible_phi(pt, sorted({(int(c), int(p)) for c, p
                                        in zip(*pb._tables()[:2])}))
        txt = "none" if iv is None else "[%s, %s]" % (iv[0], iv[1])
        print("   (C,P)=%-9s %-12s admissible phi %s"
              % (str(pt), labels[pt], txt))
    assert (1, 2) in labels, "(1,2) is not on the front"
    print("\n   (1,2) is classified: %s" % labels[(1, 2)])


if __name__ == "__main__":
    main()
