"""P0.1 regression tests for supported / unsupported Pareto classification.

The manuscript's definition of *supported* admits tied maximizers, so a point
collinear with its neighbours on a supported face is supported. The earlier
grid-and-argmax implementation lost exactly those points.

Run: python tests/test_pareto_classification.py   (exit 0 on pass)
"""
import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esdqaoa.problems import SDProblem, problem_registry  # noqa: E402

PAPER = ["III", "IV", "V", "Q13", "Q14", "Q15", "Q16", "Q17", "Q18",
         "Q19", "Q20"]


def test_collinear_face_is_supported():
    """Paper problem I: (C,P)=(1,2) is collinear with (0,0) and (4,8).

    It maximizes f_phi at phi = 1/3 exactly, tied with both neighbours, so it
    is supported under the manuscript's definition and must not be counted as
    unsupported.
    """
    pb = problem_registry()["III"]
    labels = pb.classify_pareto_points()
    assert (1, 2) in labels, "(1,2) is not on the Pareto front"
    assert labels[(1, 2)] == "face", labels[(1, 2)]
    assert pb.admissible_phi((1, 2)) == (Fraction(1, 3), Fraction(1, 3))
    assert (1, 2) not in pb.nonconvex_pareto_points()
    assert (1, 2) in pb.scalarization_reachable()
    print("  collinear face point (1,2) classified supported          OK")


def test_synthetic_three_point_face():
    """A hand-built instance whose entire front is one collinear face."""
    pb = SDProblem(name="face", performances=[2, 2, 2], costs=[1, 1, 1],
                   budget=3, dependencies=[])
    labels = pb.classify_pareto_points()
    assert all(v != "unsupported" for v in labels.values()), labels
    print("  synthetic all-collinear front has no unsupported points  OK")


def test_brute_force_agreement():
    """Exact classifier must agree with a direct search over rational phi.

    For every Pareto point, scan the candidate phi values at which the
    identity of the maximizer can change (the pairwise crossing points) and
    confirm the point does or does not attain the maximum there.
    """
    for key in PAPER[:5]:
        pb = problem_registry()[key]
        cost, perf, _ = pb._tables()
        feas = sorted({(int(c), int(p)) for c, p in zip(cost, perf)})
        cands = {Fraction(0), Fraction(1)}
        for i, (ci, pi) in enumerate(feas):
            for cj, pj in feas[i + 1:]:
                a = (pi - pj) + (ci - cj)
                if a != 0:
                    f = Fraction(ci - cj, a)
                    if 0 <= f <= 1:
                        cands.add(f)
        labels = pb.classify_pareto_points()
        for pt, lab in labels.items():
            hit = False
            for phi in cands:
                best = max(phi * p - (1 - phi) * c for c, p in feas)
                if phi * pt[1] - (1 - phi) * pt[0] == best:
                    hit = True
                    break
            assert hit == (lab != "unsupported"), (key, pt, lab, hit)
        print("  %-4s brute-force agreement over %3d crossing phi values  OK"
              % (key, len(cands)))


def test_counts_are_self_consistent():
    """|PF| must equal extreme + face + unsupported on every problem."""
    reg = problem_registry()
    total_unsup = 0
    for key in PAPER:
        pb = reg[key]
        labels = pb.classify_pareto_points()
        front = pb.pareto_front()
        assert len(labels) == len({(c, p) for c, p, _ in front})
        n_uns = sum(1 for v in labels.values() if v == "unsupported")
        assert n_uns == len(pb.nonconvex_pareto_points())
        assert len(labels) - n_uns == len(pb.supported_pareto_points())
        total_unsup += n_uns
    print("  per-problem counts partition the front                   OK")
    print("  total unsupported across the eleven problems: %d" % total_unsup)
    return total_unsup


if __name__ == "__main__":
    test_collinear_face_is_supported()
    test_synthetic_three_point_face()
    test_brute_force_agreement()
    total = test_counts_are_self_consistent()
    assert total == 49, "expected 49 unsupported points, got %d" % total
    print("\nall pareto-classification tests passed")
