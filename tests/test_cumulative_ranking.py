"""P0.5 — the ranking behind the cumulative-probability panels.

The panels place a dotted line at the rank where the feasible set is
exhausted and read the curve's height there as the probability of measuring a
feasible design. That reading is valid only if EVERY feasible configuration
is ranked ahead of EVERY infeasible one.

`cumulative_curve` guarantees it by sorting on a two-part key whose PRIMARY
component is the feasibility flag; the scalarized objective only orders the
feasible block. The property is therefore structural, not a consequence of
the penalty proof.

That distinction matters. Sorting by the penalized objective f_o instead
would NOT have the property: with a one-sided linear budget penalty, an
infeasible design that overshoots slightly can still outscore a poor feasible
one. This test asserts the property holds for the key actually used, and
demonstrates that it fails for f_o, so the caption must describe the real
rule.

Run: python tests/test_cumulative_ranking.py   (exit 0 on pass)
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from esdqaoa.problems import problem_registry  # noqa: E402
from esdqaoa.simulate import (default_alpha, default_beta_dep,  # noqa: E402
                              penalized_objective)

PAPER = ["III", "IV", "V", "Q13", "Q14", "Q15", "Q16", "Q17", "Q18",
         "Q19", "Q20"]
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"]
DISPLAY = dict(zip(PAPER, ROMAN))
PHIS = [0.0, 0.25, 0.5, 0.75, 1.0]


def ranking_key(pb, phi):
    """Exactly the key used by paper_assets.cumulative_curve."""
    keys = []
    for i in range(2 ** pb.N):
        x = np.array([(i >> b) & 1 for b in range(pb.N)])
        feas = pb.is_feasible(x)
        keys.append(((1 if feas else 0),
                     pb.scalarized(x, phi) if feas else 0.0,
                     tuple(int(v) for v in x)))
    keys.sort(key=lambda t: (-t[0], -t[1]))
    return keys


def test_feasible_block_precedes_infeasible():
    reg = problem_registry()
    for key in PAPER:
        pb = reg[key]
        n_feas = len(pb.feasible_configs())
        for phi in PHIS:
            order = ranking_key(pb, phi)
            flags = [k[0] for k in order]
            # every feasible (flag 1) must come before every infeasible (0)
            first_infeasible = flags.index(0) if 0 in flags else len(flags)
            assert all(flags[:first_infeasible]), (key, phi)
            assert not any(flags[first_infeasible:]), (key, phi)
            assert first_infeasible == n_feas, (key, phi, first_infeasible,
                                                n_feas)
    print("  feasible block precedes infeasible on 11 problems x 5 phi   OK")


def test_penalized_objective_would_not_have_the_property():
    """The explicit feasibility key is load-bearing, not decoration."""
    reg = problem_registry()
    offenders = []
    for key in PAPER:
        pb = reg[key]
        a, b = default_alpha(pb), default_beta_dep(pb)
        for phi in PHIS:
            fo, feas = [], []
            for i in range(2 ** pb.N):
                x = np.array([(i >> t) & 1 for t in range(pb.N)])
                fo.append(penalized_objective(x, pb, phi, a, b))
                feas.append(bool(pb.is_feasible(x)))
            fo = np.array(fo)
            worst_feasible = fo[np.array(feas)].min()
            best_infeasible = fo[~np.array(feas)].max()
            if best_infeasible > worst_feasible:
                offenders.append((DISPLAY[key], phi,
                                  float(best_infeasible),
                                  float(worst_feasible)))
    assert offenders, ("expected at least one case where an infeasible "
                       "design outscores a feasible one under f_o")
    print("  sorting by f_o would interleave on %d of %d (problem, phi) cases"
          % (len(offenders), len(PAPER) * len(PHIS)))
    for name, phi, bi, wf in offenders[:4]:
        print("     problem %-3s phi=%.2f  best infeasible %.2f > "
              "worst feasible %.2f" % (name, phi, bi, wf))
    return offenders


if __name__ == "__main__":
    test_feasible_block_precedes_infeasible()
    test_penalized_objective_would_not_have_the_property()
    print("\ncumulative-ranking tests passed")
