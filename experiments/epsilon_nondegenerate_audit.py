"""P0.3 / P0.4 — redefine the scored epsilon sweep to exclude degenerate bounds.

An epsilon bound is DEGENERATE when the designs satisfying C(x) <= eps and the
dependency constraints all share one performance value, so P_max == P_min and
eq:rhoeps has a zero denominator. The shipped scorer returns 1.0 there, which
puts a free perfect score into every reported mean.

This script recomputes, from the retained (non-degenerate) bounds only:

  * which bounds are retained per problem;
  * the pinned comparison position c* and register width k*, and hence the
    epsilon circuit's qubit count -- this MUST be recomputed rather than
    carried over, because the pinned width is driven by the SMALLEST retained
    epsilon (w0 = 2^c - eps - 1 grows as eps shrinks);
  * mean rho_eps per method;
  * modal Pareto recovery over the retained bounds;
  * the exact classical epsilon reference front over the retained bounds.

Usage:
    python experiments/epsilon_nondegenerate_audit.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from esdqaoa.circuits import budget_encoding, pinned_encoding  # noqa: E402
from esdqaoa.problems import problem_registry  # noqa: E402

PAPER = ["III", "IV", "V", "Q13", "Q14", "Q15", "Q16", "Q17", "Q18",
         "Q19", "Q20"]
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"]
DISPLAY = dict(zip(PAPER, ROMAN))


def is_degenerate(pb, eps):
    """True when every eps-feasible design shares one performance value."""
    vals = {int(pb.perf(x)) for x in pb.all_configs()
            if pb.is_eps_feasible(x, eps)}
    return len(vals) <= 1


def nondegenerate(pb, eps_values):
    return [e for e in eps_values if not is_degenerate(pb, e)]


def main():
    reg = problem_registry()
    eps = {}
    for f in ("eps_sweep_I_V.json", "eps_ladder.json"):
        eps.update(json.load(open(os.path.join(ROOT, "results", f)))["runs"])

    print("retained bounds and recomputed pinned width\n")
    print(" paper  n_all n_kept  dropped      c*_all k*_all  c*_kept k*_kept"
          "  q_ws  q_eps_all  q_eps_kept  width_vs_ws")
    width_rows = []
    for key in PAPER:
        pb = reg[key]
        all_eps = [e["eps"] for e in eps[key]["methods"]["esd"]["sweep"]]
        kept = nondegenerate(pb, all_eps)
        dropped = [e for e in all_eps if e not in kept]
        c_all, k_all = pinned_encoding(pb, all_eps)
        c_kept, k_kept = pinned_encoding(pb, kept)
        _, _, k_ws = budget_encoding(pb)
        q_ws = pb.N + k_ws + 1
        q_all = pb.N + k_all + 1
        q_kept = pb.N + k_kept + 1
        width_rows.append((DISPLAY[key], q_ws, q_all, q_kept))
        print("  %-5s %5d %6d  %-11s %6d %6d %8d %7d %5d %10d %11d  %+d"
              % (DISPLAY[key], len(all_eps), len(kept), str(dropped),
                 c_all, k_all, c_kept, k_kept, q_ws, q_all, q_kept,
                 q_kept - q_ws))

    same = [r[0] for r in width_rows if r[3] == r[1]]
    wider = [r[0] for r in width_rows if r[3] > r[1]]
    print("\n  epsilon width equals weighted sum on: %s"
          % (", ".join(same) if same else "none"))
    print("  epsilon width exceeds weighted sum on: %s"
          % (", ".join(wider) if wider else "none"))

    print("\n\nmean rho_eps over retained bounds\n")
    print(" paper  n_kept   ESD    Pauli     gap")
    gaps = []
    for key in PAPER:
        pb, r = reg[key], eps[key]
        all_eps = [e["eps"] for e in r["methods"]["esd"]["sweep"]]
        kept = set(nondegenerate(pb, all_eps))
        out = {}
        for m in ("esd", "pauli"):
            out[m] = float(np.mean([e["rho_eps"]
                                    for e in r["methods"][m]["sweep"]
                                    if e["eps"] in kept]))
        gaps.append(out["esd"] - out["pauli"])
        print("  %-5s %6d  %.3f  %.3f  %+.3f"
              % (DISPLAY[key], len(kept), out["esd"], out["pauli"], gaps[-1]))
    print("\n  mean gap %+.4f, positive on %d of %d, range %+.3f to %+.3f"
          % (np.mean(gaps), sum(g > 0 for g in gaps), len(gaps),
             min(gaps), max(gaps)))

    print("\n\nmodal Pareto recovery over retained bounds\n")
    print(" paper  |PF| unsup  eps_ref  ESD_eps Pauli_eps  ESD_u Pauli_u")
    t = dict(pf=0, u=0, ref=0, e=0, p=0, eu=0, pu=0)
    for key in PAPER:
        pb, r = reg[key], eps[key]
        labels = pb.classify_pareto_points()
        tf, uns = set(labels), {k for k, v in labels.items()
                                if v == "unsupported"}
        all_eps = [e["eps"] for e in r["methods"]["esd"]["sweep"]]
        kept = set(nondegenerate(pb, all_eps))
        got = {}
        for m in ("esd", "pauli"):
            pts = []
            for e in r["methods"][m]["sweep"]:
                if e["eps"] not in kept:
                    continue
                best, bp = None, -1.0
                for x, pr in e["probs"].items():
                    xa = np.array([int(c) for c in x])
                    if pb.is_eps_feasible(xa, e["eps"]) and pr > bp:
                        best, bp = xa, pr
                if best is not None:
                    pts.append((int(pb.cost(best)), int(pb.perf(best))))
            got[m] = {(c, p) for c, p in pts
                      if not any((c2 <= c and p2 >= p) and (c2 < c or p2 > p)
                                 for c2, p2 in pts)}
        ref = set()
        for e in sorted(kept):
            o = pb.epsilon_constraint_optimum(e)
            if o is not None:
                ref.add((int(o[0]), int(o[1])))
        ref = {(c, p) for c, p in ref
               if not any((c2 <= c and p2 >= p) and (c2 < c or p2 > p)
                          for c2, p2 in ref)}
        row = (len(tf), len(uns), len(ref), len(got["esd"] & tf),
               len(got["pauli"] & tf), len(got["esd"] & uns),
               len(got["pauli"] & uns))
        for k_, v in zip(("pf", "u", "ref", "e", "p", "eu", "pu"), row):
            t[k_] += v
        print("  %-5s %4d %5d %8d %8d %9d %6d %7d" % ((DISPLAY[key],) + row))
    print("  TOTAL %4d %5d %8d %8d %9d %6d %7d"
          % (t["pf"], t["u"], t["ref"], t["e"], t["p"], t["eu"], t["pu"]))


if __name__ == "__main__":
    main()
