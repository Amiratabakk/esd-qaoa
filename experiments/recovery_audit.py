"""P0.1 downstream — every recovery count that depends on the classification.

Recomputes, from the stored output distributions, the numbers the manuscript
quotes in Tables 3 and 4, Figure 9 and the Results / Discussion / Conclusions
prose, after the supported/unsupported fix.

Counting convention (P0.6): entries are DISTINCT Pareto objective vectors
(cost, performance). Two architectures with the same objective vector count
once.

Usage:
    python experiments/recovery_audit.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from esdqaoa import pareto_v2 as pv  # noqa: E402
from esdqaoa.problems import problem_registry  # noqa: E402

PAPER = ["III", "IV", "V", "Q13", "Q14", "Q15", "Q16", "Q17", "Q18",
         "Q19", "Q20"]
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"]
DISPLAY = dict(zip(PAPER, ROMAN))


def load(name):
    return json.load(open(os.path.join(ROOT, "results", name)))["runs"]


def ws_sweep(rec):
    return [(e["phi"], {tuple(int(c) for c in x): v
                        for x, v in e["probs"].items()})
            for e in rec["sweep"]]


def vecs(front):
    return {(int(c), int(p)) for c, p, *_ in front}


def main():
    reg = problem_registry()
    ws = {}
    for f in ("sweep_fixed.json", "sweep_ladder.json"):
        ws.update(load(f))
    eps = {}
    for f in ("eps_sweep_I_V.json", "eps_ladder.json"):
        eps.update(load(f))

    tot = dict(pf=0, uns=0, sup=0,
               e_mod=0, e_har=0, p_mod=0, p_har=0,
               e_mod_u=0, e_har_u=0, p_mod_u=0, p_har_u=0,
               eps_ref=0, e_eps=0, p_eps=0, e_eps_u=0, p_eps_u=0)
    rows = []

    for key in PAPER:
        pb, r = reg[key], DISPLAY[key]
        labels = pb.classify_pareto_points()
        tf = set(labels)
        uns = {pt for pt, v in labels.items() if v == "unsupported"}
        sup = len(tf) - len(uns)

        # ---- weighted sum, both encodings, both estimators ----------------
        got = {}
        for m in ("esd", "pauli"):
            rec = ws[key]["methods"][m]
            sw = ws_sweep(rec)
            got[m] = (vecs(pv.recovered_front(pb, sw)),
                      vecs(pv.harvested_front(pb, sw, 0.02)))

        # ---- epsilon sweep, modal, eps-feasibility rule -------------------
        eps_got = {}
        for m in ("esd", "pauli"):
            pts = []
            for e in eps[key]["methods"][m]["sweep"]:
                best, bp = None, -1.0
                for x, pr in e["probs"].items():
                    xa = np.array([int(c) for c in x])
                    if pb.is_eps_feasible(xa, e["eps"]) and pr > bp:
                        best, bp = xa, pr
                if best is not None:
                    pts.append((int(pb.cost(best)), int(pb.perf(best))))
            nd = {(c, p) for c, p in pts
                  if not any((c2 <= c and p2 >= p) and (c2 < c or p2 > p)
                             for c2, p2 in pts)}
            eps_got[m] = nd
        eps_ref = len(eps[key]["classical_eps_front"])

        row = dict(
            name=r, pf=len(tf), uns=len(uns), sup=sup, eps_ref=eps_ref,
            e_mod=len(got["esd"][0] & tf), e_har=len(got["esd"][1] & tf),
            p_mod=len(got["pauli"][0] & tf), p_har=len(got["pauli"][1] & tf),
            e_mod_u=len(got["esd"][0] & uns), e_har_u=len(got["esd"][1] & uns),
            p_mod_u=len(got["pauli"][0] & uns),
            p_har_u=len(got["pauli"][1] & uns),
            e_eps=len(eps_got["esd"] & tf), p_eps=len(eps_got["pauli"] & tf),
            e_eps_u=len(eps_got["esd"] & uns),
            p_eps_u=len(eps_got["pauli"] & uns))
        rows.append(row)
        for k in tot:
            tot[k] += row[k]

    hdr = ("name", "pf", "uns", "sup", "e_mod", "e_har", "p_mod", "p_har",
           "e_mod_u", "e_har_u", "p_mod_u", "p_har_u",
           "eps_ref", "e_eps", "p_eps", "e_eps_u", "p_eps_u")
    print(" ".join("%7s" % h for h in hdr))
    for row in rows:
        print(" ".join("%7s" % row[h] for h in hdr))
    print(" ".join("%7s" % ("TOTAL" if h == "name" else tot[h]) for h in hdr))

    print("\nheadline numbers for the manuscript")
    print("  unsupported present                     : %d" % tot["uns"])
    print("  weighted sum + modal      (ESD)         : %d" % tot["e_mod_u"])
    print("  weighted sum + harvesting (ESD)         : %d" % tot["e_har_u"])
    print("  epsilon      + modal      (ESD)         : %d" % tot["e_eps_u"])
    print("  weighted sum + modal      (Pauli)       : %d" % tot["p_mod_u"])
    print("  weighted sum + harvesting (Pauli)       : %d" % tot["p_har_u"])
    print("  epsilon      + modal      (Pauli)       : %d" % tot["p_eps_u"])
    print("  front points, ESD modal/harvested       : %d / %d of %d"
          % (tot["e_mod"], tot["e_har"], tot["pf"]))
    print("  front points, Pauli modal/harvested     : %d / %d of %d"
          % (tot["p_mod"], tot["p_har"], tot["pf"]))
    print("  front points, eps modal ESD / Pauli     : %d / %d"
          % (tot["e_eps"], tot["p_eps"]))
    n0 = sum(1 for r in rows if r["e_mod_u"] == 0 and r["p_mod_u"] == 0)
    print("  problems with zero unsupported under w.sum modal, "
          "either encoding: %d of %d" % (n0, len(rows)))
    return rows, tot


if __name__ == "__main__":
    main()
