"""P1.6 — sensitivity of the Pareto-recovery conclusions to two thresholds.

Both are analysis choices, not measurements:

  * the harvesting probability floor tau (reported protocol: 0.02);
  * the PCR matching radius in normalized objective space (0.15).

Neither requires a new quantum run: both are recomputed from the stored output
distributions. The question is whether the paper's conclusions -- harvesting
recovers far more of the frontier than modal tracking, and ESD-QAOA leads the
baseline on harvested coverage at the large end -- survive a reasonable range.

Emits a LaTeX table for the supplement.

Usage:
    python experiments/threshold_sensitivity.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from esdqaoa import pareto_v2 as pv  # noqa: E402
from esdqaoa.problems import problem_registry  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import paper_assets as pa  # noqa: E402

PAPER = ["III", "IV", "V", "Q13", "Q14", "Q15", "Q16", "Q17", "Q18",
         "Q19", "Q20"]
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"]
DISPLAY = dict(zip(PAPER, ROMAN))
TAUS = (0.01, 0.02, 0.05)
RADII = (0.10, 0.15, 0.20)


def sweep_of(rec):
    return [(e["phi"], {tuple(int(c) for c in x): v
                        for x, v in e["probs"].items()})
            for e in rec["sweep"]]


def main():
    reg = problem_registry()
    ws = {}
    for f in ("sweep_fixed.json", "sweep_ladder.json"):
        ws.update(json.load(open(os.path.join(ROOT, "results", f)))["runs"])

    print("harvesting threshold tau\n")
    print("  tau    ESD front  Pauli front  ESD unsup  Pauli unsup  "
          "ESD-Pauli front")
    rows_tau = []
    for tau in TAUS:
        t = dict(ef=0, pf=0, eu=0, pu=0)
        per_problem = {}
        for key in PAPER:
            pb = reg[key]
            labels = pb.classify_pareto_points()
            tf, uns = set(labels), {k for k, v in labels.items()
                                    if v == "unsupported"}
            got = {}
            for m in ("esd", "pauli"):
                h = {(int(c), int(p)) for c, p, _, _ in pv.harvested_front(
                    pb, sweep_of(ws[key]["methods"][m]), tau)}
                got[m] = h
            t["ef"] += len(got["esd"] & tf)
            t["pf"] += len(got["pauli"] & tf)
            t["eu"] += len(got["esd"] & uns)
            t["pu"] += len(got["pauli"] & uns)
            per_problem[DISPLAY[key]] = (len(got["esd"] & tf),
                                         len(got["pauli"] & tf))
        rows_tau.append((tau, t, per_problem))
        print("  %.2f %10d %12d %10d %12d %13s"
              % (tau, t["ef"], t["pf"], t["eu"], t["pu"],
                 "%+d" % (t["ef"] - t["pf"])))
    print("\n  modal-tracking reference (tau-independent): ESD 54, Pauli 47 "
          "front; ESD 2, Pauli 2 unsupported")

    print("\n\nPCR matching radius\n")
    print("  radius   ESD PCR=1.00 on   Pauli mean PCR   Pauli min PCR")
    rows_pcr = []
    for rad in RADII:
        e_ones, p_vals = 0, []
        for key in PAPER:
            pb = reg[key]
            tf = [(c, p) for c, p, _ in pb.pareto_front()]
            for m in ("esd", "pauli"):
                found = [(int(c), int(p)) for c, p, _, _
                         in pv.harvested_front(
                             pb, sweep_of(ws[key]["methods"][m]), 0.02)]
                val = pv.pcr(found, tf, eps=rad)
                if m == "esd" and abs(val - 1.0) < 1e-9:
                    e_ones += 1
                if m == "pauli":
                    p_vals.append(val)
        rows_pcr.append((rad, e_ones, float(np.mean(p_vals)), min(p_vals)))
        print("  %.2f %14s %17.3f %15.3f"
              % (rad, "%d of 11" % e_ones, np.mean(p_vals), min(p_vals)))

    # ---- supplement table ------------------------------------------------
    # Raw strings throughout: a plain literal would let Python expand \t, \r
    # and \v before the LaTeX reaches the file. Emitted through pa.wrap with
    # wide=True so the table gets the same adjustwidth / footnotesize /
    # tabcolsep treatment as the other wide tables and does not overrun the
    # text block. Column headers are kept short for the same reason.
    body = []
    for tau, t, _ in rows_tau:
        body.append(r"$%.2f$ & %d & %d & %d & %d & $%+d$ \\"
                    % (tau, t["ef"], t["pf"], t["eu"], t["pu"],
                       t["ef"] - t["pf"]))
    body.append(r"\midrule")
    body.append(r"PCR radius & \multicolumn{2}{c}{ESD-QAOA at PCR $=1$} & "
                r"\multicolumn{3}{c}{QUBO/Ising PCR, mean (min)} \\")
    body.append(r"\midrule")
    for rad, e_ones, mean_p, min_p in rows_pcr:
        body.append(r"$%.2f$ & \multicolumn{2}{c}{%d of 11} & "
                    r"\multicolumn{3}{c}{%.3f (%.3f)} \\"
                    % (rad, e_ones, mean_p, min_p))

    caption = (
        r"Sensitivity of Pareto recovery to the two analysis thresholds, "
        r"recomputed from the stored output distributions at $p=2$, seed 7. "
        r"Front points and unsupported designs are distinct Pareto objective "
        r"vectors, out of 129 and 49 respectively. Modal tracking does not "
        r"depend on $\tau$ and recovers 54 front points for ESD-QAOA and 47 "
        r"for the baseline. The upper block varies the harvesting threshold, "
        r"the lower block the PCR matching radius.")
    header = (r"& \multicolumn{2}{c}{Front points} & "
              r"\multicolumn{2}{c}{Unsupported} & \\" "\n"
              r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" "\n"
              r"$\tau$ & ESD & QUBO/Ising & ESD & QUBO/Ising & Difference \\")
    text = pa.wrap("\n".join(body), caption, "tab:sensitivity",
                   "l rr rr r", header, wide=True)

    outdir = os.path.join(ROOT, "paper", "paper_mdpi", "tables")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "tab_sensitivity.tex")
    open(path, "w", encoding="utf-8", newline="\n").write(text)
    print("\nwrote %s" % os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
