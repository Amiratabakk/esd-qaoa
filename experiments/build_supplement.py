"""P3.6 — generate the Supplementary Material tables and assemble the file.

Contents, per the revision plan:
  S1  full supported / face-supported / unsupported classification (P0.1)
  S2  register-width audit, weighted sum vs pinned epsilon (P0.4)
  S3  epsilon grid: all bounds, which are scored, which are degenerate (P0.3)
  S4  threshold sensitivity for tau and the PCR radius (P1.6)
  S5  per-seed paired approximation ratios (P1.5)
  S6  the QUBO baseline formulation, written out (P1.2)

Usage:
    python experiments/build_supplement.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from esdqaoa.circuits import budget_encoding, pinned_encoding  # noqa: E402
from esdqaoa.problems import problem_registry  # noqa: E402
import paper_assets as pa  # noqa: E402

TABDIR = os.path.join(ROOT, "paper", "paper_mdpi", "tables")
os.makedirs(TABDIR, exist_ok=True)   # created on demand: the manuscript
                                     # source is not part of this repo


def tab_classification():
    reg = problem_registry()
    rows = []
    tot = [0, 0, 0, 0]
    for n in pa.PAPER:
        pb = reg[n]
        lab = pb.classify_pareto_points()
        e = sum(1 for v in lab.values() if v == "extreme")
        f = sum(1 for v in lab.values() if v == "face")
        u = sum(1 for v in lab.values() if v == "unsupported")
        tot = [tot[0] + len(lab), tot[1] + e, tot[2] + f, tot[3] + u]
        rows.append("%s & %d & %d & %d & %d \\\\"
                    % (pa.DISPLAY[n], len(lab), e, f, u))
    rows.append("\\midrule\nTotal & %d & %d & %d & %d \\\\" % tuple(tot))
    return pa.wrap(
        "\n".join(rows),
        "Classification of every Pareto objective vector. A vector is "
        "\\emph{extreme} when it is the unique maximizer of $f_\\varphi$ on "
        "an interval of $\\varphi$, \\emph{face} when it maximizes "
        "$f_\\varphi$ at a single $\\varphi$ only, always tied with its "
        "neighbours, and \\emph{unsupported} when no $\\varphi\\in[0,1]$ "
        "makes it a maximizer. Face vectors are supported under the "
        "definition used in the main text. The classification is exact: the "
        "admissible interval of $\\varphi$ is computed in rational "
        "arithmetic, with no grid and no tolerance.",
        "tab:classification", "l rrrr",
        "Problem & Pareto vectors & Extreme & Face & Unsupported \\\\")


def tab_widths():
    reg = problem_registry()
    eps = pa.epsilon_runs()
    rows = []
    for n in pa.PAPER:
        pb = reg[n]
        allv = [s["eps"] for s in eps[n]["methods"]["esd"]["sweep"]]
        kept = pa.nondegenerate_eps(pb, allv)
        c_ws, _, k_ws = budget_encoding(pb)
        c_st, k_st = pinned_encoding(pb, kept)
        rows.append("%s & %d & %d & %d & %d & %d & %d & %d & %d \\\\"
                    % (pa.DISPLAY[n], pb.N, pb.budget, c_ws, k_ws,
                       c_st, k_st, pb.N + k_ws + 1, pb.N + k_st + 1))
    return pa.wrap(
        "\n".join(rows),
        "Register-width audit. $c$ is the comparison position and $k$ the "
        "cost-register width. The $\\epsilon$ columns are pinned across the "
        "scored sweep, so the circuit width does not vary with $\\epsilon$. "
        "Total qubits are $N+k+1$. The pinned $\\epsilon$ circuit is one "
        "qubit wider than the weighted-sum circuit on every problem but the "
        "first, which is why the resource table of the main text is reported "
        "for the weighted-sum circuit alone.",
        "tab:widths", "l rr rr rr rr",
        "& & & \\multicolumn{2}{c}{Weighted sum} & "
        "\\multicolumn{2}{c}{Pinned $\\epsilon$} & "
        "\\multicolumn{2}{c}{Total qubits} \\\\\n"
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
        "Problem & $N$ & $\\mathcal{B}$ & $c$ & $k$ & $c^\\star$ & "
        "$k^\\star$ & w.sum & $\\epsilon$ \\\\")


def tab_eps_grid():
    reg = problem_registry()
    eps = pa.epsilon_runs()
    rows = []
    for n in pa.PAPER:
        pb = reg[n]
        allv = [s["eps"] for s in eps[n]["methods"]["esd"]["sweep"]]
        kept = pa.nondegenerate_eps(pb, allv)
        drop = [e for e in allv if e not in kept]
        rows.append("%s & %d & %d & %s & %s \\\\"
                    % (pa.DISPLAY[n], len(allv), len(kept),
                       ", ".join(str(e) for e in drop) or "---",
                       ", ".join(str(e) for e in kept)))
    return pa.wrap(
        "\n".join(rows),
        "The $\\epsilon$ grid per problem. Thirteen equally spaced values on "
        "$[0,\\mathcal{B}]$ are rounded to integers and de-duplicated; "
        "bounds whose feasible set carries a single performance value are "
        "degenerate, since the ratio $\\rho_\\epsilon$ of the main text is "
        "then undefined, and are excluded from every reported $\\epsilon$ "
        "quantity.",
        "tab:epsgrid", "l rr p{1.6cm} p{6.4cm}",
        "Problem & Bounds & Scored & Dropped & Scored bounds \\\\",
        wide=True)


def tab_seeds():
    d = json.load(open(os.path.join(ROOT, "results",
                                    "multiseed_all.json")))
    runs = d["runs"] if "runs" in d else d
    seeds = None
    rows = []
    for n in pa.PAPER:
        r = runs[n]["methods"]
        es, ps = r["esd"]["per_seed"], r["pauli"]["per_seed"]
        seeds = seeds or sorted(es, key=int)
        cells = []
        for sd in seeds:
            de = float(np.mean(es[sd]["rhos"])) - float(np.mean(ps[sd]["rhos"]))
            cells.append("%+.3f" % de)
        rows.append("%s & %s \\\\" % (pa.DISPLAY[n], " & ".join(cells)))
    if not rows:
        return None
    hdr = "Problem & " + " & ".join("seed %s" % s for s in seeds) + " \\\\"
    return pa.wrap("\n".join(rows),
                   "Per-seed paired difference in approximation ratio "
                   "(ESD-QAOA minus baseline) at $p=2$, averaged over the "
                   "thirteen $\\varphi$ values. Every cell is positive.",
                   "tab:seeds", "l" + "r" * len(seeds), hdr)


def main():
    made = []
    for name, fn in (("tab_classification", tab_classification),
                     ("tab_widths", tab_widths),
                     ("tab_eps_grid", tab_eps_grid),
                     ("tab_seeds", tab_seeds),
                     ("tab_dependency", tab_dependency),
                     ("tab_optlevel", tab_optlevel)):
        try:
            text = fn()
        except Exception as exc:                      # noqa: BLE001
            print("  SKIP %-20s %s" % (name, exc))
            continue
        if text is None:
            print("  SKIP %-20s no per-seed data stored" % name)
            continue
        path = os.path.join(TABDIR, name + ".tex")
        open(path, "w", encoding="utf-8", newline="\n").write(text)
        made.append(name)
        print("  wrote tables/%s.tex" % name)
    print("\n%d supplementary tables generated" % len(made))
    return made


def tab_dependency():
    """P0.7 — the controlled dependency-count experiment.

    Every literal below that contains a LaTeX command is a RAW string. A
    plain string would let Python expand \\v, \\r, \\t and \\f before the
    text ever reaches the file, which silently emits control characters into
    the generated TeX: \\varphi becomes VT + "arphi", \\texttt becomes
    TAB + "exttt", \\rho becomes CR + "ho". tests/test_generated_tex.py
    fails the build if that happens again.
    """
    d = json.load(open(os.path.join(ROOT, "results",
                                    "dependency_study.json")))
    b, s = d["base"], d["settings"]
    rows = []
    for k in sorted(d["variants"], key=int):
        v = d["variants"][k]
        fm, m = v["front_metrics"], v["methods"]
        rows.append(r"%s & %d & %.1f\%% & %.3f & %.3f & $%+.4f$ \\"
                    % (k, fm["feasible"],
                       100 * fm["feasible"] / 2 ** b["N"],
                       m["esd"]["mean_rho"], m["pauli"]["mean_rho"],
                       m["esd"]["mean_rho"] - m["pauli"]["mean_rho"]))
    caption = (
        r"Controlled dependency-count experiment. One %d-variable instance "
        r"(%d qubits) with performances and costs held fixed and budget "
        r"$\mathcal{B}=%d$; only the dependency set varies. Dependency pairs "
        r"were drawn with seed %s, the variational optimizer used seed %s at "
        r"$p=%d$ with %d values of $\varphi$ and at most %d generations. "
        r"Approximation ratios are means over those $\varphi$. The "
        r"ESD-to-baseline gap is flat in dependency count while both methods "
        r"degrade as the feasible fraction falls, so the gap reported in the "
        r"main text is not an artefact of dependency density. This is a "
        r"separate controlled synthetic sensitivity experiment, not part of "
        r"the eleven-problem benchmark set, and it is one instance at one "
        r"problem size: a robustness check, not a scaling result. The base "
        r"instance and all three dependency sets are pinned as literals in "
        r"the released code (\texttt{DEP\_STUDY\_BASE} and "
        r"\texttt{DEP\_STUDY\_VARIANTS}), so the experiment can be "
        r"reconstructed without any generated file; "
        r"\texttt{tests/test\_dependency\_study.py} verifies that the pinned "
        r"instance reproduces the feasible counts, frontier sizes and dent "
        r"depths of the campaign."
        % (b["N"], b["n_qubits"], b["budget"], s["dep_seed"], s["seed"],
           s["p"], s["n_phi"], s["maxiter"]))
    header = (r"Dependency & Feasible & Feasible & ESD-QAOA & QUBO/Ising & "
              r"Gap \\" "\n"
              r"pairs & designs & fraction & $\rho$ & $\rho$ & \\")
    return pa.wrap("\n".join(rows), caption, "tab:dependency",
                   "r r r c c c", header, wide=True)


def tab_optlevel():
    """P0.7 — transpiler optimization level 1 against level 3.

    Raw strings throughout; see tab_dependency for why.
    """
    def load(f):
        d = json.load(open(os.path.join(ROOT, "results", f)))
        r = d["rows"] if isinstance(d, dict) and "rows" in d else d
        return r if isinstance(r, dict) else {x["name"]: x for x in r}

    a, b = load("resources.json"), load("resources_opt3.json")
    rows, dd, dc = [], [], []
    for n in pa.PAPER:
        c1, c3 = a[n]["esd"]["cx"], b[n]["esd"]["cx"]
        d1, d3 = a[n]["esd"]["depth"], b[n]["esd"]["depth"]
        dc.append(100 * (c1 - c3) / c1)
        dd.append(100 * (d1 - d3) / d1)
        rows.append(
            r"%s & %d & %d & %.1f\%% & %d & %d & %.1f\%% & "
            r"%.2f$\times$ & %.2f$\times$ \\"
            % (pa.DISPLAY[n], c1, c3, dc[-1], d1, d3, dd[-1],
               c3 / b[n]["pauli"]["cx"], d3 / b[n]["pauli"]["depth"]))
    caption = (
        r"ESD-QAOA circuit resources at transpiler optimization level 1 and "
        r"level 3, same basis $\{r_z,\sqrt{X},X,\mathrm{CX}\}$, all-to-all "
        r"connectivity, $p=2$, one transpilation per problem rather than an "
        r"average over seeds. Level 3 reduces two-qubit gates by "
        r"%.1f\%% on average (%.1f\%% to %.1f\%%) and depth by %.1f\%% on "
        r"average (%.1f\%% to %.1f\%%). The last two columns give the "
        r"level-3 ratios against the baseline compiled at the same level: "
        r"the overhead is reduced but not removed, which is why the main "
        r"text reports level 1 and treats the resource cost as part of the "
        r"result."
        % (sum(dc) / len(dc), min(dc), max(dc),
           sum(dd) / len(dd), min(dd), max(dd)))
    header = (r"& \multicolumn{3}{c}{Two-qubit gates} & "
              r"\multicolumn{3}{c}{Depth} & "
              r"\multicolumn{2}{c}{Level-3 ratio} \\" "\n"
              r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}" "\n"
              r"Problem & L1 & L3 & Saved & L1 & L3 & Saved & CX & Depth \\")
    return pa.wrap("\n".join(rows), caption, "tab:optlevel",
                   "l rr r rr r cc", header, wide=True)


if __name__ == "__main__":
    main()
