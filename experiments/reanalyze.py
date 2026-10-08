"""Re-analyze an existing sweep JSON with distribution harvesting.

No re-simulation: reads results/sweep.json, rebuilds recovered fronts from
the stored probability distributions, computes coverage + portfolio-quality
metrics, and regenerates figures and LaTeX snippets.

Usage (from repo root):
    python experiments/reanalyze.py results/sweep.json --tau 0.02

Outputs:
    figures/pareto_<name>.png        staircase true front + layered markers
    figures/pareto_<name>.tex        PGFPlots \\addplot snippets (const plot)
    figures/metrics_harvest.tex      full metrics table (LaTeX)
    stdout                           summary table
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from esdqaoa.problems import problem_registry

# includes the generated Q13..Q20 ladder when
# results/instances.json is present
PROBLEMS = problem_registry()
from esdqaoa.pareto_v2 import (harvested_front, recovered_front,
                               sweep_mass_metrics, adpf, pcr,
                               hypervolume_pct, nonconvex_coverage)


def parse_sweep(method_rec):
    """JSON sweep -> list of (phi, probs_dict with tuple keys).

    Bitstring keys are little-endian: character i = subsystem i.
    """
    out = []
    for s in method_rec["sweep"]:
        probs = {tuple(int(b) for b in k): v for k, v in s["probs"].items()}
        out.append((s["phi"], probs))
    return out


def analyze_method(pb, method_rec, tau):
    sweep = parse_sweep(method_rec)
    true_front = [(c, p) for c, p, _ in pb.pareto_front()]
    nonconvex = pb.nonconvex_pareto_points()

    hf = harvested_front(pb, sweep, tau=tau)
    found = [(c, p) for c, p, _, _ in hf]
    fm, pm = sweep_mass_metrics(pb, sweep)
    nc_hit, nc_tot = nonconvex_coverage(found, nonconvex)
    rhos = [s["rho"] for s in method_rec["sweep"]]
    return {
        "front": hf,
        "found": found,
        "metrics": {
            "ADPF": adpf(found, true_front),
            "PCR": pcr(found, true_front),
            "HV_pct": hypervolume_pct(found, true_front),
            "nonconvex": f"{nc_hit}/{nc_tot}",
            "feasible_mass": fm,
            "pareto_mass": pm,
            "mean_rho": sum(rhos) / len(rhos),
            "min_rho": min(rhos),
        },
    }


def plot_problem(name, pb, res_by_method, outpath, linestyle="segments"):
    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    tf = sorted((c, p) for c, p, _ in pb.pareto_front())
    xs, ys = zip(*tf)
    # All three styles are honest for a discrete front:
    #   staircase  step function -- literally correct, but reads as "steps"
    #   segments   connect-the-dots -- shows the trade-off curve's shape
    #   markers    points only -- makes no claim about what lies between
    # A SMOOTH SPLINE would not be: it interpolates designs that do not exist.
    if linestyle == "staircase":
        ax.plot(xs, ys, drawstyle="steps-post", color="black", lw=1.8,
                zorder=2, label="True Pareto front")
    elif linestyle == "segments":
        ax.plot(xs, ys, "-", color="black", lw=1.5, alpha=0.8,
                zorder=2, label="True Pareto front")
    else:
        ax.plot([], [], "ko", ms=4, label="True Pareto front")
    ax.plot(xs, ys, "ko", ms=4, zorder=3)
    nc = pb.nonconvex_pareto_points()
    if nc:
        ax.scatter(*zip(*nc), s=150, facecolors="none",
                   edgecolors="crimson", lw=1.4, zorder=4,
                   label="non-convex points")
    # Layering fixes marker overlap: Pauli large open, ESD smaller filled.
    if "pauli" in res_by_method:
        pts = res_by_method["pauli"]["found"]
        if pts:
            ax.scatter(*zip(*sorted(pts)), marker="^", s=110,
                       facecolors="none", edgecolors="olivedrab", lw=1.6,
                       zorder=5, label="Pauli QAOA")
    if "esd" in res_by_method:
        pts = res_by_method["esd"]["found"]
        if pts:
            ax.scatter(*zip(*sorted(pts)), marker="D", s=32, c="maroon",
                       zorder=6, label="ESD-QAOA")
    ax.set_xlabel("Total Cost")
    ax.set_ylabel("Total Performance")
    ax.set_title(f"Problem {name} (N={pb.N}, {pb.n_qubits} qubits)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(ls="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def pareto_tex(pb, res_by_method):
    lines = ["%% True Pareto front (staircase)",
             "\\addplot[black, very thick, const plot, mark=*, "
             "mark size=2.2]"]
    tf = sorted((c, p) for c, p, _ in pb.pareto_front())
    lines.append("    coordinates {"
                 + " ".join(f"({c},{p})" for c, p in tf) + "};")
    lines.append("\\addlegendentry{True Pareto}")
    style = {
        "pauli": ("OliveGreen, only marks, mark=triangle, "
                  "mark size=4.2, thick", "Pauli QAOA"),
        "esd": ("Maroon, only marks, mark=diamond*, mark size=2.6",
                "ESD-QAOA"),
    }
    for method in ("pauli", "esd"):  # open markers first, filled on top
        if method not in res_by_method:
            continue
        pts = sorted(res_by_method[method]["found"])
        st, label = style[method]
        lines.append(f"\\addplot[{st}]")
        lines.append("    coordinates {"
                     + " ".join(f"({c},{p})" for c, p in pts) + "};")
        lines.append(f"\\addlegendentry{{{label}}}")
    return "\n".join(lines) + "\n"


def metrics_tex(rows):
    header = (
        "\\begin{tabular}{clccccccc}\n\\toprule\n"
        "Problem & Method & ADPF$\\downarrow$ & PCR$\\uparrow$ & "
        "HV\\%$\\uparrow$ & NC pts & $\\bar{M}_{\\rm feas}\\uparrow$ & "
        "$\\bar{M}_{\\rm Pareto}\\uparrow$ & $\\bar\\rho\\uparrow$ \\\\\n"
        "\\midrule\n")
    body = []
    for name, method, m in rows:
        label = "ESD-QAOA" if method == "esd" else "Pauli QAOA"
        body.append(
            f"{name} & {label} & {m['ADPF']:.3f} & {100*m['PCR']:.0f}\\% & "
            f"{m['HV_pct']:.1f}\\% & {m['nonconvex']} & "
            f"{m['feasible_mass']:.3f} & {m['pareto_mass']:.3f} & "
            f"{m['mean_rho']:.3f} \\\\")
    return header + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_json")
    ap.add_argument("--tau", type=float, default=0.02,
                    help="harvest probability threshold (default 0.02)")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--linestyle", default="segments",
                    choices=["staircase", "segments", "markers"],
                    help="true-front rendering. Default 'segments' "
                         "(connect-the-dots) shows the diminishing-returns "
                         "shape; 'staircase' is literally correct for a "
                         "discrete front but reads as steps; 'markers' claims "
                         "nothing between points. All three are honest. A "
                         "smooth spline is NOT offered: it would interpolate "
                         "designs that do not exist.")
    args = ap.parse_args()

    with open(args.sweep_json) as f:
        data = json.load(f)
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    hdr = (f"{'Prob':<5}{'Method':<8}{'ADPF':>7}{'PCR':>7}{'HV%':>8}"
           f"{'NCpts':>7}{'Mfeas':>7}{'MPar':>7}{'mean_rho':>9}")
    print(f"\nHarvested-front reanalysis (tau = {args.tau})\n{hdr}")
    print("-" * len(hdr))
    for name, rec in data["runs"].items():
        pb = PROBLEMS[name]
        res = {}
        for method, mrec in rec["methods"].items():
            r = analyze_method(pb, mrec, args.tau)
            res[method] = r
            m = r["metrics"]
            rows.append((name, method, m))
            print(f"{name:<5}{method:<8}{m['ADPF']:>7.3f}"
                  f"{100*m['PCR']:>6.0f}%{m['HV_pct']:>7.1f}%"
                  f"{m['nonconvex']:>7}{m['feasible_mass']:>7.3f}"
                  f"{m['pareto_mass']:>7.3f}{m['mean_rho']:>9.3f}")
        plot_problem(name, pb, res,
                     os.path.join(args.outdir, f"pareto_{name}.png"),
                     linestyle=args.linestyle)
        with open(os.path.join(args.outdir, f"pareto_{name}.tex"), "w") as f:
            f.write(pareto_tex(pb, res))
    with open(os.path.join(args.outdir, "metrics_harvest.tex"), "w") as f:
        f.write(metrics_tex(rows))
    print(f"\nWrote figures + LaTeX to {args.outdir}/")


if __name__ == "__main__":
    main()
