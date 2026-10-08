"""Expected-front overlay figures (per problem), matching the earlier draft.

For each phi in the sweep, plots the probability-weighted expected
(cost, performance) of the FEASIBLE distribution. ESD-QAOA concentrates on
good feasible designs so its expectation hugs the true front; Pauli QAOA
wastes mass on sub-optimal/infeasible designs so its expectation sags below
the front. This is the visual analogue of the approximation-ratio gap.

IMPORTANT: an expected point is a convex mixture of designs and is NOT an
attainable configuration. These figures show an EXPECTATION, not a recovered
Pareto point; the caption must say so. Use alongside (not instead of) the
coverage scatter and rho-vs-phi figures.

Usage:  python experiments/expected_front_figures.py results/sweep.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from esdqaoa.problems import problem_registry

# includes the generated Q13..Q20 ladder when
# results/instances.json is present
PROBLEMS = problem_registry()

ESD_C, PAULI_C = "maroon", "olivedrab"
ORDER = ["I", "II", "III", "IV", "V"]


def parse(mrec):
    out = []
    for s in mrec["sweep"]:
        probs = {tuple(int(b) for b in k): v for k, v in s["probs"].items()}
        out.append((s["phi"], probs))
    return out


def expected_point(pb, probs):
    c = p = m = 0.0
    for x, pr in probs.items():
        xa = np.array(x)
        if pb.is_feasible(xa):
            c += pr * pb.cost(xa); p += pr * pb.perf(xa); m += pr
    return (c / m, p / m) if m else None


def plot_one(name, pb, data, outdir):
    tf = sorted((c, p) for c, p, _ in pb.pareto_front())
    xs, ys = zip(*tf)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    # connect-the-dots true front (consistent with coverage figures)
    ax.plot(xs, ys, "-", color="black", lw=1.8, zorder=2,
            label="True Pareto front")
    ax.plot(xs, ys, "ko", ms=4, zorder=3)
    for method, color, mk, lbl in [("pauli", PAULI_C, "^", "Pauli QAOA"),
                                   ("esd", ESD_C, "D", "ESD-QAOA")]:
        if method not in data["runs"][name]["methods"]:
            continue
        sw = parse(data["runs"][name]["methods"][method])
        pts = [expected_point(pb, pr) for _, pr in sw]
        pts = [q for q in pts if q]
        if pts:
            ax.scatter(*zip(*pts), marker=mk, s=42, c=color, alpha=0.85,
                       edgecolors="k", linewidths=0.3, zorder=5, label=lbl)
    ax.set_xlabel("Total Cost"); ax.set_ylabel("Total Performance")
    ax.set_title(f"Problem {name} (N={pb.N}, {pb.n_qubits} qubits)",
                 fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(ls="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"expected_front_{name}.png"), dpi=170)
    plt.close(fig)


def tex_snippet(name, pb, data):
    tf = sorted((c, p) for c, p, _ in pb.pareto_front())
    lines = ["%% True Pareto front",
             "\\addplot[black, very thick, mark=*, mark size=2.2]",
             "    coordinates {" + " ".join(f"({c},{p})" for c, p in tf)
             + "};", "\\addlegendentry{True Pareto}"]
    sty = {"pauli": ("OliveGreen, only marks, mark=triangle*, mark size=2.6",
                     "Pauli QAOA"),
           "esd": ("Maroon, only marks, mark=diamond*, mark size=2.6",
                   "ESD-QAOA")}
    for method in ("pauli", "esd"):
        sw = parse(data["runs"][name]["methods"][method])
        pts = [expected_point(pb, pr) for _, pr in sw]
        pts = [q for q in pts if q]
        st, lbl = sty[method]
        lines.append(f"\\addplot[{st}]")
        lines.append("    coordinates {"
                     + " ".join(f"({c:.3f},{p:.3f})" for c, p in pts) + "};")
        lines.append(f"\\addlegendentry{{{lbl}}}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_json")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    data = json.load(open(args.sweep_json))
    os.makedirs(args.outdir, exist_ok=True)
    for name in [n for n in ORDER if n in data["runs"]]:
        pb = PROBLEMS[name]
        plot_one(name, pb, data, args.outdir)
        with open(os.path.join(args.outdir,
                               f"expected_front_{name}.tex"), "w") as f:
            f.write(tex_snippet(name, pb, data))
    print(f"Wrote expected-front figures to {args.outdir}/")


if __name__ == "__main__":
    main()
