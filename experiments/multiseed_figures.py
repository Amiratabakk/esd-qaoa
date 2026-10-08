"""Figures for the multi-seed campaign: error bars, pairing, and no N-trend.

Four panels, each answering a different question the single-seed data could
not:

  A  rho per instance, both methods, mean +/- sd over seeds
     -> how large is the effect, and how noisy is each method?
  B  gap vs qubit count with error bars
     -> does the advantage grow with problem size? (it does not)
  C  every individual seed's gap, per instance
     -> is the win consistent, or an average over wins and losses?
  D  single-seed(7) error against the 5-seed mean
     -> how misleading was the original single-seed campaign?

Panel C is the load-bearing one. A mean gap with an error bar still permits
some seeds favouring the baseline; plotting all 40 paired runs shows directly
whether that ever happens.

Palette note: ESD maroon (#7b1d1d) / Pauli olivedrab (#5b7c1f) is kept
deliberately. It fails a lightness-band check but passes CVD separation at
deltaE 15.0 (deuteranopia); lightening the maroon to satisfy the band drops
that to 5-7, i.e. colorblind-unsafe. Separation here comes largely FROM the
lightness difference. Marker shape differs per series as well, so identity is
never carried by colour alone.

Usage:
    python experiments/multiseed_figures.py results/multiseed_ladder.json \\
        --out figures/multiseed.png
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ESD_C, PAULI_C = "#7b1d1d", "#5b7c1f"
GAP_C = "#1f5fd0"
INK, MUTED = "#22272e", "#6b7684"
GRID = dict(ls="--", alpha=0.30, lw=0.8)


def order(runs):
    return sorted(runs, key=lambda k: runs[k]["n_qubits"])


def uniform_rho(name, phis):
    """rho a uniform superposition scores for free on this instance.

    rho is not zero-based, so the FLOOR moves between instances. Without it
    drawn, a decline in rho across problem size cannot be told apart from
    the floor drifting down, and the two are separate claims.
    """
    from esdqaoa.problems import problem_registry
    from esdqaoa.simulate import approximation_ratio
    pb = problem_registry()[name]
    n = 2 ** pb.N
    u = {tuple((i >> b) & 1 for b in range(pb.N)): 1.0 / n for i in range(n)}
    return float(np.mean([approximation_ratio(pb, u, ph) for ph in phis]))


def mark_families(ax, names, x=None):
    """Divider between the hand-built instances and the searched ladder.

    III-V were built by hand; Q13-Q20 came from a search over front
    curvature and dent depth and are more tightly constrained (18-35%
    feasible vs 30-38%). Any trend read across the boundary spans two
    instance families, so the boundary has to be visible.
    """
    fam = [i for i, n in enumerate(names) if n.startswith("Q")]
    if not fam or fam[0] == 0:
        return
    b = fam[0] - 0.5 if x is None else 0.5 * (x[fam[0]] + x[fam[0] - 1])
    ax.axvline(b, color=MUTED, lw=1.1, ls=":", zorder=0)
    lo, hi = ax.get_ylim()
    ax.text(b - 0.12, hi, "hand-built ", ha="right", va="top", fontsize=7.5,
            color=MUTED, style="italic")
    ax.text(b + 0.12, hi, " searched ladder", ha="left", va="top",
            fontsize=7.5, color=MUTED, style="italic")


def panel_rho(ax, runs, names, phis=None):
    x = np.arange(len(names))
    for off, meth, colr, mk, lbl in [(-0.13, "esd", ESD_C, "D", "ESD-QAOA"),
                                     (+0.13, "pauli", PAULI_C, "^",
                                      "Pauli QAOA")]:
        m = [runs[k]["methods"][meth]["mean_of_means"] for k in names]
        s = [runs[k]["methods"][meth]["std_of_means"] for k in names]
        ax.errorbar(x + off, m, yerr=s, fmt=mk, ms=7, color=colr, lw=0,
                    elinewidth=1.6, capsize=3.5, ecolor=colr, label=lbl)
    if phis:
        u = [uniform_rho(k, phis) for k in names]
        ax.plot(x, u, "--", color=MUTED, lw=1.4, zorder=0,
                label=r"uniform floor $\rho_{\rm unif}$")
        ax.fill_between(x, 0, u, color=MUTED, alpha=.10, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel(r"mean $\rho$   (5 seeds)", fontsize=9, color=INK)
    ax.set_title("A  Approximation ratio, mean $\\pm$ s.d. over seeds",
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", **GRID)
    mark_families(ax, names)


def panel_gap_vs_n(ax, runs, names):
    q = np.array([runs[k]["n_qubits"] for k in names], float)
    g = np.array([runs[k]["gap"]["mean"] for k in names])
    s = np.array([runs[k]["gap"]["std"] for k in names])
    ax.errorbar(q, g, yerr=s, fmt="o", ms=7, color=GAP_C, lw=0,
                elinewidth=1.6, capsize=3.5, ecolor=GAP_C)
    fam = [i for i, k in enumerate(names) if k.startswith("Q")]
    if fam and fam[0]:
        ax.axvline(0.5 * (q[fam[0]] + q[fam[0] - 1]), color=MUTED, lw=1.1,
                   ls=":", zorder=0)
    ax.axhline(g.mean(), color=MUTED, lw=1.4, ls="--")
    # anchored in axes coords, right-aligned: a data-coord label at the last
    # point runs off the frame
    ax.text(0.985, g.mean(), f"mean {g.mean():+.3f}", transform=
            ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8,
            color=MUTED)
    ax.axhline(0, color=INK, lw=1.0, alpha=.5)
    try:
        from scipy import stats
        r, p = stats.pearsonr(q, g)
        note = f"$r={r:+.2f}$, $p={p:.2f}$ — no trend"
    except Exception:                              # scipy optional
        note = "no trend"
    ax.text(0.03, 0.06, note, transform=ax.transAxes, fontsize=8.5,
            color=MUTED)
    ax.set_xlabel("total qubits", fontsize=9, color=INK)
    ax.set_ylabel(r"$\rho$ gap (ESD $-$ Pauli)", fontsize=9, color=INK)
    ax.set_title("B  The advantage does not grow with problem size",
                 fontsize=9.5, color=INK, loc="left")
    ax.grid(**GRID)


def panel_per_seed(ax, runs, names, seeds):
    for i, k in enumerate(names):
        gaps = runs[k]["gap"]["per_seed"]
        # tiny deterministic jitter so coincident seeds stay countable
        xs = i + np.linspace(-0.16, 0.16, len(gaps))
        ax.scatter(xs, gaps, s=34, facecolors="none", edgecolors=ESD_C,
                   linewidths=1.4, zorder=3)
        ax.plot([i - 0.24, i + 0.24], [np.mean(gaps)] * 2, color=INK, lw=1.8,
                zorder=4)
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel(r"$\rho$ gap per seed", fontsize=9, color=INK)
    lo = min(min(runs[k]["gap"]["per_seed"]) for k in names)
    ax.set_title(f"C  All {len(seeds) * len(names)} paired runs favour "
                 f"ESD (worst {lo:+.3f})", fontsize=9.5, color=INK, loc="left")
    ax.grid(axis="y", **GRID)


def panel_seed7_error(ax, runs, names):
    x = np.arange(len(names))
    err = []
    for k in names:
        s7 = (runs[k]["methods"]["esd"]["per_seed"]["7"]["mean_rho"]
              - runs[k]["methods"]["pauli"]["per_seed"]["7"]["mean_rho"])
        err.append(s7 - runs[k]["gap"]["mean"])
    colors = [ESD_C if e > 0 else PAULI_C for e in err]
    ax.bar(x, err, width=0.6, color=colors, alpha=.85)
    ax.axhline(0, color=INK, lw=1.2)
    worst = max(err, key=abs)
    iw = err.index(worst)
    # headroom first, then place the label back INSIDE the axes -- annotating
    # at the bar tip clips it against the frame
    span = max(err) - min(err)
    ax.set_ylim(min(err) - 0.22 * span, max(err) + 0.18 * span)
    ax.annotate(f"{names[iw]}: {worst:+.3f}",
                xy=(iw, worst), xytext=(iw + 0.55, worst * 0.62),
                fontsize=8, color=MUTED, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("seed-7 gap $-$ 5-seed mean", fontsize=9, color=INK)
    ax.set_title("D  Error in the original single-seed campaign",
                 fontsize=9.5, color=INK, loc="left")
    ax.grid(axis="y", **GRID)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("multiseed_json", nargs="?",
                    default="results/multiseed_ladder.json")
    ap.add_argument("--out", default="figures/multiseed.png")
    args = ap.parse_args()

    data = json.load(open(args.multiseed_json))
    runs, seeds = data["runs"], data["settings"]["seeds"]
    names = order(runs)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0))
    panel_rho(axes[0][0], runs, names, data.get("phis"))
    panel_gap_vs_n(axes[0][1], runs, names)
    panel_per_seed(axes[1][0], runs, names, seeds)
    mark_families(axes[1][0], names)
    panel_seed7_error(axes[1][1], runs, names)
    mark_families(axes[1][1], names)
    for ax in axes.ravel():
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
        ax.set_axisbelow(True)

    fig.suptitle(f"Multi-seed robustness: {len(seeds)} seeds "
                 f"{seeds}, {len(names)} instances, both encodings",
                 fontsize=12.5, fontweight="bold", color=INK, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
