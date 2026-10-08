"""Per-instance hardware figures, plus a ladder summary.

Writes figures/hardware/<INST>.png for every instance in the run, and
figures/hardware/ladder_summary.png.

Each per-instance figure answers four questions:

  A  rho vs phi -- where does the device land between the uniform FLOOR and
     the noiseless ceiling? Plotting hardware alone is misleading: rho is not
     zero-based, so a decohered device still returns ~rho_uniform.
  B  is the margin over uniform real? Bars are the measured margin, the band
     is +/-3 bootstrap sigma over the shot noise. A bar inside the band is
     NOISE regardless of how large the "% retained" figure looks.
  C  how the signal is lost -- sorted output probabilities. The ideal
     distribution concentrates; a decohered one is flat at 1/2^N.
  D  feasible mass, which is what the constraint dephasing is supposed to
     buy, against the same two references.

WHY MARGIN AND NOT "% RETAINED": percent-retained divides by
(rho_ideal - rho_uniform). When that span is small, a trivial absolute
margin looks respectable -- Q18 reads "3.6% retained" off a margin of
+0.0031, which is pure shot noise. Absolute margin against a bootstrap
error bar is the honest test, and it is what these figures plot.

Usage:
    python experiments/hardware_instance_figures.py results/hardware_ladder.json
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

from esdqaoa.problems import problem_registry
from esdqaoa.simulate import default_alpha, default_beta_dep
from esdqaoa.fastsim import FastESDRunner

IDEAL_C, HW_C = "#7b1d1d", "#1f5fd0"
SIG_C, NOISE_C = "#1f5fd0", "#9aa5b1"
INK, MUTED = "#22272e", "#6b7684"
GRID = dict(ls="--", alpha=0.30, lw=0.8)
N_BOOT = 400


def value_table(pb, phi, keys):
    """Clamped scalarized value per observed design, plus the feasible range."""
    feas = [pb.scalarized(x, phi) for x in pb.feasible_configs()]
    f_max, f_min = max(feas), min(feas)
    vals = np.array([pb.scalarized(np.array(k), phi)
                     if pb.is_feasible(np.array(k)) else f_min for k in keys])
    return np.maximum(vals, f_min), f_min, f_max


def rho_bootstrap(pb, phi, counts, shots, rng):
    """(rho, bootstrap sd) -- vectorised, so 400 resamples stays cheap."""
    keys = [tuple(int(b) for b in k) for k in counts]
    p = np.array(list(counts.values()), dtype=float)
    p = p / p.sum()
    vals, f_min, f_max = value_table(pb, phi, keys)
    span = max(f_max - f_min, 1e-12)
    rho = float((p @ vals - f_min) / span)
    draws = rng.multinomial(shots, p, size=N_BOOT) / shots
    boots = (draws @ vals - f_min) / span
    return rho, float(np.std(boots))


def ideal_sorted(pb, p, phi, angles):
    r = FastESDRunner(pb, p, phi, default_alpha(pb), default_beta_dep(pb))
    return np.sort(r.probabilities(angles))[::-1]


def make_instance_fig(name, pb, rec, shots, outdir, rng):
    sweep = rec["sweep"]
    phis = [s["phi"] for s in sweep]
    stats = [rho_bootstrap(pb, s["phi"], s["counts"], shots, rng)
             for s in sweep]
    margins = [hw - s["uniform"]["rho"] for (hw, _), s in zip(stats, sweep)]
    sds = [sd for _, sd in stats]
    n_sig = sum(1 for m, sd in zip(margins, sds) if sd > 0 and m / sd > 3)

    fig, ax = plt.subplots(2, 2, figsize=(11.4, 7.4))

    # A -- rho vs phi against both references
    a = ax[0][0]
    a.plot(phis, [s["ideal"]["rho"] for s in sweep], "-D", color=IDEAL_C,
           ms=6, lw=1.8, label="noiseless simulation")
    a.plot(phis, [hw for hw, _ in stats], "-o", color=HW_C, ms=7, lw=1.8,
           label="hardware")
    a.plot(phis, [s["uniform"]["rho"] for s in sweep], "--", color=MUTED,
           lw=1.5, label="uniform (floor)")
    a.set_ylim(0, 1.0)
    a.set_xlabel(r"$\varphi$", fontsize=9, color=INK)
    a.set_ylabel(r"$\rho$", fontsize=9, color=INK)
    a.set_title("A  Device vs noiseless vs floor", fontsize=9.5, color=INK,
                loc="left")
    a.legend(fontsize=7.5, frameon=False)
    a.grid(**GRID)

    # B -- margin over uniform against a +/-3 sigma noise band
    b = ax[0][1]
    x = np.arange(len(phis))
    band = 3 * np.array(sds)
    cols = [SIG_C if m > 3 * sd else NOISE_C for m, sd in zip(margins, sds)]
    b.bar(x, margins, 0.55, color=cols)
    b.fill_between([-0.5, len(phis) - 0.5], -band.max(), band.max(),
                   color=MUTED, alpha=.18, zorder=0)
    b.axhline(0, color=INK, lw=1.1)
    b.set_xlim(-0.5, len(phis) - 0.5)
    b.set_xticks(x)
    b.set_xticklabels([f"{p:.2f}" for p in phis], fontsize=8)
    b.set_xlabel(r"$\varphi$", fontsize=9, color=INK)
    b.set_ylabel(r"$\rho_{\rm hw}-\rho_{\rm uniform}$", fontsize=9, color=INK)
    b.set_title(f"B  Margin over uniform — {n_sig}/{len(phis)} significant "
                f"(shaded = $\\pm3\\sigma$ shot noise)",
                fontsize=9.5, color=INK, loc="left")
    b.grid(axis="y", **GRID)

    # C -- where the concentration went
    c = ax[1][0]
    mid = sweep[len(sweep) // 2]
    hw_sorted = np.array(sorted(mid["counts"].values(), reverse=True))
    idl = ideal_sorted(pb, rec.get("p", 1), mid["phi"], mid["angles"])
    k = min(40, len(idl))
    c.plot(range(1, k + 1), idl[:k], "-D", color=IDEAL_C, ms=4, lw=1.6,
           label="noiseless")
    c.plot(range(1, min(k, len(hw_sorted)) + 1), hw_sorted[:k], "-o",
           color=HW_C, ms=4, lw=1.6, label="hardware")
    c.axhline(1 / 2 ** pb.N, color=MUTED, ls="--", lw=1.4,
              label=f"uniform $1/2^{{{pb.N}}}$")
    c.set_yscale("log")
    c.set_xlabel("design, ranked by probability", fontsize=9, color=INK)
    c.set_ylabel("probability", fontsize=9, color=INK)
    c.set_title(f"C  Output concentration ($\\varphi$={mid['phi']:.2f})",
                fontsize=9.5, color=INK, loc="left")
    c.legend(fontsize=7.5, frameon=False)
    c.grid(**GRID)

    # D -- feasible mass, what the constraint dephasing is meant to buy
    d = ax[1][1]
    d.plot(phis, [s["ideal"]["feasible_mass"] for s in sweep], "-D",
           color=IDEAL_C, ms=6, lw=1.8, label="noiseless simulation")
    d.plot(phis, [s["hardware"]["feasible_mass"] for s in sweep], "-o",
           color=HW_C, ms=7, lw=1.8, label="hardware")
    frac = len(pb._tables()[0]) / 2 ** pb.N
    d.axhline(frac, color=MUTED, ls="--", lw=1.5,
              label=f"uniform ({frac:.2f} of space feasible)")
    d.set_ylim(0, 1.0)
    d.set_xlabel(r"$\varphi$", fontsize=9, color=INK)
    d.set_ylabel("feasible probability mass", fontsize=9, color=INK)
    d.set_title("D  Feasible mass", fontsize=9.5, color=INK, loc="left")
    d.legend(fontsize=7.5, frameon=False)
    d.grid(**GRID)

    for a_ in ax.ravel():
        a_.tick_params(colors=MUTED, labelsize=8)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_color(MUTED)
        a_.set_axisbelow(True)

    g = sweep[0]["transpiled"]
    verdict = ("signal above uniform" if n_sig >= 4 else
               "partial" if n_sig >= 2 else
               "INDISTINGUISHABLE FROM UNIFORM")
    fig.suptitle(f"{name} — {pb.N} subsystems, {pb.n_qubits} qubits, p="
                 f"{rec.get('p', 1)} · {g['two_qubit_gates']:,} 2q gates, "
                 f"depth {g['depth']:,} · {verdict}",
                 fontsize=12, fontweight="bold", color=INK, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    path = os.path.join(outdir, f"{name}.png")
    fig.savefig(path, dpi=155)
    plt.close(fig)
    return {"name": name, "gates": g["two_qubit_gates"],
            "depth": g["depth"], "qubits": pb.n_qubits,
            "margins": margins, "sds": sds, "n_sig": n_sig,
            "mean_margin": float(np.mean(margins))}


def make_summary_fig(rows, outdir, shots):
    rows = sorted(rows, key=lambda r: r["gates"])
    g = [r["gates"] for r in rows]
    m = [r["mean_margin"] for r in rows]
    e = [float(np.mean(r["sds"])) for r in rows]
    cols = [SIG_C if r["n_sig"] >= 4 else NOISE_C for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.4))

    a = ax[0]
    a.errorbar(g, m, yerr=[3 * v for v in e], fmt="o", ms=8, lw=0,
               elinewidth=1.6, capsize=4, ecolor=MUTED, zorder=2)
    for gi, mi, ci, r in zip(g, m, cols, rows):
        a.plot([gi], [mi], "o", ms=8, color=ci, zorder=3)
        a.annotate(r["name"], (gi, mi), textcoords="offset points",
                   xytext=(0, 11), ha="center", fontsize=7.5, color=MUTED)
    a.axhline(0, color=INK, lw=1.2)
    a.set_xlabel("two-qubit gates (transpiled to device)", fontsize=9,
                 color=INK)
    a.set_ylabel(r"mean $\rho_{\rm hw}-\rho_{\rm uniform}$", fontsize=9,
                 color=INK)
    a.set_title("The depth wall, measured — error bars are $\\pm3\\sigma$ "
                "shot noise", fontsize=9.5, color=INK, loc="left")
    a.grid(**GRID)

    b = ax[1]
    x = np.arange(len(rows))
    b.bar(x, [r["n_sig"] for r in rows], 0.55, color=cols)
    b.axhline(4, color=INK, ls="--", lw=1.2)
    b.text(len(rows) - 0.4, 4.06, "4/5 = signal", ha="right", fontsize=8,
           color=MUTED)
    b.set_xticks(x)
    b.set_xticklabels([r["name"] for r in rows], fontsize=8)
    b.set_ylim(0, 5.4)
    b.set_ylabel(r"$\varphi$ points significant ($>3\sigma$)", fontsize=9,
                 color=INK)
    b.set_title(f"{sum(1 for r in rows if r['n_sig'] >= 4)} of {len(rows)} "
                f"instances clear the noise floor",
                fontsize=9.5, color=INK, loc="left")
    b.grid(axis="y", **GRID)

    for a_ in ax:
        a_.tick_params(colors=MUTED, labelsize=8)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_color(MUTED)
        a_.set_axisbelow(True)

    fig.suptitle(f"ESD-QAOA on IBM hardware: signal vs circuit size "
                 f"(p=1, {shots} shots, 5 $\\varphi$ per instance)",
                 fontsize=12, fontweight="bold", color=INK, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(outdir, "ladder_summary.png")
    fig.savefig(path, dpi=155)
    plt.close(fig)
    return path


def make_hw_vs_sim_fig(rows, data, registry, outdir, shots):
    """Replacement for the draft's Fig 9(b), with the floor included.

    The original figure plotted hardware rho beside simulator rho and read
    the gap as noise degradation. That is only half the story: rho is not
    zero-based, so the bar a fully decohered device produces is
    rho_uniform, not zero. Without the floor drawn, a hardware bar at 0.20
    looks like a partial result when it may be exactly what random sampling
    would give.
    """
    names = [r["name"] for r in rows]
    x = np.arange(len(names))
    hw, sim, unf = [], [], []
    for r in rows:
        sweep = data["runs"][r["name"]]["sweep"]
        hw.append(np.mean([s_["hardware"]["rho"] for s_ in sweep]))
        sim.append(np.mean([s_["ideal"]["rho"] for s_ in sweep]))
        unf.append(np.mean([s_["uniform"]["rho"] for s_ in sweep]))

    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.bar(x - 0.21, sim, 0.42, color=IDEAL_C, label="noiseless simulation")
    cols = [SIG_C if r["n_sig"] >= 4 else NOISE_C for r in rows]
    ax.bar(x + 0.21, hw, 0.42, color=cols, label="IBM hardware")
    ax.plot(x, unf, "_", color=INK, ms=26, mew=2.2, zorder=5,
            label="uniform floor (what noise alone gives)")
    for i, r in enumerate(rows):
        if r["n_sig"] >= 4:
            ax.text(i + 0.21, hw[i] + .015, "*", ha="center", fontsize=15,
                    color=SIG_C)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel(r"mean approximation ratio $\rho$", fontsize=10, color=INK)
    ax.set_ylim(0, 0.75)
    # the count is data-driven: it was 1 on the ladder alone and is 4 once
    # the smaller instances are included, so hard-coding it goes stale
    sig = [r["name"] for r in rows if r["n_sig"] >= 4]
    gates = [r["gates"] for r in rows if r["n_sig"] >= 4]
    span = (f"{', '.join(sig)} ($\\leq${max(gates):,} 2q gates)"
            if sig else "no instance")
    ax.set_title(f"Hardware vs noiseless simulation — * marks the {len(sig)} "
                 f"instance(s) whose hardware output is\nstatistically "
                 f"distinguishable from uniform ($\\geq$4/5 $\\varphi$ at "
                 f"$>3\\sigma$): {span}",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper right")
    ax.grid(axis="y", **GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(outdir, "hw_vs_sim.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)

    print(f"\n{'inst':>5} {'sim rho':>9} {'hw rho':>8} {'uniform':>9} "
          f"{'hw-unif':>9}  significant")
    for i, r in enumerate(rows):
        print(f"{r['name']:>5} {sim[i]:>9.3f} {hw[i]:>8.3f} {unf[i]:>9.3f} "
              f"{hw[i]-unf[i]:>+9.4f}  {'YES' if r['n_sig']>=4 else 'no'}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hardware_json", nargs="?",
                    default="results/hardware_ladder.json")
    ap.add_argument("--outdir", default="figures/hardware")
    args = ap.parse_args()

    data = json.load(open(args.hardware_json))
    shots = data["settings"]["shots"]
    registry = problem_registry()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(0)

    rows = []
    print(f"{'inst':>5} {'2q':>7} {'mean margin':>12} {'sig':>6}  verdict")
    print("-" * 52)
    for name, rec in data["runs"].items():
        r = make_instance_fig(name, registry[name], rec, shots, args.outdir,
                              rng)
        rows.append(r)
        v = ("signal" if r["n_sig"] >= 4 else
             "partial" if r["n_sig"] >= 2 else "noise")
        print(f"{name:>5} {r['gates']:>7,} {r['mean_margin']:>+12.4f} "
              f"{r['n_sig']:>4}/5  {v}")
    p = make_summary_fig(rows, args.outdir, shots)
    make_hw_vs_sim_fig(rows, data, registry, args.outdir, shots)
    print(f"\nwrote {len(rows)} instance figures + {os.path.basename(p)} "
          f"to {args.outdir}/")


if __name__ == "__main__":
    main()
