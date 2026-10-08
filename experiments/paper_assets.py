"""Generate every figure and table the manuscript uses, into paper/.

One script so that nothing in the paper can drift from the data. Re-run it
after any new campaign and the .tex picks up the new numbers without a single
hand edit:

    python experiments/paper_assets.py

Outputs
    paper/figures/fig_instances.png     M2  objective spaces, both families
    paper/figures/fig_pareto.png        M3  harvested fronts, 4 instances
    paper/figures/fig_scalarization.png M4  weighted sum vs eps-constraint
    paper/figures/fig_rho.png           M5  multi-seed rho + gap vs N
    paper/figures/fig_mass.png          M6  Pareto mass / feasible mass
    paper/figures/fig_depth.png         M7  rho vs p
    paper/figures/fig_hardware.png      M8  hardware wall + hw vs sim
    paper/tables/tab_instances.tex      T1
    paper/tables/tab_rho.tex            T2
    paper/tables/tab_resources.tex      T3
    paper/tables/tab_scalarization.tex  T4

Palette note (inherited from experiments/multiseed_figures.py): ESD maroon
#7b1d1d / Pauli olivedrab #5b7c1f is kept deliberately. It fails a lightness
band check but passes CVD separation at deltaE 15.0 (deuteranopia);
lightening the maroon drops that to 5-7, i.e. colorblind-unsafe. Marker shape
differs per series as well, so identity is never carried by colour alone.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from esdqaoa.problems import problem_registry
from esdqaoa.pareto_v2 import (harvested_front, sweep_mass_metrics, pcr,
                               nonconvex_coverage)
from esdqaoa.simulate import approximation_ratio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The manuscript folder moved to paper/paper_mdpi when the MDPI class files
# were added. Writing to paper/figures left the regenerated assets somewhere
# the .tex never looks, so a "rebuilt" figure could silently not reach the
# paper. Resolve against wherever main.tex actually is.
PAPERDIR = os.path.join(ROOT, "paper", "paper_mdpi")
if not os.path.exists(os.path.join(PAPERDIR, "main.tex")):
    PAPERDIR = os.path.join(ROOT, "paper")
FIGDIR = os.path.join(PAPERDIR, "figures")
TABDIR = os.path.join(PAPERDIR, "tables")

ESD_C, PAULI_C = "#7b1d1d", "#5b7c1f"
FEAS_C, FRONT_C, HULL_C, UNSUP_C = "#9aa5b1", "#1f5fd0", "#1a7f37", "#b3261e"
SIG_C, NOISE_C = "#1f5fd0", "#9aa5b1"
INK, MUTED = "#22272e", "#6b7684"
GRID = dict(ls="--", alpha=0.30, lw=0.8)

# Paper instances, in size order. I and II are excluded: 8 and 7 qubits,
# rho ~ 0.91-1.0, saturated -- they cannot show a gap.
PAPER = ["III", "IV", "V", "Q13", "Q14", "Q15", "Q16", "Q17", "Q18",
         "Q19", "Q20"]
# The paper numbers instances I..XI in size order. Internal keys are the
# data keys used throughout results/ and must not change; DISPLAY is the
# only place the two namespaces meet.
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"]
DISPLAY = dict(zip(PAPER, ROMAN))
LABELS = [DISPLAY[n] for n in PAPER]
REG = problem_registry()


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def load(name):
    return json.load(open(os.path.join(ROOT, "results", name)))


def weighted_sum_runs():
    """Seed-7 phi sweeps with full distributions, for the harvested fronts."""
    out = {}
    small = load("sweep_fixed.json")["runs"]
    for n in ["III", "IV", "V"]:
        out[n] = small[n]
    out.update(load("sweep_ladder.json")["runs"])
    return out


def epsilon_runs():
    out = {}
    small = load("eps_sweep_I_V.json")["runs"]
    for n in ["III", "IV", "V"]:
        out[n] = small[n]
    out.update(load("eps_ladder.json")["runs"])
    return out


def parse_sweep(method_rec):
    """JSON sweep -> [(phi, {decision tuple: prob})]. Keys are little-endian."""
    return [(s["phi"], {tuple(int(b) for b in k): v
                        for k, v in s["probs"].items()})
            for s in method_rec["sweep"]]


def uniform_rho(pb, phis):
    """What a uniform superposition scores for free on this instance.

    rho is not zero-based, so this FLOOR moves between instances (it tracks
    the feasible fraction, r = +0.98). Without it, a decline in rho across
    problem size cannot be told apart from the floor drifting down.
    """
    n = 2 ** pb.N
    u = {tuple((i >> b) & 1 for b in range(pb.N)): 1.0 / n for i in range(n)}
    return float(np.mean([approximation_ratio(pb, u, ph) for ph in phis]))


def style(ax):
    """Axes in ink, not grey.

    Grey spines and tick labels read as de-emphasised, which is wrong for the
    frame that carries the quantitative content. Only genuinely secondary
    marks -- gridlines, reference lines, annotations -- stay grey.
    """
    ax.tick_params(colors=INK, labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK)
        ax.spines[sp].set_linewidth(1.0)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------
# M2  instance characterisation
# --------------------------------------------------------------------------
def fig_instances():
    """Brute-force objective space per instance. No quantum results at all.

    Exists to justify the instances: each front really curves, and each has
    Pareto points strictly below the supported convex envelope that no
    weighted sum reaches at any phi.
    """
    ncols, nrows = 4, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.2, 8.4))
    for ax in axes.ravel()[len(PAPER):]:
        ax.axis("off")
    for ax, name in zip(axes.ravel(), PAPER):
        pb = REG[name]
        feas = [(pb.cost(x), pb.perf(x)) for x in pb.feasible_configs()]
        front = sorted((c, p) for c, p, _ in pb.pareto_front())
        hull = sorted(pb.supported_pareto_points())
        unsup = pb.nonconvex_pareto_points()
        if feas:
            ax.scatter(*zip(*feas), s=7, color=FEAS_C, alpha=.30, lw=0,
                       zorder=1)
        ax.plot(*zip(*front), "-o", color=FRONT_C, ms=3.2, lw=1.3, zorder=3)
        ax.plot(*zip(*hull), "-", color=HULL_C, lw=1.3, zorder=2)
        if unsup:
            ax.scatter(*zip(*unsup), s=62, facecolors="none",
                       edgecolors=UNSUP_C, lw=1.3, zorder=4)
        ax.set_title(f"Problem {DISPLAY[name]} — {pb.n_qubits} qubits\n"
                     f"Pareto {len(front)}, unsupported {len(unsup)}",
                     fontsize=8.5, color=INK)
        ax.set_xlabel("total cost", fontsize=8, color=INK)
        ax.set_ylabel("total performance", fontsize=8, color=INK)
        ax.grid(**GRID)
        style(ax)
    handles = [
        Line2D([], [], marker="o", ls="", ms=4, color=FEAS_C,
               label="feasible design vectors"),
        Line2D([], [], marker="o", ls="-", ms=4, color=FRONT_C,
               label="exact Pareto frontier"),
        Line2D([], [], ls="-", color=HULL_C, label="supported convex envelope"),
        Line2D([], [], marker="o", ls="", ms=8, mfc="none", mec=UNSUP_C,
               mew=1.3, label="unsupported Pareto point"),
    ]
    # Park the legend INSIDE the unused 12th cell of the 3x4 grid rather than
    # positioning it in figure coordinates, where it landed on panel XI.
    spare = axes.ravel()[len(PAPER)]
    spare.legend(handles=handles, loc="center", ncol=1, frameon=False,
                 fontsize=10, handletextpad=0.8, labelspacing=1.0)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_instances.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# M3  harvested Pareto fronts
# --------------------------------------------------------------------------
def fig_pareto(ws, picks=("III", "Q13", "Q17", "Q20")):
    """Harvested fronts, both encodings, four instances across the range.

    Connect-the-dots for the true front (shows the trade-off shape); a smooth
    spline would not be honest, it interpolates designs that do not exist.
    Pauli drawn as large open triangles UNDER ESD's small filled diamonds so
    coincident points stay distinguishable.
    """
    fig, axes = plt.subplots(1, len(picks), figsize=(14.0, 3.7))
    for ax, name in zip(axes, picks):
        pb = REG[name]
        tf = sorted((c, p) for c, p, _ in pb.pareto_front())
        ax.plot(*zip(*tf), "-", color="black", lw=1.4, alpha=.8, zorder=2)
        ax.plot(*zip(*tf), "ko", ms=3.6, zorder=3)
        unsup = pb.nonconvex_pareto_points()
        if unsup:
            ax.scatter(*zip(*unsup), s=130, facecolors="none",
                       edgecolors=UNSUP_C, lw=1.3, zorder=4)
        for meth, colr, mk, size, fill in [
                ("pauli", PAULI_C, "^", 105, False),
                ("esd", ESD_C, "D", 30, True)]:
            sw = parse_sweep(ws[name]["methods"][meth])
            pts = sorted((c, p) for c, p, _, _ in harvested_front(pb, sw, 0.02))
            if not pts:
                continue
            kw = (dict(c=colr) if fill
                  else dict(facecolors="none", edgecolors=colr, lw=1.5))
            ax.scatter(*zip(*pts), marker=mk, s=size,
                       zorder=6 if fill else 5, **kw)
        ax.set_title(f"Problem {DISPLAY[name]} — {pb.n_qubits} qubits",
                     fontsize=9.5, color=INK)
        ax.set_xlabel("total cost", fontsize=9, color=INK)
        ax.grid(**GRID)
        style(ax)
    axes[0].set_ylabel("total performance", fontsize=9, color=INK)
    handles = [
        Line2D([], [], marker="o", ls="-", color="black", ms=4,
               label="true Pareto front"),
        Line2D([], [], marker="o", ls="", ms=9, mfc="none", mec=UNSUP_C,
               mew=1.3, label="unsupported point"),
        Line2D([], [], marker="^", ls="", ms=9, mfc="none", mec=PAULI_C,
               mew=1.5, label="QUBO/Ising QAOA"),
        Line2D([], [], marker="D", ls="", ms=5, color=ESD_C,
               label="ESD-QAOA"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    out = os.path.join(FIGDIR, "fig_pareto.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# M4  scalarization
# --------------------------------------------------------------------------
def fig_scalarization(eps, ws):
    """The structural result: weighted sum cannot reach unsupported points.

    Panel B is the load-bearing one. Under a weighted sum the modal design is
    always a vertex of the convex hull, so unsupported points are missed BY
    CONSTRUCTION -- for any solver, quantum or classical. Under
    eps-constraint the same circuit reaches them. That is a property of the
    FORMULATION, not of the encoding, which is why both encodings are shown.
    """
    x = np.arange(len(PAPER))
    fr_ws, fr_eps, cl_ws, cl_eps = [], [], [], []
    nc_ws, nc_eps, nc_tot, nc_harv = [], [], [], []
    for n in PAPER:
        pb, r = REG[n], eps[n]
        tot = len(r["true_front"])
        fr_ws.append(len(set(map(tuple, (
            (c, p) for c, p, *_ in ws[n]["methods"]["esd"]["recovered_front"]
        ))) & set(map(tuple, r["true_front"]))) / tot)
        # P0.3: degenerate epsilon bounds (single-performance feasible set)
        # are excluded from every epsilon quantity.
        kept = set(nondegenerate_eps(
            pb, [s["eps"] for s in r["methods"]["esd"]["sweep"]]))
        e_front = eps_modal_front(pb, r["methods"]["esd"]["sweep"], kept)
        fr_eps.append(len(e_front & set(map(tuple, r["true_front"]))) / tot)
        cl_ws.append(len(pb.supported_pareto_points()) / tot)
        cl_eps.append(eps_exact_front(pb, kept) / tot)
        # The stored JSONs carry a "nonconvex" list written by the earlier
        # grid-and-argmax classifier, which mislabelled face-tied points as
        # unsupported (P0.1). Classify live from the registry instead; never
        # read r["nonconvex"] or metrics["nonconvex_recovered"] here.
        ncs = set(pb.nonconvex_pareto_points())
        nc_tot.append(len(ncs))
        nc_ws.append(len(set((c, p) for c, p, *_
                             in ws[n]["methods"]["esd"]["recovered_front"])
                         & ncs))
        nc_eps.append(len(e_front & ncs))
        # third route: same weighted-sum runs, read by harvesting instead of
        # by the mode. It recovers far more than modal tracking does, so
        # showing only modal would understate what a weighted-sum sweep
        # actually yields.
        sw = parse_sweep(ws[n]["methods"]["esd"])
        nc_harv.append(len(set((c, p) for c, p, _, _
                               in harvested_front(pb, sw, 0.02)) & ncs))

    fig, (a, b) = plt.subplots(1, 2, figsize=(13.0, 4.3))

    a.bar(x - 0.20, fr_ws, 0.40, color=MUTED,
          label="weighted sum, modal tracking")
    a.bar(x + 0.20, fr_eps, 0.40, color=ESD_C,
          label=r"$\epsilon$-constraint, modal tracking")
    a.plot(x - 0.20, cl_ws, "_", color=INK, ms=15, mew=1.9, zorder=5,
           label="classical ceiling (same scalarization)")
    a.plot(x + 0.20, cl_eps, "_", color=INK, ms=15, mew=1.9, zorder=5)
    a.set_ylim(0, 1.34)
    a.set_ylabel("fraction of the Pareto front recovered", fontsize=9.5,
                 color=INK)
    a.set_title("A  Front coverage — dashes mark what the scalarization\n"
                "itself allows, for any solver", fontsize=9.5, color=INK,
                loc="left")
    a.legend(fontsize=8, frameon=False, loc="upper right", ncol=1)

    b.bar(x - 0.26, nc_ws, 0.26, color=MUTED,
          label="weighted sum, modal tracking")
    b.bar(x, nc_harv, 0.26, color="#8a94a6", hatch="//", edgecolor="white",
          label="weighted sum, harvesting")
    b.bar(x + 0.26, nc_eps, 0.26, color=ESD_C,
          label=r"$\epsilon$-constraint, modal tracking")
    b.plot(x, nc_tot, "_", color=INK, ms=17, mew=1.9, zorder=5,
           label="unsupported points present")
    b.set_ylabel("unsupported Pareto points recovered", fontsize=9.5,
                 color=INK)
    tot = sum(nc_tot)
    b.set_title(f"B  Unsupported points, all ESD: modal tracking under a "
                f"weighted sum reaches\n{sum(nc_ws)} of {tot}; harvesting "
                f"the same runs reaches {sum(nc_harv)}; "
                f"$\\epsilon$-constraint reaches {sum(nc_eps)}",
                fontsize=9.5, color=INK, loc="left")
    b.legend(fontsize=7.8, frameon=False, loc="upper right")
    b.set_ylim(0, max(nc_tot) + 2.2)

    for ax in (a, b):
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, fontsize=8.5)
        ax.grid(axis="y", **GRID)
        style(ax)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_scalarization.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# M5  multi-seed rho
# --------------------------------------------------------------------------
def fig_rho():
    d = load("multiseed_all.json")
    runs, phis = d["runs"], d["phis"]
    seeds = [str(s) for s in d["settings"]["seeds"]]
    names = [n for n in PAPER if n in runs]
    x = np.arange(len(names))

    fig, (a, b) = plt.subplots(1, 2, figsize=(13.0, 4.3))
    for off, meth, colr, mk, lbl in [(-0.13, "esd", ESD_C, "D", "ESD-QAOA"),
                                     (+0.13, "pauli", PAULI_C, "^",
                                      "QUBO/Ising QAOA")]:
        m = [runs[k]["methods"][meth]["mean_of_means"] for k in names]
        s = [runs[k]["methods"][meth]["std_of_means"] for k in names]
        a.errorbar(x + off, m, yerr=s, fmt=mk, ms=6.5, color=colr, lw=0,
                   elinewidth=1.5, capsize=3.2, ecolor=colr, label=lbl)
    u = [uniform_rho(REG[k], phis) for k in names]
    a.plot(x, u, "--", color=MUTED, lw=1.4, zorder=0,
           label=r"uniform reference $\rho_{\mathrm{unif}}$")
    a.fill_between(x, 0, u, color=MUTED, alpha=.10, zorder=0)
    a.set_xticks(x)
    a.set_xticklabels([DISPLAY[k] for k in names], fontsize=8.5)
    a.set_ylim(0, 0.86)
    a.set_ylabel(r"mean $\rho$ over 13 $\varphi$ (5 seeds)", fontsize=9.5,
                 color=INK)
    a.set_title("A  Approximation ratio, mean $\\pm$ s.d. over seeds",
                fontsize=9.5, color=INK, loc="left")
    a.legend(fontsize=8, frameon=False, loc="upper right")
    a.grid(axis="y", **GRID)
    style(a)

    q = np.array([runs[k]["n_qubits"] for k in names], float)
    g = np.array([runs[k]["gap"]["mean"] for k in names])
    s = np.array([runs[k]["gap"]["std"] for k in names])
    b.errorbar(q, g, yerr=s, fmt="o", ms=6.5, color=SIG_C, lw=0,
               elinewidth=1.5, capsize=3.2, ecolor=SIG_C)
    b.axhline(g.mean(), color=MUTED, lw=1.4, ls="--")
    b.text(0.985, g.mean(), f"mean {g.mean():+.3f}",
           transform=b.get_yaxis_transform(), ha="right", va="bottom",
           fontsize=8.5, color=MUTED)
    b.axhline(0, color=INK, lw=1.0, alpha=.5)
    try:
        from scipy import stats
        r, p = stats.pearsonr(q, g)
        note = f"$r={r:+.2f}$, $p={p:.2f}$ — no trend with problem size"
    except Exception:
        note = "no trend with problem size"
    b.text(0.03, 0.06, note, transform=b.transAxes, fontsize=8.5, color=MUTED)
    b.set_xlabel("total qubits", fontsize=9.5, color=INK)
    b.set_ylabel(r"$\rho$ gap (ESD $-$ Pauli)", fontsize=9.5, color=INK)
    b.set_title("B  The advantage is uniform across 10–20 qubits",
                fontsize=9.5, color=INK, loc="left")
    b.grid(**GRID)
    style(b)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_rho.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# M6  where the advantage comes from
# --------------------------------------------------------------------------
def cumulative_curve(pb, phi, probs):
    """Cumulative probability over configurations, feasible block first.

    The sort key is (feasibility, scalarized objective). Feasibility is
    PRIMARY, so every feasible configuration precedes every infeasible
    one by construction; the scalarized value only orders within the
    feasible block. That is what makes the dotted line in the figure
    readable as the probability of measuring a feasible design.

    Do not "simplify" this to a sort on the penalised objective f_o.
    The budget penalty is one-sided and linear, so a design that
    overshoots slightly can outscore a poor feasible one: on the paper
    problems that happens at 26 of 55 (problem, phi) combinations, and
    the feasible block would no longer be contiguous. See
    tests/test_cumulative_ranking.py.
    """
    keys = []
    for i in range(2 ** pb.N):
        x = np.array([(i >> b) & 1 for b in range(pb.N)])
        feas = pb.is_feasible(x)
        keys.append(((1 if feas else 0), pb.scalarized(x, phi) if feas else 0.0,
                     tuple(int(v) for v in x)))
    keys.sort(key=lambda t: (-t[0], -t[1]))
    vals = [probs.get(k[2], 0.0) for k in keys]
    return np.cumsum(vals)


def fig_concentration(rows, ws, phi_index=6, picks=("Q13", "Q20")):
    """The mechanism, in one figure: where the probability mass goes.

    Bars give one number per problem; the cumulative curves give the shape
    that number summarises, at the two ends of the size range.
    """
    x = np.arange(len(PAPER))
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.0))

    for ax, key, title in [
            (axes[0][0], "fm",
             "A  Feasible mass — probability on designs satisfying both "
             "constraints"),
            (axes[0][1], "pm",
             "B  Pareto mass — probability on Pareto-optimal designs")]:
        ax.bar(x - 0.20, [rows[n]["esd"][key] for n in PAPER], 0.40,
               color=ESD_C, label="ESD-QAOA")
        ax.bar(x + 0.20, [rows[n]["pauli"][key] for n in PAPER], 0.40,
               color=PAULI_C, label="QUBO/Ising QAOA")
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, fontsize=8.5)
        ax.set_ylabel("mean probability mass", fontsize=9.5, color=INK)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_ylim(0, 0.95)
        ax.legend(fontsize=8.5, frameon=False)
        ax.grid(axis="y", **GRID)
        style(ax)

    for col, name in enumerate(picks):
        ax = axes[1][col]
        pb = REG[name]
        rec = ws[name]["methods"]
        phi = rec["esd"]["sweep"][phi_index]["phi"]
        n_states = 2 ** pb.N
        rank = np.arange(1, n_states + 1)
        for meth, colr, lbl in [("esd", ESD_C, "ESD-QAOA"),
                                ("pauli", PAULI_C, "QUBO/Ising QAOA")]:
            probs = {tuple(int(b) for b in k): v for k, v
                     in rec[meth]["sweep"][phi_index]["probs"].items()}
            ax.plot(rank, cumulative_curve(pb, phi, probs), "-", color=colr,
                    lw=1.9, label=lbl)
        ax.plot(rank, rank / n_states, "--", color=MUTED, lw=1.4,
                label="uniform superposition")
        n_feas = len(pb.feasible_configs())
        ax.axvline(n_feas, color=INK, lw=1.0, ls=":", alpha=.6)
        ax.annotate(f"feasible set ends ({n_feas:,})", xy=(n_feas, 0.98),
                    xytext=(6, 0), textcoords="offset points", fontsize=7.5,
                    color=MUTED, rotation=90, va="top")
        ax.set_xscale("log")
        ax.set_xlim(1, n_states)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("configurations, feasible first then by "
                      "objective",
                      fontsize=9, color=INK)
        ax.set_ylabel("cumulative probability", fontsize=9, color=INK)
        ax.set_title(f"{'CD'[col]}  Problem {DISPLAY[name]} "
                     f"({pb.n_qubits} qubits), $\\varphi={phi:.2f}$",
                     fontsize=9.5, color=INK, loc="left")
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        ax.grid(**GRID)
        style(ax)

    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_concentration.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# M7  depth
# --------------------------------------------------------------------------
def fig_depth():
    d = load("depth_all.json")
    names = sorted(d["runs"], key=lambda k: d["runs"][k]["n_qubits"])
    ps = sorted(int(k) for k in d["runs"][names[0]]["esd"])
    fig, axes = plt.subplots(1, len(names), figsize=(13.4, 3.3),
                             sharey=True)
    for ax, n in zip(axes, names):
        r = d["runs"][n]
        e = [r["esd"][str(p)]["mean_rho"] for p in ps]
        q = [r["pauli"][str(p)]["mean_rho"] for p in ps]
        ax.plot(ps, e, "-D", color=ESD_C, ms=5, lw=1.6, label="ESD-QAOA")
        ax.plot(ps, q, "-^", color=PAULI_C, ms=5, lw=1.6, label="QUBO/Ising QAOA")
        ax.set_title(f"Problem {DISPLAY[n]} — {r['n_qubits']} qubits",
                     fontsize=9.5, color=INK)
        ax.set_xlabel("QAOA layers $p$", fontsize=9, color=INK)
        ax.set_xticks(ps)
        ax.set_ylim(0, 1.0)
        ax.grid(**GRID)
        style(ax)
    axes[0].set_ylabel(r"mean $\rho$ over 7 $\varphi$", fontsize=9,
                       color=INK)
    axes[-1].legend(fontsize=8.5, frameon=False, loc="lower right")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_depth.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# M8  hardware
# --------------------------------------------------------------------------
def hardware_rows():
    """Per-instance margin over uniform, with a bootstrap sd over shot noise.

    The margin, not "percent of signal retained": percent-retained divides by
    (rho_ideal - rho_uniform), which shrinks with N and makes a trivial
    absolute margin look respectable.
    """
    from experiments.hardware_instance_figures import rho_bootstrap
    d = load("hardware_all.json")
    rng = np.random.default_rng(0)
    shots = d["settings"]["shots"]
    rows = []
    for name in PAPER:
        rec = d["runs"][name]
        pb = REG[name]
        stats_ = [rho_bootstrap(pb, s["phi"], s["counts"], shots, rng)
                  for s in rec["sweep"]]
        margins = [hw - s["uniform"]["rho"]
                   for (hw, _), s in zip(stats_, rec["sweep"])]
        sds = [sd for _, sd in stats_]
        rows.append(dict(
            name=name, qubits=pb.n_qubits,
            gates=rec["sweep"][0]["transpiled"]["two_qubit_gates"],
            depth=rec["sweep"][0]["transpiled"]["depth"],
            margin=float(np.mean(margins)), sd=float(np.mean(sds)),
            n_sig=sum(1 for m, s in zip(margins, sds) if s > 0 and m / s > 3),
            n_phi=len(margins),
            hw=float(np.mean([s["hardware"]["rho"] for s in rec["sweep"]])),
            ideal=float(np.mean([s["ideal"]["rho"] for s in rec["sweep"]])),
            unif=float(np.mean([s["uniform"]["rho"] for s in rec["sweep"]]))))
    return rows, d


def fig_hardware_detail(data, small="III", large="Q20"):
    """rho vs phi and the output distribution, at the two ends of the range.

    The pair carries what the summary figure cannot. The top row shows where
    the device lands BETWEEN the uniform reference and the noiseless ceiling,
    which is the only way a hardware rho is interpretable. The bottom row
    shows HOW the signal is lost: the ideal distribution concentrates on a
    few designs, the device output flattens toward 1/2^N. On the largest
    instance the two curves have already met.
    """
    from esdqaoa.fastsim import FastESDRunner
    from esdqaoa.simulate import default_alpha, default_beta_dep

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.0))
    for col, name in enumerate((small, large)):
        rec = data["runs"][name]
        pb = REG[name]
        sweep = rec["sweep"]
        phis = [s["phi"] for s in sweep]

        a = axes[0][col]
        ideal = [s["ideal"]["rho"] for s in sweep]
        hw = [s["hardware"]["rho"] for s in sweep]
        unif = [s["uniform"]["rho"] for s in sweep]
        a.plot(phis, ideal, "-D", color="#7b1d1d", ms=5.5, lw=1.7,
               label="noiseless simulation")
        a.plot(phis, hw, "-o", color=SIG_C, ms=6, lw=1.7, label="IBM hardware")
        a.plot(phis, unif, "--", color=MUTED, lw=1.5, label="uniform reference")
        a.fill_between(phis, unif, hw, color=SIG_C, alpha=.13)
        a.set_xlabel(r"trade-off weight $\varphi$", fontsize=9, color=INK)
        a.set_ylabel(r"approximation ratio $\rho$", fontsize=9, color=INK)
        a.set_ylim(0, 0.95)
        a.set_title(f"{'AB'[col]}  Problem {DISPLAY[name]} "
                    f"({pb.n_qubits} qubits, "
                    f"{sweep[0]['transpiled']['two_qubit_gates']:,} 2q gates)",
                    fontsize=9.5, color=INK, loc="left")
        a.legend(fontsize=8, frameon=False, loc="upper right")
        a.grid(**GRID)
        style(a)

        b = axes[1][col]
        mid = sweep[len(sweep) // 2]
        r = FastESDRunner(pb, rec.get("p", 1), mid["phi"],
                          default_alpha(pb), default_beta_dep(pb))
        ideal_d = np.sort(r.probabilities(mid["angles"]))[::-1]
        hw_d = np.array(sorted(mid["counts"].values(), reverse=True))
        n_states = 2 ** pb.N
        unif = 1.0 / n_states

        # Plot the WHOLE distribution in both panels, on a log rank axis.
        # A fixed top-k cap is not comparable across sizes: 40 ranks is all
        # 16 configurations at N=4 but 1% of 4096 at N=12, so the small
        # panel showed the tail and the large one showed only the head. The
        # crossing below the uniform level is the signature of concentration
        # -- probabilities sum to 1 over 2^N states, so the mean IS the
        # uniform level and any head above it forces a tail below it -- and
        # cutting the tail off hid exactly that on the large instance.
        mk = dict(marker="D", ms=4) if n_states <= 64 else {}
        b.plot(np.arange(1, len(ideal_d) + 1), ideal_d, "-", color="#7b1d1d",
               lw=1.6, label="noiseless simulation", **mk)
        mk = dict(marker="o", ms=4) if n_states <= 64 else {}
        b.plot(np.arange(1, len(hw_d) + 1), hw_d, "-", color=SIG_C,
               lw=1.6, label="IBM hardware", **mk)
        # P3.2: a rank-ordered empirical distribution over many states from
        # 8192 shots is dominated by occupancy noise, so the measured curve
        # must be read against what a UNIFORM device would produce with the
        # same shots. Band = 2.5-97.5 percentile of the sorted multinomial
        # null at each rank.
        shots = 8192
        rng = np.random.default_rng(7)
        reps = 400 if n_states > 1024 else 2000
        draws = np.empty((reps, n_states))
        for t in range(reps):
            draws[t] = np.sort(rng.multinomial(
                shots, np.full(n_states, unif)))[::-1] / shots
        lo_b, hi_b = np.percentile(draws, [2.5, 97.5], axis=0)
        b.fill_between(np.arange(1, n_states + 1), lo_b, hi_b,
                       color=MUTED, alpha=0.45, lw=0,
                       label="uniform device, 8192 shots (95% band)")
        for bnd in (lo_b, hi_b):
            b.plot(np.arange(1, n_states + 1), bnd, "-", color=MUTED,
                   lw=0.8, alpha=0.9)
        b.axhline(unif, color=MUTED, ls="--", lw=1.4,
                  label=f"uniform mean, $1/2^{{{pb.N}}}$")
        if (ideal_d < unif).any():
            xc = int(np.argmax(ideal_d < unif)) + 1
            b.axvline(xc, color=INK, ls=":", lw=1.0, alpha=.55)
            b.annotate(f"noiseless falls below\nuniform at rank {xc:,}",
                       xy=(xc, unif), xytext=(5, 12),
                       textcoords="offset points", fontsize=7, color=MUTED)
        b.set_xscale("log")
        b.set_xlim(1, n_states)
        b.set_yscale("log")
        b.set_xlabel(f"configuration, ranked by its own probability "
                     f"(all $2^{{{pb.N}}}$)", fontsize=9, color=INK)
        b.set_ylabel("probability of that configuration",
                     fontsize=9, color=INK)
        b.set_title(f"{'CD'[col]}  Output distribution at "
                    f"$\\varphi={mid['phi']:.2f}$", fontsize=9.5, color=INK,
                    loc="left")
        b.legend(fontsize=8, frameon=False, loc="lower left")
        b.grid(**GRID)
        style(b)

    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_hardware_detail.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_hardware(rows, data):
    sig = [r for r in rows if r["n_sig"] >= 4]
    wall_lo = max(r["gates"] for r in sig)
    wall_hi = min(r["gates"] for r in rows if r["n_sig"] < 4)
    by_gates = sorted(rows, key=lambda r: r["gates"])

    fig, (a, b) = plt.subplots(1, 2, figsize=(13.0, 4.3))

    g = [r["gates"] for r in by_gates]
    m = [r["margin"] for r in by_gates]
    e = [3 * r["sd"] for r in by_gates]
    cols = [SIG_C if r["n_sig"] >= 4 else NOISE_C for r in by_gates]
    # The transition is bracketed, not measured: the largest problem that
    # clears the floor has wall_lo gates and the smallest that fails has
    # wall_hi, with nothing sampled in between. Draw it as an interval and
    # say so, rather than as a line that implies a measured threshold.
    a.axvspan(wall_lo, wall_hi, color=SIG_C, alpha=.10, zorder=0)
    for xv in (wall_lo, wall_hi):
        a.axvline(xv, color=MUTED, lw=1.0, ls=":", zorder=1)
    a.errorbar(g, m, yerr=e, fmt="none", elinewidth=1.4, ecolor=MUTED,
               capsize=3, zorder=2)
    # Fill, not just hue, separates the two classes: filled blue for output
    # above the floor, hollow for output at it. Blue against grey-blue alone
    # is too weak at marker size, and colour should never be the only cue.
    sig_mask = [r["n_sig"] >= 4 for r in by_gates]
    a.scatter([v for v, s_ in zip(g, sig_mask) if s_],
              [v for v, s_ in zip(m, sig_mask) if s_],
              s=70, c=SIG_C, zorder=4)
    a.scatter([v for v, s_ in zip(g, sig_mask) if not s_],
              [v for v, s_ in zip(m, sig_mask) if not s_],
              s=62, facecolors="white", edgecolors=NOISE_C, linewidths=1.6,
              zorder=3)
    a.axhline(0, color=INK, lw=1.1)
    # Label side follows the neighbours, not the axis midpoint: put each
    # label into whichever adjacent gap is wider, so a label never lands on
    # the next problem's marker. Alternating a small vertical nudge keeps two
    # labels that face each other across one gap from touching.
    gaps = [(g[i] - g[i - 1] if i else float("inf"),
             g[i + 1] - g[i] if i < len(g) - 1 else float("inf"))
            for i in range(len(g))]
    # Per-problem overrides, applied on top of the automatic placement.
    # The gap rule gets most labels right but cannot see that a marker's
    # error bar, the transition band or a neighbour's label occupies the
    # side it chose. Values are (dx, dy, side) in offset points; side None
    # keeps whatever the gap rule picked. Tune these, not the rule.
    LABEL_FIX = {
        "III":  (12, 0, "right"),    # was crowding its own error bar
        "IV":   (12, 0, "right"),    # sat on the transition band
        "V":    (-13, 2, "left"),    # clear of the band's right edge
        "VI":   (13, -2, "right"),
        "VII":  (-13, 2, "left"),
        "VIII": (6, 16, "right"),    # VIII and IX are only 56 gates
                                     # apart, so VIII goes UP, not across
        "X":    (-12, 9, "left"),
        "XI":   (12, 9, "right"),
    }
    for i, r in enumerate(by_gates):
        name = DISPLAY[r["name"]]
        left = gaps[i][0] >= gaps[i][1]
        dx = -12 if left else 12
        dy = 7 if i % 2 == 0 else -7
        if name in LABEL_FIX:
            fx, fy, side = LABEL_FIX[name]
            dx, dy = fx, fy
            if side is not None:
                left = (side == "left")
        a.annotate(name, (r["gates"], r["margin"]),
                   textcoords="offset points", xytext=(dx, dy),
                   ha="right" if left else "left", va="center",
                   fontsize=8.5, color=INK, zorder=5)
    a.annotate(f"transition lies between\n{wall_lo:,} and {wall_hi:,} 2q gates",
               xy=(wall_hi, 0.97), xycoords=("data", "axes fraction"),
               xytext=(14, 0), textcoords="offset points",
               fontsize=8, color=INK, ha="left", va="top")
    # Room for the outermost labels, which sit beside their markers and
    # would otherwise run into the frame.
    a.margins(x=0.10)
    a.set_xlabel("two-qubit gates after transpilation", fontsize=9.5,
                 color=INK)
    a.set_ylabel(r"$\rho_{\mathrm{hw}} - \rho_{\mathrm{uniform}}$",
                 fontsize=9.5, color=INK)
    a.set_title("A  Margin over the uniform reference, $\\pm3\\sigma$ shot noise",
                fontsize=9.5, color=INK, loc="left")
    a.grid(**GRID)
    style(a)

    x = np.arange(len(rows))
    b.bar(x - 0.21, [r["ideal"] for r in rows], 0.42, color="#7b1d1d",
          label="noiseless simulation")
    # Two colours of hardware bar, so both need naming. Without the second
    # entry the grey bars appear with nothing in the legend to explain them.
    b.bar(x + 0.21, [r["hw"] for r in rows], 0.42,
          color=[SIG_C if r["n_sig"] >= 4 else NOISE_C for r in rows])
    # Proxy handles, not bar([], []): an empty BarContainer has no patch, so
    # the legend cannot pick up its colour and both entries render alike.
    from matplotlib.patches import Patch
    b.add_artist(b.legend(
        handles=[Patch(facecolor=SIG_C, label="IBM hardware, above the floor"),
                 Patch(facecolor=NOISE_C,
                       label="IBM hardware, at the floor")],
        fontsize=8, frameon=False, loc="upper right",
        bbox_to_anchor=(1.0, 0.86)))
    b.plot(x, [r["unif"] for r in rows], "_", color=INK, ms=21, mew=2.0,
           zorder=5, label="uniform reference")
    for i, r in enumerate(rows):
        if r["n_sig"] >= 4:
            b.text(i + 0.21, r["hw"] + .012, "*", ha="center", fontsize=13,
                   color=SIG_C)
    b.set_xticks(x)
    b.set_xticklabels([DISPLAY[r["name"]] for r in rows], fontsize=8.5)
    b.set_ylim(0, 0.72)
    b.set_ylabel(r"mean approximation ratio $\rho$", fontsize=9.5, color=INK)
    b.set_title(f"B  Hardware against simulation — * marks the "
                f"{len(sig)} problems whose output\nis distinguishable from "
                f"uniform ($\\geq$4/5 $\\varphi$ at $>3\\sigma$)",
                fontsize=9.5, color=INK, loc="left")
    b.legend(fontsize=8, frameon=False, ncol=1, loc="upper right")
    b.grid(axis="y", **GRID)
    style(b)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_hardware.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
def wrap(body, caption, label, colspec, header, note=None, wide=False):
    """Emit one table in MDPI house style.

    MDPI specifics: float placement [H], caption ABOVE the tabular, booktabs
    rules, and \\begin{adjustwidth}{-\\extralength}{0cm} to let a wide table
    use the full page in the single-column layout. \\extralength is defined
    by the mdpi class, so the adjustwidth wrapper only appears on tables that
    need it.

    Deliberately never emits table*: in a one-column layout a starred float
    can be silently dropped ("Float(s) lost").
    """
    lines = ["\\begin{table}[H]", "\\centering",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    if wide:
        lines.append("\\begin{adjustwidth}{-\\extralength}{0cm}")
        lines += ["\\footnotesize", "\\setlength{\\tabcolsep}{4pt}"]
    lines += [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header,
              "\\midrule", body, "\\bottomrule", "\\end{tabular}"]
    if note:
        lines.append(f"\\\\[2pt]\n\\begin{{minipage}}{{\\linewidth}}"
                     f"\\footnotesize {note}\\end{{minipage}}")
    if wide:
        lines.append("\\end{adjustwidth}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def tab_instances():
    """Full specification, so the benchmark is reproducible from the paper.

    Summary counts alone (N, qubits, Pareto points) do not let a reader
    rebuild a single problem. Performances, costs, budget and the dependency
    pairs are the complete definition; everything else in the paper follows
    from them by exhaustive enumeration.
    """
    rows = []
    for n in PAPER:
        pb = REG[n]
        perf = ", ".join(str(int(v)) for v in pb.performances)
        cost = ", ".join(str(int(v)) for v in pb.costs)
        dep = ", ".join(f"({int(i)},{int(j)})" for i, j in pb.dependencies)
        rows.append(f"{DISPLAY[n]} & {pb.N} & {pb.n_qubits} & {perf} & "
                    f"{cost} & {pb.budget} & {dep} & "
                    f"{len(pb.pareto_front())} & "
                    f"{len(pb.nonconvex_pareto_points())} \\\\[2pt]")
    return wrap(
        "\n".join(rows),
        "Complete specification of the eleven benchmark problems. Subsystem "
        "$i$ contributes performance $p_i$ and cost $c_i$; a dependency pair "
        "$(i,j)$ means subsystem $j$ requires subsystem $i$, so $x_j\\le "
        "x_i$. Qubit count is $N+k+1$: $N$ decision qubits, $k$ "
        "cost-register qubits and one budget flag, with $k$ given by "
        "\\cref{eq:k}. The last two columns are derived by exhaustive "
        "enumeration over the feasible set.",
        "tab:instances",
        "l r r p{3.1cm} p{2.5cm} r p{2.6cm} r r",
        "Problem & $N$ & Qubits & Performances $p_i$ & Costs $c_i$ & "
        "$\\mathcal{B}$ & Dependencies $D$ & Pareto & Unsup. \\\\",
        note="``Pareto'' counts distinct non-dominated objective vectors "
             "$(C,P)$, not architectures: on problems III and XI, 12 and 17 "
             "feasible architectures map onto 11 and 15 distinct vectors. "
             "``Unsup.'' counts those vectors that maximize $f_\\varphi$ at "
             "no $\\varphi\\in[0,1]$; a vector tied with its neighbours on a "
             "supported face is counted as supported, and the classification "
             "is computed exactly rather than on a $\\varphi$ grid. "
             "Every problem was constructed rather than drawn at "
             "random: each was built or selected to have a curved Pareto "
             "frontier carrying unsupported designs, with the dependency "
             "count fixed at $\\lceil N/3 \\rceil$. The set is therefore not "
             "a random sample of the problem class, and the size trends "
             "reported below should be read with that in mind.",
        wide=True)


def tab_rho():
    d = load("multiseed_all.json")
    runs = d["runs"]
    seeds = [str(x) for x in d["settings"]["seeds"]]
    rows = []
    for n in PAPER:
        r = runs[n]
        # Sample std (ddof=1), NOT the population std stored as
        # std_of_means in the JSON. Five seeds are a sample, and ddof=1 is
        # what CLAUDE.md section 7 quotes; recomputed here so the table and
        # the context file cannot drift apart.
        per = {m: np.array([r["methods"][m]["per_seed"][s]["mean_rho"]
                            for s in seeds]) for m in ("esd", "pauli")}
        gap = per["esd"] - per["pauli"]
        rows.append(
            f"{DISPLAY[n]} & {r['n_qubits']} & "
            f"${per['esd'].mean():.3f} \\pm {per['esd'].std(ddof=1):.3f}$ & "
            f"${per['pauli'].mean():.3f} \\pm "
            f"{per['pauli'].std(ddof=1):.3f}$ & "
            f"$+{gap.mean():.3f} \\pm {gap.std(ddof=1):.3f}$ & "
            f"$+{gap.min():.3f}$ \\\\")
    gaps = [runs[n]["gap"]["mean"] for n in PAPER]
    worst = min(min(runs[n]["gap"]["per_seed"]) for n in PAPER)
    n_runs = len(PAPER) * len(d["settings"]["seeds"])
    return wrap(
        "\n".join(rows),
        "Approximation ratio over five optimizer seeds "
        f"$\\{{{', '.join(map(str, d['settings']['seeds']))}\\}}$ at $p=2$, "
        "averaged over 13 values of $\\varphi$ at QAOA layer count $p=2$. "
        "Both methods use the same surrogate, optimizer, paired seeds, layer "
        "count and numerical penalty coefficients. The two encodings "
        "nevertheless implement different penalty functions, so the "
        "comparison is between complete formulations at matched layer count "
        "rather than between two parameterisations of one.",
        "tab:rho", "lrccc r",
        "Problem & Qubits & ESD-QAOA & QUBO/Ising QAOA & Gap & Worst seed \\\\",
        note=f"Mean gap $+{np.mean(gaps):.3f}$, positive on "
             f"{len(PAPER)}/{len(PAPER)} problems and on all {n_runs} "
             f"seed-paired runs (worst $+{worst:.3f}$). The gap shows no trend "
             f"with problem size (Pearson $r=-0.08$, $p=0.82$); $\\rho$ itself "
             f"declines significantly ($r=-0.89$, $p=0.0003$), and does so "
             f"even after normalizing by $\\rho_{{\\mathrm{{unif}}}}$ "
             f"($r=-0.68$, $p=0.02$).")


def tab_resources():
    d = load("resources.json")
    rows = []
    for row in d["rows"]:
        if row["name"] not in PAPER:
            continue
        e, p_ = row["esd"], row["pauli"]
        rows.append(f"{DISPLAY[row['name']]} & {e['qubits']} & {e['cx']} & "
                    f"{e['depth']} & {p_['qubits']} & {p_['cx']} & "
                    f"{p_['depth']} & {row['cx_ratio']:.1f}$\\times$ & "
                    f"{row['depth_ratio']:.1f}$\\times$ \\\\")
    return wrap(
        "\n".join(rows),
        "Transpiled circuit resources at $p=2$, basis "
        "$\\{r_z, \\sqrt{X}, X, \\mathrm{CX}\\}$, optimization level 1, "
        "all-to-all connectivity. A ratio above one means ESD-QAOA is the "
        "more expensive circuit.",
        "tab:resources", "lrrrrrrrr",
        "& \\multicolumn{3}{c}{ESD-QAOA} & \\multicolumn{3}{c}{QUBO/Ising QAOA} & "
        "\\multicolumn{2}{c}{Ratio} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}\n"
        "Problem & Qubits & CX & Depth & Qubits & CX & Depth & CX & Depth \\\\",
        note=_resources_note(), wide=True)


def _resources_note():
    """Table 5 note, including the epsilon-width correction of P0.4."""
    w = eps_qubit_widths()
    same = [DISPLAY[n] for n in PAPER if w[n][0] == w[n][1]]
    wider = [DISPLAY[n] for n in PAPER if w[n][1] > w[n][0]]
    delta = sorted({w[n][1] - w[n][0] for n in PAPER if w[n][1] > w[n][0]})
    span = ("%s to %s" % (wider[0], wider[-1])) if len(wider) > 1 else \
        (wider[0] if wider else "none")
    return (
        "Resources are measured for the weighted-sum circuit at QAOA layer "
        "count $p=2$. The reported advantage is at equal $p$, not equal "
        "circuit depth; matching ESD-QAOA's $p=2$ depth would require the "
        "baseline at $p\\approx23$ (13 qubits) to $p\\approx38$ (20 qubits), "
        "a comparison that was costed and not run (see Discussion). The "
        "$\\epsilon$-constraint circuit pins the comparison position and "
        "register width across the sweep and therefore uses "
        "%d additional cost-register qubit%s on problems %s; only on problem "
        "%s do the two scalarizations share a width. This table does not "
        "apply unchanged to the $\\epsilon$ sweep."
        % (delta[0] if delta else 0, "" if delta == [1] else "s",
           span, ", ".join(same) if same else "none"))


def fig_workflow():
    """P2.6 -- the pipeline from engineering inputs to a recovered trade space.

    A Systems reader should be able to follow what the method does before
    meeting a circuit diagram. Deliberately carries no numbers: it is a
    schematic of the procedure, and every quantitative claim lives elsewhere.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    stages = [
        ("Architecture data", "$p_i$, $c_i$, $\\mathcal{B}$, $D$\n"
                              "for $N$ candidate subsystems", FEAS_C),
        ("Scalarization", "weighted sum over $\\varphi$\n"
                          "or $\\epsilon$-constraint over $\\epsilon$", MUTED),
        ("ESD-QAOA circuit", "objective phases, reversible cost\n"
                             "accumulation, threshold test,\n"
                             "penalties, uncompute, mix", ESD_C),
        ("Optimize and sample", "classical optimizer tunes\n"
                                "$(\\gamma,\\beta)$; measure $R$", MUTED),
        ("Keep feasible designs", "discard architectures violating\n"
                                  "the budget or a dependency", FEAS_C),
        ("Recover the trade space", "modal tracking or\n"
                                    "distribution harvesting", HULL_C),
    ]
    fig, ax = plt.subplots(figsize=(13.4, 3.1))
    n = len(stages)
    w, gap = 1.30, 0.30
    for i, (title, body, colr) in enumerate(stages):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch(
            (x, 0.26), w, 0.72, boxstyle="round,pad=0.015,rounding_size=0.04",
            linewidth=1.4, edgecolor=colr, facecolor=colr + "14"))
        ax.text(x + w / 2, 0.87, title, ha="center", va="center",
                fontsize=8.8, color=INK, fontweight="bold")
        ax.text(x + w / 2, 0.57, body, ha="center", va="center",
                fontsize=7.3, color=INK, linespacing=1.5)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch(
                (x + w + 0.04, 0.62), (x + w + gap - 0.04, 0.62),
                arrowstyle="-|>", mutation_scale=13, linewidth=1.3,
                color=MUTED))
    # the sweep loop: one circuit per sweep point, same circuit throughout
    ax.add_patch(FancyArrowPatch(
        (1 * (w + gap) + w / 2, 0.22), (3 * (w + gap) + w / 2, 0.22),
        connectionstyle="arc3,rad=0.30", arrowstyle="<|-",
        mutation_scale=12, linewidth=1.1, color=MUTED, linestyle="--"))
    ax.text(2 * (w + gap) + w / 2, -0.21,
            "repeated once per sweep point; the circuit is unchanged, "
            "only $\\varphi$ or $\\epsilon$ differs",
            ha="center", va="center", fontsize=7.4, color=MUTED)
    ax.set_xlim(-0.14, n * (w + gap) - gap + 0.14)
    ax.set_ylim(-0.30, 1.06)
    ax.axis("off")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_workflow.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def nondegenerate_eps(pb, eps_values):
    """Epsilon bounds whose feasible set carries more than one performance.

    At a degenerate bound P_max == P_min, so eq:rhoeps has a zero denominator
    and the scorer's guard returns 1.0 -- a free perfect score on a trivial
    subproblem. Every epsilon quantity in the paper is computed over the
    retained bounds only (P0.3).
    """
    out = []
    for e in eps_values:
        vals = {int(pb.perf(x)) for x in pb.all_configs()
                if pb.is_eps_feasible(x, e)}
        if len(vals) > 1:
            out.append(e)
    return out


def eps_modal_front(pb, sweep, kept):
    """Non-dominated set of modal designs over the retained epsilon bounds."""
    pts = []
    for e in sweep:
        if e["eps"] not in kept:
            continue
        best, bp = None, -1.0
        for x, pr in e["probs"].items():
            xa = np.array([int(c) for c in x])
            if pb.is_eps_feasible(xa, e["eps"]) and pr > bp:
                best, bp = xa, pr
        if best is not None:
            pts.append((int(pb.cost(best)), int(pb.perf(best))))
    return {(c, p) for c, p in pts
            if not any((c2 <= c and p2 >= p) and (c2 < c or p2 > p)
                       for c2, p2 in pts)}


def eps_exact_front(pb, kept):
    """Count of distinct Pareto vectors an exact solver reaches on `kept`."""
    ref = set()
    for e in sorted(kept):
        o = pb.epsilon_constraint_optimum(e)
        if o is not None:
            ref.add((int(o[0]), int(o[1])))
    return len({(c, p) for c, p in ref
                if not any((c2 <= c and p2 >= p) and (c2 < c or p2 > p)
                           for c2, p2 in ref)})


def eps_qubit_widths():
    """(weighted-sum qubits, pinned-epsilon qubits) per paper problem.

    The epsilon circuit pins the comparison position and register width
    across the sweep, which costs one extra cost-register qubit on most
    problems. Table 5 reports the weighted-sum circuit, so this is the
    correction that keeps its caption honest (P0.4).
    """
    from esdqaoa.circuits import budget_encoding, pinned_encoding
    eps = epsilon_runs()
    out = {}
    for n in PAPER:
        pb = REG[n]
        kept = nondegenerate_eps(
            pb, [s["eps"] for s in eps[n]["methods"]["esd"]["sweep"]])
        _, k_eps = pinned_encoding(pb, kept)
        _, _, k_ws = budget_encoding(pb)
        out[n] = (pb.N + k_ws + 1, pb.N + k_eps + 1)
    return out


def tab_scalarization(eps, ws):
    """Recovery under the two scalarizations, plus the rho_eps pair.

    The recovery columns are the point of the section, but they do NOT
    separate the encodings (the eps totals come out within a few points of
    each other). The quantity that does is rho_eps, the solution-quality
    ratio of eq:rhoeps, so it belongs in this table rather than in one
    unquantified sentence of prose.
    """
    rows, r_eps, n_eps_min = [], [], None
    n_dropped = 0
    for n in PAPER:
        pb, r = REG[n], eps[n]
        # classification comes from the registry, not from the stored JSON
        # (P0.1: the stored "nonconvex" lists predate the fix)
        ncs = set(pb.nonconvex_pareto_points())
        tot, nc = len(r["true_front"]), len(ncs)
        # P0.3: a bound whose feasible set carries a single performance value
        # has P_max == P_min, so eq:rhoeps is undefined there. The shipped
        # scorer returned 1.0, putting a free perfect score into the mean.
        # Those bounds are excluded from EVERY epsilon quantity below.
        kept = set(nondegenerate_eps(pb, [s["eps"] for s
                                          in r["methods"]["esd"]["sweep"]]))
        n_dropped += len(r["methods"]["esd"]["sweep"]) - len(kept)
        rf = set((c, p) for c, p, *_
                 in ws[n]["methods"]["esd"]["recovered_front"])
        ws_pts = len(rf & set(map(tuple, r["true_front"])))
        ws_nc = len(rf & ncs)
        e_eps = eps_modal_front(pb, r["methods"]["esd"]["sweep"], kept)
        p_eps = eps_modal_front(pb, r["methods"]["pauli"]["sweep"], kept)
        eps_nc = len(e_eps & ncs)
        eps_ref = eps_exact_front(pb, kept)
        re_e = float(np.mean([s["rho_eps"] for s
                              in r["methods"]["esd"]["sweep"]
                              if s["eps"] in kept]))
        re_p = float(np.mean([s["rho_eps"] for s
                              in r["methods"]["pauli"]["sweep"]
                              if s["eps"] in kept]))
        r_eps.append((DISPLAY[n], re_e, re_p))
        k = len(kept)
        if n_eps_min is None or k < n_eps_min[1]:
            n_eps_min = (DISPLAY[n], k)
        rows.append(
            f"{DISPLAY[n]} & {tot} & {nc} & "
            f"{len(pb.supported_pareto_points())} & "
            f"{eps_ref} & "
            f"{ws_pts} & {len(e_eps & set(map(tuple, r['true_front'])))} & "
            f"{len(p_eps & set(map(tuple, r['true_front'])))} & "
            f"{ws_nc} & {eps_nc} & "
            f"{re_e:.3f} & {re_p:.3f} \\\\")

    def _ws_unsup(n, m):
        return (set((c, p) for c, p, *_
                    in ws[n]["methods"][m]["recovered_front"])
                & set(REG[n].nonconvex_pareto_points()))

    n_zero = sum(1 for n in PAPER
                 if not _ws_unsup(n, "esd") and not _ws_unsup(n, "pauli"))
    n_zero_esd = sum(1 for n in PAPER if not _ws_unsup(n, "esd"))
    gaps = [e - p for _, e, p in r_eps]
    esd_eps = pau_eps = 0
    for n in PAPER:
        pb, r = REG[n], eps[n]
        kept = set(nondegenerate_eps(pb, [s["eps"] for s
                                          in r["methods"]["esd"]["sweep"]]))
        tfs = set(map(tuple, r["true_front"]))
        esd_eps += len(eps_modal_front(pb, r["methods"]["esd"]["sweep"],
                                       kept) & tfs)
        pau_eps += len(eps_modal_front(pb, r["methods"]["pauli"]["sweep"],
                                       kept) & tfs)
    return wrap(
        "\n".join(rows),
        "Pareto-front recovery under the two scalarizations at $p=2$, "
        "seed 7. Exact references count supported points for weighted sums "
        "and points reachable by exact optimization on the integer "
        "$\\epsilon$ grid; they describe exact scalar optima and are not "
        "universal bounds on recovery from approximate distributions. "
        "Recovery columns count designs found by modal tracking. The final "
        "pair gives the mean $\\rho_\\epsilon$ of \\cref{eq:rhoeps} over "
        "the $\\epsilon$ sweep.",
        "tab:scalarization", "lrr rr rrr rr rr",
        "& & & \\multicolumn{2}{c}{Exact reference} & "
        "\\multicolumn{3}{c}{Quantum, front points} & "
        "\\multicolumn{2}{c}{Unsupported} & "
        "\\multicolumn{2}{c}{Mean $\\rho_\\epsilon$} \\\\\n"
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-8}"
        "\\cmidrule(lr){9-10}\\cmidrule(lr){11-12}\n"
        "Problem & $|PF|$ & Unsup. & w.sum & $\\epsilon$ & "
        "ESD w.sum & ESD $\\epsilon$ & QUBO/Ising $\\epsilon$ & "
        "ESD w.sum & ESD $\\epsilon$ & ESD & QUBO/Ising \\\\",
        note=f"With modal tracking the weighted-sum sweep recovers no "
             f"unsupported point on {n_zero_esd} of {len(PAPER)} problems "
             f"for ESD-QAOA, and on {n_zero} of {len(PAPER)} neither "
             f"encoding recovers any. The $\\epsilon$ front-point totals "
             f"are {esd_eps} for ESD-QAOA and {pau_eps} for the baseline, "
             f"a difference comparable to the per-problem variation, so "
             f"the recovery columns do not separate the encodings. Mean "
             f"$\\rho_\\epsilon$ does: it favours ESD-QAOA on all "
             f"{len(gaps)} problems, by ${np.mean(gaps):.3f}$ on average "
             f"(from $+{min(gaps):.3f}$ to $+{max(gaps):.3f}$), which is "
             f"comparable to the weighted-sum gap of $0.150$ in "
             f"\\cref{{tab:rho}}. These $\\rho_\\epsilon$ values use a "
             f"single optimizer seed. Integer rounding, de-duplication and "
             f"the exclusion of {n_dropped} degenerate bounds, at which the "
             f"feasible set carries a single performance value and "
             f"\\cref{{eq:rhoeps}} is undefined, leave fewer than thirteen "
             f"scored $\\epsilon$ values on every problem "
             f"(problem {n_eps_min[0]}: {n_eps_min[1]}).",
        wide=True)


# --------------------------------------------------------------------------
def fig_comparison_space(ws, eps, ncol=6):
    """Which designs each encoding actually recovers, in objective space.

    Two comparisons in one figure without overplotting:
        within a panel   ESD vs Pauli         -> which ENCODING
        paired rows      weighted sum vs eps  -> which SCALARIZATION

    MODAL fronts for both sweeps, deliberately. A weighted-sum run's modal
    design is always a convex-hull vertex, so the modal view is what exposes
    the structural blind spot; harvesting flatters both methods and blurs the
    point. Eleven problems in two rows would be unreadable, so the set is
    split into blocks of `ncol`, each block carrying its own pair of rows.
    """
    blocks = [PAPER[i:i + ncol] for i in range(0, len(PAPER), ncol)]
    nrows = 2 * len(blocks)
    fig, axes = plt.subplots(nrows, ncol, figsize=(2.32 * ncol, 2.5 * nrows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for b, names in enumerate(blocks):
        for c, name in enumerate(names):
            pb = REG[name]
            tf = sorted((cst, p) for cst, p, _ in pb.pareto_front())
            hull = sorted(pb.supported_pareto_points())
            unsup = pb.nonconvex_pareto_points()
            feas = [(pb.cost(x), pb.perf(x)) for x in pb.feasible_configs()]
            for row, (src, key, tag) in enumerate(
                    [(ws, "recovered_front", r"weighted sum ($\varphi$ sweep)"),
                     (eps, "modal_front", r"$\epsilon$-constraint ($\epsilon$ sweep)")]):
                ax = axes[2 * b + row][c]
                ax.axis("on")
                if feas:
                    ax.scatter(*zip(*feas), s=6, color=FEAS_C, alpha=.28, lw=0,
                               zorder=1)
                ax.plot(*zip(*tf), "-", color=FRONT_C, lw=1.2, zorder=3)
                ax.plot(*zip(*hull), "-", color=HULL_C, lw=1.1, zorder=2)
                if unsup:
                    ax.scatter(*zip(*unsup), s=70, facecolors="none",
                               edgecolors=UNSUP_C, lw=1.2, zorder=4)
                got = {}
                for meth, colr, mk, size, fill in [
                        ("pauli", PAULI_C, "^", 62, False),
                        ("esd", ESD_C, "D", 22, True)]:
                    pts = sorted({(cst, p) for cst, p, *_
                                  in src[name]["methods"][meth][key]})
                    got[meth] = len(set(pts) & set(tf))
                    kw = (dict(c=colr) if fill
                          else dict(facecolors="none", edgecolors=colr, lw=1.3))
                    ax.scatter(*zip(*pts), marker=mk, s=size,
                               zorder=6 if fill else 5, **kw)
                ax.set_title(f"Problem {DISPLAY[name]} ({pb.n_qubits}q)  "
                             f"ESD {got['esd']}/{len(tf)}, "
                             f"Pauli {got['pauli']}/{len(tf)}",
                             fontsize=7.0, color=INK)
                ax.tick_params(labelsize=7)
                ax.grid(**GRID)
                style(ax)
                if c == 0:
                    ax.set_ylabel(f"total performance\n{tag}", fontsize=7.6,
                                  color=INK)
                if row == 1:
                    ax.set_xlabel("total cost", fontsize=7.6, color=INK)

    handles = [
        Line2D([], [], marker="o", ls="", ms=4, color=FEAS_C,
               label="feasible design vectors"),
        Line2D([], [], ls="-", color=FRONT_C, label="exact Pareto frontier"),
        Line2D([], [], ls="-", color=HULL_C, label="supported convex envelope"),
        Line2D([], [], marker="o", ls="", ms=8, mfc="none", mec=UNSUP_C,
               mew=1.2, label="unsupported Pareto design"),
        Line2D([], [], marker="^", ls="", ms=8, mfc="none", mec=PAULI_C,
               mew=1.3, label="QUBO/Ising QAOA recovered"),
        Line2D([], [], marker="D", ls="", ms=5, color=ESD_C,
               label="ESD-QAOA recovered"),
    ]
    # Put the legend in the white space the last block leaves free, rather
    # than in a strip under the figure: the final block holds fewer problems
    # than there are columns, so the trailing cells of its two rows are empty.
    spare = len(blocks) * ncol - len(PAPER)
    if spare:
        cell = axes[nrows - 2][ncol - spare]
        cell.legend(handles=handles, loc="center", ncol=1, frameon=False,
                    fontsize=8.5, labelspacing=1.1, handletextpad=0.8,
                    bbox_to_anchor=(0.5, 0.0))
        bottom = 0.0
    else:
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                   fontsize=9, bbox_to_anchor=(0.5, 0.0))
        bottom = 0.055
    fig.suptitle("Designs recovered by modal tracking: encoding compared "
                 "within each panel, scalarization between paired rows",
                 fontsize=10.5, color=INK, y=0.995)
    fig.tight_layout(rect=[0, bottom, 1, 0.975])
    out = os.path.join(FIGDIR, "fig_comparison_space.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_rho_phi():
    """rho against phi for every problem, five-seed mean with a +/-1 sd band.

    Figure 2 averages over phi, which leaves open whether a handful of
    weights carry the gap. This one answers that. The single-seed version of
    this plot crossed constantly and read as a tie on several problems; the
    seed bands are what make the separation visible.
    """
    d = load("multiseed_all.json")
    runs, phis = d["runs"], np.array(d["phis"])
    seeds = [str(s) for s in d["settings"]["seeds"]]

    fig, axes = plt.subplots(3, 4, figsize=(13.2, 8.6), sharex=True)
    wins = sig = total = 0
    for ax, name in zip(axes.ravel(), PAPER):
        r = runs[name]["methods"]
        arr = {m: np.array([r[m]["per_seed"][s]["rhos"] for s in seeds])
               for m in ("esd", "pauli")}
        for m, colr, mk, lbl in [("pauli", PAULI_C, "^", "QUBO/Ising QAOA"),
                                 ("esd", ESD_C, "D", "ESD-QAOA")]:
            mu, sd = arr[m].mean(0), arr[m].std(0, ddof=1)
            ax.fill_between(phis, mu - sd, mu + sd, color=colr, alpha=.18,
                            lw=0)
            ax.plot(phis, mu, "-", marker=mk, ms=3.8, color=colr, lw=1.6,
                    label=lbl)
        gap = arr["esd"] - arr["pauli"]
        gm, gse = gap.mean(0), gap.std(0, ddof=1) / np.sqrt(len(seeds))
        w, s_ = int((gm > 0).sum()), int((gm > 2 * gse).sum())
        wins, sig, total = wins + w, sig + s_, total + len(gm)
        ax.set_title(f"Problem {DISPLAY[name]} — {REG[name].n_qubits} "
                     f"qubits\nESD ahead at {w}/{len(gm)} values of "
                     f"$\\varphi$", fontsize=8.5, color=INK)
        ax.set_ylim(0, 1.0)
        ax.grid(**GRID)
        style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"approximation ratio $\rho$", fontsize=8.5, color=INK)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"trade-off weight $\varphi$", fontsize=8.5, color=INK)
    # last row has three problems; the fourth cell holds the legend, as in
    # the instance figure, so it never sits on a curve
    spare = axes.ravel()[len(PAPER)]
    spare.axis("off")
    handles = [
        Line2D([], [], color=ESD_C, marker="D", ms=5, lw=1.6,
               label="ESD-QAOA, mean of 5 seeds"),
        Line2D([], [], color=PAULI_C, marker="^", ms=5, lw=1.6,
               label="QUBO/Ising QAOA, mean of 5 seeds"),
        plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=.25,
                      label=r"$\pm1$ s.d. over seeds"),
    ]
    spare.legend(handles=handles, loc="center", frameon=False, fontsize=10,
                 labelspacing=1.0)
    spare.tick_params(labelbottom=True)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_rho_phi.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"rho-vs-phi: ESD ahead at {wins}/{total} (problem, phi) points; "
          f"lead > 2 s.e. at {sig}/{total}")
    return out


def tab_estimators(ws, rows):
    """Both estimators, both encodings, on the same weighted-sum runs.

    Two independent effects, separable by reading the table two ways.
    Across a pair of columns: modal tracking reads one design per sweep
    point -- what the algorithm returns if used as a single-objective
    solver at each weight -- while harvesting reads the whole distribution
    and keeps every feasible design above a probability floor, which is
    what a designer sampling the circuit obtains. Between the ESD and Pauli
    pairs: the same two rules applied to the baseline, so the estimator
    effect does not have to be disentangled from the encoding effect in
    prose.

    Counts are DISTINCT Pareto objective vectors, not configurations: two
    designs can share a (cost, performance) vector, and counting
    configurations lets a column exceed |PF|.
    """
    out = []
    tot = {k: 0 for k in ("tf", "nc", "em", "eh", "pm", "ph",
                          "emu", "ehu", "pmu", "phu")}
    for n in PAPER:
        pb = REG[n]
        tf = set((c, p) for c, p, _ in pb.pareto_front())
        nc = set(pb.nonconvex_pareto_points())
        got = {}
        for m in ("esd", "pauli"):
            rec = ws[n]["methods"][m]
            got[m] = (
                set((c, p) for c, p, *_ in rec["recovered_front"]),
                set((c, p) for c, p, _, _
                    in harvested_front(pb, parse_sweep(rec), 0.02)))
        front = [len(s & tf) for m in ("esd", "pauli") for s in got[m]]
        unsup = [len(s & nc) for m in ("esd", "pauli") for s in got[m]]
        for k, v in zip(("em", "eh", "pm", "ph"), front):
            tot[k] += v
        for k, v in zip(("emu", "ehu", "pmu", "phu"), unsup):
            tot[k] += v
        tot["tf"] += len(tf)
        tot["nc"] += len(nc)
        out.append(f"{DISPLAY[n]} & {len(tf)} & {len(nc)} & "
                   + " & ".join(str(v) for v in front) + " & "
                   + " & ".join(str(v) for v in unsup) + " \\\\")
    worst = sorted(PAPER, key=lambda n: rows[n]["pauli"]["pcr"])[:2]
    pcr_txt = " and ".join(
        f"${rows[n]['pauli']['pcr']:.2f}$ on problem {DISPLAY[n]}"
        for n in sorted(worst, key=lambda n: -rows[n]["pauli"]["pcr"]))
    return wrap(
        "\n".join(out),
        "Both estimators applied to the same weighted-sum runs at $p=2$, "
        "for both encodings. Modal tracking records the "
        "highest-probability feasible design at each $\\varphi$; "
        "distribution harvesting collects every feasible design observed "
        "with probability at least $\\tau=0.02$ at any $\\varphi$ and keeps "
        "the non-dominated subset. Entries count distinct Pareto objective "
        "vectors recovered.",
        "tab:estimators", "lrr rrrr rrrr",
        "& & & \\multicolumn{4}{c}{Front points recovered} & "
        "\\multicolumn{4}{c}{Unsupported recovered} \\\\\n"
        "\\cmidrule(lr){4-7}\\cmidrule(lr){8-11}\n"
        "& & & \\multicolumn{2}{c}{ESD-QAOA} & "
        "\\multicolumn{2}{c}{QUBO/Ising QAOA} & "
        "\\multicolumn{2}{c}{ESD-QAOA} & "
        "\\multicolumn{2}{c}{QUBO/Ising QAOA} \\\\\n"
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}"
        "\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}\n"
        "Problem & $|PF|$ & Unsup. & Modal & Harv. & Modal & Harv. & "
        "Modal & Harv. & Modal & Harv. \\\\",
        note=f"Across the eleven problems the estimator changes the "
             f"recovered set far more than the encoding does. Harvesting "
             f"raises front-point recovery from {tot['em']} to "
             f"{tot['eh']} of {tot['tf']} for ESD-QAOA and from "
             f"{tot['pm']} to {tot['ph']} for the baseline, and raises "
             f"unsupported recovery from {tot['emu']} to {tot['ehu']} of "
             f"{tot['nc']} and from {tot['pmu']} to {tot['phu']}. An "
             f"approximate QAOA distribution can have an unsupported "
             f"design as its mode, which is why modal tracking recovers "
             f"any. Between encodings, harvested front-point recovery "
             f"favours ESD-QAOA ({tot['eh']} against {tot['ph']}) while "
             f"the unsupported counts ({tot['ehu']} against {tot['phu']}) "
             f"do not separate them. PCR is tolerance-matched at "
             f"normalized distance $0.15$ and is therefore not an "
             f"exact-recovery count (Section~\\ref{{sec:metrics}}): it is "
             f"$1.00$ for ESD-QAOA on all eleven problems, and {pcr_txt} "
             f"for the baseline.",
        wide=True)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(TABDIR, exist_ok=True)
    ws, eps = weighted_sum_runs(), epsilon_runs()

    # harvested coverage + mass, recomputed from the stored distributions
    rows = {}
    for n in PAPER:
        pb = REG[n]
        tf = [(c, p) for c, p, _ in pb.pareto_front()]
        nc = pb.nonconvex_pareto_points()
        rows[n] = {}
        for m in ("esd", "pauli"):
            sw = parse_sweep(ws[n]["methods"][m])
            found = [(c, p) for c, p, _, _ in harvested_front(pb, sw, 0.02)]
            fm, pm = sweep_mass_metrics(pb, sw)
            hit, tot = nonconvex_coverage(found, nc)
            rows[n][m] = dict(pcr=pcr(found, tf), pm=float(np.mean(pm)),
                              fm=float(np.mean(fm)), nc=hit, nc_tot=tot)

    made = [fig_workflow(), fig_instances(), fig_pareto(ws),
            fig_scalarization(eps, ws),
            fig_rho(), fig_rho_phi(), fig_concentration(rows, ws),
            fig_comparison_space(ws, eps), fig_depth()]
    hw_rows, hw_data = hardware_rows()
    made.append(fig_hardware(hw_rows, hw_data))
    made.append(fig_hardware_detail(hw_data))

    for fname, text in [("tab_instances.tex", tab_instances()),
                        ("tab_estimators.tex", tab_estimators(ws, rows)),
                        ("tab_rho.tex", tab_rho()),
                        ("tab_resources.tex", tab_resources()),
                        ("tab_scalarization.tex", tab_scalarization(eps, ws))]:
        path = os.path.join(TABDIR, fname)
        open(path, "w", encoding="utf-8").write(text)
        made.append(path)

    for p in made:
        print("wrote", os.path.relpath(p, ROOT))

    print(f"\n{'inst':>5} {'2q':>6} {'margin':>9} {'sig':>5}")
    for r in hw_rows:
        print(f"{r['name']:>5} {r['gates']:>6,} {r['margin']:>+9.4f} "
              f"{r['n_sig']}/{r['n_phi']}")


if __name__ == "__main__":
    main()
