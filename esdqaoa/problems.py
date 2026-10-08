"""Systems design problem definitions and classical reference analysis.

Convention: a dependency pair (i, j) means subsystem j REQUIRES subsystem i,
i.e. the constraint x_j <= x_i. Selecting j without i is infeasible.
"""
from dataclasses import dataclass, field
from itertools import product
import json
import math
import os
import numpy as np


@dataclass
class SDProblem:
    """A multi-objective systems design problem instance."""
    name: str
    performances: list
    costs: list
    budget: int
    dependencies: list = field(default_factory=list)  # (i, j): x_j <= x_i

    def __post_init__(self):
        self.performances = np.asarray(self.performances, dtype=int)
        self.costs = np.asarray(self.costs, dtype=int)
        self.N = len(self.costs)
        self.c_max = int(self.costs.sum())
        # Cost-register width: register holds C(x) + w0 with the threshold
        # offset w0 = 2^c - B - 1, c = floor(log2(B)) + 1. Width must avoid
        # modular wrap-around and provide >= 1 overflow-detection bit.
        self.c_bits = math.floor(math.log2(self.budget)) + 1
        self.w0 = 2**self.c_bits - self.budget - 1
        self.k = max(math.ceil(math.log2(self.c_max + self.w0 + 1)),
                     self.c_bits + 1)
        self.n_qubits = self.N + self.k + 1            # + flag qubit

    # ---- classical evaluation --------------------------------------------
    def cost(self, x):
        return int(np.dot(x, self.costs))

    def perf(self, x):
        return int(np.dot(x, self.performances))

    def deps_satisfied(self, x):
        """Dependency constraints only: x_j <= x_i for every pair (i, j)."""
        return all(not (x[j] == 1 and x[i] == 0) for (i, j) in self.dependencies)

    def is_feasible(self, x):
        return self.cost(x) <= self.budget and self.deps_satisfied(x)

    def is_eps_feasible(self, x, eps):
        """Feasible for the epsilon-constraint problem: deps + C(x) <= eps.

        The budget bound is replaced by eps; dependency constraints always
        apply, since they are structural rather than budgetary.
        """
        return self.cost(x) <= eps and self.deps_satisfied(x)

    def scalarized(self, x, phi):
        """phi * performance - (1 - phi) * cost (no penalty)."""
        return phi * self.perf(x) - (1 - phi) * self.cost(x)

    def all_configs(self):
        for bits in product([0, 1], repeat=self.N):
            yield np.array(bits, dtype=int)

    def feasible_configs(self):
        return [x for x in self.all_configs() if self.is_feasible(x)]

    # ---- cached feasible-set tables --------------------------------------
    # Everything below is O(2^N) once, then vectorised. Without the cache the
    # classical references re-scan the feasible set thousands of times, which
    # is unnoticeable at N=6 and prohibitive at N=13-14.
    def _tables(self):
        """(costs, perfs, bit-matrix) over the FEASIBLE set, cached."""
        cached = getattr(self, "_tab", None)
        if cached is None:
            bits = np.array([b for b in product([0, 1], repeat=self.N)],
                            dtype=np.int8)
            cost = bits @ self.costs
            perf = bits @ self.performances
            ok = cost <= self.budget
            for (i, j) in self.dependencies:
                ok &= ~((bits[:, j] == 1) & (bits[:, i] == 0))
            cached = (cost[ok], perf[ok], bits[ok])
            self._tab = cached
        return cached

    # ---- brute-force references ------------------------------------------
    def brute_force_optimum(self, phi):
        """Best feasible configuration for a given trade-off weight."""
        cost, perf, bits = self._tables()
        vals = phi * perf - (1 - phi) * cost
        i = int(np.argmax(vals))
        return bits[i].astype(int), float(vals[i])

    def pareto_front(self):
        """Non-dominated (cost, perf) pairs of the feasible set, sorted.

        Staircase sweep in O(n log n): after keeping the best performance at
        each distinct cost, a point survives iff it beats every cheaper one.
        """
        cost, perf, bits = self._tables()
        order = np.lexsort((-perf, cost))       # cost asc, then perf desc
        p_s = perf[order]
        # best performance strictly earlier in the order; a point survives iff
        # it beats it. Sorting perf descending within a cost tier means the
        # tier's runner-up compares against its own leader and is dropped, so
        # duplicate (cost, perf) pairs fall out without a separate dedup pass.
        prev_max = np.maximum.accumulate(
            np.concatenate(([np.iinfo(perf.dtype).min], p_s[:-1])))
        idx = order[p_s > prev_max]
        return [(int(cost[i]), int(perf[i]), tuple(int(b) for b in bits[i]))
                for i in idx]

    def admissible_phi(self, point):
        """Exact interval of phi in [0,1] on which `point` maximizes f_phi.

        Returns a (Fraction, Fraction) pair, or None when no such phi exists.

        A point (C_i, P_i) is a maximizer at phi iff for every feasible
        (C_j, P_j),

            phi*P_i - (1-phi)*C_i  >=  phi*P_j - (1-phi)*C_j ,

        which rearranges to the linear constraint ``phi*A_j >= B_j`` with
        ``A_j = (P_i-P_j) + (C_i-C_j)`` and ``B_j = C_i-C_j``. The admissible
        set is the intersection of those half-lines with [0,1]. Fractions keep
        the comparison exact, so a point that ties on a face is not lost to
        floating-point error the way a sampled phi grid loses it.
        """
        from fractions import Fraction

        cost, perf, _ = self._tables()
        ci, pi = int(point[0]), int(point[1])
        lo, hi = Fraction(0), Fraction(1)
        for cj, pj in zip(cost.tolist(), perf.tolist()):
            if (cj, pj) == (ci, pi):
                continue
            a = (pi - pj) + (ci - cj)
            b = ci - cj
            if a == 0:
                if b > 0:
                    return None
                continue
            bound = Fraction(b, a)
            if a > 0:
                lo = max(lo, bound)
            else:
                hi = min(hi, bound)
            if lo > hi:
                return None
        return (lo, hi)

    def classify_pareto_points(self):
        """Label every Pareto point extreme / face / unsupported.

        The manuscript's definition of *supported* is "maximizes f_phi for
        some phi", with the maximizer NOT required to be unique, so a point
        tied with its neighbours on a supported face is supported. This is the
        authoritative classifier for that definition.

        * ``extreme``     -- unique maximizer on an interval of phi, i.e. a
                             vertex of the upper convex boundary;
        * ``face``        -- maximizer only at a single phi, always tied with
                             its neighbours (collinear on a supported face);
        * ``unsupported`` -- never a maximizer at any phi in [0,1].

        An earlier implementation sampled phi on a grid and took ``argmax``,
        which records only the first of several tied maximizers, so every
        face-tied point was misreported as unsupported. On the eleven paper
        problems that moved three points (problems I, III and XI) out of the
        supported class. Do not reintroduce a grid-and-argmax test here.
        """
        out = {}
        for c, p, _ in self.pareto_front():
            iv = self.admissible_phi((c, p))
            if iv is None:
                out[(c, p)] = "unsupported"
            elif iv[0] == iv[1]:
                out[(c, p)] = "face"
            else:
                out[(c, p)] = "extreme"
        return out

    def scalarization_reachable(self):
        """(cost, perf) pairs a classical weighted-sum sweep can attain.

        Supported points, ties on a face included -- see
        classify_pareto_points() for why the tie handling matters.
        """
        return {pt for pt, lab in self.classify_pareto_points().items()
                if lab != "unsupported"}

    def supported_pareto_points(self):
        """Supported Pareto points, sorted by cost.

        Includes points lying on a supported face, which are exact
        weighted-sum optima through a tie.
        """
        return sorted(self.scalarization_reachable())

    def epsilon_constraint_optimum(self, eps):
        """max P(x) s.t. C(x) <= eps, over the feasible set.

        Ties on performance are broken toward the lowest cost, so the result
        is always a Pareto point rather than a padded one. Returns
        (cost, perf, x) or None when no feasible design fits under eps.
        """
        cost, perf, bits = self._tables()
        m = cost <= eps
        if not m.any():
            return None
        c_m, p_m, b_m = cost[m], perf[m], bits[m]
        best_p = p_m.max()
        j = int(np.argmin(np.where(p_m == best_p, c_m, np.iinfo(np.int64).max)))
        return int(c_m[j]), int(best_p), tuple(int(b) for b in b_m[j])

    def epsilon_constraint_reachable(self, eps_values=None):
        """(cost, perf) pairs recovered by an epsilon-constraint sweep.

        Unlike weighted-sum scalarization this has NO convexity blind spot:
        it recovers unsupported (non-convex) Pareto points too. The cost is
        one constrained single-objective solve per eps value.
        """
        if eps_values is None:
            eps_values = range(0, int(self.budget) + 1)
        pts = set()
        for eps in eps_values:
            best = self.epsilon_constraint_optimum(eps)
            if best is not None:
                pts.add((best[0], best[1]))
        return pts

    def nonconvex_pareto_points(self):
        """Pareto points a weighted-sum sweep can never reach."""
        reach = self.scalarization_reachable()
        return [(c, p) for c, p, _ in self.pareto_front() if (c, p) not in reach]


# ----------------------------------------------------------------------------
# The five case-study instances (Problems I-V).
# Pareto structure verified by brute force; see tests/test_problems.py.
# ----------------------------------------------------------------------------
PROBLEMS = {
    "I": SDProblem(
        name="I", performances=[10, 2], costs=[12, 1], budget=13,
        dependencies=[(1, 0)],
    ),  # N=2, 7 qubits, 3 Pareto pts, fully convex — hardware-friendly anchor
    "II": SDProblem(
        name="II", performances=[1, 2, 4], costs=[1, 2, 3], budget=3,
        dependencies=[(2, 1)],
    ),  # N=3, 7 qubits, 3 Pareto pts, 1 non-convex
    "III": SDProblem(
        name="III", performances=[2, 6, 7, 12], costs=[1, 3, 4, 6], budget=9,
        dependencies=[(0, 1), (1, 3)],
    ),  # N=4, 9 qubits, 5 Pareto pts, 2 non-convex, chained dependencies
    "IV": SDProblem(
        name="IV", performances=[3, 5, 9, 11, 14], costs=[1, 2, 5, 6, 8],
        budget=13, dependencies=[(0, 2), (1, 3)],
    ),  # N=5, 11 qubits, 8 Pareto pts, 3 non-convex
    "V": SDProblem(
        name="V", performances=[2, 4, 7, 9, 12, 15], costs=[1, 2, 3, 5, 6, 8],
        budget=14, dependencies=[(0, 2), (1, 4), (3, 5)],
    ),  # N=6, 12 qubits, 11 Pareto pts, 7 non-convex — headline instance
}


# ----------------------------------------------------------------------------
# Generated instance ladder Q13-Q20 (experiments/generate_instances.py).
#
# Pinned here as literals rather than left in results/instances.json, which is
# gitignored: the ladder is the experimental subject of the paper and must not
# depend on a regenerable, untracked file.
#
# Selected for CURVED fronts (diminishing returns: cheap subsystems have a high
# performance-per-cost ratio, expensive ones less) with unsupported Pareto
# points that are real dents rather than rounding -- mean dent depth 3-7% of the
# performance range, versus <1% for an earlier near-linear ladder whose fronts
# were flat and whose "non-convexity" was numerically meaningless.
#
# Dependency count is exactly ceil(N/3), matching the density of Problems I-V.
# Sampled with i < j so the requirement graph is acyclic by construction.
#
# NOTE these instances were SEARCHED for these properties. Say so in the
# manuscript; they are not generic draws.
# ----------------------------------------------------------------------------
LADDER = {
    "Q13": SDProblem(
        name="Q13", performances=[5, 15, 14, 6, 8, 13, 4],
        costs=[1, 6, 5, 1, 2, 6, 1],
        budget=12, dependencies=[(1, 6), (2, 6), (5, 6)],
    ),  # N=7,  13 qubits, 11 Pareto pts,  4 unsupported, 3 deps
    "Q14": SDProblem(
        name="Q14", performances=[7, 4, 14, 11, 3, 10, 9, 5],
        costs=[1, 1, 5, 6, 1, 4, 7, 1],
        budget=15, dependencies=[(0, 2), (0, 3), (4, 5)],
    ),  # N=8,  14 qubits, 13 Pareto pts,  5 unsupported, 3 deps
    "Q15": SDProblem(
        name="Q15", performances=[12, 9, 17, 14, 4, 11, 8, 9],
        costs=[7, 2, 6, 7, 1, 4, 7, 4],
        budget=23, dependencies=[(1, 4), (4, 5), (5, 7)],
    ),  # N=8,  15 qubits, 13 Pareto pts,  5 unsupported, 3 deps
    "Q16": SDProblem(
        name="Q16", performances=[10, 6, 9, 14, 16, 7, 9, 11, 15],
        costs=[4, 1, 4, 8, 8, 2, 3, 3, 7],
        budget=18, dependencies=[(2, 8), (3, 8), (4, 5)],
    ),  # N=9,  16 qubits, 13 Pareto pts,  5 unsupported, 3 deps
    "Q17": SDProblem(
        name="Q17", performances=[16, 15, 14, 15, 14, 19, 16, 24, 17, 13],
        costs=[5, 7, 8, 5, 4, 3, 8, 5, 3, 5],
        budget=27, dependencies=[(2, 3), (4, 9), (6, 7), (8, 9)],
    ),  # N=10, 17 qubits, 13 Pareto pts,  5 unsupported, 4 deps
    "Q18": SDProblem(
        name="Q18", performances=[11, 13, 9, 10, 6, 11, 10, 8, 11, 10, 11],
        costs=[10, 4, 5, 3, 1, 4, 5, 5, 8, 4, 3],
        budget=29, dependencies=[(1, 2), (4, 6), (4, 10), (5, 9)],
    ),  # N=11, 18 qubits, 15 Pareto pts,  6 unsupported, 4 deps
    "Q19": SDProblem(
        name="Q19", performances=[23, 16, 10, 12, 18, 3, 15, 9, 25, 14, 16],
        costs=[7, 6, 10, 10, 8, 1, 9, 2, 7, 9, 8],
        budget=48, dependencies=[(0, 1), (3, 5), (7, 8), (7, 9)],
    ),  # N=11, 19 qubits, 12 Pareto pts,  4 unsupported, 4 deps
    "Q20": SDProblem(
        name="Q20", performances=[10, 15, 11, 12, 12, 13, 13, 10, 15, 11, 15, 16],
        costs=[10, 7, 9, 7, 8, 3, 8, 3, 6, 3, 6, 7],
        budget=40, dependencies=[(0, 11), (1, 11), (9, 10), (10, 11)],
    ),  # N=12, 20 qubits, 15 Pareto pts,  6 unsupported, 4 deps
}


# ----------------------------------------------------------------------------
# Controlled dependency-count sensitivity study.
#
# This is NOT part of the eleven-problem benchmark set. It is a separate
# synthetic experiment: one base instance with performances, costs and budget
# held fixed, and only the dependency set varied over 2, 4 and 6 pairs, used
# to test whether the encoding comparison is sensitive to dependency density.
#
# It was originally drawn from a generated instance file under results/, which
# is gitignored and regenerable, so the experiment could not be reconstructed
# from the repository alone. The instance and all three dependency sets are
# therefore pinned here as literals. tests/test_dependency_study.py checks
# them against the stored campaign output.
# ----------------------------------------------------------------------------
DEP_STUDY_BASE = dict(
    performances=[23, 25, 22, 22, 17, 18, 15, 14, 16, 21, 19, 13],
    costs=[11, 10, 4, 10, 4, 6, 4, 4, 6, 4, 11, 3],
    budget=39,
)

DEP_STUDY_VARIANTS = {
    2: [(2, 8), (10, 11)],
    4: [(0, 1), (4, 11), (7, 8), (10, 11)],
    6: [(0, 1), (0, 11), (2, 10), (7, 8), (7, 10), (10, 11)],
}


def dependency_study_problems():
    """The three dependency-count variants as SDProblem objects.

    Reconstructs the controlled study exactly from pinned literals, with no
    dependence on any file under results/.
    """
    return {k: SDProblem(name=f"dep{k}", dependencies=list(v),
                         **DEP_STUDY_BASE)
            for k, v in sorted(DEP_STUDY_VARIANTS.items())}


def problem_registry(instances=None):
    """PROBLEMS (I-V) plus the pinned LADDER (Q13-Q20).

    Analysis and figure scripts should look problems up here rather than in
    PROBLEMS directly, so they work on both the I-V campaign and the ladder
    without a KeyError.

    Pass ``instances`` to overlay a generated JSON on top -- for evaluating a
    freshly searched ladder before pinning it. Left at None, the registry is
    fully determined by this file, so results cannot silently change when an
    untracked JSON in results/ is regenerated.
    """
    reg = dict(PROBLEMS)
    reg.update(LADDER)
    if instances and os.path.exists(instances):
        reg.update(load_instances(instances))
    return reg


def load_instances(path="results/instances.json"):
    """Load a generated instance ladder as {name: SDProblem}.

    Written by experiments/generate_instances.py and keyed by total circuit
    width, e.g. "Q20". These are searched for high Pareto non-convexity, so
    they are NOT generic instances -- see CLAUDE.md section 3.
    """
    with open(path) as f:
        data = json.load(f)
    out = {}
    for r in data["instances"]:
        name = f"Q{r['n_qubits']}"
        pb = SDProblem(name=name, performances=r["performances"],
                       costs=r["costs"], budget=r["budget"],
                       dependencies=[tuple(d) for d in r["dependencies"]])
        if pb.n_qubits != r["n_qubits"]:
            raise ValueError(
                f"{name}: stored width {r['n_qubits']} != recomputed "
                f"{pb.n_qubits}; instances.json is stale")
        out[name] = pb
    return out
