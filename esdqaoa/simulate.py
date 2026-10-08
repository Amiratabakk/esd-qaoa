"""Noiseless statevector simulation and SHGO parameter optimization."""
from functools import partial
import math
import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector

from .circuits import build_esd_qaoa
from .problems import SDProblem


# ---------------------------------------------------------------------------
# Objective used by the classical optimizer (matches the circuit's penalties)
# ---------------------------------------------------------------------------
def penalized_objective(x, problem, phi, alpha, beta_dep):
    perf, cost = problem.perf(x), problem.cost(x)
    val = phi * perf - (1 - phi) * cost
    if cost > problem.budget:
        val -= alpha * (cost - problem.budget)
    n_viol = sum(1 for (i, j) in problem.dependencies
                 if x[j] == 1 and x[i] == 0)
    val -= beta_dep * n_viol
    return val


def default_alpha(problem):
    """Penalty slope guaranteeing infeasible < feasible optima."""
    return float(max(problem.performances)) + 1.0


def default_beta_dep(problem):
    return float(sum(problem.performances)) + 1.0


# ---------------------------------------------------------------------------
# ESD-QAOA expectation and optimization
# ---------------------------------------------------------------------------
class ESDQAOARunner:
    """Builds the circuit once; binds (gamma, beta) per evaluation."""

    def __init__(self, problem, p, phi, alpha, beta_dep):
        self.problem, self.p, self.phi = problem, p, phi
        self.alpha, self.beta_dep = alpha, beta_dep
        self.qc, self.gammas, self.betas = build_esd_qaoa(
            problem, p, phi, alpha, beta_dep, measure=False)
        # Precompute per-config penalized objective for fast expectation.
        self._values = {}
        for x in problem.all_configs():
            self._values[tuple(x)] = penalized_objective(
                x, problem, phi, alpha, beta_dep)

    def decision_probs(self, angles):
        binding = {}
        for l in range(self.p):
            binding[self.gammas[l]] = angles[2 * l]
            binding[self.betas[l]] = angles[2 * l + 1]
        sv = Statevector(self.qc.assign_parameters(binding))
        probs = sv.probabilities_dict(qargs=list(range(self.problem.N)))
        out = {}
        for bitstr, pr in probs.items():
            x = tuple(int(b) for b in bitstr[::-1])
            out[x] = out.get(x, 0.0) + float(pr)
        return out

    def expectation(self, angles):
        probs = self.decision_probs(angles)
        return sum(pr * self._values[x] for x, pr in probs.items())

    def gibbs(self, angles, eta=5.0):
        """Gibbs objective F_eta = eta^-1 * log <e^{eta f_o}> (maximization
        form; Li et al., Phys. Rev. Research 2, 023074, 2020). Rewards
        concentration of probability on high-objective states and is
        optimum-agnostic; used as the default classical surrogate for both
        ESD-QAOA and the Pauli baseline."""
        probs = self.decision_probs(angles)
        m = max(self._values.values())
        s = sum(pr * math.exp(eta * (self._values[x] - m))
                for x, pr in probs.items())
        return m + math.log(max(s, 1e-300)) / eta

    def cvar(self, angles, alpha_cvar=0.2):
        """CVaR_alpha of the penalized objective (maximization form):
        mean over the best alpha-quantile of the output distribution
        (Barkoutsos et al., Quantum 4, 256, 2020). Improves concentration
        for constrained problems; applied identically to both methods."""
        probs = self.decision_probs(angles)
        items = sorted(probs.items(), key=lambda kv: -self._values[kv[0]])
        acc_p, acc_v = 0.0, 0.0
        for x, pr in items:
            take = min(pr, alpha_cvar - acc_p)
            if take <= 0:
                break
            acc_v += take * self._values[x]
            acc_p += take
        return acc_v / max(acc_p, 1e-12)


def decision_probs(problem, p, phi, alpha, beta_dep, angles):
    """Probability dict over decision bitstrings (little-endian tuple keys)."""
    return ESDQAOARunner(problem, p, phi, alpha, beta_dep) \
        .decision_probs(angles)


def expectation(problem, p, phi, alpha, beta_dep, angles):
    return ESDQAOARunner(problem, p, phi, alpha, beta_dep).expectation(angles)


def optimize_angles(problem, p, phi, alpha, beta_dep,
                    gamma_max=4 * math.pi, objective="gibbs", eta=5.0,
                    cvar_alpha=0.2, seed=7, maxiter=80, popsize=20,
                    runner=None):
    """Maximize the chosen surrogate ("gibbs", "cvar", or "expectation")
    over (gamma_1, beta_1, ..., gamma_p, beta_p) with seeded differential
    evolution (reproducible)."""
    from scipy.optimize import differential_evolution
    runner = runner or ESDQAOARunner(problem, p, phi, alpha, beta_dep)
    if objective == "gibbs":
        fn = lambda a: -runner.gibbs(a, eta)
    elif objective == "cvar":
        fn = lambda a: -runner.cvar(a, cvar_alpha)
    else:
        fn = lambda a: -runner.expectation(a)
    bounds = [(0, gamma_max), (0, math.pi)] * p
    result = differential_evolution(fn, bounds, seed=seed, maxiter=maxiter,
                                    popsize=popsize, tol=1e-9, polish=True)
    return result.x, -result.fun


def run_esd_qaoa(problem, p, phi, alpha=None, beta_dep=None, backend="fast",
                 **kw):
    """Optimize parameters and return (probs, angles, objective_value).

    backend="fast" uses the ancilla-free 2^N simulator (fastsim), which is
    mathematically exact here -- the cost register and flag are uncomputed
    every layer, so they factor out. It agrees with the full circuit to
    ~1e-12 (tests/test_fastsim.py, tests/test_instance_q20.py) and is what
    makes 20-qubit instances tractable: at that width the full statevector
    takes ~17 s per evaluation versus well under a millisecond.

    backend="full" builds the real Qiskit circuit instead. Kept for
    verification and for anything that needs the actual gate sequence.
    """
    alpha = default_alpha(problem) if alpha is None else alpha
    beta_dep = default_beta_dep(problem) if beta_dep is None else beta_dep
    if backend == "fast":
        from .fastsim import FastESDRunner
        runner = FastESDRunner(problem, p, phi, alpha, beta_dep)
    elif backend == "full":
        runner = ESDQAOARunner(problem, p, phi, alpha, beta_dep)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    angles, obj_val = optimize_angles(problem, p, phi, alpha, beta_dep,
                                      runner=runner, **kw)
    probs = runner.decision_probs(angles)
    return probs, angles, obj_val


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def approximation_ratio(problem, probs, phi, shift_to_worst=True):
    """Shift-normalized approximation ratio.

    rho = (<f> - f_min) / (f_max - f_min) over the FEASIBLE range, with <f>
    the expected *unpenalized* scalarized objective and infeasible samples
    clamped to f_min. Well-defined for all phi (unlike <f>/f_max, which
    breaks when f_max <= 0, e.g. at phi = 0).
    """
    feas_vals = [problem.scalarized(x, phi) for x in problem.feasible_configs()]
    f_max, f_min = max(feas_vals), min(feas_vals)
    if math.isclose(f_max, f_min):
        return 1.0
    exp_f = 0.0
    for x, pr in probs.items():
        xa = np.array(x)
        v = problem.scalarized(xa, phi) if problem.is_feasible(xa) else f_min
        v = max(v, f_min)
        exp_f += pr * v
    return (exp_f - f_min) / (f_max - f_min)


def epsilon_approximation_ratio(problem, probs, eps):
    """Shift-normalized approximation ratio for the epsilon-constraint mode.

    rho_eps = (<P> - P_min) / (P_max - P_min) over the designs satisfying
    C(x) <= eps and the dependency constraints, with violators clamped to
    P_min. This is the direct analogue of approximation_ratio(): same
    shift-normalization, with performance replacing the scalarized objective
    and the eps bound replacing the fixed budget, so the two sweeps produce
    comparable numbers.
    """
    vals = [problem.perf(x) for x in problem.all_configs()
            if problem.is_eps_feasible(x, eps)]
    if not vals:
        return 0.0
    p_max, p_min = max(vals), min(vals)
    if math.isclose(p_max, p_min):
        return 1.0
    exp_p = 0.0
    for x, pr in probs.items():
        xa = np.array(x)
        v = problem.perf(xa) if problem.is_eps_feasible(xa, eps) else p_min
        exp_p += pr * max(v, p_min)
    return (exp_p - p_min) / (p_max - p_min)


def epsilon_prob_of_optimum(problem, probs, eps):
    """Probability mass on an optimal design of max P s.t. C(x) <= eps."""
    best = problem.epsilon_constraint_optimum(eps)
    if best is None:
        return 0.0
    total = 0.0
    for x, pr in probs.items():
        xa = np.array(x)
        if problem.is_eps_feasible(xa, eps) and problem.perf(xa) == best[1]:
            total += pr
    return total


def prob_of_optimum(problem, probs, phi):
    x_opt, v_opt = problem.brute_force_optimum(phi)
    total = 0.0
    for x, pr in probs.items():
        xa = np.array(x)
        if problem.is_feasible(xa) and math.isclose(
                problem.scalarized(xa, phi), v_opt):
            total += pr
    return total
