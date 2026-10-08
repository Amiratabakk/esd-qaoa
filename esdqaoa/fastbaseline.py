"""Fast exact simulator for the Pauli QAOA baseline.

The same observation that collapses ESD-QAOA to its decision register applies
to the baseline, for a different reason. Here there are no ancillas to
uncompute -- the slack bits are genuine variables -- but the cost operator is
a sum of Z and ZZ strings and is therefore diagonal in the computational
basis, while the mixer is Rx on every qubit. So one layer is

    exp(-i * gamma * E(z))  followed by  Rx(2*beta) on all n qubits

which is a diagonal phase plus a product mixer, exactly as in fastsim. E(z)
is the QUBO objective built by baseline._build_qubo.

This matters for fairness as much as for speed: both methods must get the
identical optimiser, surrogate, seed and depth, so if ESD-QAOA runs on an
exact fast simulator the baseline has to as well. Equivalence with the
Qiskit QAOAAnsatz path is asserted in tests/test_fastbaseline.py; the
parameter order and mixer convention are taken from that check rather than
assumed.
"""
import math
import numpy as np

from .baseline import _build_qubo
from .fastsim import _apply_mixer, _bit_matrix, objective_values
from .simulate import default_alpha, default_beta_dep


def qubo_diagonal(problem, phi, lam_b, lam_d, budget=None, slack_bits=None):
    """(E, n): the QUBO objective over all 2^n assignments, little-endian.

    Variables are x_0..x_{N-1} then the slack bits, matching _build_qubo.
    """
    Q, coeff, const, n = _build_qubo(problem, phi, lam_b, lam_d,
                                     budget, slack_bits)
    bits = _bit_matrix(n).astype(float)
    # x^T Q x with Q strictly upper triangular == sum_{v<w} Q[v,w] x_v x_w
    E = bits @ coeff + ((bits @ Q) * bits).sum(axis=1) + const
    return E, n


class FastPauliRunner:
    """Exact 2^n simulator for the QAOAAnsatz baseline.

    Parameter order follows Qiskit's QAOAAnsatz: all betas, then all gammas.
    """

    def __init__(self, problem, p, phi, lam_b=None, lam_d=None, budget=None,
                 slack_bits=None):
        self.problem, self.p, self.phi = problem, p, phi
        alpha, beta_dep = default_alpha(problem), default_beta_dep(problem)
        self.alpha, self.beta_dep = alpha, beta_dep
        lam_b = alpha if lam_b is None else lam_b
        lam_d = beta_dep if lam_d is None else lam_d
        self.budget = problem.budget if budget is None else budget

        self._E, self.n = qubo_diagonal(problem, phi, lam_b, lam_d,
                                        budget, slack_bits)
        self.N = problem.N
        self._value = objective_values(problem, phi, alpha, beta_dep, budget)
        self._keys = [tuple(int(b) for b in row)
                      for row in _bit_matrix(problem.N)]

    def marginal_probs_array(self, params):
        """Probabilities over decision assignments, slack bits summed out."""
        betas, gammas = params[:self.p], params[self.p:]
        dim = 2 ** self.n
        psi = np.full(dim, 1.0 / math.sqrt(dim), dtype=complex)
        for l in range(self.p):
            psi *= np.exp(-1j * gammas[l] * self._E)
            psi = _apply_mixer(psi, self.n, betas[l])
        pr = np.abs(psi) ** 2
        # decision variables are the low N bits, so slack bits are the
        # leading axis of this view and sum away directly
        return pr.reshape(2 ** (self.n - self.N), 2 ** self.N).sum(axis=0)

    def decision_probs(self, params):
        return {k: float(v) for k, v in
                zip(self._keys, self.marginal_probs_array(params))}

    def expectation(self, params):
        return float(self.marginal_probs_array(params) @ self._value)

    def gibbs(self, params, eta=5.0):
        pr = self.marginal_probs_array(params)
        m = self._value.max()
        s = float(pr @ np.exp(eta * (self._value - m)))
        return m + math.log(max(s, 1e-300)) / eta


def extend_params(x0, p):
    """Pad depth-(p-1) QAOAAnsatz parameters to depth p.

    Qiskit's layout is [beta_0..beta_{p-1}, gamma_0..gamma_{p-1}], so a new
    layer cannot simply be appended: the fresh beta belongs at the end of the
    beta block, not after the gammas. Appending naively would reinterpret
    every gamma as a beta and make the warm start actively harmful.
    """
    x0 = list(x0)
    prev_p = len(x0) // 2
    betas, gammas = x0[:prev_p], x0[prev_p:]
    while len(betas) < p:
        betas.append(0.1)
    while len(gammas) < p:
        gammas.append(0.1)
    return np.array(betas[:p] + gammas[:p])


def run_pauli_qaoa_fast(problem, p, phi, lam_b=None, lam_d=None, seed=7,
                        maxiter=80, popsize=20, eta=5.0, budget=None,
                        slack_bits=None, x0=None):
    """Drop-in replacement for baseline.run_pauli_qaoa using the fast path.

    ``x0`` warm-starts the optimizer (e.g. from depth p-1). Depth sweeps must
    pass it for BOTH methods or the comparison is biased: differential
    evolution degrades as the parameter count grows, so whichever method is
    left cold-started looks progressively worse with p for reasons that have
    nothing to do with its encoding.
    """
    from scipy.optimize import differential_evolution
    runner = FastPauliRunner(problem, p, phi, lam_b, lam_d, budget, slack_bits)
    bounds = [(0, math.pi)] * p + [(0, 4 * math.pi)] * p
    init = None
    if x0 is not None:
        init = np.clip(extend_params(x0, p),
                       [b[0] for b in bounds], [b[1] for b in bounds])
    result = differential_evolution(lambda prm: -runner.gibbs(prm, eta),
                                    bounds, seed=seed, maxiter=maxiter,
                                    popsize=popsize, tol=1e-9, polish=True,
                                    x0=init)
    return runner.decision_probs(result.x), result.x, -result.fun
