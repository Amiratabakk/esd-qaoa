"""Exact N-qubit simulator for ESD-QAOA (ancilla-free reformulation).

Within every layer the cost register A and flag F are computed and then
uncomputed (see circuits.build_esd_qaoa). For each decision basis state |x>
the ancillas therefore return to |0> and the whole block acts on R as a pure
diagonal phase:

    U_phase |x>|0>_A |0>_F = exp(-i * gamma * V(x)) |x>|0>_A |0>_F

with V(x) read directly off the gate angles in circuits.py:

    V(x) = phi*P(x) - (1-phi)*C(x) + eps*sum(x)          objective_phase
           - alpha * (C(x) - B) * [C(x) > B]             budget_penalty
           - beta_dep * n_violations(x)                  dependency_penalty

The mixer acts only on R. The dynamics restricted to R are consequently
*exact*, not an approximation, at cost 2^N instead of 2^(N+k+1). This is what
makes 20-qubit-class instances tractable: the ancilla width k drops out.

V(x) equals simulate.penalized_objective exactly, so the classical surrogate
the optimiser maximises and the phases the circuit actually applies are the
same function. (Before the budget-penalty fix they differed by alpha on
over-budget designs, which let a design overshooting by exactly one unit go
unpenalised.) Verified against the full circuit to ~1e-12 by
tests/test_fastsim.py.
"""
import math
import numpy as np


def _bit_matrix(N):
    """(2^N, N) array of decision bits; row index has bit i == x_i."""
    idx = np.arange(2 ** N, dtype=np.uint32)
    return ((idx[:, None] >> np.arange(N)[None, :]) & 1).astype(np.int8)


def _scalarized_and_cost(problem, phi, eps, bits):
    perf = bits @ problem.performances.astype(float)
    cost = bits @ problem.costs.astype(float)
    val = phi * perf - (1 - phi) * cost
    if eps:
        val = val + eps * bits.sum(axis=1)
    return val, cost


def _dependency_deduction(problem, beta_dep, bits):
    d = np.zeros(bits.shape[0])
    for (i, j) in problem.dependencies:
        d += beta_dep * (bits[:, j] & (1 - bits[:, i]))
    return d


def circuit_values(problem, phi, alpha, beta_dep, eps=0.0, budget=None):
    """V(x) as implemented by the circuit, indexed little-endian.

    Reproduces the gate angles in circuits.py exactly; the layer applies
    exp(-i * gamma * V(x)).

    ``budget`` overrides problem.budget, which is how the epsilon-constraint
    method is realised (phi=1, budget=eps). Note ``eps`` here is the
    per-qubit objective offset from circuits.objective_phase -- NOT the
    epsilon-constraint bound, which is ``budget``.
    """
    b = problem.budget if budget is None else budget
    bits = _bit_matrix(problem.N)
    val, cost = _scalarized_and_cost(problem, phi, eps, bits)
    over = cost > b
    val[over] -= alpha * (cost[over] - b)
    return val - _dependency_deduction(problem, beta_dep, bits)


def objective_values(problem, phi, alpha, beta_dep, budget=None):
    """Vectorised simulate.penalized_objective over all configs.

    Identical to circuit_values() at eps=0: since the budget-penalty fix the
    circuit implements the penalized objective exactly, so the optimiser
    surrogate and the circuit phases can no longer drift apart.
    tests/test_fastsim.py asserts this against the scalar reference.
    """
    return circuit_values(problem, phi, alpha, beta_dep, 0.0, budget)


def _apply_mixer(psi, N, beta, scratch=None):
    """Rx(2*beta) on every qubit: cos(beta) I - i sin(beta) X.

    Written with two reusable scratch buffers and in-place numpy ops. The
    naive version allocates six temporaries per qubit, which at n=18 (the
    Pauli baseline's width on the 20-qubit instance) costs ~10x more than
    the arithmetic itself.
    """
    c, s = math.cos(beta), -1j * math.sin(beta)
    half = psi.size // 2
    if scratch is None:
        scratch = (np.empty(half, dtype=psi.dtype),
                   np.empty(half, dtype=psi.dtype))
    tb, ub = scratch
    for q in range(N):
        # axis 1 of this view is bit q (index is little-endian)
        v = psi.reshape(-1, 2, 2 ** q)
        a0, a1 = v[:, 0, :], v[:, 1, :]
        t, u = tb.reshape(a0.shape), ub.reshape(a0.shape)
        np.multiply(a1, s, out=u)
        np.multiply(a0, c, out=t)
        t += u                      # t = c*a0 + s*a1   (new a0)
        np.multiply(a0, s, out=u)   # a0 still holds the original values
        np.multiply(a1, c, out=a1)
        a1 += u                     # a1 = c*a1 + s*a0  (new a1)
        a0[...] = t
    return psi


class FastESDRunner:
    """Drop-in replacement for simulate.ESDQAOARunner using 2^N amplitudes.

    Exposes the same decision_probs / expectation / gibbs / cvar API, so it
    can be handed to simulate.optimize_angles via its ``runner`` argument.
    """

    def __init__(self, problem, p, phi, alpha, beta_dep, eps=0.0, budget=None):
        self.problem, self.p, self.phi = problem, p, phi
        self.alpha, self.beta_dep = alpha, beta_dep
        self.budget = problem.budget if budget is None else budget
        self.N = problem.N
        self._phase = circuit_values(problem, phi, alpha, beta_dep, eps, budget)
        self._value = objective_values(problem, phi, alpha, beta_dep, budget)
        self._keys = [tuple(int(b) for b in row)
                      for row in _bit_matrix(problem.N)]

    def probabilities(self, angles):
        """Raw probability vector over the 2^N decision states."""
        dim = 2 ** self.N
        psi = np.full(dim, 1.0 / math.sqrt(dim), dtype=complex)
        for l in range(self.p):
            gamma, beta = angles[2 * l], angles[2 * l + 1]
            psi *= np.exp(-1j * gamma * self._phase)
            psi = _apply_mixer(psi, self.N, beta)
        return np.abs(psi) ** 2

    def decision_probs(self, angles):
        pr = self.probabilities(angles)
        return {k: float(v) for k, v in zip(self._keys, pr)}

    def expectation(self, angles):
        return float(self.probabilities(angles) @ self._value)

    def gibbs(self, angles, eta=5.0):
        """Gibbs objective F_eta = eta^-1 log <e^{eta f_o}> (maximization)."""
        pr = self.probabilities(angles)
        m = self._value.max()
        s = float(pr @ np.exp(eta * (self._value - m)))
        return m + math.log(max(s, 1e-300)) / eta

    def cvar(self, angles, alpha_cvar=0.2):
        """CVaR_alpha of the penalized objective (maximization form)."""
        pr = self.probabilities(angles)
        order = np.argsort(-self._value)
        p_sorted, v_sorted = pr[order], self._value[order]
        take = np.minimum(p_sorted, np.maximum(
            alpha_cvar - np.concatenate(([0.0], np.cumsum(p_sorted)[:-1])), 0.0))
        acc_p = take.sum()
        return float(take @ v_sorted / max(acc_p, 1e-12))
