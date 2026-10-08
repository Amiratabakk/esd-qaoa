"""Pauli Hamiltonian QAOA baseline (standard Qiskit implementation).

The scalarized objective with quadratic constraint penalties is converted to
a QUBO, then to an Ising Hamiltonian (weighted Pauli-Z strings), and run
through qiskit.circuit.library.QAOAAnsatz — the standard, structure-agnostic
QAOA pipeline. Both methods are optimized with the same SHGO settings, the
same depth p, and matched penalty strengths so the comparison isolates the
encoding, not the optimizer.

QUBO construction (maximization -> minimize -Q):
  f(x)      = phi * sum p_i x_i - (1 - phi) * sum c_i x_i
  budget    = lam_b * (sum c_i x_i - B - s)^2 slack-free quadratic penalty
              approximated by the standard unbalanced penalty
              lam_b * max(0, C - B)^2 expanded exactly over binary products
              is not representable as QUBO without slack variables, so the
              conventional slack-register QUBO is used.
  dependency= lam_d * x_j (1 - x_i) per pair (exact QUBO term).

To keep qubit counts comparable to ESD-QAOA, the budget slack register uses
ceil(log2(B + 1)) binary slack variables (standard knapsack QUBO encoding).
"""
import math
import numpy as np
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp, Statevector

from .simulate import penalized_objective, default_alpha, default_beta_dep


def _build_qubo(problem, phi, lam_b, lam_d, budget=None, slack_bits=None):
    """Return (Q, linear, const, n_vars) for minimization of -f + penalties.

    Variables: x_0..x_{N-1}, then slack bits s_0..s_{m-1} encoding
    S = sum 2^l s_l with constraint C(x) + S = B enforced softly by
    lam_b * (C(x) + S - B)^2.

    ``budget`` overrides problem.budget, which is how the baseline is run in
    epsilon-constraint mode (phi=1, budget=eps). Pass ``slack_bits`` to pin
    the slack register width across an eps sweep -- otherwise m shrinks with
    eps and the qubit count changes from one sweep point to the next.
    """
    N = problem.N
    b = problem.budget if budget is None else int(budget)
    m = math.ceil(math.log2(b + 1)) if b > 0 else 1
    if slack_bits is not None:
        if int(slack_bits) < m:
            raise ValueError(f"slack_bits={slack_bits} too narrow; needs >= {m}")
        m = int(slack_bits)
    n = N + m
    coeff = np.zeros(n)              # linear terms
    Q = np.zeros((n, n))             # quadratic (upper triangle)
    const = 0.0

    # objective (minimize negative)
    for i in range(N):
        coeff[i] += -(phi * problem.performances[i]
                      - (1 - phi) * problem.costs[i])

    # dependency penalties: lam_d * x_j (1 - x_i) = lam_d x_j - lam_d x_i x_j
    for (i, j) in problem.dependencies:
        coeff[j] += lam_d
        a, b = min(i, j), max(i, j)
        Q[a, b] += -lam_d

    # budget penalty: lam_b * (sum a_v z_v - B)^2 with a = costs + slack powers
    a = np.zeros(n)
    a[:N] = problem.costs
    for l in range(m):
        a[N + l] = 2**l
    const += lam_b * b**2
    for v in range(n):
        coeff[v] += lam_b * (a[v]**2 - 2 * b * a[v])
        for w in range(v + 1, n):
            Q[v, w] += 2 * lam_b * a[v] * a[w]
    return Q, coeff, const, n


def _qubo_to_ising(Q, coeff, const, n):
    """Map x_v = (1 - z_v) / 2 to a SparsePauliOp over n qubits."""
    h = np.zeros(n)
    J = {}
    offset = const
    for v in range(n):
        offset += coeff[v] / 2
        h[v] += -coeff[v] / 2
    for v in range(n):
        for w in range(v + 1, n):
            q = Q[v, w]
            if q == 0:
                continue
            offset += q / 4
            h[v] += -q / 4
            h[w] += -q / 4
            J[(v, w)] = J.get((v, w), 0.0) + q / 4
    terms, coeffs = [], []
    for v in range(n):
        if h[v] != 0:
            label = ["I"] * n
            label[n - 1 - v] = "Z"
            terms.append("".join(label))
            coeffs.append(h[v])
    for (v, w), val in J.items():
        if val != 0:
            label = ["I"] * n
            label[n - 1 - v] = "Z"
            label[n - 1 - w] = "Z"
            terms.append("".join(label))
            coeffs.append(val)
    op = SparsePauliOp(terms, coeffs=np.array(coeffs, dtype=float))
    return op, offset


def run_pauli_qaoa(problem, p, phi, lam_b=None, lam_d=None,
                   shgo_iters=10):
    """Optimize the Pauli QAOA baseline; return (probs, angles, exp_val).

    probs is a dict over decision-variable tuples (slack bits marginalized).
    exp_val is the expectation of the SAME penalized objective used for
    ESD-QAOA, evaluated on the marginal decision distribution, so the two
    methods are scored identically.
    """
    alpha = default_alpha(problem)
    beta_dep = default_beta_dep(problem)
    lam_b = alpha if lam_b is None else lam_b
    lam_d = beta_dep if lam_d is None else lam_d

    Q, coeff, const, n = _build_qubo(problem, phi, lam_b, lam_d)
    op, offset = _qubo_to_ising(Q, coeff, const, n)
    ansatz = QAOAAnsatz(cost_operator=op, reps=p)
    ansatz = ansatz.decompose(reps=3)

    n_params = ansatz.num_parameters

    def marginal_probs(params):
        sv = Statevector(ansatz.assign_parameters(params))
        probs = sv.probabilities_dict(qargs=list(range(problem.N)))
        out = {}
        for bitstr, pr in probs.items():
            x = tuple(int(b) for b in bitstr[::-1])
            out[x] = out.get(x, 0.0) + float(pr)
        return out

    values = {}
    import itertools
    for bits in itertools.product([0, 1], repeat=problem.N):
        import numpy as _np
        x = _np.array(bits)
        values[bits] = penalized_objective(x, problem, phi, alpha, beta_dep)

    def gibbs(params, eta=5.0):
        probs = marginal_probs(params)
        m = max(values.values())
        s = sum(pr * math.exp(eta * (values[x] - m))
                for x, pr in probs.items())
        return m + math.log(max(s, 1e-300)) / eta

    # QAOAAnsatz parameter order: betas then gammas (Qiskit convention).
    from scipy.optimize import differential_evolution
    bounds = [(0, math.pi)] * (n_params // 2) + \
             [(0, 4 * math.pi)] * (n_params - n_params // 2)
    result = differential_evolution(lambda prm: -gibbs(prm), bounds,
                                    seed=7, maxiter=80, popsize=20,
                                    tol=1e-9, polish=True)
    probs = marginal_probs(result.x)
    return probs, result.x, -result.fun
