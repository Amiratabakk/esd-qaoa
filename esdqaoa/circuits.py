"""ESD-QAOA circuit construction.

Registers (paper notation):
  R : N decision qubits (one per subsystem)
  A : k cost-register qubits
  F : 1 budget-violation flag qubit

Cost register holds C(x) + w0 with offset w0 = 2^c - B - 1 where
c = floor(log2(B)) + 1, so that C(x) > B  <=>  some bit at position >= c is 1.
Register width k = max(ceil(log2(C_max + w0 + 1)), c + 1) guarantees no
modular wrap-around and at least one overflow-detection bit.

One QAOA layer:
  U_f(gamma)                       objective phase (single-qubit P gates)
  U_TB                             QFT add w0 + sum c_i x_i, IQFT, flag test
  U_pen_budget(gamma)              flag-controlled overshoot dephasing
  U_pen_dep(gamma)                 per-pair controlled-phase dephasing on R
  U_TB^dagger                      uncompute flag + arithmetic
  U_M(beta)                        Rx mixing on R

Dependency convention: (i, j) in problem.dependencies means x_j <= x_i;
a violation (x_j = 1, x_i = 0) receives phase gamma * beta_dep, summed over
violated pairs. This per-pair dephasing implements the paper's
P_dependency(x) = -beta * sum_{(i,j)} x_j (1 - x_i) exactly and additively,
avoiding the parity (XOR) artifact of accumulating multiple violations on a
single shared flag qubit.
"""
import math
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter


def budget_encoding(problem, budget=None, c_bits=None, k=None):
    """Threshold encoding (c, w0, k) for the constraint C(x) <= budget.

    Register A holds C(x) + w0 with w0 = 2^c - budget - 1, so that
    C(x) > budget iff some bit at position >= c is set.

    Pass ``c_bits`` (and ``k``) to PIN the comparison threshold and register
    width across a sweep of budgets. That is required by the
    epsilon-constraint method, where budget is the swept parameter and the
    qubit count must not change with epsilon. Defaults reproduce the
    original budget-derived encoding exactly.
    """
    b = int(problem.budget if budget is None else budget)
    if b < 0:
        raise ValueError(f"budget must be non-negative, got {b}")
    if c_bits is None:
        c = 1 if b == 0 else math.floor(math.log2(b)) + 1
    else:
        c = int(c_bits)
    if b >= 2**c:
        raise ValueError(f"budget {b} does not fit in c_bits={c}")
    w0 = 2**c - b - 1
    k_min = max(math.ceil(math.log2(problem.c_max + w0 + 1)), c + 1)
    if k is None:
        return c, w0, k_min
    if int(k) < k_min:
        raise ValueError(f"k={k} too narrow; needs >= {k_min}")
    return c, w0, int(k)


def pinned_encoding(problem, eps_values):
    """(c_bits, k) wide enough for every epsilon in an eps-constraint sweep.

    w0 grows as epsilon shrinks, so the widest register is needed at the
    smallest epsilon; both are held fixed so the circuit width is constant
    over the sweep.
    """
    c = max(budget_encoding(problem, e)[0] for e in eps_values)
    k = max(budget_encoding(problem, e, c_bits=c)[2] for e in eps_values)
    return c, k


def register_width(problem, budget=None, c_bits=None):
    """Width of the cost register A (see module docstring)."""
    return budget_encoding(problem, budget, c_bits)[2]


# ---------------------------------------------------------------------------
# QFT-based arithmetic (Draper adder)
# ---------------------------------------------------------------------------
def qft(circuit, reg):
    for idx in reversed(range(len(reg))):
        circuit.h(reg[idx])
        for c_idx in reversed(range(idx)):
            k = idx - c_idx + 1
            circuit.cp(2 * np.pi / 2**k, reg[idx], reg[c_idx])


def iqft(circuit, reg):
    for idx in range(len(reg)):
        for c_idx in range(idx):
            k = idx - c_idx + 1
            circuit.cp(-2 * np.pi / 2**k, reg[idx], reg[c_idx])
        circuit.h(reg[idx])


def add_const(circuit, reg, n, control=None):
    """Add integer n (mod 2^len(reg)) to a Fourier-basis register."""
    binary = list(map(int, reversed(bin(n)[2:])))
    for k, bit in enumerate(binary):
        if not bit:
            continue
        for idx in range(len(reg)):
            l = idx + 1
            if l > k:
                angle = 2 * np.pi / 2 ** (l - k)
                if control is None:
                    circuit.p(angle, reg[idx])
                else:
                    circuit.cp(angle, control, reg[idx])


# ---------------------------------------------------------------------------
# Layer subroutines
# ---------------------------------------------------------------------------
def objective_phase(circuit, r_reg, problem, phi, gamma, eps=0.0):
    """U_f: parallel phase gates encoding f(x) = phi*p_i - (1-phi)*c_i."""
    for i in range(problem.N):
        weight = phi * float(problem.performances[i]) \
            - (1 - phi) * float(problem.costs[i]) + eps
        circuit.p(-gamma * weight, r_reg[i])


def cost_accumulate(circuit, r_reg, a_reg, problem, inverse=False, enc=None):
    """Load (or unload) C(x) + w0 into register A."""
    _, w0, _ = enc or budget_encoding(problem)
    k = len(a_reg)
    qft(circuit, a_reg)
    if not inverse:
        if w0 > 0:
            add_const(circuit, a_reg, w0)
        for i in range(problem.N):
            add_const(circuit, a_reg, int(problem.costs[i]), control=r_reg[i])
    else:
        for i in reversed(range(problem.N)):
            add_const(circuit, a_reg, 2**k - int(problem.costs[i]),
                      control=r_reg[i])
        if w0 > 0:
            add_const(circuit, a_reg, 2**k - w0)
    iqft(circuit, a_reg)


def budget_flag(circuit, a_reg, f_reg, problem, enc=None):
    """Toggle F so that F = 1 iff C(x) > B (self-inverse).

    Assumes F enters as |1>; the multi-controlled X on the inverted
    high-order bits resets it to |0> exactly when all high bits are zero
    (budget satisfied).
    """
    c, _, _ = enc or budget_encoding(problem)
    high = [a_reg[l] for l in range(c, len(a_reg))]
    for q in high:
        circuit.x(q)
    circuit.mcx(high, f_reg[0])
    for q in high:
        circuit.x(q)


def budget_penalty(circuit, a_reg, f_reg, problem, alpha, gamma, enc=None):
    """Flag-controlled dephasing by the budget overshoot alpha * (C(x) - B).

    Register A holds C(x) + w0, so sum_l 2^l A[l] = C(x) + w0. Subtracting
    (2^c - 1) leaves exactly C(x) - B, since w0 = 2^c - B - 1.

    The subtracted constant is (2^c - 1), NOT 2^c: using 2^c would penalise
    alpha * (C - B - 1), which is zero when C = B + 1 and would therefore let
    a design that overshoots the budget by exactly one unit escape unpenalised
    -- the circuit would enforce C <= B + 1 rather than C <= B. See
    tests/test_epsilon.py, which pins this down.
    """
    c, _, _ = enc or budget_encoding(problem)
    for l in range(len(a_reg)):
        circuit.cp(2**l * gamma * alpha, f_reg[0], a_reg[l])
    circuit.p(-(2**c - 1) * gamma * alpha, f_reg[0])


def dependency_penalty(circuit, r_reg, problem, beta_dep, gamma):
    """Per-pair dephasing: phase gamma*beta_dep when x_j = 1 and x_i = 0."""
    for (i, j) in problem.dependencies:
        circuit.x(r_reg[i])
        circuit.cp(gamma * beta_dep, r_reg[j], r_reg[i])
        circuit.x(r_reg[i])


def mixing(circuit, r_reg, beta):
    for i in range(len(r_reg)):
        circuit.rx(2 * beta, r_reg[i])


# ---------------------------------------------------------------------------
# Full circuit
# ---------------------------------------------------------------------------
def build_esd_qaoa(problem, p, phi, alpha, beta_dep, measure=True, eps=0.0,
                   budget=None, c_bits=None, k=None):
    """Build the p-layer ESD-QAOA circuit with symbolic (gamma_l, beta_l).

    Returns (circuit, gammas, betas). Decision qubits are R[0..N-1]
    (the lowest N bits of the measurement bitstring).

    EPSILON-CONSTRAINT MODE: pass phi=1 (pure performance objective) and
    ``budget=eps`` to realise max P(x) s.t. C(x) <= eps. Pin ``c_bits`` and
    ``k`` (see pinned_encoding) so the width stays constant across the eps
    sweep. Defaults reproduce the linear-scalarization circuit exactly.
    """
    enc = budget_encoding(problem, budget, c_bits, k)
    k = enc[2]
    r_reg = QuantumRegister(problem.N, "R")
    a_reg = QuantumRegister(k, "A")
    f_reg = QuantumRegister(1, "F")
    qc = QuantumCircuit(r_reg, a_reg, f_reg)

    gammas = [Parameter(f"gamma_{l}") for l in range(p)]
    betas = [Parameter(f"beta_{l}") for l in range(p)]

    qc.h(r_reg)
    for l in range(p):
        # objective phases
        objective_phase(qc, r_reg, problem, phi, gammas[l], eps)
        # compute C(x) + w0 and budget flag (F enters |0>, X sets it to |1>)
        qc.x(f_reg[0])
        cost_accumulate(qc, r_reg, a_reg, problem, enc=enc)
        budget_flag(qc, a_reg, f_reg, problem, enc=enc)
        # penalty dephasing
        budget_penalty(qc, a_reg, f_reg, problem, alpha, gammas[l], enc=enc)
        dependency_penalty(qc, r_reg, problem, beta_dep, gammas[l])
        # uncompute (flag test is self-inverse; then reverse arithmetic and X)
        budget_flag(qc, a_reg, f_reg, problem, enc=enc)
        cost_accumulate(qc, r_reg, a_reg, problem, inverse=True, enc=enc)
        qc.x(f_reg[0])
        # mixing
        mixing(qc, r_reg, betas[l])

    if measure:
        c_reg = ClassicalRegister(problem.N, "meas")
        qc.add_register(c_reg)
        qc.measure(r_reg, c_reg)
    return qc, gammas, betas
