"""Verify the epsilon-constraint mode of the ESD-QAOA circuit.

epsilon-constraint solves  max P(x)  s.t.  C(x) <= eps, swept over eps. In
this encoding that is just phi=1 with the budget register threshold set to
eps -- no new circuit structure, only a re-parameterisation of the existing
cost register and flag.

Three things must hold:
  1. default arguments reproduce the linear-scalarization circuit exactly
     (no regression on the existing campaign);
  2. with a pinned encoding the circuit WIDTH is constant across the sweep,
     so qubit counts are comparable from one eps to the next;
  3. the diagonal phase matches V(x) computed at that eps, i.e. the flag
     really tests C(x) <= eps and not C(x) <= B.
"""
import sys, os, cmath, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from qiskit.quantum_info import Statevector

from esdqaoa.problems import PROBLEMS
from esdqaoa.circuits import build_esd_qaoa, budget_encoding, pinned_encoding
from esdqaoa.fastsim import circuit_values
from esdqaoa.simulate import default_alpha, default_beta_dep

GAMMA = 0.137
# Penalty strengths must be the real ones: default_alpha = max(perf) + 1 is
# what guarantees an over-budget design can never out-score a feasible one.
# A weaker alpha lets a 1-unit overshoot buy more performance than it costs,
# and the argmax check below would fail for that reason alone.
ALPHA, BETA_DEP = 3.0, 4.0
all_ok = True


def diagonal_phase_error(pb, phi, budget, c_bits, k, alpha, beta_dep):
    """max phase error between the circuit and circuit_values() at `budget`."""
    qc, gammas, betas = build_esd_qaoa(pb, 1, phi, alpha, beta_dep,
                                       measure=False, budget=budget,
                                       c_bits=c_bits, k=k)
    sv = Statevector(qc.assign_parameters({gammas[0]: GAMMA, betas[0]: 0.0}))
    V = circuit_values(pb, phi, alpha, beta_dep, budget=budget)

    worst, ref = 0.0, None
    for idx, amp in enumerate(sv.data):
        if abs(amp) < 1e-9:
            continue
        r_bits = idx & (2 ** pb.N - 1)
        assert idx >> pb.N == 0, f"ancilla not uncomputed: |{idx:b}>"
        want = -GAMMA * V[r_bits]
        if ref is None:                       # fix the irrelevant global phase
            ref = (cmath.phase(amp) - want) % (2 * math.pi)
        d = (cmath.phase(amp) - want - ref) % (2 * math.pi)
        worst = max(worst, min(d, 2 * math.pi - d))
    return worst, qc.num_qubits


# --- 1. regression: defaults must equal the original budget-derived encoding
for name, pb in PROBLEMS.items():
    c, w0, k = budget_encoding(pb)
    ok = (c == pb.c_bits and w0 == pb.w0 and k == pb.k
          and build_esd_qaoa(pb, 1, 0.5, ALPHA, BETA_DEP,
                             measure=False)[0].num_qubits == pb.n_qubits)
    all_ok &= ok
    print(f"Problem {name}: default encoding (c={c},w0={w0},k={k}) "
          f"{'PASS' if ok else 'FAIL'}")

# --- 2 & 3. epsilon sweep with pinned width
print()
for name in ["II", "III", "IV", "V"]:
    pb = PROBLEMS[name]
    alpha, beta_dep = default_alpha(pb), default_beta_dep(pb)
    eps_values = list(range(0, int(pb.budget) + 1))
    c_bits, k = pinned_encoding(pb, eps_values)

    widths, worst = set(), 0.0
    for eps in eps_values:
        # phase correctness is independent of penalty strength, so the weak
        # ALPHA/BETA_DEP are fine (and a sharper probe) here
        err, nq = diagonal_phase_error(pb, 1.0, eps, c_bits, k,
                                       ALPHA, BETA_DEP)
        widths.add(nq)
        worst = max(worst, err)

    # the quantum eps-constraint optimum must match the classical one
    mism = []
    for eps in eps_values:
        V = circuit_values(pb, 1.0, alpha, beta_dep, budget=eps)
        got = int(np.argmax(V))
        x = np.array([(got >> i) & 1 for i in range(pb.N)])
        ref = pb.epsilon_constraint_optimum(eps)
        if ref is None:
            continue
        if (pb.cost(x), pb.perf(x)) != (ref[0], ref[1]):
            mism.append((eps, (pb.cost(x), pb.perf(x)), (ref[0], ref[1])))

    ok = len(widths) == 1 and worst < 1e-6 and not mism
    all_ok &= ok
    print(f"Problem {name}: eps in [0,{pb.budget}]  width={sorted(widths)} "
          f"(pinned c={c_bits},k={k})  max_phase_err={worst:.2e}  "
          f"argmax_mismatches={len(mism)}  {'PASS' if ok else 'FAIL'}")
    for m in mism[:3]:
        print(f"    eps={m[0]}: circuit {m[1]} vs classical {m[2]}")

print(f"\noverall: {'PASS' if all_ok else 'FAIL'}")
sys.exit(0 if all_ok else 1)
