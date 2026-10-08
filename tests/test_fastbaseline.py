"""Verify FastPauliRunner reproduces the Qiskit QAOAAnsatz baseline exactly.

The baseline's cost operator is diagonal and its mixer is a product of Rx, so
a 2^n numpy simulation should be exact. Qiskit's parameter ordering and mixer
convention are assumptions until checked -- that is what this test is for. If
it passes, the baseline can use the fast path and both methods stay on
identical footing.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import Statevector

from esdqaoa.problems import PROBLEMS
from esdqaoa.baseline import _build_qubo, _qubo_to_ising
from esdqaoa.fastbaseline import FastPauliRunner
from esdqaoa.simulate import default_alpha, default_beta_dep

TOL = 1e-10
rng = np.random.default_rng(5)
all_ok, worst = True, 0.0


def qiskit_marginal(pb, p, phi, params):
    """The exact code path used by baseline.run_pauli_qaoa."""
    lam_b, lam_d = default_alpha(pb), default_beta_dep(pb)
    Q, coeff, const, n = _build_qubo(pb, phi, lam_b, lam_d)
    op, _ = _qubo_to_ising(Q, coeff, const, n)
    ansatz = QAOAAnsatz(cost_operator=op, reps=p).decompose(reps=3)
    sv = Statevector(ansatz.assign_parameters(params))
    probs = sv.probabilities_dict(qargs=list(range(pb.N)))
    out = {}
    for bitstr, pr in probs.items():
        x = tuple(int(b) for b in bitstr[::-1])
        out[x] = out.get(x, 0.0) + float(pr)
    return out


for name in ["I", "II", "III", "IV", "V"]:
    pb = PROBLEMS[name]
    ok, errs, t_qk, t_fast = True, [], 0.0, 0.0
    for p in (1, 2):
        for phi in (0.0, 1 / 3, 0.7, 1.0):
            fast = FastPauliRunner(pb, p, phi)
            for _ in range(2):
                params = rng.uniform(0.0, 2.0, size=2 * p)
                t0 = time.perf_counter()
                ref = qiskit_marginal(pb, p, phi, params)
                t_qk += time.perf_counter() - t0
                t0 = time.perf_counter()
                got = fast.decision_probs(params)
                t_fast += time.perf_counter() - t0
                d = max(abs(ref[x] - got[x]) for x in ref)
                errs.append(d)
                if d > TOL:
                    ok = False
    e = max(errs)
    worst = max(worst, e)
    all_ok &= ok
    print(f"Problem {name}: n_vars={FastPauliRunner(pb, 1, 0.5).n} "
          f"max_err={e:.2e} speedup={t_qk / max(t_fast, 1e-9):.0f}x "
          f"{'PASS' if ok else 'FAIL'}")

print(f"\noverall max error {worst:.2e} (tol {TOL:.0e}): "
      f"{'PASS' if all_ok else 'FAIL'}")
sys.exit(0 if all_ok else 1)
