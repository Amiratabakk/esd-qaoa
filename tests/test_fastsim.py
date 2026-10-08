"""Verify FastESDRunner reproduces the full (N+k+1)-qubit statevector exactly.

The ancilla registers A and F are uncomputed inside every layer, so the R
register evolves under a pure diagonal phase. If that holds, the 2^N
simulator is exact rather than approximate -- which is what licenses running
20-qubit-class instances without simulating the cost register.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from esdqaoa.problems import PROBLEMS
from esdqaoa.simulate import (ESDQAOARunner, penalized_objective,
                              default_alpha, default_beta_dep)
from esdqaoa.fastsim import FastESDRunner, objective_values, _bit_matrix

TOL = 1e-10
rng = np.random.default_rng(11)
worst = 0.0
all_ok = True

for name in ["I", "II", "III", "IV", "V"]:
    pb = PROBLEMS[name]
    alpha, beta_dep = default_alpha(pb), default_beta_dep(pb)
    ok, errs, t_full, t_fast = True, [], 0.0, 0.0

    # the vectorised objective table must equal the scalar reference exactly,
    # otherwise the optimiser surrogate would drift from the existing campaign
    for phi in (0.0, 1 / 3, 1.0):
        vec = objective_values(pb, phi, alpha, beta_dep)
        ref = np.array([penalized_objective(x, pb, phi, alpha, beta_dep)
                        for x in _bit_matrix(pb.N)], dtype=float)
        d = float(np.max(np.abs(vec - ref)))
        errs.append(d)
        if d > 0.0:
            ok = False
            print(f"  {name}: vectorised objective_values mismatch {d:.2e}")

    for p in (1, 2):
        for phi in (0.0, 1 / 3, 0.7, 1.0):
            full = ESDQAOARunner(pb, p, phi, alpha, beta_dep)
            fast = FastESDRunner(pb, p, phi, alpha, beta_dep)
            for _ in range(3):
                angles = rng.uniform(0.0, 2 * np.pi, size=2 * p)

                t0 = time.perf_counter()
                pf = full.decision_probs(angles)
                t_full += time.perf_counter() - t0
                t0 = time.perf_counter()
                pq = fast.decision_probs(angles)
                t_fast += time.perf_counter() - t0

                d = max(abs(pf[x] - pq[x]) for x in pf)
                # surrogates must agree too, not just the distribution
                d = max(d, abs(full.gibbs(angles) - fast.gibbs(angles)))
                d = max(d, abs(full.cvar(angles) - fast.cvar(angles)))
                d = max(d, abs(full.expectation(angles)
                               - fast.expectation(angles)))
                errs.append(d)
                if d > TOL:
                    ok = False

    e = max(errs)
    worst = max(worst, e)
    all_ok &= ok
    print(f"Problem {name}: N={pb.N} full_qubits={pb.n_qubits} "
          f"max_err={e:.2e} speedup={t_full / max(t_fast, 1e-9):.0f}x "
          f"{'PASS' if ok else 'FAIL'}")

print(f"\noverall max error {worst:.2e} (tol {TOL:.0e}): "
      f"{'PASS' if all_ok else 'FAIL'}")
sys.exit(0 if all_ok else 1)
