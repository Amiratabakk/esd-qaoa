"""Independent verification of the generated 20-qubit instance.

The generator's own accounting is not evidence. This re-derives everything
from scratch with naive brute force, and -- the load-bearing check -- runs
the FULL 20-qubit circuit once to confirm the ancilla-free fast simulator
is still exact at this width. Every downstream 20-qubit result depends on
that equivalence.
"""
import sys, os, cmath, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from qiskit.quantum_info import Statevector

from esdqaoa.problems import LADDER
from esdqaoa.circuits import build_esd_qaoa
from esdqaoa.fastsim import circuit_values, FastESDRunner
from esdqaoa.simulate import default_alpha, default_beta_dep

# Read the PINNED instance, not results/instances.json: that file is
# gitignored and regenerable, so a test reading it would break on a clean
# checkout and would silently verify a different problem after any re-search.
pb = LADDER["Q20"]
ok = True
print(f"instance: N={pb.N} k={pb.k} qubits={pb.n_qubits} B={pb.budget}")
print(f"  costs={list(map(int, pb.costs))}")
print(f"  perfs={list(map(int, pb.performances))}")
print(f"  deps ={pb.dependencies}   (i,j) means x_j <= x_i\n")

# --- 1. width is really 20
ok &= (pb.n_qubits == 20)
print(f"[1] qubit count 20: {'PASS' if pb.n_qubits == 20 else 'FAIL'}")

# --- 2. naive O(n^2) dominance check reproduces the fast Pareto front
feas = [(pb.cost(x), pb.perf(x)) for x in pb.feasible_configs()]
naive = set()
for c, p in feas:
    if not any((c2 <= c and p2 >= p) and (c2 < c or p2 > p) for c2, p2 in feas):
        naive.add((c, p))
fast = {(c, p) for c, p, _ in pb.pareto_front()}
ok &= (naive == fast)
print(f"[2] naive vs vectorised front: {len(naive)} vs {len(fast)} "
      f"{'PASS' if naive == fast else 'FAIL'}")

# --- 3. unsupported points really are unreachable by any weighted sum
sup = set(pb.supported_pareto_points())
unsup = fast - sup
bad = []
for phi in np.linspace(0, 1, 5001):          # dense independent phi sweep
    x, _ = pb.brute_force_optimum(float(phi))
    hit = (pb.cost(x), pb.perf(x))
    if hit in unsup:
        bad.append((phi, hit))
ok &= not bad
print(f"[3] {len(unsup)}/{len(fast)} unsupported, none hit by 5001-point "
      f"phi sweep: {'PASS' if not bad else 'FAIL ' + str(bad[:3])}")

# --- 4. eps-constraint recovers the whole front
cov = pb.epsilon_constraint_reachable() & fast
ok &= (cov == fast)
print(f"[4] eps-constraint coverage {len(cov)}/{len(fast)}: "
      f"{'PASS' if cov == fast else 'FAIL'}")

# --- 5. THE LOAD-BEARING CHECK: full 20-qubit circuit == fast simulator
alpha, beta_dep = default_alpha(pb), default_beta_dep(pb)
gamma, phi = 0.113, 1 / 3
t0 = time.time()
qc, gammas, betas = build_esd_qaoa(pb, 1, phi, alpha, beta_dep, measure=False)
sv = Statevector(qc.assign_parameters({gammas[0]: gamma, betas[0]: 0.0}))
t_full = time.time() - t0
print(f"\n[5] full circuit: {qc.num_qubits} qubits, {qc.size()} gates, "
      f"{t_full:.1f}s to simulate")

V = circuit_values(pb, phi, alpha, beta_dep)
worst, ref, leaked = 0.0, None, 0
for idx, amp in enumerate(sv.data):
    if abs(amp) < 1e-9:
        continue
    if idx >> pb.N:                      # amplitude outside ancilla |0>
        leaked += 1
        continue
    want = -gamma * V[idx & (2 ** pb.N - 1)]
    if ref is None:
        ref = (cmath.phase(amp) - want) % (2 * math.pi)
    d = (cmath.phase(amp) - want - ref) % (2 * math.pi)
    worst = max(worst, min(d, 2 * math.pi - d))
ok &= (leaked == 0 and worst < 1e-6)
print(f"    ancillas uncomputed: {leaked} leaked amplitudes")
print(f"    max phase error vs fastsim: {worst:.2e} "
      f"{'PASS' if leaked == 0 and worst < 1e-6 else 'FAIL'}")

# --- 6. probabilities agree end to end at p=2
angles = [0.7, 0.4, 1.1, 0.9]
t0 = time.time()
qc2, g2, b2 = build_esd_qaoa(pb, 2, phi, alpha, beta_dep, measure=False)
sv2 = Statevector(qc2.assign_parameters(
    {g2[0]: angles[0], b2[0]: angles[1], g2[1]: angles[2], b2[1]: angles[3]}))
full_probs = sv2.probabilities_dict(qargs=list(range(pb.N)))
t_full2 = time.time() - t0
t0 = time.time()
fastp = FastESDRunner(pb, 2, phi, alpha, beta_dep).decision_probs(angles)
t_fast2 = time.time() - t0
err = max(abs(pr - fastp[tuple(int(c) for c in bs[::-1])])
          for bs, pr in full_probs.items())
ok &= (err < 1e-9)
print(f"[6] p=2 probabilities: max|dP|={err:.2e}  "
      f"full={t_full2:.1f}s fast={t_fast2:.4f}s "
      f"({t_full2 / max(t_fast2, 1e-9):.0f}x)  "
      f"{'PASS' if err < 1e-9 else 'FAIL'}")

print(f"\noverall: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
