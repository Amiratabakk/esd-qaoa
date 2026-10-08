"""Verify the phase unitary acts diagonally as e^{-i*gamma*f_o(x)}."""
import sys, os, cmath, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from qiskit.quantum_info import Statevector
from esdqaoa.problems import PROBLEMS
from esdqaoa.circuits import build_esd_qaoa
from esdqaoa.simulate import penalized_objective

def check(pb, phi, gamma=0.137, alpha=3.0, beta_dep=4.0):
    qc, gammas, betas = build_esd_qaoa(pb, 1, phi, alpha, beta_dep,
                                       measure=False)
    sv = Statevector(qc.assign_parameters({gammas[0]: gamma, betas[0]: 0.0}))
    dim_anc = 2 ** (pb.k + 1)
    ok = True
    amps = sv.data.reshape(dim_anc, 2 ** pb.N)  # [anc, R] little-endian? check
    # qiskit ordering: qubit 0 = least significant. R are qubits 0..N-1.
    data = sv.data
    ref_phase = None
    for idx, amp in enumerate(data):
        if abs(amp) < 1e-9:
            continue
        r_bits = idx & (2**pb.N - 1)
        anc = idx >> pb.N
        assert anc == 0, f"ancilla not cleaned: state {idx:b}"
        x = np.array([(r_bits >> i) & 1 for i in range(pb.N)])
        # No offset correction: budget_penalty now applies alpha*(C-B)
        # exactly, matching penalized_objective on every configuration.
        f_o = penalized_objective(x, pb, phi, alpha, beta_dep)
        measured = cmath.phase(amp)
        expected = (-gamma * f_o) % (2 * math.pi)
        if ref_phase is None:
            ref_phase = (measured - (-gamma * f_o)) % (2 * math.pi)
        diff = (measured - (-gamma * f_o) - ref_phase) % (2 * math.pi)
        diff = min(diff, 2 * math.pi - diff)
        if diff > 1e-6:
            ok = False
            print(f"  MISMATCH x={x} f_o={f_o:.3f} diff={diff:.2e}")
    return ok

for name in ["I", "II", "III", "IV", "V"]:
    pb = PROBLEMS[name]
    res = all(check(pb, phi) for phi in [0.0, 0.3, 0.7, 1.0])
    print(f"Problem {name}: phase check {'PASS' if res else 'FAIL'}")
