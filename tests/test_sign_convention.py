"""P0.2 — which sign does the implemented cost unitary actually realise?

The manuscript writes

    H_C |x> = -f_o(x) |x>                           (eq:HC)
    U_C     = exp(-i gamma H_C)                     (eq:qaoa_ansatz)

so the implemented layer should multiply |x> by exp(+i gamma f_o(x)).

This test measures the phase directly. The decision register is prepared in a
uniform superposition, one cost-unitary layer is applied with the mixer angle
set to zero, and the resulting amplitude of each basis state is compared with
the analytic prediction. The global phase is divided out using one reference
state, so only the RELATIVE phases -- which are what the algorithm uses -- are
tested.

Run: python tests/test_sign_convention.py   (exit 0 on pass)
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esdqaoa.circuits import build_esd_qaoa  # noqa: E402
from esdqaoa.problems import SDProblem  # noqa: E402
from esdqaoa.simulate import (default_alpha, default_beta_dep,  # noqa: E402
                              penalized_objective)

try:
    from qiskit.quantum_info import Statevector
except ImportError:  # pragma: no cover
    print("qiskit not available; skipping")
    sys.exit(0)


def measured_phases(pb, gamma):
    """Relative phase that one cost layer puts on each decision state."""
    alpha, beta = default_alpha(pb), default_beta_dep(pb)
    qc, gammas, betas = build_esd_qaoa(pb, 1, 0.5, alpha, beta, measure=False)
    binding = {gammas[0]: gamma, betas[0]: 0.0}
    sv = Statevector(qc.assign_parameters(binding)).data
    n_tot = int(math.log2(len(sv)))
    amps = {}
    for idx, a in enumerate(sv):
        if abs(a) < 1e-9:
            continue
        bits = [(idx >> b) & 1 for b in range(n_tot)]
        if any(bits[pb.N:]):          # ancillas must return to |0>
            raise AssertionError("ancilla not uncomputed at index %d" % idx)
        amps[tuple(bits[:pb.N])] = a
    return amps


def main():
    pb = SDProblem(name="tiny", performances=[3, 2], costs=[2, 1],
                   budget=2, dependencies=[(0, 1)])
    gamma = 0.37
    alpha, beta = default_alpha(pb), default_beta_dep(pb)
    amps = measured_phases(pb, gamma)

    ref = sorted(amps)[0]
    ref_f = penalized_objective(np.array(ref), pb, 0.5, alpha, beta)
    ref_phase = np.angle(amps[ref])

    print("  state   f_o(x)    measured dphase   exp(+i g f)   exp(-i g f)")
    plus_ok = minus_ok = True
    for x in sorted(amps):
        f = penalized_objective(np.array(x), pb, 0.5, alpha, beta)
        dmeas = np.angle(amps[x]) - ref_phase
        dplus = gamma * (f - ref_f)
        dminus = -gamma * (f - ref_f)
        wrap = lambda d: (d + math.pi) % (2 * math.pi) - math.pi  # noqa: E731
        ep, em = abs(wrap(dmeas - dplus)), abs(wrap(dmeas - dminus))
        plus_ok &= ep < 1e-6
        minus_ok &= em < 1e-6
        print("  %s  %7.2f   %+12.6f   %+11.6f  %+11.6f"
              % ("".join(map(str, x)), f, wrap(dmeas), wrap(dplus),
                 wrap(dminus)))

    print()
    if plus_ok:
        print("  circuit realises exp(+i gamma f_o(x))  -> matches "
              "H_C|x> = -f_o(x)|x> with U_C = exp(-i gamma H_C)")
    elif minus_ok:
        print("  circuit realises exp(-i gamma f_o(x))  -> manuscript sign "
              "convention must be corrected")
    else:
        raise AssertionError("circuit matches neither sign convention")
    assert plus_ok or minus_ok
    return "plus" if plus_ok else "minus"


if __name__ == "__main__":
    main()
    print("\nsign-convention test passed")
