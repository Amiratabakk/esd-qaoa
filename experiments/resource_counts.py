"""Transpiled quantum-resource comparison: ESD-QAOA vs the Pauli baseline.

This answers "which circuit is actually cheaper to RUN", which is a different
question from which is cheaper to simulate. ESD-QAOA simulates far faster
only because its cost register uncomputes and factors out; on hardware both
circuits run at their full width, and the Pauli baseline generally uses
FEWER qubits (slack register vs cost register). So resource claims must be
gate- and depth-based after transpilation -- never qubit-count-based.

Both circuits are transpiled to the same basis with the same optimization
level. Default is all-to-all connectivity, which measures intrinsic gate
cost; pass --heavy-hex to add IBM-style connectivity and pay for routing.

Usage:
    python experiments/resource_counts.py --level 1
    python experiments/resource_counts.py --level 3 --heavy-hex --out results/resources.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz

from esdqaoa.problems import PROBLEMS, load_instances, problem_registry
from esdqaoa.circuits import build_esd_qaoa
from esdqaoa.baseline import _build_qubo, _qubo_to_ising
from esdqaoa.simulate import default_alpha, default_beta_dep

BASIS = ["rz", "sx", "x", "cx"]


def esd_circuit(pb, p, phi):
    qc, gammas, betas = build_esd_qaoa(pb, p, phi, default_alpha(pb),
                                       default_beta_dep(pb), measure=False)
    binding = {}
    for l in range(p):
        binding[gammas[l]] = 0.31 + 0.1 * l
        binding[betas[l]] = 0.47 - 0.05 * l
    return qc.assign_parameters(binding)


def pauli_circuit(pb, p, phi):
    Q, coeff, const, n = _build_qubo(pb, phi, default_alpha(pb),
                                     default_beta_dep(pb))
    op, _ = _qubo_to_ising(Q, coeff, const, n)
    ansatz = QAOAAnsatz(cost_operator=op, reps=p).decompose(reps=3)
    return ansatz.assign_parameters(
        [0.47 - 0.05 * l for l in range(p)]
        + [0.31 + 0.1 * l for l in range(p)])


def measure(qc, level, coupling):
    t0 = time.time()
    tq = transpile(qc, basis_gates=BASIS, optimization_level=level,
                   coupling_map=coupling, seed_transpiler=7)
    ops = tq.count_ops()
    return {"qubits": tq.num_qubits, "cx": int(ops.get("cx", 0)),
            "depth": tq.depth(), "total_gates": int(sum(ops.values())),
            "transpile_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--phi", type=float, default=1 / 3)
    ap.add_argument("--level", type=int, default=1, choices=[0, 1, 2, 3])
    ap.add_argument("--heavy-hex", action="store_true",
                    help="add IBM heavy-hex connectivity (adds SWAP routing)")
    ap.add_argument("--instances", default="results/instances.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # PROBLEMS (I-V) + the pinned LADDER (Q13-Q20); --instances
    # only needed to evaluate a freshly generated ladder
    registry = problem_registry()
    if os.path.exists(args.instances):
        # only ADD instances not already pinned. Letting the regenerable
        # JSON override LADDER produced a stale resources.json (Q14 listed
        # at N=7 when the pinned Q14 is N=8) -- CLAUDE.md section 3 says
        # no result may depend on that file.
        for name, pb in load_instances(args.instances).items():
            registry.setdefault(name, pb)

    coupling = None
    if args.heavy_hex:
        from qiskit.transpiler import CouplingMap
        coupling = CouplingMap.from_heavy_hex(7)

    print(f"p={args.p} phi={args.phi:.3f} basis={BASIS} "
          f"optimization_level={args.level} "
          f"coupling={'heavy-hex' if args.heavy_hex else 'all-to-all'}\n")
    print(f"{'inst':>5} {'ESD qubits':>10} {'CX':>7} {'depth':>7}   "
          f"{'Pauli qubits':>12} {'CX':>7} {'depth':>7}   "
          f"{'CX ratio':>9} {'depth ratio':>11}")
    print("-" * 92)

    rows = []
    for name, pb in registry.items():
        e = measure(esd_circuit(pb, args.p, args.phi), args.level, coupling)
        q = measure(pauli_circuit(pb, args.p, args.phi), args.level, coupling)
        cx_r = e["cx"] / max(q["cx"], 1)
        d_r = e["depth"] / max(q["depth"], 1)
        rows.append({"name": name, "N": pb.N, "esd": e, "pauli": q,
                     "cx_ratio": cx_r, "depth_ratio": d_r})
        print(f"{name:>5} {e['qubits']:>10} {e['cx']:>7} {e['depth']:>7}   "
              f"{q['qubits']:>12} {q['cx']:>7} {q['depth']:>7}   "
              f"{cx_r:>8.2f}x {d_r:>10.2f}x")

    print("\nratio > 1 means ESD-QAOA is MORE expensive on that metric.")
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"settings": vars(args), "rows": rows}, f, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
