"""Smoke tests: circuit correctness on small instances."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from esdqaoa.problems import PROBLEMS
from esdqaoa.simulate import (run_esd_qaoa, approximation_ratio,
                              prob_of_optimum)

def test_problem_structure():
    for name, pb in PROBLEMS.items():
        front = pb.pareto_front()
        nc = pb.nonconvex_pareto_points()
        print(f"{name}: N={pb.N} k={pb.k} Q={pb.n_qubits} "
              f"pareto={len(front)} nonconvex={len(nc)}")

def test_esd_qaoa_problem_II():
    pb = PROBLEMS["II"]
    probs, angles, exp_val = run_esd_qaoa(pb, p=2, phi=0.5, maxiter=40)
    rho = approximation_ratio(pb, probs, 0.5)
    p_opt = prob_of_optimum(pb, probs, 0.5)
    print(f"Problem II phi=0.5 p=2: rho={rho:.3f}, P(opt)={p_opt:.3f}")
    top = sorted(probs.items(), key=lambda kv: -kv[1])[:4]
    print("top:", [(x, round(pr,3)) for x, pr in top])
    assert rho > 0.5, rho

if __name__ == "__main__":
    test_problem_structure()
    test_esd_qaoa_problem_II()
    print("OK")
