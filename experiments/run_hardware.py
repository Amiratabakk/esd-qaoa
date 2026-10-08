"""Execute ESD-QAOA circuits on IBM Quantum hardware (or a device noise model).

SCOPE -- read before interpreting any output. At the depths this encoding
needs, a hardware run is a FEASIBILITY demonstration, not a performance
result. Transpiled to heavy-hex connectivity the ESD circuit costs:

    Q13 p=1    659 CX     fidelity ~0.14 (Heron)  ~0.01 (Eagle)
    Q13 p=2   1265 CX     ~0.02                   ~1e-4
    Q17 p=2   2498 CX     ~5e-4                   ~2e-8
    Q20 p=2   4159 CX     ~4e-6                   ~2e-13

so Q17/Q20 at p=2 would return noise indistinguishable from a uniform
distribution. The defensible experiment is the SMALLEST paper instance at
p=1, and the claim is "the circuit runs end-to-end and retains measurable
signal", never "hardware matches simulation".

Every run therefore reports three numbers, not one:

    rho_hardware   what the device produced
    rho_ideal      the SAME angles simulated noiselessly (the ceiling)
    rho_uniform    what a uniform superposition scores for free (the floor)

A bare rho_hardware is uninterpretable: rho is not zero-based, so a
fully-decohered device still returns rho ~ rho_uniform. The claim is only
meaningful if rho_hardware sits clearly above rho_uniform.

Angles come from a noiseless optimization (no on-device optimization, so
queue time is not multiplied across optimizer iterations).

Usage:
    # predict the result first -- no credentials, no queue
    python experiments/run_hardware.py --problems Q13 --p 1 --dry-run
    python experiments/run_hardware.py --problems Q13 --p 1 --dry-run \\
        --noise-backend FakeTorino --phis 0.25 0.5 0.75

    # real hardware (needs a saved account)
    python experiments/run_hardware.py --problems Q13 --p 1 \\
        --backend ibm_torino --shots 8192 --out results/hardware_Q13.json

Credentials, once:
    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(channel="ibm_quantum_platform",
                                      token="<TOKEN>", overwrite=True)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from esdqaoa.problems import problem_registry
from esdqaoa.circuits import build_esd_qaoa
from esdqaoa.simulate import (run_esd_qaoa, approximation_ratio,
                              prob_of_optimum, default_alpha,
                              default_beta_dep)
from esdqaoa.fastsim import FastESDRunner


def counts_to_probs(counts, N):
    """Measurement counts -> probability dict over little-endian decision tuples."""
    total = sum(counts.values())
    out = {}
    for bitstr, ct in counts.items():
        clean = bitstr.replace(" ", "")
        x = tuple(int(b) for b in clean[::-1])[:N]
        out[x] = out.get(x, 0.0) + ct / total
    return out


def uniform_probs(pb):
    n = 2 ** pb.N
    return {tuple((i >> b) & 1 for b in range(pb.N)): 1.0 / n
            for i in range(n)}


def scorecard(pb, probs, phi):
    feas = sum(pr for x, pr in probs.items()
               if pb.is_feasible(np.array(x)))
    return {
        "rho": approximation_ratio(pb, probs, phi),
        "p_opt": prob_of_optimum(pb, probs, phi),
        "feasible_mass": float(feas),
    }


def tvd(p, q):
    """Total variation distance between two decision distributions."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def build_bound_circuit(pb, p, phi, angles):
    alpha, beta_dep = default_alpha(pb), default_beta_dep(pb)
    qc, gammas, betas = build_esd_qaoa(pb, p, phi, alpha, beta_dep,
                                       measure=True)
    binding = {}
    for l in range(p):
        binding[gammas[l]] = angles[2 * l]
        binding[betas[l]] = angles[2 * l + 1]
    return qc.assign_parameters(binding)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="+", default=["Q13"])
    ap.add_argument("--p", type=int, default=1,
                    help="p=1 is the only depth with usable fidelity on the "
                         "ladder; p=2 on Q13 is already ~0.02")
    ap.add_argument("--phis", nargs="+", type=float, default=[0.5],
                    help="one or more trade-off weights. Several phi are "
                         "BATCHED into a single Sampler submission, so a "
                         "small hardware sweep costs about the same queue "
                         "time as one point.")
    ap.add_argument("--shots", type=int, default=8192)
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--opt-level", type=int, default=3)
    ap.add_argument("--backend", default="ibm_torino",
                    help="real device; Heron (torino/fez) has ~2x lower CX "
                         "error than Eagle (brisbane)")
    ap.add_argument("--dry-run", action="store_true",
                    help="simulate a device NOISE MODEL locally instead of "
                         "queueing: predicts the hardware result for free")
    ap.add_argument("--noise-backend", default="FakeTorino",
                    help="fake backend supplying the dry-run noise model")
    ap.add_argument("--no-dd", action="store_true",
                    help="disable dynamical decoupling (on by default)")
    ap.add_argument("--out", default="results/hardware.json")
    args = ap.parse_args()

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    registry = problem_registry()
    unknown = [n for n in args.problems if n not in registry]
    if unknown:
        ap.error(f"unknown problem(s) {unknown}; available: {sorted(registry)}")

    # ---- pick the execution target -------------------------------------
    if args.dry_run:
        from qiskit_ibm_runtime import fake_provider
        from qiskit_aer import AerSimulator
        fake = getattr(fake_provider, args.noise_backend)()
        backend = AerSimulator.from_backend(fake)
        target_desc = f"noise model of {args.noise_backend} (local)"
        coupling_source = fake
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        service = QiskitRuntimeService()
        backend = service.backend(args.backend)
        target_desc = f"{args.backend} (hardware)"
        coupling_source = backend

    print(f"target: {target_desc}   p={args.p}  "
          f"phi={[round(v, 3) for v in args.phis]}  shots={args.shots}\n")

    pm = generate_preset_pass_manager(optimization_level=args.opt_level,
                                      backend=coupling_source,
                                      seed_transpiler=args.seed)
    out = {"settings": vars(args), "target": target_desc, "runs": {}}
    for name in args.problems:
        pb = registry[name]
        sweep, isas, ideals = [], [], []

        # one circuit per phi, angles from a noiseless optimization
        for phi in args.phis:
            probs_ideal, angles, _ = run_esd_qaoa(
                pb, args.p, phi, seed=args.seed, maxiter=args.maxiter)
            bound = build_bound_circuit(pb, args.p, phi, angles)
            t0 = time.time()
            isa = pm.run(bound)
            n_cx = sum(v for k, v in isa.count_ops().items()
                       if k in ("cx", "ecr", "cz"))
            isas.append(isa)
            ideals.append((phi, probs_ideal))
            sweep.append({"phi": phi, "angles": list(map(float, angles)),
                          "transpiled": {"qubits": isa.num_qubits,
                                         "two_qubit_gates": n_cx,
                                         "depth": isa.depth()},
                          "transpile_s": time.time() - t0})
        t0 = sweep[0]["transpiled"]
        print(f"[{name}] {len(isas)} circuit(s), {t0['qubits']}q, "
              f"{t0['two_qubit_gates']} 2q-gates, depth {t0['depth']}")

        # execute -- all phi go in ONE submission so queue cost is paid once
        t_start = time.time()
        job_meta = {}
        if args.dry_run:
            counts_list = [backend.run(isa, shots=args.shots,
                                       seed_simulator=args.seed)
                           .result().get_counts() for isa in isas]
        else:
            from qiskit_ibm_runtime import SamplerV2
            sampler = SamplerV2(mode=backend)
            if not args.no_dd:
                sampler.options.dynamical_decoupling.enable = True
                sampler.options.dynamical_decoupling.sequence_type = "XY4"
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.enable_measure = True
            job = sampler.run(isas, shots=args.shots)
            print(f"    job id {job.job_id()} submitted; waiting...")
            res = job.result()
            counts_list = [r.data.meas.get_counts() for r in res]
            job_meta = job.metrics() if hasattr(job, "metrics") else {}
        t_exec = time.time() - t_start

        # score every phi against BOTH baselines
        for rec, counts, (phi, probs_ideal) in zip(sweep, counts_list, ideals):
            probs_hw = counts_to_probs(counts, pb.N)
            hw = scorecard(pb, probs_hw, phi)
            ideal = scorecard(pb, probs_ideal, phi)
            unif = scorecard(pb, uniform_probs(pb), phi)
            span = ideal["rho"] - unif["rho"]
            retained = (hw["rho"] - unif["rho"]) / span if span > 1e-9 else 0.0
            rec.update({
                "hardware": hw, "ideal": ideal, "uniform": unif,
                "signal_retained": float(retained),
                "tvd_hw_vs_ideal": float(tvd(probs_hw, probs_ideal)),
                "counts": {"".join(map(str, x)): pr
                           for x, pr in sorted(probs_hw.items()) if pr > 0},
            })
            print(f"    phi={phi:.3f}  rho: hw {hw['rho']:.3f} | ideal "
                  f"{ideal['rho']:.3f} | uniform {unif['rho']:.3f}"
                  f"   retained {retained:6.1%}")
            if retained < 0.05:
                print("      WARNING: indistinguishable from uniform -- "
                      "supports no claim beyond 'the circuit executed'.")

        mean_ret = float(np.mean([r["signal_retained"] for r in sweep]))
        out["runs"][name] = {"p": args.p, "sweep": sweep,
                             "mean_signal_retained": mean_ret,
                             "exec_s": t_exec, "job_metrics": job_meta}
        print(f"    mean signal retained across {len(sweep)} phi: "
              f"{mean_ret:.1%}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
