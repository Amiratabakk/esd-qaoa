# ESD-QAOA

**A Multi-Objective QAOA Framework for Engineering Systems Design Optimization**  
Amiratabak Bahengam, Hao Chen &middot; Complex Engineering Systems Lab, Stevens Institute of Technology

[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey)](https://doi.org/10.XXXX/XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Code and campaign data behind the paper. If you use either, please [cite it](#citation).

## The problem

Choose a subset of `N` candidate subsystems for a system architecture. Each
candidate `i` contributes performance `p_i` and costs `c_i`. The architecture
must stay inside a budget `B`, and a directed pair `(i, j)` means subsystem
`j` requires subsystem `i`. Performance and cost conflict, so the answer is a
set of non-dominated trade-offs rather than one architecture.

ESD-QAOA encodes this with `N + k + 1` qubits: `N` decision qubits, a
`k`-qubit cost register and one budget flag. Each layer accumulates the total
cost by reversible QFT-based arithmetic, compares it against a threshold,
charges a phase penalty for any overshoot, charges each dependency violation
individually on the decision register, then uncomputes the arithmetic so only
the decision register is measured. Because the threshold is a classical
circuit parameter, the same construction serves both a weighted-sum sweep
over `phi` and an epsilon-constraint sweep that fixes `phi = 1` and varies the
threshold.

The comparison throughout is against a QUBO/Ising QAOA baseline (Qiskit
implementation) with slack-variable budget encoding, run with an identical
optimizer, surrogate, seed and layer count.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate        # POSIX: source .venv/bin/activate
pip install -r requirements.txt
```

Run on Python 3.11.9 with Qiskit 2.4.1, Qiskit Aer 0.17.2, SciPy 1.17.1 and
NumPy 2.4.4, on an Intel Core Ultra 9 285K (24 cores) with 64 GB of memory
under Windows 11. No GPU.

## Rebuilding the figures and tables

```bash
python experiments/paper_assets.py          # 11 figures, 5 main tables
python experiments/build_supplement.py      # 6 supplementary tables
python experiments/threshold_sensitivity.py # sensitivity table
```

Output goes to `paper/figures` and `paper/tables`, created on demand and not
tracked. Everything is computed from `results/` at generation time.

## Layout

### `esdqaoa/`

| module | contents |
|---|---|
| `problems.py` | `SDProblem`; the eleven pinned benchmarks; exact Pareto classification |
| `circuits.py` | QFT adder, threshold comparison, budget and dependency penalties, uncomputation, mixer |
| `simulate.py` | penalized objective, Gibbs and CVaR surrogates, differential evolution, approximation ratios |
| `fastsim.py` | exact ancilla-free `2^N` simulator for ESD-QAOA |
| `baseline.py` | QUBO construction, Ising mapping, Qiskit `QAOAAnsatz` baseline |
| `fastbaseline.py` | exact `2^(N+s)` simulator for the baseline |
| `pareto_v2.py` | distribution harvesting, PCR, hypervolume, mass metrics |
| `pareto.py` | legacy modal-tracking estimator, kept for comparison |

The fast simulators are **exact, not approximate**. In ESD-QAOA the cost
register and flag are computed and uncomputed inside each layer, so they
return to `|0>` for every decision-register basis state and the constraint
block acts as a pure diagonal phase on `R`; the mixer touches only `R`. The
dynamics are therefore reproducible at cost `2^N` instead of `2^(N+k+1)`.
The baseline's slack bits never uncompute, but its cost operator is diagonal
and its mixer is a product of single-qubit rotations, so `2^(N+s)` is
likewise exact. Both were verified to `1e-12` against full circuit simulation.
This is a simulation convenience, not a result: on hardware ESD-QAOA is the
more expensive circuit.

### `experiments/`

Campaigns, writing to `results/`: `run_sweep.py`, `run_epsilon_sweep.py`,
`multiseed_sweep.py`, `depth_sweep.py`, `run_hardware.py`,
`resource_counts.py`, `generate_instances.py`, `dependency_study.py`.

Audits: `classify_audit.py` (supported/unsupported classification),
`recovery_audit.py` (Pareto recovery counts),
`epsilon_nondegenerate_audit.py` (degenerate bounds, pinned register widths),
`threshold_sensitivity.py` (`tau` and PCR radius),
`hardware_null_audit.py` (measured distributions against a shot-matched
multinomial null).

Asset generation: `paper_assets.py`, `build_supplement.py`.

### `results/`

| file | used for |
|---|---|
| `multiseed_all.json` | the reported approximation ratios (5 seeds, 11 problems) |
| `sweep_fixed.json`, `sweep_ladder.json` | seed-7 runs with full output distributions |
| `eps_sweep_I_V.json`, `eps_ladder.json` | epsilon-constraint sweeps, both methods |
| `depth_all.json` | the reported layer-count result |
| `hardware_all.json` | the IBM Heron campaign |
| `resources.json`, `resources_opt3.json` | transpiled resources at optimization level 1 and 3 |
| `dependency_study.json` | controlled dependency-count experiment |

`depth_sweep_coldstart_INVALID.json` and `depth_sweep_esdwarm_only.json` are
kept as an audit trail and must **not** be used for results: in the first the
warm start never fired, and in the second only ESD-QAOA was warm-started,
which inflates the gap by up to 0.248.

## Conventions that matter

Getting any of these wrong silently corrupts results downstream.

**Dependency direction.** `(i, j)` means `j` requires `i`, i.e.
`x_j <= x_i`; a violation is `x_j = 1` with `x_i = 0`. Prerequisites are
assumed acyclic, and the budget-penalty argument relies on a sink existing in
the selected subgraph.

**Bitstring endianness.** Probability keys in the result JSONs are
little-endian: character `i` is subsystem `i`.

**Dependency penalties are per pair**, as controlled-phase gates on the
decision register — never accumulated on the shared flag. A shared flag
records whether the violation count is *odd*, so two simultaneous violations
cancel and go unpenalised.

**The budget-penalty constant is `2^c - 1`, not `2^c`.** Using `2^c` gives a
penalty proportional to `C - B - 1`, which vanishes at `C = B + 1`, so a
design one unit over budget escapes and the circuit silently enforces
`C <= B + 1`.

**The two penalty strengths differ and must stay different.**
`alpha = max(p) + 1` prices the budget, charged per unit of overshoot, so a
marginal bound suffices. `beta = sum(p) + 1` prices a dependency violation,
which has no magnitude, so it must dominate the whole objective range.

**Supported means "maximises `f_phi` for some `phi`", ties included.** A
point collinear with its neighbours on a supported face is supported.
`classify_pareto_points()` decides this exactly, in rational arithmetic. A
`phi` grid with `argmax` does not implement the definition: it records only
the first of several tied maximisers.

**Depth sweeps must warm-start both methods.** Warm-starting only ESD-QAOA
inflates the gap by up to 0.248. Note that Qiskit's `QAOAAnsatz` orders
parameters `[betas..., gammas...]`, so a new layer cannot simply be appended.

## Tests

```bash
for t in tests/test_*.py; do python "$t"; done
```

Plain scripts, not pytest; exit 0 on pass. They cover the exact Pareto
classification (including a collinear-face regression case), the implemented
cost-unitary sign convention, the cumulative-probability ranking rule, the
pinned dependency-study instance, generated-TeX integrity, diagonal phase
correctness, both exact simulators, epsilon-constraint mode, and the largest
pinned instance.

## Scope

The comparison is at **matched QAOA layer count**, not matched circuit
resources: ESD-QAOA uses 2.5 to 4.0 times the two-qubit gates and 7.4 to 13.7
times the transpiled depth. No computational advantage over classical
optimization is claimed, and the hardware results describe one device under
one protocol.

## Citation

If you use this code or data, please cite the paper:

> A. Bahengam and H. Chen, "A Multi-Objective QAOA Framework for Engineering
> Systems Design Optimization," *Systems*, MDPI, 2026.
> doi: [10.XXXX/XXXXXXX](https://doi.org/10.XXXX/XXXXXXX)

```bibtex
@article{bahengam2026esdqaoa,
  title   = {A Multi-Objective QAOA Framework for Engineering Systems
             Design Optimization},
  author  = {Bahengam, Amiratabak and Chen, Hao},
  journal = {Systems},
  year    = {2026},
  volume  = {},
  number  = {},
  pages   = {},
  doi     = {10.XXXX/XXXXXXX},
  url     = {https://github.com/Amiratabakk/esd-qaoa}
}
```

The DOI, volume, issue and pages are placeholders until the paper is
published; `CITATION.cff` carries the same fields and drives the
**Cite this repository** button in the sidebar. Update both on acceptance.

## Licence and copyright

Copyright &copy; 2026 Amiratabak Bahengam and Hao Chen.

Released under the MIT Licence; see [`LICENSE`](LICENSE). The licence covers
the code and the campaign data in this repository. Please cite the paper
rather than only the repository when the work informs a publication.
