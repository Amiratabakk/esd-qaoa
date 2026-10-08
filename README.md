# ESD-QAOA

Code, campaign data and manuscript source for **"A Multi-Objective QAOA
Framework for Engineering Systems Design Optimization"** (Amiratabak Bahengam
and Hao Chen, Complex Engineering Systems Lab, Stevens Institute of
Technology).

Everything reported in the paper can be regenerated from this repository. The
figures and tables are not committed by hand: they are produced by the
scripts below from the result files in `results/`, and a consistency gate
checks that every headline number in the manuscript still matches the data.

---

## The problem

Choose a subset of `N` candidate subsystems for a system architecture. Each
candidate `i` contributes performance `p_i` and costs `c_i`. The architecture
must stay inside a budget `B`, and a directed pair `(i, j)` means subsystem
`j` requires subsystem `i`. Performance and cost conflict, so the answer is
not one architecture but a **set** of non-dominated trade-offs.

ESD-QAOA encodes this for a gate-based quantum computer with `N + k + 1`
qubits: `N` decision qubits, a `k`-qubit cost register and one budget flag.
Within every layer the circuit accumulates the total cost by reversible
(QFT-based) arithmetic, compares it against a threshold, charges a phase
penalty proportional to any overshoot, charges each dependency violation
individually on the decision register, then uncomputes the arithmetic so only
the decision register is measured.

Because the threshold is a **classical circuit parameter**, the same
construction serves both scalarizations: a weighted-sum sweep over `phi`, and
an epsilon-constraint sweep that fixes `phi = 1` and varies the threshold.

The comparison throughout is against a QUBO/Ising QAOA baseline (Qiskit
implementation) with slack-variable budget encoding, run with an identical
optimizer, surrogate, seed and layer count.

---

## Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows;  source .venv/bin/activate on POSIX
pip install -r requirements.txt
```

Developed and run on Python 3.11.9 with Qiskit 2.4.1, Qiskit Aer 0.17.2,
Qiskit IBM Runtime 0.47.0, SciPy 1.17.1 and NumPy 2.4.4, on an Intel Core
Ultra 9 285K (24 cores) with 64 GB of memory under Windows 11. No GPU: at
these register widths a CPU state-vector evaluation beats the host-to-device
transfer.

---

## Reproducing the paper

Using the committed result files, this takes a couple of minutes and rebuilds
every figure and table reported in the paper:

```bash
python experiments/paper_assets.py          # 11 figures, 5 main tables
python experiments/build_supplement.py      # 6 supplementary tables
python experiments/threshold_sensitivity.py # the sensitivity table
```

Output is written to `paper/figures/*.png` and `paper/tables/*.tex`, created
on demand. That directory is not tracked: the manuscript source is not part
of this repository, so the figures and tables here are build products you
regenerate rather than files you read.

Nothing is hand-written. Every number in those tables, and every point in
those figures, is computed from `results/` at generation time, so a changed
result file changes the output.

Re-running the campaigns from scratch is much slower; see
**Regenerating the data** below.

---

## Repository layout

### `esdqaoa/` — the library

| module | contents |
|---|---|
| `problems.py` | `SDProblem`; the eleven pinned benchmarks; exact Pareto classification; the pinned dependency-study instance |
| `circuits.py` | the ESD-QAOA circuit: QFT adder, threshold comparison, budget and dependency penalties, uncomputation, mixer |
| `simulate.py` | penalized objective, Gibbs and CVaR surrogates, differential-evolution driver, approximation ratios |
| `fastsim.py` | **exact** ancilla-free `2^N` simulator for ESD-QAOA |
| `baseline.py` | QUBO construction, Ising mapping, Qiskit `QAOAAnsatz` baseline |
| `fastbaseline.py` | **exact** `2^(N+s)` simulator for the baseline |
| `pareto_v2.py` | distribution harvesting, PCR, hypervolume, mass metrics |
| `pareto.py` | legacy modal-tracking estimator, kept for comparison |

**Why the fast simulators are exact, not approximate.** In ESD-QAOA the cost
register and flag are computed and uncomputed inside each layer, so for every
decision-register basis state they return to `|0>` and the whole constraint
block acts as a pure diagonal phase on `R`. The mixer touches only `R`. The
restricted dynamics can therefore be reproduced exactly at cost `2^N` instead
of `2^(N+k+1)` — the ancilla width drops out. The baseline admits the same
treatment for a different reason: its slack bits never uncompute, but its
cost operator is diagonal and its mixer is a product of single-qubit
rotations, so a `2^(N+s)` simulation is also exact. Both were verified
against full circuit simulation to `1e-12` and against the reference ansatz
to `1e-14`.

This is a **simulation** convenience and is not reported as a result. On
hardware the ordering reverses: ESD-QAOA is the more expensive circuit.

### `experiments/` — campaigns, analysis, asset generation

**Campaigns** (these produce `results/*.json`)

| script | produces |
|---|---|
| `run_sweep.py` | weighted-sum sweep, both methods |
| `run_epsilon_sweep.py` | epsilon-constraint sweep, both methods |
| `multiseed_sweep.py` | five seed-paired repeats at `p=2` |
| `depth_sweep.py` | layer-count sweep `p = 1..6`, both methods warm-started |
| `run_hardware.py` | IBM Quantum execution |
| `resource_counts.py` | transpiled CX and depth |
| `generate_instances.py` | instance search (curvature, dent depth, dependency density) |
| `dependency_study.py` | controlled dependency-count experiment |

**Analysis and audits**

| script | answers |
|---|---|
| `classify_audit.py` | exact supported / face / unsupported classification per problem |
| `recovery_audit.py` | every Pareto-recovery count in Tables 3 and 4 |
| `epsilon_nondegenerate_audit.py` | which epsilon bounds are degenerate; recomputed pinned register widths |
| `threshold_sensitivity.py` | recovery as a function of `tau` and the PCR radius |
| `hardware_null_audit.py` | measured distributions against a shot-matched multinomial null |

**Asset generation**

| script | produces |
|---|---|
| `paper_assets.py` | all main-text figures and the five main tables |
| `build_supplement.py` | the six supplementary tables |

### `results/` — campaign output

Reported results:

| file | used for |
|---|---|
| `multiseed_all.json` | **the reported approximation ratios** (5 seeds, 11 problems) |
| `sweep_fixed.json`, `sweep_ladder.json` | seed-7 runs with full output distributions, used wherever a distribution is needed |
| `eps_sweep_I_V.json`, `eps_ladder.json` | epsilon-constraint sweeps, both methods |
| `depth_all.json` | the reported layer-count result |
| `hardware_all.json` | the reported IBM Heron campaign |
| `resources.json`, `resources_opt3.json` | transpiled resources at optimization level 1 and 3 |
| `dependency_study.json` | controlled dependency-count experiment |

Two files are kept deliberately as an audit trail and must **not** be used
for results: `depth_sweep_coldstart_INVALID.json`, where the warm start never
fired, and `depth_sweep_esdwarm_only.json`, where only ESD-QAOA was
warm-started and the gap is inflated by as much as 0.248. They are retained
because the bug they document is easy to reintroduce.

### `paper/` — build output, not tracked

Created when you run the generators. The manuscript source is not
distributed here; this repository provides the code and data behind it.

Caption and note text for every table lives in the generators, not in any
`.tex` file, because embedded table bodies drift from the data they claim to
report.

---

## Conventions that matter

Getting any of these wrong silently corrupts results downstream.

**Dependency direction.** A pair `(i, j)` means subsystem `j` requires
subsystem `i`, i.e. `x_j <= x_i`. A violation is `x_j = 1` with `x_i = 0`.
Prerequisite relations are assumed acyclic; all eleven benchmarks are DAGs,
and the budget-penalty argument relies on the existence of a sink in the
selected subgraph.

**Bitstring endianness.** Probability keys in the result JSONs are
little-endian: character `i` of the bitstring is subsystem `i`.

**Dependency penalties are per pair**, applied as controlled-phase gates on
the decision register — never accumulated on the shared flag qubit. A shared
flag records whether the number of violations is *odd*, so two simultaneous
violations cancel and receive no penalty at all.

**The budget-penalty constant is `2^c - 1`, not `2^c`.** Using `2^c` yields a
penalty proportional to `C - B - 1`, which vanishes at `C = B + 1`, so a
design exceeding the budget by exactly one unit escapes unpenalised and the
circuit silently enforces `C <= B + 1`.

**The two penalty strengths are different and must stay different.**
`alpha = max(p) + 1` prices the budget, which is charged per unit of
overshoot, so a marginal bound suffices. `beta = sum(p) + 1` prices a
dependency violation, which has no magnitude, so it must dominate the whole
objective range.

**Supported means "maximises `f_phi` for some `phi`", ties included.** A
Pareto point collinear with its neighbours on a supported face is supported.
`classify_pareto_points()` decides this exactly, in rational arithmetic. A
`phi` grid with `argmax` does **not** implement the definition: it records
only the first of several tied maximisers, which mislabels face-tied points
as unsupported.

**Depth sweeps must warm-start both methods.** Warm-starting only ESD-QAOA
inflates the gap by up to 0.248, because cold-start differential evolution
does not converge as the parameter count grows. Note also that Qiskit's
`QAOAAnsatz` orders parameters `[betas..., gammas...]`, so a new layer cannot
simply be appended.

---

## Tests

```bash
for t in tests/test_*.py; do python "$t"; done
```

Plain scripts, not pytest; exit 0 on pass.

| test | checks |
|---|---|
| `test_pareto_classification.py` | exact supported/unsupported classification, including a collinear-face regression case and brute-force agreement over every pairwise crossing `phi` |
| `test_sign_convention.py` | the implemented cost unitary realises `exp(-i gamma f_o(x))`, and the ancillas return to `|0>` |
| `test_cumulative_ranking.py` | the cumulative-probability panels rank every feasible configuration ahead of every infeasible one, and that sorting by `f_o` instead would not |
| `test_dependency_study.py` | the pinned dependency-study instance reproduces the campaign's feasible counts, frontier sizes and dent depths |
| `test_generated_tex.py` | generated TeX contains no Python escape corruption, no orphaned command fragments, no single-backslash row endings |
| `test_phases.py` | diagonal phase correctness on all instances |
| `test_fastsim.py`, `test_fastbaseline.py` | the exact simulators against full circuit simulation and the reference ansatz |
| `test_epsilon.py` | pinned register width, phase and argmax in epsilon-constraint mode |
| `test_instance_q20.py` | independent verification of the largest pinned instance |

---

## Regenerating the data

Only needed if you want to rebuild `results/` rather than use the committed
files. Costs scale as `2^n * p^3` for the differential-evolution work, so the
large instances are slow.

```bash
python experiments/run_sweep.py --instances ladder --backend fast
python experiments/run_epsilon_sweep.py
python experiments/multiseed_sweep.py        # 5 seeds x 11 problems
python experiments/depth_sweep.py            # p = 1..6, both warm-started
python experiments/resource_counts.py --level 1 --out results/resources.json
python experiments/resource_counts.py --level 3 --out results/resources_opt3.json
```

Hardware runs need an IBM Quantum account saved once, in your own session:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum_platform",
                                  token="<YOUR TOKEN>", overwrite=True)
```

No token is stored in this repository. `run_hardware.py --dry-run` estimates
a job against a fake backend before spending queue time — but note that the
fake-backend noise model overestimated retained signal by a factor of 3.6 on
the one problem where both a prediction and a measurement exist, so use it to
decide *whether* to run, never to report a predicted number.

---

## What the results do and do not show

The paper is deliberately narrow about this, and so is the code.

**Supported.** At matched QAOA layer count, the ESD-QAOA formulation attains
a higher approximation ratio than the QUBO/Ising baseline on all eleven
benchmarks, positive on all 55 seed-paired runs, and places more probability
on feasible and Pareto-optimal architectures. The same difference appears
under the epsilon-constraint sweep. Frontier recovery depends strongly on the
extraction rule and on the scalarization, which is a methodological result
about how such comparisons should be reported.

**Not supported.** No claim of computational advantage over classical
optimization. No claim of equal-resource superiority: the comparison is at
equal layer count, and ESD-QAOA costs 2.5 to 4.0 times the two-qubit gates
and 7.4 to 13.7 times the transpiled depth. An equal-transpiled-depth
experiment was not performed. The hardware campaign characterises retained
signal under one device and one protocol; it does not establish a universal
gate-count threshold, and failing to distinguish a measured distribution from
uniform is not evidence that it is uniform.

**Also not a result:** ESD-QAOA simulates faster than the baseline purely
because its ancillas factor out of the state space. That is a property of the
simulator, not of the method.

---

## Licence

MIT, see `LICENSE`.
