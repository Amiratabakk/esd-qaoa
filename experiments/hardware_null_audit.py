"""P3.2 — is the measured output distribution distinguishable from uniform?

The manuscript currently reports the fraction of configurations whose measured
probability exceeds 1/2^N. With 8192 shots over up to 4096 configurations that
statistic is dominated by finite-shot occupancy noise: most configurations
receive 0, 1 or 2 counts, so "above 1/2^N" is largely a statement about
Poisson fluctuations rather than about the device.

This script compares each measurement against a uniform multinomial null with
the SAME number of shots and configurations, and also computes two statistics
that are stable under shot noise:

  * total variation distance from uniform;
  * collision probability sum_x p(x)^2, normalized by its uniform value
    2^-N (the inverse participation ratio), which is 1 for a uniform
    distribution and larger for a concentrated one.

Each is reported with the null's mean and 95% interval from resampling, so a
measured value can be called distinguishable from uniform or not.

Usage:
    python experiments/hardware_null_audit.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from esdqaoa.problems import problem_registry  # noqa: E402

PAPER = ["III", "IV", "V", "Q13", "Q14", "Q15", "Q16", "Q17", "Q18",
         "Q19", "Q20"]
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"]
DISPLAY = dict(zip(PAPER, ROMAN))
SHOTS = 8192
NULL_REPS = 2000
RNG = np.random.default_rng(7)


def stats(p, n_states):
    """(fraction at or above uniform, TVD from uniform, IPR)."""
    u = 1.0 / n_states
    frac = float(np.sum(p >= u)) / n_states
    tvd = 0.5 * float(np.sum(np.abs(p - u))) + 0.5 * (n_states - len(p)) * u
    ipr = float(np.sum(p ** 2)) / u
    return frac, tvd, ipr


def null_envelope(n_states, shots, reps=NULL_REPS):
    """Null distribution of the three statistics for a uniform device."""
    out = []
    for _ in range(reps):
        counts = RNG.multinomial(shots, np.full(n_states, 1.0 / n_states))
        p = counts / shots
        out.append(stats(p, n_states))
    a = np.array(out)
    return a.mean(axis=0), np.percentile(a, [2.5, 97.5], axis=0)


def main():
    reg = problem_registry()
    d = json.load(open(os.path.join(ROOT, "results",
                                    "hardware_all.json")))["runs"]
    print("measured vs uniform multinomial null, phi = 0.50, 8192 shots\n")
    for key in ("III", "Q20"):
        pb = reg[key]
        n_states = 2 ** pb.N
        s = d[key]["sweep"]
        mid = s[len(s) // 2]
        p = np.zeros(n_states)
        for i, (x, v) in enumerate(mid["counts"].items()):
            p[i] = v
        p = p[:len(mid["counts"])]
        p = np.concatenate([p, np.zeros(n_states - len(p))])
        meas = stats(p, n_states)
        mean, (lo, hi) = null_envelope(n_states, SHOTS)
        names = ("fraction at or above 1/2^N", "TVD from uniform",
                 "IPR (1 = uniform)")
        print("problem %s  (N=%d, %d configurations)"
              % (DISPLAY[key], pb.N, n_states))
        for i, nm in enumerate(names):
            verdict = ("INSIDE null" if lo[i] <= meas[i] <= hi[i]
                       else "outside null")
            print("   %-28s measured %9.4f   null %7.4f "
                  "[%7.4f, %7.4f]   %s"
                  % (nm, meas[i], mean[i], lo[i], hi[i], verdict))
        print()


if __name__ == "__main__":
    main()
