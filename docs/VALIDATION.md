# Independent validation corpus

Synthetic trajectories test implementation invariants, but they cannot establish
recovery accuracy on logs produced by other systems. Keep a separate corpus of
orientation and angular-rate CSV files whose conventions were recorded independently
of this package's predictions. A source specification, reference implementation, or
original logger configuration should establish each label.

By default, the independent generator writes four cases from closed-form ZYX
rotations and analytic body/world angular velocities. It does not import the
inference package or its synthetic helper. These cases exercise quaternion storage
variants, direction and rate-frame changes, rate-channel reordering, quaternion sign
flips, irregular timestamps with dropped samples, Euler/degrees inputs, and seeded
orientation/rate measurement noise. The noise case records its seed and standard
deviations in the manifest and assessment. They remain **synthetic**, so passing
them is not real-world validation.

With the checked-in generator and current inference settings, the noisy case is
ambiguous but retains the known class; the other three are uniquely resolved.
This is a deterministic regression fixture, not a Monte Carlo robustness rate or
confidence calibration.

For a paired noise sensitivity study, the generator offers low, moderate, and high
per-axis standard deviations: orientation/rate `(0.0015 rad, 0.012 rad/s)`,
`(0.003 rad, 0.025 rad/s)`, and `(0.006 rad, 0.05 rad/s)`. It uses the same seed
set across all levels and motion profiles; increase `--noise-replicates` for a
larger study. `--noise-sample-counts 161 401` compares 4-second and 10-second
recordings at 25 ms spacing. Separate seeded streams for orientation/rate noise,
timestamp jitter, and dropout are reused across acquisition-factor levels. Each
shorter paired trial is an exact prefix of the longer one, with configured duration
endpoints protected from dropout; the assessment groups results by sample count and
period.

Acquisition stress factors can be varied independently with
`--timestamp-jitter-amplitudes-s`, `--dropout-probabilities`,
`--orientation-latencies-s`, and `--rate-latencies-s`. Supplying several values to a
flag creates a factorial sweep, so case counts multiply across the selected motion,
noise, duration, and acquisition conditions. Timestamp jitter is uniform in the
requested symmetric amplitude and must be less than half the 25 ms sample period.
Dropout removes a whole row (orientation and rate together), is seeded per sample,
and the manifest records both its requested probability and realized count; scheduled
duration endpoints are retained to preserve paired duration comparisons. A positive
channel latency means that channel's value at row
time `t` was generated from the analytic signal at `t - latency`. Thus a mismatch
between orientation and rate latency is an intentional synchronization error, not
an inferred convention. The assessment groups by requested acquisition condition,
not each seed's realized dropout count, and reports observed/dropped sample-count
ranges and means for those replicates.

For example, this compares baseline and perturbed acquisition conditions with paired
seeds at two durations:

```bash
python scripts/generate_validation_corpus.py \
  --noise-profile low --noise-replicates 30 --noise-sample-counts 161 401 \
  --timestamp-jitter-amplitudes-s 0 0.002 \
  --dropout-probabilities 0 0.1 \
  --orientation-latencies-s 0 0.01 --rate-latencies-s 0 0.025
```

This produces 32 conditions per seed (two durations × two jitter levels × two
dropout levels × two orientation delays × two rate delays), or 960 noisy cases for
30 seeds. Start with fewer factors or replicates when exploring because the sweep
grows multiplicatively. These remain controlled synthetic stress tests: they can
expose sensitivity and implementation defects, but they do not model every logger
or substitute for independently labeled real trajectories.

```bash
python scripts/generate_validation_corpus.py \
  --noise-profile all --motion-profile all --noise-replicates 30 \
  --noise-sample-counts 161 401
python scripts/validate_corpus.py .validation-corpus/manifest.json
```

The motion profiles are rich three-axis, low-excitation, single-axis, and near-gimbal
trajectories. The assessment groups replicates by representation, motion profile,
and measurement conditions; it reports evidence, data-winner, and operational miss
rates, plus indeterminate-confidence and evidence-insufficiency rates. It also
records Python/platform/NumPy/SciPy/package versions. The typed `report.decision`
recommends a class only when evidence is sufficient and one data-only contender remains;
otherwise it abstains with a reason. The report still preserves its deterministic
`operational_class_id` as a best candidate for diagnostics, not as confirmation. These
paired synthetic trials are not accuracy estimates for real loggers or independent
recording sessions. The assessment records each decision and reports recommendation
coverage, accuracy among recommendations, and wrong-recommendation counts separately
from the always-present operational class metrics.

## Observed synthetic snapshot

On 2026-10-30, commit `3e9173c` was evaluated with 30 paired seeds for each of four
motion profiles and three noise levels (360 noisy cases total). Each trajectory had
161 samples at 25 ms spacing. All 360 true classes remained in the evidence contender
set, but 171 operational classes differed from truth; 280 cases were evidence-
insufficient, all 360 confidence labels were indeterminate, and no incorrect result
was labeled evidence-sufficient. The table gives operational misses / evidence-
insufficient cases; both counts are out of 30 per cell.

| Motion profile | Low noise | Moderate noise | High noise |
|---|---:|---:|---:|
| Low excitation | 11 / 30 | 16 / 30 | 24 / 30 |
| Near gimbal | 2 / 30 | 14 / 30 | 23 / 30 |
| Rich three-axis | 0 / 0 | 0 / 0 | 0 / 10 |
| Single axis | 27 / 30 | 27 / 30 | 27 / 30 |

Noise levels are orientation/rate standard deviations of `(0.0015 rad, 0.012 rad/s)`,
`(0.003 rad, 0.025 rad/s)`, and `(0.006 rad, 0.05 rad/s)`, respectively. The run
used Python 3.14.6, NumPy 2.5.2, SciPy 1.18.0, and package version
`0.0.1a0.post1.dev16+g3e9173cb8`. This is a reproducible point-in-time synthetic
diagnostic, not a real-world success rate, confidence calibration, or release gate.
It demonstrates that an operational class can be wrong even while truth remains a
plausible contender; consumers should inspect evidence sufficiency, confidence, and
the contender set rather than treating the operational class as confirmed.

## Duration-aware decision snapshot

The evidence-gated decision was evaluated on commit `c54c51d` with 30 paired seeds,
four motion profiles, three noise levels, and 4-second/10-second recordings: 720 noisy
cases plus the three original control cases. In the rich-three-axis, low-noise
condition, the decision abstained on all 30 four-second trials and recommended the
correct class on all 30 ten-second trials:

| Duration | Recommendations | Correct recommendations | Abstentions |
|---|---:|---:|---:|
| 4 seconds | 0 / 30 | — | 30 / 30 |
| 10 seconds | 30 / 30 | 30 / 30 | 0 / 30 |

Across all 720 noisy cases, 30 were recommended and 690 abstained; none of the 30
recommendations was wrong (4.17% noisy-case coverage, 100% accuracy among those
recommendations). The three control cases were also recommended correctly. All other
noisy motion/noise/duration combinations abstained. This demonstrates useful sensitivity
to recording duration for one informative synthetic condition, but the overall coverage
is intentionally low and does not establish real-world precision or calibration.

The run used Python 3.14.6, NumPy 2.5.2, SciPy 1.18.0, and package version
`0.0.1a0.post1.dev18+gc54c51d8d`. The counts are synthetic diagnostic results only; do
not interpret the 30/30 outcome as a statistical guarantee.

## Acquisition-impairment snapshot

Commit `6400de0` was evaluated on one rich-three-axis motion profile at low
measurement noise, with five paired seeds at 4-second and 10-second durations. The
factorial sweep varied timestamp jitter (0 or ±5 ms), per-sample dropout probability
(0 or 0.35), orientation delay (0 or 50 ms), and rate delay (0 or 50 ms): 32
conditions per seed, or 160 noisy cases plus the three control cases. The table groups
trials by duration and the number of nonzero acquisition factors; it does not imply
that factors have equal physical severity. The run used Python 3.14.6, NumPy 2.5.2,
SciPy 1.18.0, and package version `0.0.1a0.post1.dev20+g6400de09c`.

| Duration | Nonzero factors (of 4) | Trials | Correct recommendations | Abstentions | Wrong recommendations |
|---|---:|---:|---:|---:|---:|
| 4 seconds | 0 | 5 | 0 | 5 | 0 |
| 4 seconds | 1 | 20 | 3 | 17 | 0 |
| 4 seconds | 2 | 30 | 3 | 27 | 0 |
| 4 seconds | 3 | 20 | 1 | 19 | 0 |
| 4 seconds | 4 | 5 | 1 | 4 | 0 |
| 10 seconds | 0 | 5 | 5 | 0 | 0 |
| 10 seconds | 1 | 20 | 20 | 0 | 0 |
| 10 seconds | 2 | 30 | 30 | 0 | 0 |
| 10 seconds | 3 | 20 | 20 | 0 | 0 |
| 10 seconds | 4 | 5 | 5 | 0 | 0 |

All 88 recommendations among the 160 noisy trials were correct; there were no
evidence misses or operational-class misses in this run. The 4-second results are
non-monotonic: the unperturbed baseline abstained on all five trials, while some
dropout conditions produced recommendations. Do not infer that dropout improves
recovery; this needs investigation with more seeds and motion/noise profiles. At
10 seconds all 80 trials were recommended correctly, including the combined maximum
settings. This is a small, controlled synthetic diagnostic—not evidence of logger
robustness, real-world accuracy, or confidence calibration.

The run can be regenerated with:

```bash
python scripts/generate_validation_corpus.py \
  --outdir .validation-acquisition-stress-20261030 \
  --noise-profile low --motion-profile rich-three-axis --noise-replicates 5 \
  --noise-sample-counts 161 401 \
  --timestamp-jitter-amplitudes-s 0 0.005 \
  --dropout-probabilities 0 0.35 \
  --orientation-latencies-s 0 0.05 --rate-latencies-s 0 0.05
python scripts/validate_corpus.py \
  .validation-acquisition-stress-20261030/manifest.json \
  --output .validation-acquisition-stress-20261030/assessment.json
```

```bash
python scripts/generate_validation_corpus.py
python scripts/validate_corpus.py .validation-corpus/manifest.json \
  --output .validation-corpus/assessment.json
```

The generated directory is ignored by Git. Choose a new directory with `--outdir`
when you need another copy; the generator refuses to overwrite an existing one.

Each manifest case records the CSV path, a source for its label, the complete known
convention, and the search configuration. Paths are relative to the manifest. The
validator checks that the labeled hypothesis is actually in the configured search;
otherwise it rejects the case instead of counting it as a success or failure.

```json
{
  "schema_version": 1,
  "cases": [
    {
      "case_id": "demo-only-synthetic",
      "csv": "../examples/synthetic_quaternion.csv",
      "label_source": "scripts/generate_examples.py (demo only; not independent validation)",
      "representation": "quaternion",
      "orientation_columns": ["q0", "q1", "q2", "q3"],
      "rate_columns": ["w0", "w1", "w2"],
      "time_column": "time",
      "truth": {
        "attitude_direction": "world_to_body",
        "quaternion_order": "scalar_first",
        "rate_frame": "world",
        "rate_order": "rqp",
        "rate_unit": "radians_per_second"
      },
      "config": {}
    }
  ]
}
```

Run the evaluator after installing development dependencies:

```bash
python scripts/validate_corpus.py path/to/manifest.json --output assessment.json
python scripts/validate_corpus.py path/to/manifest.json --fail-on-miss
```

The result distinguishes a unique correct evidence class, ambiguity that contains
the known class, and an evidence miss (truth outside the contender classes). It
separately reports when the data-winning or operational class differs from truth.
Condition summaries expose each miss rate, evidence insufficiency, and false
evidence sufficiency by group. `--fail-on-miss` exits nonzero for either an evidence
miss or an operational miss. If a config fixes a convention dimension, the case
does **not** measure recovery of that dimension; record what was known before
running inference.

Start with logs from at least two independent sources and include quiet motion,
three-axis motion, timestamp jitter, missing samples, and sensor latency when those
conditions occur naturally. Keep acquisition details, label evidence, permissions,
and source versions with the corpus. Split by source or recording session when
tuning thresholds; adjacent samples from one recording are not independent
validation cases. The packaged `common-engineering-v1` profile and confidence labels
remain uncalibrated until held-out measurements support a stronger claim.
