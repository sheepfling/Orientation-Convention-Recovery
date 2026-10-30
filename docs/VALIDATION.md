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
records the per-case noise estimate, residual lag-one autocorrelation, evidence
prefix/likelihood diagnostics, and inference warnings; condition summaries include
the mean residual lag-one autocorrelation. Python/platform/NumPy/SciPy/package
versions are recorded as well. The typed `report.decision`
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

Commit `6400de0` was evaluated on rich-three-axis trajectories at 4 seconds
(161 scheduled samples, 25 ms spacing), using 30 paired noise seeds. The first
acquisition sweep crossed timestamp jitter (0 or ±5 ms), whole-row dropout
probability (0 or 0.35), orientation delay (0 or 50 ms), and rate delay (0 or 50 ms):
16 conditions per seed, or 480 noisy cases plus three controls. The table combines
the two jitter levels; each row therefore contains 60 cases. The run used Python
3.14.6, NumPy 2.5.2, SciPy 1.18.0, and package version
`0.0.1a0.post1.dev20+g6400de09c`.

| Dropout p | Orient delay | Rate delay | Trials | Recommended | Abstain | Wrong |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 ms | 0 ms | 60 | 0 | 60 | 0 |
| 0 | 0 ms | 50 ms | 60 | 0 | 60 | 0 |
| 0 | 50 ms | 0 ms | 60 | 0 | 60 | 0 |
| 0 | 50 ms | 50 ms | 60 | 0 | 60 | 0 |
| 0.35 | 0 ms | 0 ms | 60 | 26 | 34 | 0 |
| 0.35 | 0 ms | 50 ms | 60 | 0 | 60 | 0 |
| 0.35 | 50 ms | 0 ms | 60 | 2 | 58 | 0 |
| 0.35 | 50 ms | 50 ms | 60 | 20 | 40 | 0 |

Across the 480 noisy trials, 48 recommendations were correct and 432 cases abstained;
there were no evidence misses or operational-class misses. All noisy cases met the
current evidence-sufficiency test, but every case's separate heuristic confidence
label remained `indeterminate`. In 240 paired comparisons of no dropout versus 35%
dropout, 48 contender sets changed from two classes to one, always retaining truth;
the other 192 remained two-class sets. The ±5 ms jitter changed no contender set or
decision in the 240 paired jitter comparisons. At 35% requested dropout, the generator
retained an average of 105.8 of 161 samples because duration endpoints are protected.

A plausible generator-specific explanation is that longer gaps let the integrated
orientation change stand out from independent per-sample measurement noise. That is
not evidence that real dropout improves inference: this is one analytic motion and
white-noise model, and the inference diagnostics warn that adjacent residuals are
correlated while the evidence calculation uses an approximate i.i.d. likelihood.
More importantly, the apparent benefit disappears at higher noise:

| Noise profile | Dropout probability | Trials | Recommendations | Abstentions |
|---|---:|---:|---:|---:|
| Low | 0 | 60 | 0 | 60 |
| Low | 0.35 | 60 | 26 | 34 |
| Moderate | 0 | 60 | 0 | 60 |
| Moderate | 0.35 | 60 | 0 | 60 |
| High | 0 | 60 | 0 | 60 |
| High | 0.35 | 60 | 0 | 60 |

This companion 4-second run crossed all three noise profiles with the same two jitter
and dropout levels over 30 seeds (360 noisy cases plus controls). The 26 recommendations
were all correct and occurred only at low noise with 35% dropout. All moderate- and
high-noise cases abstained; the 29 evidence-insufficient cases were all high-noise.
These paired synthetic results help expose sensitivity, but do not establish
real-world accuracy, logger robustness, or confidence calibration.

## Temporally correlated measurement-noise stress

The independent generator also supports stationary Gaussian AR(1) perturbations
for the three orientation rotation-vector components and three angular-rate
components. `--orientation-noise-ar1-coefficients` and
`--rate-noise-ar1-coefficients` accept one or more sample-lag coefficients
`rho` in `(-1, 1)`. A coefficient of zero is white noise and preserves the
generator's previous seeded sequence exactly. Positive values model slowly
varying measurement error; negative values model alternating error. Each
coefficient is the correlation between adjacent scheduled samples, not a
continuous-time constant. Marginal standard deviations remain those of the
selected noise profile. The manifest and assessment retain both coefficients,
and summaries keep different correlation conditions separate.

Noise is generated on the scheduled sample grid before timestamp jitter and row
dropout are applied. Thus, after a row is removed, adjacent observed rows may
span multiple scheduled lags. Seeds and underlying innovation draws are paired
across coefficient levels, as well as across the existing profile and duration
factors. For example, this compares white and positively correlated orientation
and rate noise over 30 paired 4-second trials per combination:

```bash
python scripts/generate_validation_corpus.py \
  --outdir .validation-corpus-ar1-30seed \
  --noise-profile all --motion-profile rich-three-axis --noise-replicates 30 \
  --noise-sample-counts 161 \
  --orientation-noise-ar1-coefficients 0 0.8 \
  --rate-noise-ar1-coefficients 0 0.8
python scripts/validate_corpus.py \
  .validation-corpus-ar1-30seed/manifest.json \
  --output .validation-corpus-ar1-30seed/assessment.json
```

The factorial sweep produces 12 correlation/noise-profile conditions, or 360
noisy cases. AR(1) is a controlled stress model, not a claim that every sensor's
errors follow this process. The evaluator's likelihood still has approximate
i.i.d. assumptions, so these runs are specifically useful for checking whether
evidence sufficiency or recommendations become misleading under temporal
dependence. Report any such result as a synthetic warning; do not tune the
inference thresholds solely to fit this generator.

### Observed AR(1) snapshot

On 2026-10-30, 30 seeds were evaluated for each noise profile and each paired
orientation/rate coefficient combination at 161 samples (4 seconds): 360 noisy
trials plus three control cases. `rho=0.8` was compared with white noise
`rho=0` independently on each channel. All 360 true classes remained in their
evidence contender sets; there were no operational misses, evidence misses, or
incorrect evidence-sufficient cases. The recommendation counts below are out of
30 per condition; all recommendations were correct.

| Noise profile | `(rho_orientation, rho_rate) = (0, 0)` | `(0, 0.8)` | `(0.8, 0)` | `(0.8, 0.8)` |
|---|---:|---:|---:|---:|
| Low | 0 | 0 | 27 | 14 |
| Moderate | 0 | 0 | 0 | 0 |
| High | 0 | 0 | 0 | 0 |

Across the 360 noisy trials, the evaluator recommended 41 and abstained on 319;
all 360 heuristic confidence labels remained indeterminate. The 32
evidence-insufficient cases occurred in high-noise conditions. Correlated
orientation noise therefore changed decision coverage substantially at low
marginal noise, despite no false recommendation in these 30-seed cells. That is
a useful warning about sensitivity to dependence, not proof of calibration or
overconfidence: this finite synthetic run cannot establish the false-positive
rate, and the evidence calculation still uses an approximate i.i.d. likelihood.
The three control cases bring the complete assessment to 44 recommendations
(including those controls) and 319 abstentions. The run used Python 3.14.6,
NumPy 2.5.2, SciPy 1.18.0, and inference package version
`0.0.1a0.post1.dev20+g6400de09c`.

The generator's `rho` values describe raw per-sample measurement perturbations;
the inference diagnostic estimates lag-one correlation on interval rotation
residuals, so those numbers are not expected to match. In fact, differencing
independent orientation errors naturally induces negative adjacent interval
correlation. In the low-noise cells, the mean measured residual correlation and
number of cases carrying the correlation warning were:

| Requested `(rho_orientation, rho_rate)` | Mean interval-residual lag-1 | Warning cases |
|---|---:|---:|
| `(0, 0)` | -0.476 | 30 / 30 |
| `(0, 0.8)` | -0.463 | 30 / 30 |
| `(0.8, 0)` | -0.084 | 0 / 30 |
| `(0.8, 0.8)` | -0.042 | 0 / 30 |

For an ideal scalar AR(1) orientation perturbation, first differences alone
have lag-one correlation `-(1-rho)/2` (approximately -0.5 at `rho=0` and -0.1
at `rho=0.8`). The observed estimates are close to that pattern, with small
effects from rate noise and the full rotation residual calculation. This makes
the warning interpretable as a property of interval residuals, not a direct
sensor-noise-correlation estimate; it also reinforces that the evidence model's
independence assumption needs a more suitable residual covariance model before
its likelihood ratio can be treated as calibrated.

### 200-seed low-noise follow-up

To reduce Monte Carlo uncertainty in the only noise profile that produced
recommendations in the preceding 30-seed sweep, 200 paired seeds were run for
each of the same four correlation conditions at 4 seconds (800 noisy cases plus
three controls). Truth remained in
every evidence contender set; there were no evidence misses, operational
misses, or insufficient-evidence cases. The table reports recommendations out
of 200, the mean fixed-noise log-likelihood ratio, and the mean interval-residual
lag-1 estimate:

| Requested `(rho_orientation, rho_rate)` | Recommendations | Mean log LR | Mean interval-residual lag-1 |
|---|---:|---:|---:|
| `(0, 0)` | 0 / 200 | 88.93 | -0.466 |
| `(0, 0.8)` | 0 / 200 | 88.35 | -0.452 |
| `(0.8, 0)` | 196 / 200 | 425.50 | -0.066 |
| `(0.8, 0.8)` | 89 / 200 | 411.47 | -0.018 |

All 285 recommendations were correct, and the remaining 515 noisy cases
abstained. With zero observed errors, the one-sided 95% exact-binomial upper
bounds on the conditional error rate are 1.52% for the 196 recommendations and
3.31% for the 89 recommendations. These bounds apply only to these
generator-defined conditions; they are not a real-world precision guarantee.
The large shift in log LR and recommendation coverage under correlated
orientation noise is now reproducible at 200 seeds, so dependence-aware evidence
is a substantive next correctness task—not a threshold-tuning exercise.
The assessment is saved locally at
`.validation-corpus-ar1-low-200seed/assessment.json` (Git-ignored) and can be
regenerated with:

```bash
python scripts/generate_validation_corpus.py \
  --outdir .validation-corpus-ar1-low-200seed \
  --noise-profile low --motion-profile rich-three-axis --noise-replicates 200 \
  --noise-sample-counts 161 \
  --orientation-noise-ar1-coefficients 0 0.8 \
  --rate-noise-ar1-coefficients 0 0.8
python scripts/validate_corpus.py \
  .validation-corpus-ar1-low-200seed/manifest.json \
  --output .validation-corpus-ar1-low-200seed/assessment.json
```

The 30-seed acquisition and noise sweeps can be regenerated with:

```bash
python scripts/generate_validation_corpus.py \
  --outdir .validation-corpus-short-duration-30seed \
  --noise-profile low --motion-profile rich-three-axis --noise-replicates 30 \
  --noise-sample-counts 161 \
  --timestamp-jitter-amplitudes-s 0 0.005 \
  --dropout-probabilities 0 0.35 \
  --orientation-latencies-s 0 0.05 --rate-latencies-s 0 0.05
python scripts/validate_corpus.py \
  .validation-corpus-short-duration-30seed/manifest.json \
  --output .validation-corpus-short-duration-30seed/assessment.json
python scripts/generate_validation_corpus.py \
  --outdir .validation-corpus-short-duration-noise-30seed \
  --noise-profile all --motion-profile rich-three-axis --noise-replicates 30 \
  --noise-sample-counts 161 --timestamp-jitter-amplitudes-s 0 0.005 \
  --dropout-probabilities 0 0.35
python scripts/validate_corpus.py \
  .validation-corpus-short-duration-noise-30seed/manifest.json \
  --output .validation-corpus-short-duration-noise-30seed/assessment.json
```

```bash
python scripts/generate_validation_corpus.py
python scripts/validate_corpus.py .validation-corpus/manifest.json \
  --output .validation-corpus/assessment.json
```

The default `.validation-corpus/` and `.validation-corpus-*` directories are ignored
by Git. Keep generated outputs under that prefix, choose a new name with `--outdir`
for each run, and the generator refuses to overwrite an existing directory.

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
