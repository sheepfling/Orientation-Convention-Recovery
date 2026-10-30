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
larger study:

```bash
python scripts/generate_validation_corpus.py \
  --noise-profile all --motion-profile all --noise-replicates 10
python scripts/validate_corpus.py .validation-corpus/manifest.json
```

The motion profiles are rich three-axis, low-excitation, single-axis, and near-gimbal
trajectories. The assessment groups replicates by representation, motion profile,
and measurement conditions; it reports evidence, data-winner, and operational miss
rates, plus indeterminate-confidence and evidence-insufficiency rates. It also
records Python/platform/NumPy/SciPy/package versions. Evidence insufficiency is a
diagnostic abstention signal; the API still returns an operational class. These
paired synthetic trials are not accuracy estimates for real loggers or independent
recording sessions.

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
