# Independent validation corpus

Synthetic trajectories test implementation invariants, but they cannot establish
recovery accuracy on logs produced by other systems. Keep a separate corpus of
orientation and angular-rate CSV files whose conventions were recorded independently
of this package's predictions. A source specification, reference implementation, or
original logger configuration should establish each label.

For a reproducible synthetic starting point, generate three cases from closed-form
ZYX rotations and analytic body/world angular velocities. This generator does not
import the inference package or its synthetic helper. The cases exercise quaternion
storage variants, direction and rate-frame changes, rate-channel reordering,
quaternion sign flips, irregular timestamps with dropped samples, and Euler/degrees
inputs. They remain **synthetic**, so passing them is not real-world validation.

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
the known class, and a miss. It also counts misses for which the diagnostic evidence
threshold was reached. The operational decoder and data-winning classes are reported
separately. If a config fixes a convention dimension, the case does **not** measure
recovery of that dimension; record what was known before running inference.

Start with logs from at least two independent sources and include quiet motion,
three-axis motion, timestamp jitter, missing samples, and sensor latency when those
conditions occur naturally. Keep acquisition details, label evidence, permissions,
and source versions with the corpus. Split by source or recording session when
tuning thresholds; adjacent samples from one recording are not independent
validation cases. The packaged `common-engineering-v1` profile and confidence labels
remain uncalibrated until held-out measurements support a stronger claim.
