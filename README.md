# Attitude Convention Inference

Infer unknown attitude and angular-rate conventions by checking whether integrated rates reproduce a sampled orientation trajectory.

The package has one inference engine and two ways to use it:

- import `infer_conventions` as a Python library;
- run the `attitude-conventions` command as an application.

## Inferred convention dimensions

The search can cover:

- quaternion scalar-first versus scalar-last storage;
- body-to-world versus world-to-body attitude direction;
- all six Tait–Bryan Euler orders;
- intrinsic versus extrinsic Euler interpretation;
- all six `p/q/r` channel permutations;
- body-expressed versus world-expressed angular velocity;
- radians versus degrees when enabled;
- positive versus reversed rate signs when enabled;
- a constant sample period when timestamps are missing.

Every dimension can be fixed, restricted to an allowed subset, left unknown, or softly preferred.

## Installation and command-line use

Run without a persistent installation:

```bash
uvx attitude-convention-inference trajectory.csv
```

Install as an isolated application:

```bash
pipx install attitude-convention-inference
attitude-conventions trajectory.csv
```

Install into the current environment:

```bash
python -m pip install attitude-convention-inference
attitude-conventions trajectory.csv
```

Run from a checkout:

```bash
python -m pip install -e .
python -m attitude_convention_inference trajectory.csv
```

`attitude-convention-inference` remains as a compatibility command alias in 0.4.x. The canonical command is `attitude-conventions`.

## Automatic CSV detection

A conventional CSV often needs no column flags:

```text
time_s,qx,qy,qz,qw,p,q,r
```

```bash
attitude-conventions trajectory.csv
```

Recognized column families include:

- quaternion: `qw qx qy qz`, `qx qy qz qw`, `q0 q1 q2 q3`;
- Euler: `roll pitch yaw`, `phi theta psi`;
- rates: `p q r`, `wx wy wz`, `omega_x omega_y omega_z`;
- time: `time_s`, `time`, `sim_time`, `timestamp`, `t`.

Inspect the detected layout without scoring hypotheses:

```bash
attitude-conventions trajectory.csv --inspect
attitude-conventions trajectory.csv --inspect --format json
```

Override detection when necessary:

```bash
attitude-conventions trajectory.csv \
  --representation quaternion \
  --orientation-columns qx qy qz qw \
  --rate-columns p q r \
  --time-column time_s
```

## Canonical Python API

`infer_conventions` accepts lists, tuples, NumPy arrays, or a validated `TrajectoryData` object.

```python
from __future__ import annotations

from attitude_convention_inference import infer_conventions

report = infer_conventions(
    orientation=[
        [0.0, 0.0, 0.000, 1.000],
        [0.0, 0.0, 0.005, 0.9999875],
        [0.0, 0.0, 0.010, 0.9999500],
    ],
    rates=[
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ],
    time_s=[0.00, 0.01, 0.02],
    representation="quaternion",
    constraints={
        "quaternion_order": "scalar_last",
        "rate_frame": "body",
        "rate_order": ["pqr", "rqp"],
    },
)

print(report.best.hypothesis.model_dump_json(indent=2))
print(report.diagnostics.model_dump_json(indent=2))
####
```

The same function accepts explicit model objects:

```python
from __future__ import annotations

import numpy as np

from attitude_convention_inference import (
    AttitudeRepresentation,
    InferenceConfig,
    TrajectoryData,
    infer_conventions,
)

trajectory = TrajectoryData(
    representation=AttitudeRepresentation.QUATERNION,
    orientation=np.asarray(quaternions),
    rates=np.asarray(angular_rates),
    time_s=np.asarray(timestamps),
)

report = infer_conventions(trajectory, InferenceConfig())
####
```

`infer_attitude_conventions` remains as a compatibility alias, but new code should use `infer_conventions`.

## Partial information

A scalar fixes a convention dimension. A sequence permits only that subset. An omitted value remains unknown.

```python
from attitude_convention_inference import ConventionConstraints, InferenceConfig

config = InferenceConfig(
    constraints=ConventionConstraints(
        quaternion_order="scalar_first",
        rate_frame="body",
        rate_order=("pqr", "rqp"),
    )
)
####
```

Equivalent JSON:

```json
{
  "constraints": {
    "quaternion_order": "scalar_first",
    "rate_frame": "body",
    "rate_order": ["pqr", "rqp"]
  }
}
```

Equivalent CLI flags:

```bash
attitude-conventions trajectory.csv \
  --quaternion-order scalar_first \
  --rate-frame body \
  --rate-order pqr rqp
```

## Soft priors

Relative prior weights can rank an otherwise unresolved equivalence class without changing the evidence-only diagnostics.

```python
from attitude_convention_inference import ConventionPriors, InferenceConfig

config = InferenceConfig(
    priors=ConventionPriors(
        rate_frame={"body": 4.0, "world": 1.0},
    )
)
####
```

The report preserves both evidence-only and prior-weighted rankings.

## CSV helpers

```python
from attitude_convention_inference import infer_csv, inspect_csv

layout = inspect_csv("trajectory.csv")
report = infer_csv(
    "trajectory.csv",
    constraints={"quaternion_order": "scalar_last"},
)
####
```

## Machine-readable output

Write the complete report to standard output:

```bash
attitude-conventions trajectory.csv --format json
```

Write the report and winning-hypothesis interval diagnostics to files:

```bash
attitude-conventions trajectory.csv \
  --report-json convention-report.json \
  --details-csv interval-deltas.csv
```

The interval file includes canonical midpoint rates, observed and predicted delta quaternions, geodesic error vectors, path angles, and alias flags.

## Ambiguity handling

The tool does not force a unique answer when the trajectory cannot identify one. It reports:

- evidence near ties and unresolved dimensions;
- posterior near ties after soft priors;
- low-excitation and coaxial-motion warnings;
- near-alias and hard-alias fractions;
- timing/rate-unit confounding when timestamps are absent;
- fixed, inferred, ambiguous, and not-applicable status per dimension.

## Numerical model

Every attitude hypothesis is converted to the active body-to-world rotation `R_BW`:

```text
v_world = R_BW.apply(v_body)
```

The tool integrates the candidate canonical angular velocity, compares predicted and observed rotation increments on `SO(3)`, and scores geodesic residuals across multiple window lengths.

## Dependencies

Runtime dependencies are intentionally limited to:

- NumPy;
- SciPy.

CSV, JSON, CLI parsing, configuration models, and report serialization use the Python standard library. Pydantic and pandas are not required.

## Development

```bash
python -m pip install -e '.[dev]'
pytest -q
ruff check .
pyright
```

The reproducible adversarial harness can be run with:

```bash
PYTHONPATH=src python examples/stress_scenarios.py --seed 20260801
```

It exercises noisy/sign-flipped quaternions, burst outliers, timestamp jitter, and
dropped samples. The seed is fixed by default so results can be compared across
changes.

Python 3.12 or newer is required.
