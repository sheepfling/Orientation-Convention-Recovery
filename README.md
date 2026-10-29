# Attitude Convention Inference

Infer unknown attitude and angular-rate conventions by checking whether integrated rates
reproduce a sampled orientation trajectory.

The identifiable result is generally an **equivalence class**, not one uniquely
data-determined convention. In particular, simultaneously inverting the attitude
direction, switching the rate frame, and reversing the global rate sign is an exact
trajectory symmetry. With timestamps absent, rate-unit scale and sample period are also
confounded. Hard constraints and soft priors can choose a representative, but the report
keeps those choices separate from trajectory evidence.

The package has one inference engine and two ways to use it:

- import `infer_conventions` as a Python library;
- run the `attitude-conventions` command as an application.

## Five-module recovery engine

The accompanying [research draft](paper/orientation-convention-recovery.tex) separates
five decisions that should not be collapsed into one score:

1. generate and normalize syntactically valid hypotheses;
2. quotient exact structural duplicates and diagnose excitation;
3. compute representation-only unit, wrapping, Euler-range, and branch heuristics;
4. apply provenance/domain information as a class-level prior;
5. integrate, score, accumulate evidence, and eliminate candidates only under an exact
   bound or an explicit statistical rule.

The report keeps data-only rankings, prior-weighted rankings, exact-class members, and
evidence contenders separate. A policy-selected representative is not additional
trajectory evidence.

The implementation boundaries are documented in the
[architecture spine](docs/ARCHITECTURE.md): coercion, hypothesis construction,
kinematics, small numerical adapters, representation heuristics, statistical
diagnostics, and evidence-preserving orchestration are separate lanes.

Each report also carries a result label such as `trajectory-quotient`, `scale-gauge`,
`attitude-heuristic`, or `rate-non-ID`, the evidence-preserving contender class IDs,
the data-winning class IDs, the empirical gap to the next distinct class, the
operational class/decoder IDs, and the reason for instantiating that decoder.

## Data availability limits

The inference API intentionally requires paired attitude and rate streams. That is the
regime where integrated rates provide an independent kinematic witness. Rate-only
self-integration can always produce a mathematically valid curve from a chosen initial
attitude, so it is not treated as a convention test. Attitude-only range, branch, or
smoothness checks are exposed as heuristics rather than trajectory-confirmed recovery.

## Inferred convention dimensions

The search can cover:

- quaternion scalar-first versus scalar-last storage;
- body-to-world versus world-to-body attitude direction;
- all twelve standard Euler sequences: six Tait–Bryan orders and six proper-Euler orders;
- intrinsic versus extrinsic Euler interpretation;
- all six `p/q/r` channel permutations;
- body-expressed versus world-expressed angular velocity;
- radians, degrees, turns, or grads when enabled;
- one global positive versus reversed rate sign when enabled (not all 48 signed channel maps);
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

`attitude-convention-inference` remains as a compatibility command alias in 0.0.x. The
canonical command is `attitude-conventions`.

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

For a minimal downstream-library example, see
[`examples/quickstart.py`](examples/quickstart.py).

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

A scalar fixes a convention dimension. A sequence permits only that subset. An omitted
value remains unknown.

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

Relative field priors can rank otherwise unresolved candidates without changing the
evidence-only diagnostics. They are operational information, not trajectory evidence;
exact-class members retain the same data risk and are reported together.

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

For a domain-conditioned class prior, key weights by the reported structural class ID:

```python
from attitude_convention_inference import DomainPrior, infer_conventions

report = infer_conventions(
    trajectory,
    domain_prior=DomainPrior(
        context={"domain": "robotics", "role": "imu"},
        class_weights={"<equivalence-class-id>": 3.0},
    ),
)
```

The domain prior is applied only inside the trajectory-supported contender set. If
none of those classes has positive support, it abstains (`domain_prior_abstained=True`)
and the scientific result remains set-valued; a caller that requires one decoder gets
the deterministic `canonical-policy` fallback.

## Persistent preference profiles

When an application has a stable house convention, keep it in the configuration as
a versioned `PreferenceProfile`. Class weights can choose among surviving quotient
classes; representative preferences choose one decoder *inside* the selected class.
They are operational policy, not additional trajectory evidence.

```python
from attitude_convention_inference import (
    ConventionPriors,
    InferenceConfig,
    PreferenceProfile,
)

config = InferenceConfig(
    preference_profile=PreferenceProfile(
        profile_id="robotics-stack-v1",
        source="internal-convention-policy",
        context={"domain": "robotics", "role": "imu"},
        class_weights={"<evidence-contender-class-id>": 2.0},
        representative_preferences=ConventionPriors(
            quaternion_order={"scalar_last": 2.0},
            rate_frame={"body": 2.0},
            rate_unit={"radians_per_second": 2.0},
        ),
    )
)
```

Persist that exact policy with `config.model_dump_json(indent=2)` and pass the file
to the CLI using `--config`. The report records the profile ID/source, supported
class IDs, whether it abstained, and whether it changed the representative inside
an exact class. A profile cannot split an exact quotient class or turn a policy
choice into a data-identified result.

The starter template is
[`examples/preference_profile.json`](examples/preference_profile.json).

Before assigning weights from external documentation, record the source, version,
artifact role, exact fields supported by that source, and whether the claim applies
to a class or only to a representative. See
[`docs/PREFERENCE_RESEARCH.md`](docs/PREFERENCE_RESEARCH.md)
for the research worksheet.

The package also includes a versioned `neutral-v1` profile. It has no survey-derived
weights and is the implicit default, so ordinary deployments remain data-led. Packaged
profiles can be loaded explicitly with `load_preference_profile("profile-id")`; their
ID and source are retained in the report. The package also includes the opinionated
`common-engineering-v1` policy profile. It is a curated relative-ranking policy, not
a calibrated survey or universal truth.

Inspect packaged profiles before using them:

```bash
attitude-conventions --list-profiles
attitude-conventions --inspect-profile common-engineering-v1 --format json
attitude-conventions trajectory.csv --profile common-engineering-v1
```

A domain survey should be added as another opt-in profile only after its sources and
calibration are documented.

Profiles can also be composed into a new persisted policy. Composition uses a
normalized weighted geometric combination of relative weights; missing values remain
neutral, while explicit zero class weights remain exclusions.

```python
from attitude_convention_inference import (
    WeightedPreferenceProfile,
    compose_preference_profiles,
    load_preference_profile,
)

blended = compose_preference_profiles(
    (
        WeightedPreferenceProfile(load_preference_profile("common-engineering-v1"), 3.0),
        WeightedPreferenceProfile(load_preference_profile("neutral-v1"), 1.0),
    ),
    profile_id="common-engineering-blended-v1",
    context={"consumer": "my-deployment"},
)
```

The resulting profile contains composition members and weights in its context, can be
serialized with `blended.model_dump_json(indent=2)`, and can then be supplied through
`InferenceConfig` or the CLI `--config` option. Composition changes policy weights;
it does not add trajectory evidence or combine hard constraints.

## Evidence packets and cohort fusion

The local report can be serialized as an immutable, prior-separated evidence packet.
The packet is the product boundary: it retains the ordered quotient-class universe,
max-zero relative log evidence, the local prior, evidence contenders, calibration and
excitation metadata, hard anchors, provenance, and atomic source lineage.

```python
from attitude_convention_inference import (
    CalibrationDescriptor,
    EvidencePacket,
    EvidenceSemantics,
    LineageDescriptor,
)

packet = EvidencePacket.from_inference_report(
    report,
    packet_id="trajectory-C",
    quotient_space_id="orientation-qspace-v1",
    evidence_semantics=EvidenceSemantics.CALIBRATED_LIKELIHOOD,
    calibration=CalibrationDescriptor(
        calibration_id="orientation-risk-temp-v1",
        validated=True,
        validation_domain="held-out-imu-domain",
        temperature=1.0,
    ),
    lineage=LineageDescriptor(atomic_source_ids=("raw-file-C",)),
)
packet_json = packet.model_dump_json(indent=2)
```

Evidence semantics are explicit. `calibrated-likelihood` and
`calibrated-generalized` packets may produce a posterior only when every packet passes
the calibration and lineage gates. `generalized-weight` and `uncalibrated-risk`
packets still support ranking and borrowing, but the fusion result is labeled
`cohort-weight-only`.

Reference packets can be coalesced into a shared quotient-class prototype and then
borrowed by a weak target through the robust sharedness mixture:

```python
from attitude_convention_inference import (
    coalesce_evidence_packets,
    fuse_target_with_robust_borrowing,
)

prototype = coalesce_evidence_packets([packet_a, packet_b])
fusion = fuse_target_with_robust_borrowing(
    packet_c,
    [packet_a, packet_b],
    prior_sharedness=0.8,
)
print(fusion.status, fusion.fused_posterior, fusion.posterior_sharedness)
```

Strict fusion rejects mismatched quotient spaces, duplicate class ordering, overlapping
atomic sources, shared dependency groups, and direct packet derivation. The local target
packet remains unchanged; the result reports local, prototype, borrowed, and fused
class distributions side by side. The same workflow is available as
`attitude-cohort-fusion --target ... --reference ... --prior-sharedness 0.8 --output ...`.

Pydantic generates optional local exchange-schema files under the ignored
`schemas/` directory when needed; they are not source-controlled. Runnable A/B/C
fixtures and the other example data files live under
[`examples`](examples). Regenerate all schemas and example artifacts from the
Pydantic-backed models and deterministic generators with:

```bash
python scripts/generate_schemas.py --fix
```

Verify that checked-in artifacts are current without changing files:

```bash
python scripts/generate_schemas.py --check
```

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

The interval file includes canonical midpoint rates, observed and predicted delta
quaternions, geodesic error vectors, path angles, and alias flags.

## Ambiguity handling

The tool does not force a unique answer when the trajectory cannot identify one. It reports:

- exact-equivalence-class members for the selected result;
- evidence near ties and unresolved dimensions;
- posterior near ties after soft priors;
- low-excitation and coaxial-motion warnings;
- near-alias and hard-alias fractions;
- timing/rate-unit confounding when timestamps are absent;
- fixed, inferred, ambiguous, and not-applicable status per dimension.
- `data_winning_class_ids` and `empirical_gap_to_next_class`, which distinguish
  multiple equal minima from a gap to the next distinct quotient class.

## Numerical model

Every attitude hypothesis is converted to the active body-to-world rotation `R_BW`:

```text
v_world = R_BW.apply(v_body)
```

The tool integrates the candidate canonical angular velocity, compares predicted and observed
rotation increments on `SO(3)`, and ranks candidates by a raw geometric risk. Each window
scale gets its own upper-trimmed mean of `geodesic_error**2 / window_size`; the final risk
is their explicit equal-weight average. The report also includes the untrimmed risk and
per-scale risks, so trimming cannot silently hide the intervals that distinguish models.

The displayed `fit_score` is descriptive only. It uses one reference motion scale shared
by every candidate in the run and never affects ranking, near ties, or confidence.

The raw-rate excitation diagnostic is computed before signed-permutation decoding. Its
thresholded rank and weakest-axis ratio are invariant to common positive rate scale,
channel order, and channel signs. Rank three is a conditioning diagnostic, not a
uniqueness theorem.

For Euler inputs, the report also exposes `euler_range_penalty_rad2`,
`euler_branch_switch_count`, `euler_branch_cost_rad2`, and
`euler_singularity_margin`. These are branch, principal-range, and chart-singularity
heuristics; they do not override trajectory evidence or make a nonprincipal Euler log
physically invalid.

Euler candidates also expose `heuristic_features.wrap_unit_status`. It reports
`unit-undetermined-by-wrap` when no visible period crossing separates the tested raw
angle units. Wrap evidence concerns angle representation periods, not physical clock
scale, and uncalibrated feature weights are not probabilities.

The multi-window implementation uses cached decoder results and noncommutative prefix
products. Compare the direct reference with the optimized window kernel using:

```bash
python examples/benchmark_acceleration.py
```

The benchmark checks numerical agreement before reporting timing. Its speed is a kernel
measurement, not a claim about statistical recovery or screening recall.

Exact computational pruning is exposed separately through
`should_prune_by_exact_lower_bound(lower_bound, incumbent_score)`: it returns true only
for a verified strict lower-bound exceedance. This has different semantics from the
statistical evidence contender set, which requires explicit bounded-loss/dependence
assumptions.

## Statistical evidence and stopping

The report also fits an explicit diagnostic noise model to the winning data-ranked
hypothesis. It treats rotation-vector residuals as approximately isotropic Gaussian
errors, estimates a robust per-axis standard deviation in radians, and reports the
lag-one residual correlation. The relevant fields are:

- `estimated_noise_sigma_rad`: estimated residual noise scale per rotation-vector axis;
- `noise_residual_count`: number of residual intervals retained after robust trimming;
- `noise_lag1_autocorrelation`: a diagnostic for temporal dependence, not a proof of an
  independent-noise process;
- `evidence_log_likelihood_ratio`: the winning hypothesis' log likelihood advantage over
  the best competing hypothesis at the estimated noise scale;
- `evidence_sufficient`, `evidence_intervals_used`, and `evidence_fraction`: whether the
  winner stayed ahead of every competitor by the configured threshold for the required
  number of consecutive prefixes.

By default the threshold is 5.0 log-likelihood units, with at least 20 intervals and
three stable prefixes. If `evidence_sufficient` is false, use the full file and treat the
ranking as unresolved or collect more informative motion. This is a plug-in
quasi-likelihood diagnostic, not a calibrated posterior probability or a Bayes factor:
adjacent increments share attitude and rate samples, so an i.i.d. model is structurally
approximate. A genuine probabilistic claim would additionally require validated
dependence assumptions, priors, and model calibration. Noticeable residual correlation
is reported as a warning because the i.i.d. approximation can make evidence accumulate
too quickly.

## Dependencies

Runtime dependencies are intentionally limited to:

- NumPy;
- SciPy;
- Pydantic.

CSV and CLI parsing use the Python standard library. Pydantic validates configuration
and report models and generates exchange schemas. pandas is not required.

## Development

```bash
python -m pip install -e '.[dev]'
python scripts/ci.py
```

Use `python scripts/ci.py --fix` to apply the autofixes before running the same
checks. The runner handles mdrepo, rumdl, generated schemas and examples, Ruff,
Pyright, scope markers, and tests in the CI order.

The reproducible adversarial harness can be run with:

```bash
PYTHONPATH=src python examples/stress_scenarios.py --seed 20260801
```

It exercises noisy/sign-flipped quaternions, burst outliers, timestamp jitter and
drops, zero-motion abstention, hard-alias warnings, float32 storage, one-sample rate
latency, and slow timestamp drift. The test suite additionally covers nonfinite
streams, nonincreasing timestamps, zero-norm quaternions, partial single-axis
excitation, too-short trajectories, Euler branch crossings, scalar-first/last
variants, and cohort packet failures. The seed is fixed by default so results can be
compared across changes.

For real-world accuracy, record independently labeled logs in a separate corpus and
evaluate them with [`scripts/validate_corpus.py`](scripts/validate_corpus.py). The
[validation guide](docs/VALIDATION.md) defines the manifest and interpretation. An
independently calculated but still synthetic starter corpus can be generated with
[`scripts/generate_validation_corpus.py`](scripts/generate_validation_corpus.py).
The remaining release gates are tracked in the [productization roadmap](docs/ROADMAP.md).
The [performance baseline](docs/PERFORMANCE.md) records measured latency and traced
allocation for small and default quaternion searches.

Build and smoke-test a source distribution and its wheel with:

```bash
python -m pip install -e '.[dev,publish]'
python scripts/check_distribution.py
```

Python 3.11 or newer is required.
