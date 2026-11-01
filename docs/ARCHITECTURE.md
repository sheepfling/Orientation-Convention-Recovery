# Architecture spine

The implementation follows the same separation as the research argument: first
describe the valid hypothesis space, then decode and integrate data, then add
representation-only diagnostics, and finally compare candidates using trajectory
evidence and policy.

```text
public API / CLI
       |
       v
coercion -> models facade -> hypotheses
               |-- model_common (enums / validation)
               |-- search_models (search / constraints)
               |-- policy_models (priors / config)
               |-- hypothesis_model (one convention)
               `-- report_models (results / diagnostics)
                         |
                         v
                    inference engine
                    |-- trajectory -> decoding -------+
                    |-- timing -> integration -------+--> scoring
                    |-- sparse_inference -> anchor integration/scoring
                    |-- heuristics (non-evidence) ---+
                    |-- equivalence -> ranking
                    |-- selection (operational policy)
                    |-- statistics -> evidence
                    |-- assessments -> diagnostics
                    `-- reporting
```

## Module responsibilities

| Module          | Responsibility                                                                                                  | Must not own                          |
|-----------------|-----------------------------------------------------------------------------------------------------------------|---------------------------------------|
| `model_common.py` | Shared enums, Pydantic model behavior, and normalization helpers                                               | Search policy or reports              |
| `search_models.py` | Typed search space and hard constraints                                                                        | Data-dependent evidence               |
| `policy_models.py` | Priors, preference profiles, and inference configuration                                                      | Trajectory-fit evidence               |
| `hypothesis_model.py` | One fully specified convention and its exact structural dual                                              | Candidate scoring                     |
| `report_models.py` | Candidate results, diagnostics, and serialized inference reports                                               | Search policy or numerical scoring    |
| `models.py`     | Stable re-exports for the established model import surface                                                      | New model implementations             |
| `immutable.py`  | Copied read-only mappings with Pydantic-compatible serialization                                                | Domain-specific validation            |
| `coercion.py`   | Mapping/array-facing input coercion for the convenience API                                                     | Convention search or trajectory math  |
| `hypotheses.py` | Search dimensions and normalized hypothesis generation                                                          | Data-dependent evidence               |
| `trajectory.py` | Owned, validated aligned streams and separately timestamped sparse anchors/rates | Convention decoding or fit evidence   |
| `decoding.py`   | Convert declared attitude/rate conventions to canonical rotations and body rates                                    | Search policy or trajectory integration |
| `integration.py` | Rate integration, sparse anchor-range composition, and inference-scoped rotation caches | Time-period estimation or report policy |
| `timing.py`     | Resolve timestamp/configured intervals or infer a constant sample period from trajectory consistency                 | Input ownership or convention decoding |
| `pruning.py`    | Exact lower-bound candidate pruning, distinct from statistical evidence                                               | Heuristic or probabilistic elimination |
| `core.py`       | Backward-compatible re-exports from focused implementation modules                                                     | New implementation logic              |
| `math_utils.py` | Small NumPy/SciPy adapters and reusable numerical transforms                                                    | Domain decisions                      |
| `heuristics.py` | Euler range/branch, gimbal, and visible-wrap diagnostics                                                        | Trajectory-fit evidence               |
| `statistics.py` | Excitation rank and residual-noise summaries                                                                    | Operational representative selection  |
| `scoring.py`    | Increment-sequence scoring for aligned or sparse-anchor observations, with optional interval diagnostics | Class ranking and policy selection    |
| `equivalence.py` | Exact observational symmetry and scale-gauge grouping                                                           | Soft priors or representative policy  |
| `ranking.py`   | Data ranks, soft-prior support weights, and deterministic tie ordering                                           | Operational class selection           |
| `selection.py` | Class contender filtering and operational representative policy                                                 | Geometric scoring                     |
| `evidence.py` | Fixed-noise evidence sufficiency and residual-noise estimates                                                   | Operational representative selection  |
| `assessments.py` | Hypothesis-space and per-dimension assessments                                                                | Candidate scoring                     |
| `diagnostics.py` | Overall and per-interval diagnostics                                                                          | Operational class selection           |
| `calibration_diagnostics.py` | Opt-in rate-polarity fit sensitivity and counts at caller-declared input rails                    | Automatic calibration or default inference |
| `reporting.py` | Final serialized report assembly                                                                                | Trajectory integration                |
| `inference.py` | Scoring orchestration and stable inference/interval-diagnostic imports                                          | Policy and report implementation      |
| `sparse_inference.py` | Sparse-anchor orchestration and conservative non-overlapping-interval evidence | CSV parsing or convention policy |
| `io.py`         | CSV/config/report serialization and layout detection                                                            | Candidate scoring                     |
| `fusion.py`     | Cross-trajectory evidence fusion                                                                                | Single-trajectory convention decoding |

The small adapter modules are deliberately dependency-light. In particular,
`math_utils.py` is the only home for the repeated typed SciPy `Rotation` boundary,
and `coercion.py` is the only home for convenience-input model conversion. Internal
consumers import trajectory validation, convention decoding, integration, and timing
from their owning modules directly. `core.py` remains a compatibility facade for
callers using its historical import path; it is not an implementation dependency.
This keeps the public API readable while making the mechanics reusable and testable.

## Evidence boundary

Heuristic features are allowed to explain representation behavior, but they do
not become trajectory evidence. `trajectory.py` validates and owns sample arrays;
`decoding.py` interprets declared conventions; `integration.py` produces observed
and predicted rotations; and `timing.py` resolves sample intervals.
`heuristics.py` describes non-identifying representation behavior;
`scoring.py` computes candidate-level fit and representation diagnostics;
`equivalence.py` groups exact observational symmetries; `ranking.py` assigns
data and soft-prior ranks; `selection.py` applies operational policy only after
data contenders exist; `statistics.py` and `evidence.py` summarize excitation,
residuals, and sufficiency; `assessments.py` and `diagnostics.py` explain the
result; and `reporting.py` assembles the serialized contract.

Evidence sufficiency uses a fixed independent-increment scale and is withheld when
measurable residuals are nearly persistent (absolute lag-one correlation at least
`0.95`) and the estimated residual scale exceeds `residual_floor_rad`. This guard
prevents one extreme dependence regime from masquerading as many independent
intervals; it is not a full temporal noise model.

The class contender set is a separate geometric-fit screen: a class remains a
contender when the log ratio of its floor-adjusted `class_data_risk` to the minimum
floor-adjusted class risk is at most `near_tie_delta_criterion` (default `2.0`, a
risk ratio of about `7.39`). The floor is `residual_floor_rad**2`. This is a
heuristic relative-fit margin, not a likelihood ratio or a calibrated uncertainty
interval. Because the score averages fixed sample-count windows, its physical time
scales change with sampling cadence. `evidence_sufficient` is a separate sequential
quasi-likelihood diagnostic and should not be conflated with contender membership.

`TrajectoryData` owns read-only copies of validated input arrays, so later caller
mutations cannot change a result. Exact quotient grouping uses disjoint-set
connections for structural duals and clock-scale gauges. Prefix evidence uses
cumulative residual energy under one fixed noise scale; this preserves the
documented fixed-noise likelihood contrast while avoiding repeated rescoring of
each earlier interval.

`SparseOrientationRateData` is a separate observation contract: orientation
anchors and dense rate samples have distinct timestamp arrays on one shared clock.
The integration layer inserts anchor times into the piecewise-linear rate grid,
integrates every resulting interval, and composes the non-commuting increments
between anchor pairs. Sparse inference scores those anchor-to-anchor increments;
its sequential evidence uses alternating pairs to avoid reusing endpoint
orientations. The report names both stream counts and records the sparse-anchor
observation model rather than pretending the arrays are sample-aligned.

Frozen configuration, profile, packet, and report models copy their mapping fields
into read-only views, including the inner maps in a hypothesis-space summary.
`model_common.py` owns their shared Pydantic-facing behavior;
`immutable.py` supplies the serializer that converts those views back to ordinary
JSON objects, so `model_dump_json()` and generated validation schemas stay usable.
Callers can edit a dumped dictionary without changing the model it came from.

Normal inference retains each candidate's result and one squared angular error per
interval rather than a full set of per-candidate residual and rotation arrays. It
keeps decoded rates per unique rate convention, reuses predicted interval
integrations when the exact resolved durations match, and caches observed rotations
only for the current attitude convention. Evidence prefixes use the stored scalar
energies. Overall and per-interval diagnostics recompute the selected candidate's
large arrays on demand; if the data-winning hypothesis differs from the operational
selection, its residual vectors are also recomputed for noise estimation. These
scoped caches trade a bounded amount of memory for less repeated work; see [the
performance baseline](PERFORMANCE.md).

## Cohort fusion lineage

`fusion.py` treats packet independence as an explicit input contract, not as a
property inferred from different packet IDs. Strict fusion requires atomic source
IDs, rejects shared source IDs, dependency groups, and direct derivation, and checks
sensor-time overlap. If two packets name the same sensor, each must provide both
inclusive nanosecond time bounds; overlapping windows (including a shared endpoint)
are rejected. Missing bounds leave dependence unknown. Declare other persistent
dependencies, such as shared calibration or preprocessing, with dependency groups
even when time windows do not overlap. The opt-out skips all lineage checks and is
appropriate only when independence has been verified externally. These checks can
catch contradictory metadata; they cannot prove that supplied lineage is complete.
