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
               `-- search_models (search / constraints)
                         |
                         v
                    inference engine
                    |-- scoring
                    |    |-- core (kinematics)
                    |    `-- heuristics (non-evidence)
                    |-- equivalence -> ranking
                    |-- statistics (evidence summaries)
                    `-- policy, report
```

## Module responsibilities

| Module          | Responsibility                                                                                                  | Must not own                          |
|-----------------|-----------------------------------------------------------------------------------------------------------------|---------------------------------------|
| `model_common.py` | Shared enums, Pydantic model behavior, and normalization helpers                                               | Search policy or reports              |
| `search_models.py` | Typed search space and hard constraints                                                                        | Data-dependent evidence               |
| `models.py`     | Priors, hypotheses, report contracts, and stable re-exports of common/search models                             | Numerical scoring or file I/O         |
| `immutable.py`  | Copied read-only mappings with Pydantic-compatible serialization                                                | Domain-specific validation            |
| `coercion.py`   | Mapping/array-facing input coercion for the convenience API                                                     | Convention search or trajectory math  |
| `hypotheses.py` | Search dimensions and normalized hypothesis generation                                                          | Data-dependent evidence               |
| `core.py`       | Trajectory validation, convention decoding, interval integration, and window composition                        | Prior policy and report selection     |
| `math_utils.py` | Small NumPy/SciPy adapters and reusable numerical transforms                                                    | Domain decisions                      |
| `heuristics.py` | Euler range/branch, gimbal, and visible-wrap diagnostics                                                        | Trajectory-fit evidence               |
| `statistics.py` | Excitation rank and residual-noise summaries                                                                    | Operational representative selection  |
| `scoring.py`    | Complete- and partial-stream candidate scoring, with optional interval diagnostics                              | Class ranking and policy selection    |
| `equivalence.py` | Exact observational symmetry and scale-gauge grouping                                                           | Soft priors or representative policy  |
| `ranking.py`   | Data ranks, soft-prior support weights, and deterministic tie ordering                                           | Operational class selection           |
| `inference.py` | Scoring orchestration, evidence sufficiency, operational policy, and report assembly                             | CSV parsing and model validation      |
| `io.py`         | CSV/config/report serialization and layout detection                                                            | Candidate scoring                     |
| `fusion.py`     | Cross-trajectory evidence fusion                                                                                | Single-trajectory convention decoding |

The small adapter modules are deliberately dependency-light. In particular,
`math_utils.py` is the only home for the repeated typed SciPy `Rotation` boundary,
and `coercion.py` is the only home for convenience-input model conversion. This
keeps the public API readable while making the mechanics reusable and testable.

## Evidence boundary

Heuristic features are allowed to explain representation behavior, but they do
not become trajectory evidence. `core.py` produces decoded kinematic quantities;
`heuristics.py` describes non-identifying representation behavior;
`scoring.py` computes candidate-level fit and representation diagnostics;
`equivalence.py` groups exact observational symmetries; `ranking.py` assigns
data and soft-prior ranks; `statistics.py` summarizes excitation and residual
processes; and `inference.py` combines those inputs according to the documented
evidence and policy rules.

`TrajectoryData` owns read-only copies of validated input arrays, so later caller
mutations cannot change a result. Exact quotient grouping uses disjoint-set
connections for structural duals and clock-scale gauges. Prefix evidence uses
cumulative residual energy under one fixed noise scale; this preserves the
documented fixed-noise likelihood contrast while avoiding repeated rescoring of
each earlier interval.

Frozen configuration, profile, packet, and report models copy their mapping fields
into read-only views, including the inner maps in a hypothesis-space summary.
`model_common.py` owns their shared Pydantic-facing behavior;
`immutable.py` supplies the serializer that converts those views back to ordinary
JSON objects, so `model_dump_json()` and generated validation schemas stay usable.
Callers can edit a dumped dictionary without changing the model it came from.

Normal inference retains each candidate's result and one squared angular error per
interval, but not its three-component residual vectors, decoded rates, interval
durations, predicted/observed rotations, or path angles. Evidence prefixes use the
stored scalar energies. Overall and per-interval diagnostics recompute the selected
candidate's large arrays on demand; if the data-winning hypothesis differs from
the operational selection, its residual vectors are also recomputed for noise
estimation. This reduces memory across a broad search at a modest runtime cost;
see [the performance baseline](PERFORMANCE.md).
