# Architecture spine

The implementation follows the same separation as the research argument: first
describe the valid hypothesis space, then decode and integrate data, then add
representation-only diagnostics, and finally compare candidates using trajectory
evidence and policy.

```text
public API / CLI
       |
       v
coercion -> models -> hypotheses
                         |
                         v
                    inference engine
                    /      |       \\
                   /       |        \\
             core      heuristics  statistics
          (kinematics) (non-evidence) (evidence summaries)
                   \\       |        /
                    \\      |       /
                         report
```

## Module responsibilities

| Module          | Responsibility                                                                                                  | Must not own                          |
|-----------------|-----------------------------------------------------------------------------------------------------------------|---------------------------------------|
| `models.py`     | Typed domain contracts, validation, and serialized report models                                                | Numerical scoring or file I/O         |
| `coercion.py`   | Mapping/array-facing input coercion for the convenience API                                                     | Convention search or trajectory math  |
| `hypotheses.py` | Search dimensions and normalized hypothesis generation                                                          | Data-dependent evidence               |
| `core.py`       | Trajectory validation, convention decoding, interval integration, and window composition                        | Prior policy and report selection     |
| `math_utils.py` | Small NumPy/SciPy adapters and reusable numerical transforms                                                    | Domain decisions                      |
| `heuristics.py` | Euler range/branch, gimbal, and visible-wrap diagnostics                                                        | Trajectory-fit evidence               |
| `statistics.py` | Excitation rank and residual-noise summaries                                                                    | Operational representative selection  |
| `inference.py`  | Evidence-preserving scoring orchestration, equivalence handling, ranking, policy selection, and report assembly | CSV parsing and model validation      |
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
`statistics.py` summarizes excitation and residual processes; and `inference.py`
is responsible for combining those inputs according to the documented evidence
and policy rules.
