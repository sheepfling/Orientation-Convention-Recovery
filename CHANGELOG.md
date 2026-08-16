# Changelog

## 0.0.1

- Added Pydantic back as a runtime dependency for dataclass validation and JSON Schema generation.
- Kept the configuration and report models as frozen, slotted Pydantic dataclasses.
- Made `infer_conventions` the canonical public API for both array-like inputs and `TrajectoryData`.
- Routed the CSV API and command-line application through the same canonical function.
- Kept `infer_attitude_conventions` and the full-name executable as compatibility aliases.
- Made `attitude-conventions` the canonical console command.
- Kept runtime dependencies limited to NumPy, Pydantic, and SciPy.
- Expanded the regression suite to 68 tests.

## 0.3.0

- Made the project a first-class library and command-line application from one package.
- Added the `attitude-convention-inference` executable while retaining `attitude-conventions`.
- Added `python -m attitude_convention_inference` support.
- Added high-level `infer_attitude_conventions`, `infer_csv`, and `inspect_csv` APIs.
- Added automatic, conservative discovery of common quaternion, Euler, rate, and time columns.
- Added case- and spacing-tolerant explicit column resolution.
- Added human-readable and JSON standard-output modes plus layout-only inspection.
- Added direct constant-sample-period CLI override.
- Added package version export and PEP 561 `py.typed` marker.
- Expanded the test suite from 54 to 63 tests.

## 0.2.0

- Added concise hard constraints for fixed or allowed-subset convention knowledge.
- Added relative soft-prior weights without removing evidence-only ambiguity reporting.
- Added direct CLI flags for every convention dimension.
- Added hypothesis-space reduction summaries and per-dimension status reporting.
- Added separate data-only and prior-weighted ranks, support, criteria, and near-tie sets.
- Added contradiction detection when constraints do not overlap the base search space.
- Expanded the test suite to 54 tests.

## 0.1.0

- Initial Euler/quaternion and angular-rate convention inference engine.
