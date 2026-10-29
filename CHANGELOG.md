# Changelog

## Unreleased

- Stored per-candidate squared residual energies instead of full residual vectors,
  retaining detailed vectors only when diagnostics require them.
- Reused increment-prefix rotations across scoring windows and replaced per-interval
  SciPy object composition with a normalized scalar quaternion accumulator.
- Added Euler benchmark workloads and fresh-process cold-start/peak-memory
  measurements; limited decoded-attitude retention to the current convention group.
- Separated candidate scoring from evidence-class ranking and policy selection,
  keeping strict typing and the public inference API unchanged.
- Made nested maps in frozen configuration, profile, packet, and report models
  read-only snapshots without changing Pydantic JSON output; normalized nested
  hypotheses and prior keys through every construction path.
- Reduced retained per-candidate diagnostic arrays, added a repeatable inference
  benchmark, and extended strict Pyright coverage to development scripts.
- Made the auto-fix sequence idempotent by stripping scope markers before Ruff formatting,
  and made CI check Ruff formatting on a temporary marker-free copy.
- Added independent-corpus evaluation and a wheel-from-sdist distribution smoke test.
- Added an analytic synthetic validation corpus generator with three labeled cases.
- Added Windows to the Python 3.11/3.12 CI matrix, sped up exact-class grouping
  and evidence-prefix evaluation, and made trajectory input buffers read-only copies.
- Separated source-checkout version testing from installed-wheel metadata testing.

## 0.0.1a0

- Added Pydantic back as a runtime dependency for dataclass validation and JSON Schema generation.
- Kept the configuration and report models as frozen, slotted Pydantic dataclasses.
- Made `infer_conventions` the canonical public API for both array-like inputs and `TrajectoryData`.
- Routed the CSV API and command-line application through the same canonical function.
- Kept `infer_attitude_conventions` and the full-name executable as compatibility aliases.
- Made `attitude-conventions` the canonical console command.
- Kept runtime dependencies limited to NumPy, Pydantic, and SciPy.
- Expanded the regression suite to 68 tests.

## Earlier development work

The following were internal milestones before the `0.0.1a0` tag, not release versions.

### Library and CLI

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

### Constraints and priors

- Added concise hard constraints for fixed or allowed-subset convention knowledge.
- Added relative soft-prior weights without removing evidence-only ambiguity reporting.
- Added direct CLI flags for every convention dimension.
- Added hypothesis-space reduction summaries and per-dimension status reporting.
- Added separate data-only and prior-weighted ranks, support, criteria, and near-tie sets.
- Added contradiction detection when constraints do not overlap the base search space.
- Expanded the test suite to 54 tests.

### Initial prototype

- Initial Euler/quaternion and angular-rate convention inference engine.
