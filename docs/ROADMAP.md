# Productization roadmap

The `0.0.1a0` tag is an alpha checkpoint. A green synthetic test suite establishes
implementation consistency, not recovery accuracy on outside data.

## Current engineering baseline

- `scripts/ci.py --fix` and check-only CI pass locally, including strict Pyright,
  Ruff formatting on marker-free copies, and scope markers restored last.
- The sdist-to-wheel build, Twine check, and installed-wheel smoke test pass on
  local Windows/Python 3.14. The Python 3.11/3.12 Linux/Windows matrix remains a
  separate release gate.
- The analytic synthetic corpus has labeled recovery/ambiguity/miss evaluation;
  it is not evidence of recovery accuracy on independent systems.
- Exact-class grouping and evidence prefixes avoid known quadratic paths.
- Exact symmetry grouping, ranking, operational selection, evidence summaries,
  dimension assessments, diagnostics, and report assembly now have separate
  modules; `inference.py` orchestrates them.
- Shared model primitives, search contracts, policy configuration, convention
  hypotheses, and report contracts now have separate modules with stable
  `models.py` re-exports.
- [Performance baselines](PERFORMANCE.md) include 10,001-sample compact and
  48-hypothesis quaternion searches, plus 576-hypothesis Euler searches through
  10,001 samples with isolated-process peak memory and cold-call timing.

## Next evidence gates

1. Assemble a permission-cleared corpus from at least two external systems.
   Record the exact logger/API version, independent convention label, sensor
   timing contract, and source citation for each case. Hold out entire sources
   or recording sessions for evaluation.
2. Measure class-level miss rate, abstention rate, and false diagnostic evidence
   sufficiency across motion, noise, timing, and sampling conditions. Tune
   thresholds only on training sources and publish held-out results with the
   configuration used.
3. Calibrate any probability or confidence claim against held-out outcomes and
   temporal dependence. Until then, keep the current heuristic and weight-only
   labels explicit.

## Next engineering gates

1. Repeat broad Euler timing and memory measurements across supported Python,
   NumPy, and SciPy versions, including representative nonanalytic workloads.
   Avoid promising long-log support until repeated measurements justify it.
2. Compare public imports and serialized configuration/report contracts against
   the alpha tag so the module moves do not silently change the library surface.
3. Confirm the release matrix passes on GitHub, then test an actual `0.0.1`
   prerelease artifact on TestPyPI before a public package release.
