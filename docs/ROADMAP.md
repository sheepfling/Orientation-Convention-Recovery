# Productization roadmap

The `0.0.1a0` tag is an alpha checkpoint. A green synthetic test suite establishes
implementation consistency, not recovery accuracy on outside data.

## Current work

- Keep `scripts/ci.py --fix` repeatable and make the normal CI path check Ruff
  formatting on a temporary copy without scope markers.
- Build an sdist and a wheel from it, validate both with Twine, and smoke-test
  the installed wheel outside the checkout on Python 3.11/3.12 and Linux/Windows.
- Measure recovery, ambiguity, and misses with the analytic synthetic corpus and
  the same evaluator intended for independently labeled logs.
- Reduce avoidable quadratic work in exact-class grouping and evidence prefixes.

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

1. Extend the [initial performance baseline](PERFORMANCE.md) to larger logs,
   Euler searches, process RSS, and cold starts. Keep the per-candidate score
   storage compact and avoid promising long-log support until measured.
2. Review nested mutability in frozen configuration, profile, packet, and report
   models. Preserve Pydantic serialization while preventing caller mutations
   from changing a previously validated result.
3. Split the large `models.py` and `inference.py` modules along the architecture
   spine, with stable public imports and focused tests for each moved contract.
4. Confirm the release matrix passes on GitHub, then test an actual `0.0.1`
   prerelease artifact on TestPyPI before a public package release.
