"""Regression checks for the inference engine's retained candidate state."""

from __future__ import annotations

from numpy import arange, float64
from numpy.testing import assert_allclose

from attitude_convention_inference import (
    AttitudeRepresentation,
    InferenceConfig,
    TrajectoryData,
    best_interval_diagnostics,
    infer_conventions,
)
from attitude_convention_inference.inference import (
    _score_hypothesis,  # pyright: ignore[reportPrivateUsage]
)
from scripts.generate_validation_corpus import analytic_motion


def test_full_scoring_retains_interval_details_only_on_request() -> None:
    motion = analytic_motion(arange(31, dtype=float64) * 0.025)
    data = TrajectoryData(
        representation=AttitudeRepresentation.QUATERNION,
        orientation=motion.quaternion_xyzw,
        rates=motion.body_rates_rad_s,
        time_s=motion.time_s,
    )
    config = InferenceConfig()
    report = infer_conventions(data, config)
    hypothesis = report.best.hypothesis

    compact = _score_hypothesis(data, hypothesis, config)
    detailed = _score_hypothesis(data, hypothesis, config, retain_interval_details=True)

    assert compact.interval_details is None
    assert detailed.interval_details is not None
    assert_allclose(compact.interval_error_rotvecs, detailed.interval_error_rotvecs)
    assert len(best_interval_diagnostics(data, report, config)) == data.sample_count - 1
####
