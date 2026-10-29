"""Regression checks for the inference engine's retained candidate state."""

from __future__ import annotations

from dataclasses import replace
from weakref import ReferenceType, ref

import pytest
from numpy import arange, float64
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

import attitude_convention_inference.inference as inference_module
from attitude_convention_inference import (
    AttitudeDirection,
    AttitudeRepresentation,
    ConventionHypothesis,
    EulerMode,
    InferenceConfig,
    RateFrame,
    SearchSpace,
    TrajectoryData,
    best_interval_diagnostics,
    infer_conventions,
)
from attitude_convention_inference.scoring import score_hypothesis
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

    compact = score_hypothesis(data, hypothesis, config)
    detailed = score_hypothesis(data, hypothesis, config, retain_interval_details=True)

    assert compact.interval_details is None
    assert detailed.interval_details is not None
    assert_allclose(compact.interval_error_rotvecs, detailed.interval_error_rotvecs)
    assert len(best_interval_diagnostics(data, report, config)) == data.sample_count - 1
####


def test_euler_search_does_not_retain_past_attitude_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    motion = analytic_motion(arange(21, dtype=float64) * 0.025)
    data = TrajectoryData(
        representation=AttitudeRepresentation.EULER,
        orientation=motion.euler_xyz_rad,
        rates=motion.body_rates_rad_s,
        time_s=motion.time_s,
    )
    config = InferenceConfig(
        search_space=replace(
            SearchSpace(),
            attitude_directions=(
                AttitudeDirection.BODY_TO_WORLD,
                AttitudeDirection.WORLD_TO_BODY,
            ),
            euler_orders=("xyz", "zyx"),
            euler_modes=(EulerMode.EXTRINSIC,),
            rate_frames=(RateFrame.BODY,),
            rate_orders=("pqr", "rqp"),
        )
    )
    original_decode = inference_module.decode_attitudes
    decoded_refs: list[ReferenceType[Rotation]] = []
    live_before_decode: list[int] = []

    def track_decode(data: TrajectoryData, hypothesis: ConventionHypothesis) -> Rotation:
        live_before_decode.append(sum(item() is not None for item in decoded_refs))
        rotation = original_decode(data, hypothesis)
        decoded_refs.append(ref(rotation))
        return rotation
    ####

    monkeypatch.setattr(inference_module, "decode_attitudes", track_decode)
    report = infer_conventions(data, config)

    assert report.hypothesis_count == 8
    assert len(decoded_refs) == 4
    assert max(live_before_decode) <= 1
####
