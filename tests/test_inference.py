from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from attitude_convention_inference import (
    AngleUnit,
    AttitudeDirection,
    AttitudeRepresentation,
    EulerMode,
    InferenceConfig,
    QuaternionOrder,
    RateFrame,
    RateUnit,
    SearchSpace,
    infer_conventions,
)
from attitude_convention_inference.core import TrajectoryData
from attitude_convention_inference.synthetic import (
    SyntheticConvention,
    encode_synthetic_trajectory,
    rich_three_axis_rates,
)


def _time(sample_count: int = 161, dt: float = 0.025) -> np.ndarray:
    return np.arange(sample_count, dtype=np.float64) * dt


####


def test_recovers_scalar_first_world_to_body_world_rates_and_rqp_order() -> None:
    time_s = _time()
    rates = rich_three_axis_rates(time_s)
    truth = SyntheticConvention(
        representation=AttitudeRepresentation.QUATERNION,
        attitude_direction=AttitudeDirection.WORLD_TO_BODY,
        quaternion_order=QuaternionOrder.SCALAR_FIRST,
        rate_frame=RateFrame.WORLD,
        rate_order="rqp",
    )
    data = encode_synthetic_trajectory(time_s, rates, truth)
    report = infer_conventions(data)

    best = report.best.hypothesis
    assert best.quaternion_order is truth.quaternion_order
    assert best.attitude_direction is truth.attitude_direction
    assert best.rate_frame is truth.rate_frame
    assert best.rate_order == truth.rate_order
    assert report.best.interval_rmse_rad < 1.0e-7
    assert report.diagnostics.confidence > 0.70


####


def test_recovers_euler_intrinsic_order_direction_and_degree_units() -> None:
    time_s = _time(sample_count=181, dt=0.02)
    rates = rich_three_axis_rates(time_s, scale=0.7)
    truth = SyntheticConvention(
        representation=AttitudeRepresentation.EULER,
        attitude_direction=AttitudeDirection.BODY_TO_WORLD,
        euler_order="zxy",
        euler_mode=EulerMode.INTRINSIC,
        angle_unit=AngleUnit.DEGREES,
        rate_frame=RateFrame.BODY,
        rate_order="qpr",
        rate_unit=RateUnit.DEGREES_PER_SECOND,
    )
    data = encode_synthetic_trajectory(time_s, rates, truth)
    config = InferenceConfig(
        search_space=SearchSpace(
            angle_units=(AngleUnit.RADIANS, AngleUnit.DEGREES),
            rate_units=(RateUnit.RADIANS_PER_SECOND, RateUnit.DEGREES_PER_SECOND),
        )
    )
    report = infer_conventions(data, config)

    # With both the Euler axis order and p/q/r channel order free, a pure
    # attitude/rate trajectory has an exact global-axis relabeling symmetry.
    # The truth must remain in the tied equivalence class rather than being
    # replaced by a falsely confident arbitrary winner.
    tied = tuple(
        candidate.hypothesis
        for candidate in report.rankings
        if candidate.delta_criterion <= config.near_tie_delta_criterion
    )
    assert any(
        candidate.euler_order == truth.euler_order
        and candidate.euler_mode is truth.euler_mode
        and candidate.angle_unit is truth.angle_unit
        and candidate.attitude_direction is truth.attitude_direction
        and candidate.rate_frame is truth.rate_frame
        and candidate.rate_order == truth.rate_order
        and candidate.rate_unit is truth.rate_unit
        for candidate in tied
    )
    assert report.diagnostics.near_tie_count >= 2
    assert "euler_order" in report.diagnostics.unresolved_dimensions
    assert "rate_order" in report.diagnostics.unresolved_dimensions
    assert report.diagnostics.confidence < 0.50

    constrained = InferenceConfig(
        search_space=SearchSpace(
            angle_units=(AngleUnit.RADIANS, AngleUnit.DEGREES),
            rate_units=(RateUnit.DEGREES_PER_SECOND,),
            rate_frames=(RateFrame.BODY,),
            rate_orders=(truth.rate_order,),
        )
    )
    constrained_report = infer_conventions(data, constrained)
    best = constrained_report.best.hypothesis
    assert best.euler_order == truth.euler_order
    assert best.euler_mode is truth.euler_mode
    assert best.angle_unit is truth.angle_unit
    assert best.attitude_direction is truth.attitude_direction
    assert best.rate_frame is truth.rate_frame
    assert best.rate_order == truth.rate_order
    assert best.rate_unit is truth.rate_unit
    assert constrained_report.best.interval_rmse_rad < 1.0e-7


####


def test_infers_missing_constant_sample_period() -> None:
    time_s = _time(sample_count=201, dt=0.04)
    rates = rich_three_axis_rates(time_s, scale=0.8)
    truth = SyntheticConvention(
        representation=AttitudeRepresentation.QUATERNION,
        attitude_direction=AttitudeDirection.BODY_TO_WORLD,
        quaternion_order=QuaternionOrder.SCALAR_LAST,
        rate_frame=RateFrame.BODY,
        rate_order="prq",
    )
    timed_data = encode_synthetic_trajectory(time_s, rates, truth)
    data = TrajectoryData(
        representation=timed_data.representation,
        orientation=timed_data.orientation,
        rates=timed_data.rates,
        time_s=None,
    )
    report = infer_conventions(data)

    best = report.best.hypothesis
    assert best.quaternion_order is truth.quaternion_order
    assert best.attitude_direction is truth.attitude_direction
    assert best.rate_frame is truth.rate_frame
    assert best.rate_order == truth.rate_order
    assert report.best.inferred_sample_period_s is not None
    assert abs(report.best.inferred_sample_period_s - 0.04) < 2.0e-4
    assert report.diagnostics.time_source == "inferred_sample_period"


####


def test_quaternion_sign_flips_do_not_change_inference() -> None:
    time_s = _time()
    rates = rich_three_axis_rates(time_s)
    truth = SyntheticConvention(
        representation=AttitudeRepresentation.QUATERNION,
        attitude_direction=AttitudeDirection.BODY_TO_WORLD,
        quaternion_order=QuaternionOrder.SCALAR_LAST,
        rate_frame=RateFrame.BODY,
        rate_order="pqr",
    )
    data = encode_synthetic_trajectory(
        time_s,
        rates,
        truth,
        quaternion_sign_flips=True,
    )
    report = infer_conventions(data)

    assert report.best.hypothesis.quaternion_order is QuaternionOrder.SCALAR_LAST
    assert report.best.hypothesis.attitude_direction is AttitudeDirection.BODY_TO_WORLD
    assert report.best.interval_rmse_rad < 1.0e-7


####


def test_coaxial_motion_marks_body_vs_world_rate_frame_ambiguous() -> None:
    time_s = _time(sample_count=121, dt=0.02)
    rates = np.zeros((time_s.size, 3), dtype=np.float64)
    rates[:, 2] = 0.8
    truth = SyntheticConvention(
        representation=AttitudeRepresentation.QUATERNION,
        attitude_direction=AttitudeDirection.BODY_TO_WORLD,
        quaternion_order=QuaternionOrder.SCALAR_LAST,
        rate_frame=RateFrame.BODY,
        rate_order="pqr",
    )
    data = encode_synthetic_trajectory(
        time_s,
        rates,
        truth,
        initial_attitude=Rotation.identity(),
    )
    config = InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=(AttitudeDirection.BODY_TO_WORLD,),
            quaternion_orders=(QuaternionOrder.SCALAR_LAST,),
            rate_frames=(RateFrame.BODY, RateFrame.WORLD),
            rate_orders=("pqr",),
        )
    )
    report = infer_conventions(data, config)

    assert report.diagnostics.near_tie_count == 2
    assert "rate_frame" in report.diagnostics.unresolved_dimensions
    assert report.diagnostics.confidence < 0.50
    assert report.diagnostics.effective_excitation_rank < 1.05


####


def test_very_slow_motion_caps_confidence() -> None:
    time_s = _time(sample_count=81, dt=0.02)
    rates = rich_three_axis_rates(time_s, scale=1.0e-4)
    truth = SyntheticConvention(
        representation=AttitudeRepresentation.QUATERNION,
        attitude_direction=AttitudeDirection.BODY_TO_WORLD,
        quaternion_order=QuaternionOrder.SCALAR_LAST,
        rate_frame=RateFrame.BODY,
        rate_order="pqr",
    )
    data = encode_synthetic_trajectory(time_s, rates, truth)
    constrained = InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=(AttitudeDirection.BODY_TO_WORLD,),
            quaternion_orders=(QuaternionOrder.SCALAR_LAST,),
            rate_frames=(RateFrame.BODY,),
            rate_orders=("pqr",),
        )
    )
    report = infer_conventions(data, constrained)

    assert report.diagnostics.confidence <= 0.35
    assert any(
        "little total rotational excitation" in warning
        for warning in report.diagnostics.warnings
    )


####


def test_high_rate_per_sample_wrap_caps_confidence_and_warns() -> None:
    time_s = _time(sample_count=61, dt=0.01)
    rates = np.zeros((time_s.size, 3), dtype=np.float64)
    rates[:, 0] = 380.0
    rates[:, 1] = 90.0 * np.sin(2.0 * time_s)
    truth = SyntheticConvention(
        representation=AttitudeRepresentation.QUATERNION,
        attitude_direction=AttitudeDirection.BODY_TO_WORLD,
        quaternion_order=QuaternionOrder.SCALAR_LAST,
        rate_frame=RateFrame.BODY,
        rate_order="pqr",
    )
    data = encode_synthetic_trajectory(time_s, rates, truth)
    constrained = InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=(AttitudeDirection.BODY_TO_WORLD,),
            quaternion_orders=(QuaternionOrder.SCALAR_LAST,),
            rate_frames=(RateFrame.BODY,),
            rate_orders=("pqr",),
        )
    )
    report = infer_conventions(data, constrained)

    assert report.best.hard_alias_fraction > 0.0
    assert report.diagnostics.confidence <= 0.35
    assert any("wrap count" in warning for warning in report.diagnostics.warnings)


####


def test_partial_constraints_reduce_hypothesis_count() -> None:
    config = InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=(AttitudeDirection.BODY_TO_WORLD,),
            quaternion_orders=(QuaternionOrder.SCALAR_FIRST,),
            rate_frames=(RateFrame.BODY,),
            rate_orders=("pqr", "rqp"),
        )
    )
    time_s = _time(sample_count=31)
    truth = SyntheticConvention(
        representation=AttitudeRepresentation.QUATERNION,
        attitude_direction=AttitudeDirection.BODY_TO_WORLD,
        quaternion_order=QuaternionOrder.SCALAR_FIRST,
        rate_frame=RateFrame.BODY,
        rate_order="pqr",
    )
    data = encode_synthetic_trajectory(time_s, rich_three_axis_rates(time_s), truth)
    report = infer_conventions(data, config)
    assert report.hypothesis_count == 2


####
