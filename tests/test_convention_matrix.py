from __future__ import annotations

from itertools import product
from typing import Any, cast

import numpy as np
import pytest
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
    TrajectoryData,
    infer_conventions,
)
from attitude_convention_inference.synthetic import (
    SyntheticConvention,
    encode_synthetic_trajectory,
    rich_three_axis_rates,
)


def _time(sample_count: int = 81, dt: float = 0.025) -> np.ndarray:
    return np.arange(sample_count, dtype=np.float64) * dt
####




def _from_quat(quaternions: np.ndarray, *, scalar_first: bool = False) -> Rotation:
    return cast(
        Rotation,
        cast(Any, Rotation).from_quat(quaternions, scalar_first=scalar_first),
    )
####




def _from_rotvec(rotvecs: np.ndarray) -> Rotation:
    return cast(Rotation, cast(Any, Rotation).from_rotvec(rotvecs))
####




def _quaternions(rotation: Rotation, *, scalar_first: bool = False) -> np.ndarray:
    return np.asarray(cast(Any, rotation).as_quat(scalar_first=scalar_first), dtype=np.float64)
####




QUATERNION_CASES = tuple(
    product(
        tuple(AttitudeDirection),
        tuple(QuaternionOrder),
        tuple(RateFrame),
        ("pqr", "rqp"),
    )
)


@pytest.mark.parametrize(
    ("direction", "quaternion_order", "rate_frame", "rate_order"),
    QUATERNION_CASES,
)
def test_quaternion_convention_matrix(
    direction: AttitudeDirection,
    quaternion_order: QuaternionOrder,
    rate_frame: RateFrame,
    rate_order: str,
) -> None:
    time_s = _time()
    truth = SyntheticConvention(
        representation=AttitudeRepresentation.QUATERNION,
        attitude_direction=direction,
        quaternion_order=quaternion_order,
        rate_frame=rate_frame,
        rate_order=rate_order,
    )
    data = encode_synthetic_trajectory(time_s, rich_three_axis_rates(time_s), truth)
    config = InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=tuple(AttitudeDirection),
            quaternion_orders=tuple(QuaternionOrder),
            rate_frames=tuple(RateFrame),
            rate_orders=("pqr", "rqp"),
        ),
        window_sizes=(1, 2, 4),
    )

    report = infer_conventions(data, config)

    best = report.best.hypothesis
    assert best.attitude_direction is direction
    assert best.quaternion_order is quaternion_order
    assert best.rate_frame is rate_frame
    assert best.rate_order == rate_order
    assert report.best.interval_rmse_rad < 1.0e-6
####




EULER_CASES = tuple(product(("xyz", "zxy", "zyx"), tuple(EulerMode), tuple(AttitudeDirection)))


@pytest.mark.parametrize(("euler_order", "euler_mode", "direction"), EULER_CASES)
def test_euler_convention_matrix_with_rate_axes_anchored(
    euler_order: str,
    euler_mode: EulerMode,
    direction: AttitudeDirection,
) -> None:
    time_s = _time(sample_count=101, dt=0.02)
    truth = SyntheticConvention(
        representation=AttitudeRepresentation.EULER,
        attitude_direction=direction,
        euler_order=euler_order,
        euler_mode=euler_mode,
        angle_unit=AngleUnit.RADIANS,
        rate_frame=RateFrame.BODY,
        rate_order="pqr",
    )
    data = encode_synthetic_trajectory(time_s, rich_three_axis_rates(time_s, scale=0.65), truth)
    config = InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=tuple(AttitudeDirection),
            euler_orders=("xyz", "zxy", "zyx"),
            euler_modes=tuple(EulerMode),
            angle_units=(AngleUnit.RADIANS,),
            rate_frames=(RateFrame.BODY,),
            rate_orders=("pqr",),
        ),
        window_sizes=(1, 2, 4),
    )

    report = infer_conventions(data, config)

    best = report.best.hypothesis
    assert best.euler_order == euler_order
    assert best.euler_mode is euler_mode
    assert best.attitude_direction is direction
    assert report.best.interval_rmse_rad < 1.0e-6
####




def test_trimmed_multiscale_score_tolerates_sparse_attitude_outliers() -> None:
    time_s = _time(sample_count=161, dt=0.02)
    truth = SyntheticConvention(
        representation=AttitudeRepresentation.QUATERNION,
        attitude_direction=AttitudeDirection.WORLD_TO_BODY,
        quaternion_order=QuaternionOrder.SCALAR_FIRST,
        rate_frame=RateFrame.WORLD,
        rate_order="qpr",
    )
    clean = encode_synthetic_trajectory(time_s, rich_three_axis_rates(time_s), truth)
    if clean.orientation is None or clean.rates is None or clean.time_s is None:
        raise AssertionError("synthetic trajectory must contain all streams")
    ####
    raw = _from_quat(clean.orientation, scalar_first=True)
    quaternions = _quaternions(raw)
    for index, error in zip((37, 101), ([0.08, -0.03, 0.05], [-0.06, 0.04, 0.02]), strict=True):
        quaternions[index] = _quaternions(
            _from_rotvec(np.asarray(error, dtype=np.float64)) * raw[index]
        )
    ####
    contaminated = TrajectoryData(
        representation=clean.representation,
        orientation=_quaternions(_from_quat(quaternions), scalar_first=True),
        rates=clean.rates,
        time_s=clean.time_s,
    )
    config = InferenceConfig(
        search_space=SearchSpace(
            quaternion_orders=tuple(QuaternionOrder),
            attitude_directions=tuple(AttitudeDirection),
            rate_frames=tuple(RateFrame),
            rate_orders=("pqr", "qpr", "rqp"),
        ),
        trim_fraction=0.04,
    )

    report = infer_conventions(contaminated, config)

    best = report.best.hypothesis
    assert best.quaternion_order is truth.quaternion_order
    assert best.attitude_direction is truth.attitude_direction
    assert best.rate_frame is truth.rate_frame
    assert best.rate_order == truth.rate_order
####




def test_rate_unit_and_missing_time_are_reported_as_unidentifiable_when_both_free() -> None:
    time_s = _time(sample_count=121, dt=0.04)
    truth = SyntheticConvention(
        representation=AttitudeRepresentation.QUATERNION,
        attitude_direction=AttitudeDirection.BODY_TO_WORLD,
        quaternion_order=QuaternionOrder.SCALAR_LAST,
        rate_frame=RateFrame.BODY,
        rate_order="pqr",
        rate_unit=RateUnit.RADIANS_PER_SECOND,
    )
    timed = encode_synthetic_trajectory(time_s, rich_three_axis_rates(time_s), truth)
    if truth.quaternion_order is None:
        raise AssertionError("truth must define a quaternion order")
    ####
    untimed = TrajectoryData(
        representation=timed.representation,
        orientation=timed.orientation,
        rates=timed.rates,
    )
    config = InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=(truth.attitude_direction,),
            quaternion_orders=(truth.quaternion_order,),
            rate_frames=(truth.rate_frame,),
            rate_orders=(truth.rate_order,),
            rate_units=tuple(RateUnit),
        )
    )

    report = infer_conventions(untimed, config)

    assert report.diagnostics.near_tie_count == 2
    assert "rate_unit" in report.diagnostics.unresolved_dimensions
    periods = sorted(
        candidate.inferred_sample_period_s
        for candidate in report.rankings[:2]
        if candidate.inferred_sample_period_s is not None
    )
    assert periods[0] < periods[1]
    assert report.diagnostics.confidence < 0.50
####


