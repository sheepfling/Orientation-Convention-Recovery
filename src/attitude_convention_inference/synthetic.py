"""Synthetic trajectory generation for examples and contract tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from numpy import (
    all as np_all,
)
from numpy import (
    arange,
    asarray,
    column_stack,
    cos,
    diff,
    float64,
    int64,
    newaxis,
    rad2deg,
    sin,
    stack,
    where,
)
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from .integration import integrate_interval_increments
from .math_utils import FloatArray
from .math_utils import rotation_from_euler as _rotation_from_euler
from .math_utils import rotation_from_quat as _rotation_from_quat
from .math_utils import rotation_product as _rotation_product
from .math_utils import rotation_quaternions as _rotation_quaternions
from .models import (
    AngleUnit,
    AttitudeDirection,
    AttitudeRepresentation,
    EulerMode,
    QuaternionOrder,
    RateFrame,
    RateUnit,
    angle_unit_scale,
)
from .trajectory import TrajectoryData

__all__ = [
    "SyntheticConvention",
    "encode_synthetic_trajectory",
    "integrate_attitude_trajectory",
    "rich_three_axis_rates",
]


@dataclass(frozen=True, slots=True)
class SyntheticConvention:
    representation: AttitudeRepresentation
    attitude_direction: AttitudeDirection
    rate_frame: RateFrame
    rate_order: str = "pqr"
    rate_unit: RateUnit = RateUnit.RADIANS_PER_SECOND
    quaternion_order: QuaternionOrder | None = None
    euler_order: str | None = None
    euler_mode: EulerMode | None = None
    angle_unit: AngleUnit | None = None
####


def integrate_attitude_trajectory(
    time_s: FloatArray,
    angular_rates_rad_s: FloatArray,
    rate_frame: RateFrame,
    initial_attitude: Rotation | None = None,
    integration_substeps: int = 32,
) -> Rotation:
    times = asarray(time_s, dtype=float64)
    rates = asarray(angular_rates_rad_s, dtype=float64)
    if times.ndim != 1 or rates.shape != (times.size, 3):
        raise ValueError("time_s must be (N,) and angular_rates_rad_s must be (N, 3).")
    ####
    if not np_all(diff(times) > 0.0):
        raise ValueError("time_s must be strictly increasing.")
    ####

    integrated = integrate_interval_increments(
        rates,
        diff(times),
        rate_frame,
        integration_substeps,
    )
    increments = integrated.predicted_increments
    current = (
        initial_attitude
        if initial_attitude is not None
        else _rotation_from_euler("xyz", asarray([0.31, -0.22, 0.47], dtype=float64))
    )
    quaternions: list[NDArray[float64]] = [_rotation_quaternions(current)]
    for increment in increments:
        current = (
            _rotation_product(current, increment)
            if rate_frame is RateFrame.BODY
            else _rotation_product(increment, current)
        )
        quaternions.append(_rotation_quaternions(current))
    ####
    return _rotation_from_quat(stack(quaternions))
####


def encode_synthetic_trajectory(
    time_s: FloatArray,
    angular_rates_rad_s: FloatArray,
    convention: SyntheticConvention,
    initial_attitude: Rotation | None = None,
    quaternion_sign_flips: bool = False,
) -> TrajectoryData:
    canonical_attitudes = integrate_attitude_trajectory(
        time_s,
        angular_rates_rad_s,
        convention.rate_frame,
        initial_attitude=initial_attitude,
    )
    raw_attitudes = (
        canonical_attitudes
        if convention.attitude_direction is AttitudeDirection.BODY_TO_WORLD
        else canonical_attitudes.inv()
    )

    if convention.representation is AttitudeRepresentation.QUATERNION:
        if convention.quaternion_order is None:
            raise ValueError("Quaternion synthetic data require quaternion_order.")
        ####
        orientation: FloatArray = asarray(
            cast(Any, raw_attitudes).as_quat(
                scalar_first=convention.quaternion_order is QuaternionOrder.SCALAR_FIRST
            ),
            dtype=float64,
        )
        if quaternion_sign_flips:
            indices: NDArray[int64] = arange(orientation.shape[0], dtype=int64)
            signs: FloatArray = where(indices % 2 == 0, 1.0, -1.0)
            orientation = orientation * signs[:, newaxis]
        ####
    else:
        if convention.euler_order is None or convention.euler_mode is None:
            raise ValueError("Euler synthetic data require euler_order and euler_mode.")
        ####
        if convention.angle_unit is None:
            raise ValueError("Euler synthetic data require angle_unit.")
        ####
        sequence = (
            convention.euler_order.upper()
            if convention.euler_mode is EulerMode.INTRINSIC
            else convention.euler_order.lower()
        )
        orientation = asarray(
            cast(Any, raw_attitudes).as_euler(sequence, degrees=False),
            dtype=float64,
        )
        orientation /= angle_unit_scale(convention.angle_unit)
    ####

    canonical_rates = asarray(angular_rates_rad_s, dtype=float64).copy()
    if convention.rate_unit is RateUnit.DEGREES_PER_SECOND:
        canonical_rates = rad2deg(canonical_rates)
    ####
    component = {"p": 0, "q": 1, "r": 2}
    raw_rates = canonical_rates[:, [component[name] for name in convention.rate_order]]

    return TrajectoryData(
        representation=convention.representation,
        orientation=asarray(orientation, dtype=float64),
        rates=raw_rates,
        time_s=asarray(time_s, dtype=float64),
    )
####


def rich_three_axis_rates(time_s: FloatArray, scale: float = 1.0) -> FloatArray:
    times = asarray(time_s, dtype=float64)
    return scale * column_stack(
        (
            0.72 * sin(0.83 * times) + 0.31 * cos(2.17 * times + 0.2),
            0.67 * cos(1.11 * times + 0.4) + 0.26 * sin(2.63 * times),
            0.61 * sin(1.47 * times + 0.8) + 0.29 * cos(0.53 * times),
        )
    )
####
