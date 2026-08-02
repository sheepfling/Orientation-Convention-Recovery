from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from .core import FloatArray, TrajectoryData, integrate_interval_increments
from .models import (
    AngleUnit,
    AttitudeDirection,
    AttitudeRepresentation,
    EulerMode,
    QuaternionOrder,
    RateFrame,
    RateUnit,
)


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
    times = np.asarray(time_s, dtype=np.float64)
    rates = np.asarray(angular_rates_rad_s, dtype=np.float64)
    if times.ndim != 1 or rates.shape != (times.size, 3):
        raise ValueError("time_s must be (N,) and angular_rates_rad_s must be (N, 3).")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("time_s must be strictly increasing.")

    increments, _, _ = integrate_interval_increments(
        rates,
        np.diff(times),
        rate_frame,
        integration_substeps,
    )
    current = initial_attitude or Rotation.from_euler("xyz", [0.31, -0.22, 0.47])
    quaternions: list[NDArray[np.float64]] = [
        np.asarray(current.as_quat(), dtype=np.float64)
    ]
    for increment in increments:
        current = current * increment if rate_frame is RateFrame.BODY else increment * current
        quaternions.append(np.asarray(current.as_quat(), dtype=np.float64))
    return Rotation.from_quat(np.stack(quaternions))


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
        orientation = raw_attitudes.as_quat(
            scalar_first=convention.quaternion_order is QuaternionOrder.SCALAR_FIRST
        )
        if quaternion_sign_flips:
            signs = np.where(np.arange(orientation.shape[0]) % 2 == 0, 1.0, -1.0)
            orientation = orientation * signs[:, np.newaxis]
    else:
        if convention.euler_order is None or convention.euler_mode is None:
            raise ValueError("Euler synthetic data require euler_order and euler_mode.")
        if convention.angle_unit is None:
            raise ValueError("Euler synthetic data require angle_unit.")
        sequence = (
            convention.euler_order.upper()
            if convention.euler_mode is EulerMode.INTRINSIC
            else convention.euler_order.lower()
        )
        orientation = raw_attitudes.as_euler(
            sequence,
            degrees=convention.angle_unit is AngleUnit.DEGREES,
        )

    canonical_rates = np.asarray(angular_rates_rad_s, dtype=np.float64).copy()
    if convention.rate_unit is RateUnit.DEGREES_PER_SECOND:
        canonical_rates = np.rad2deg(canonical_rates)
    component = {"p": 0, "q": 1, "r": 2}
    raw_rates = canonical_rates[:, [component[name] for name in convention.rate_order]]

    return TrajectoryData(
        representation=convention.representation,
        orientation=np.asarray(orientation, dtype=np.float64),
        rates=raw_rates,
        time_s=np.asarray(time_s, dtype=np.float64),
    )


####


def rich_three_axis_rates(time_s: FloatArray, scale: float = 1.0) -> FloatArray:
    times = np.asarray(time_s, dtype=np.float64)
    return scale * np.column_stack(
        (
            0.72 * np.sin(0.83 * times) + 0.31 * np.cos(2.17 * times + 0.2),
            0.67 * np.cos(1.11 * times + 0.4) + 0.26 * np.sin(2.63 * times),
            0.61 * np.sin(1.47 * times + 0.8) + 0.29 * np.cos(0.53 * times),
        )
    )


####
