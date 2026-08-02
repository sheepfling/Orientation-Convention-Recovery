from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from .models import (
    AngleUnit,
    AttitudeDirection,
    AttitudeRepresentation,
    ConventionHypothesis,
    EulerMode,
    InferenceConfig,
    QuaternionOrder,
    RateFrame,
    RateUnit,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TrajectoryData:
    representation: AttitudeRepresentation
    orientation: FloatArray
    rates: FloatArray
    time_s: FloatArray | None = None

    def __post_init__(self) -> None:
        orientation = np.asarray(self.orientation, dtype=np.float64)
        rates = np.asarray(self.rates, dtype=np.float64)
        time_s = None if self.time_s is None else np.asarray(self.time_s, dtype=np.float64)

        expected_width = 4 if self.representation is AttitudeRepresentation.QUATERNION else 3
        if orientation.ndim != 2 or orientation.shape[1] != expected_width:
            raise ValueError(
                f"orientation must have shape (N, {expected_width}) for {self.representation}."
            )
        if rates.ndim != 2 or rates.shape[1] != 3:
            raise ValueError("rates must have shape (N, 3).")
        if orientation.shape[0] != rates.shape[0]:
            raise ValueError("orientation and rates must contain the same number of samples.")
        if orientation.shape[0] < 3:
            raise ValueError("At least three samples are required for convention inference.")
        if not np.all(np.isfinite(orientation)) or not np.all(np.isfinite(rates)):
            raise ValueError("orientation and rates must contain only finite values.")

        if time_s is not None:
            if time_s.ndim != 1 or time_s.shape[0] != orientation.shape[0]:
                raise ValueError("time_s must have shape (N,).")
            if not np.all(np.isfinite(time_s)):
                raise ValueError("time_s must contain only finite values.")
            if not np.all(np.diff(time_s) > 0.0):
                raise ValueError("time_s must be strictly increasing.")

        if self.representation is AttitudeRepresentation.QUATERNION:
            norms = np.linalg.norm(orientation, axis=1)
            if np.any(norms < 1.0e-12):
                raise ValueError("Quaternion samples must have nonzero norm.")

        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "rates", rates)
        object.__setattr__(self, "time_s", time_s)

    ####

    @property
    def sample_count(self) -> int:
        return int(self.orientation.shape[0])

    ####


####


def decode_attitudes(data: TrajectoryData, hypothesis: ConventionHypothesis) -> Rotation:
    if data.representation is AttitudeRepresentation.QUATERNION:
        if hypothesis.quaternion_order is None:
            raise ValueError("Quaternion data require a quaternion-order hypothesis.")
        scalar_first = hypothesis.quaternion_order is QuaternionOrder.SCALAR_FIRST
        decoded = Rotation.from_quat(data.orientation, scalar_first=scalar_first)
    else:
        if hypothesis.euler_order is None or hypothesis.euler_mode is None:
            raise ValueError("Euler data require Euler order and mode hypotheses.")
        if hypothesis.angle_unit is None:
            raise ValueError("Euler data require an angle-unit hypothesis.")
        sequence = (
            hypothesis.euler_order.upper()
            if hypothesis.euler_mode is EulerMode.INTRINSIC
            else hypothesis.euler_order.lower()
        )
        decoded = Rotation.from_euler(
            sequence,
            data.orientation,
            degrees=hypothesis.angle_unit is AngleUnit.DEGREES,
        )

    if hypothesis.attitude_direction is AttitudeDirection.WORLD_TO_BODY:
        decoded = decoded.inv()
    return decoded


####


def decode_rates(data: TrajectoryData, hypothesis: ConventionHypothesis) -> FloatArray:
    positions = {component: index for index, component in enumerate(hypothesis.rate_order)}
    canonical = data.rates[:, [positions["p"], positions["q"], positions["r"]]].copy()
    if hypothesis.rate_unit is RateUnit.DEGREES_PER_SECOND:
        canonical *= pi / 180.0
    canonical *= float(hypothesis.rate_sign)
    return canonical


####


def observed_interval_increments(attitudes: Rotation, frame: RateFrame) -> Rotation:
    previous = attitudes[:-1]
    following = attitudes[1:]
    if frame is RateFrame.BODY:
        return previous.inv() * following
    return following * previous.inv()


####


def infer_constant_sample_period(
    attitudes: Rotation,
    rates_rad_s: FloatArray,
    frame: RateFrame,
    config: InferenceConfig,
) -> tuple[float, tuple[str, ...]]:
    observed = observed_interval_increments(attitudes, frame).as_rotvec()
    midpoint_rates = 0.5 * (rates_rad_s[:-1] + rates_rad_s[1:])
    rate_energy = np.einsum("ij,ij->i", midpoint_rates, midpoint_rates)
    valid = rate_energy > 1.0e-20
    warnings: list[str] = []

    lower, configured_upper = config.inferred_sample_period_bounds_s
    if not np.any(valid):
        warnings.append(
            "Sample period is unidentifiable because all candidate rates are near zero."
        )
        return lower, tuple(warnings)

    numerator = float(np.sum(np.einsum("ij,ij->i", midpoint_rates[valid], observed[valid])))
    denominator = float(np.sum(rate_energy[valid]))
    estimate = numerator / denominator

    rate_norm = np.sqrt(rate_energy[valid])
    representative_rate = float(np.quantile(rate_norm, 0.99))
    no_alias_upper = (
        config.near_alias_step_rad / representative_rate
        if representative_rate > 1.0e-15
        else configured_upper
    )
    upper = max(lower * 1.01, min(configured_upper, no_alias_upper))

    if not np.isfinite(estimate) or estimate <= 0.0:
        warnings.append(
            "The unconstrained sample-period fit was nonpositive; this hypothesis is "
            "directionally inconsistent."
        )
        estimate = lower
    if estimate > upper:
        warnings.append(
            "The fitted sample period crossed the no-alias bound and was clipped; "
            "timing is ambiguous."
        )
    estimate = float(np.clip(estimate, lower, upper))
    return estimate, tuple(warnings)


####


def resolve_interval_durations(
    data: TrajectoryData,
    attitudes: Rotation,
    rates_rad_s: FloatArray,
    frame: RateFrame,
    config: InferenceConfig,
) -> tuple[FloatArray, float | None, str, tuple[str, ...]]:
    if data.time_s is not None:
        return np.diff(data.time_s), None, "timestamps", ()
    if config.sample_period_s is not None:
        intervals = np.full(data.sample_count - 1, config.sample_period_s, dtype=np.float64)
        return intervals, None, "provided_sample_period", ()
    if not config.infer_sample_period_when_missing:
        raise ValueError(
            "No timestamps or sample_period_s were supplied and sample-period "
            "inference is disabled."
        )

    inferred, warnings = infer_constant_sample_period(attitudes, rates_rad_s, frame, config)
    intervals = np.full(data.sample_count - 1, inferred, dtype=np.float64)
    return intervals, inferred, "inferred_sample_period", warnings


####


def integrate_interval_increments(
    rates_rad_s: FloatArray,
    interval_durations_s: FloatArray,
    frame: RateFrame,
    substeps: int,
) -> tuple[Rotation, FloatArray, FloatArray]:
    interval_count = rates_rad_s.shape[0] - 1
    predicted = Rotation.identity(num=interval_count)
    path_angles = np.zeros(interval_count, dtype=np.float64)

    for substep in range(substeps):
        alpha = (substep + 0.5) / substeps
        interpolated = (1.0 - alpha) * rates_rad_s[:-1] + alpha * rates_rad_s[1:]
        substep_dt = interval_durations_s / substeps
        increments = Rotation.from_rotvec(interpolated * substep_dt[:, np.newaxis])
        path_angles += np.linalg.norm(interpolated, axis=1) * substep_dt
        if frame is RateFrame.BODY:
            predicted = predicted * increments
        else:
            predicted = increments * predicted

    rate_change_angles = (
        np.linalg.norm(rates_rad_s[1:] - rates_rad_s[:-1], axis=1) * interval_durations_s
    )
    return predicted, path_angles, rate_change_angles


####


def compose_increment_windows(
    interval_increments: Rotation,
    window_size: int,
    frame: RateFrame,
) -> tuple[Rotation, NDArray[np.int64]]:
    interval_count = len(interval_increments)
    starts = np.arange(0, interval_count - window_size + 1, window_size, dtype=np.int64)
    quaternions: list[FloatArray] = []

    for start in starts:
        total = Rotation.identity()
        for offset in range(window_size):
            increment = interval_increments[int(start + offset)]
            if frame is RateFrame.BODY:
                total = total * increment
            else:
                total = increment * total
        quaternions.append(np.asarray(total.as_quat(), dtype=np.float64))

    if not quaternions:
        return Rotation.from_quat(np.empty((0, 4))), starts
    return Rotation.from_quat(np.stack(quaternions)), starts


####


def observed_window_increments(
    attitudes: Rotation,
    starts: NDArray[np.int64],
    window_size: int,
    frame: RateFrame,
) -> Rotation:
    previous = attitudes[starts]
    following = attitudes[starts + window_size]
    if frame is RateFrame.BODY:
        return previous.inv() * following
    return following * previous.inv()


####


def quaternion_norm_statistics(data: TrajectoryData) -> tuple[float, float] | None:
    if data.representation is not AttitudeRepresentation.QUATERNION:
        return None
    norms = np.linalg.norm(data.orientation, axis=1)
    return float(np.mean(np.abs(norms - 1.0))), float(np.max(np.abs(norms - 1.0)))


####
