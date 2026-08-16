"""Typed trajectory validation, convention decoding, and kinematic integration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite, pi
from typing import Any, Protocol, cast

import scipy.optimize as scipy_optimize
from numpy import (
    abs,
    amax,
    arange,
    argmin,
    asarray,
    clip,
    diff,
    empty,
    float64,
    full,
    geomspace,
    int64,
    mean,
    newaxis,
    quantile,
    sqrt,
    zeros,
)
from numpy import (
    all as np_all,
)
from numpy import (
    any as np_any,
)
from numpy import (
    isfinite as np_isfinite,
)
from numpy import (
    sum as np_sum,
)
from numpy.linalg import norm
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from .math_utils import (
    FloatArray,
    rotation_from_euler,
    rotation_from_quat,
    rotation_from_rotvec,
    rotation_magnitudes,
    trimmed_mean_square,
)
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
    angle_unit_scale,
)

__all__ = [
    "FloatArray",
    "IntegratedIntervals",
    "IntervalDurationResolution",
    "QuaternionNormStatistics",
    "SamplePeriodEstimate",
    "TrajectoryData",
    "WindowComposition",
    "compose_increment_windows",
    "decode_attitudes",
    "decode_rates",
    "infer_constant_sample_period",
    "integrate_interval_increments",
    "observed_interval_increments",
    "observed_window_increments",
    "quaternion_norm_statistics",
    "refine_constant_sample_period",
    "resolve_interval_durations",
    "should_prune_by_exact_lower_bound",
]


@dataclass(frozen=True, slots=True)
class SamplePeriodEstimate:
    """Estimated constant sample period and diagnostic warnings."""

    period_s: float
    warnings: tuple[str, ...]
####


@dataclass(frozen=True, slots=True)
class IntervalDurationResolution:
    """Resolved interval durations together with their provenance and warnings."""

    durations_s: FloatArray
    inferred_period_s: float | None
    source: str
    warnings: tuple[str, ...]
####


@dataclass(frozen=True, slots=True)
class IntegratedIntervals:
    """Predicted interval increments and interval-level integration diagnostics."""

    predicted_increments: Rotation
    path_angles_rad: FloatArray
    rate_change_angles_rad: FloatArray
####


@dataclass(frozen=True, slots=True)
class WindowComposition:
    """Window-composed rotations and the source interval start indices."""

    window_increments: Rotation
    starts: NDArray[int64]
####


@dataclass(frozen=True, slots=True)
class QuaternionNormStatistics:
    """Quaternion unit-norm error statistics."""

    mean_abs_error: float
    max_abs_error: float
####


class _ScalarMinimizeResult(Protocol):
    success: bool
    x: float
####


_typed_minimize_scalar = cast(
    Callable[..., _ScalarMinimizeResult],
    getattr(scipy_optimize, "minimize_scalar"),  # noqa: B009
)


def should_prune_by_exact_lower_bound(
    partial_lower_bound: float,
    incumbent_score: float,
) -> bool:
    """Return whether an exact partial bound can abandon a candidate safely.

    This predicate is deliberately separate from statistical evidence elimination:
    it is valid only when the caller supplies a mathematically verified lower bound
    on the same final risk and a verified incumbent score.
    """

    lower_bound = float(partial_lower_bound)
    incumbent = float(incumbent_score)
    if not isfinite(lower_bound) or not isfinite(incumbent):
        raise ValueError("Exact pruning scores must be finite.")
    ####
    return lower_bound > incumbent
####


@dataclass(frozen=True, slots=True)
class TrajectoryData:
    representation: AttitudeRepresentation
    orientation: FloatArray | None
    rates: FloatArray | None
    time_s: FloatArray | None = None

    def __post_init__(self) -> None:
        orientation = None if self.orientation is None else asarray(self.orientation, dtype=float64)
        rates = None if self.rates is None else asarray(self.rates, dtype=float64)
        time_s = None if self.time_s is None else asarray(self.time_s, dtype=float64)

        if orientation is None and rates is None:
            raise ValueError("At least one of orientation or rates must be supplied.")
        ####
        expected_width = 4 if self.representation is AttitudeRepresentation.QUATERNION else 3
        if orientation is not None and (
            orientation.ndim != 2 or orientation.shape[1] != expected_width
        ):
            raise ValueError(
                f"orientation must have shape (N, {expected_width}) for {self.representation}."
            )
        ####
        if rates is not None and (rates.ndim != 2 or rates.shape[1] != 3):
            raise ValueError("rates must have shape (N, 3).")
        ####
        if orientation is None:
            if rates is None:
                raise RuntimeError("TrajectoryData has no streams.")
            ####
            sample_count = rates.shape[0]
        else:
            sample_count = orientation.shape[0]
        ####
        if orientation is not None and rates is not None and orientation.shape[0] != rates.shape[0]:
            raise ValueError("orientation and rates must contain the same number of samples.")
        ####
        if sample_count < 3:
            raise ValueError("At least three samples are required for convention inference.")
        ####
        if orientation is not None and not np_all(np_isfinite(orientation)):
            raise ValueError("orientation must contain only finite values.")
        ####
        if rates is not None and not np_all(np_isfinite(rates)):
            raise ValueError("orientation and rates must contain only finite values.")
        ####

        if time_s is not None:
            if time_s.ndim != 1 or time_s.shape[0] != sample_count:
                raise ValueError("time_s must have shape (N,).")
            ####
            if not np_all(np_isfinite(time_s)):
                raise ValueError("time_s must contain only finite values.")
            ####
            if not np_all(diff(time_s) > 0.0):
                raise ValueError("time_s must be strictly increasing.")
            ####
        ####

        if orientation is not None and self.representation is AttitudeRepresentation.QUATERNION:
            norms = norm(orientation, axis=1)
            if np_any(norms < 1.0e-12):
                raise ValueError("Quaternion samples must have nonzero norm.")
            ####
        ####

        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "rates", rates)
        object.__setattr__(self, "time_s", time_s)
    ####

    @property
    def sample_count(self) -> int:
        if self.orientation is not None:
            return int(self.orientation.shape[0])
        ####
        if self.rates is not None:
            return int(self.rates.shape[0])
        ####
        raise RuntimeError("TrajectoryData has no streams.")
    ####
####


def decode_attitudes(data: TrajectoryData, hypothesis: ConventionHypothesis) -> Rotation:
    if data.orientation is None:
        raise ValueError("No attitude stream is available for this operation.")
    ####
    if data.representation is AttitudeRepresentation.QUATERNION:
        if hypothesis.quaternion_order is None:
            raise ValueError("Quaternion data require a quaternion-order hypothesis.")
        ####
        scalar_first = hypothesis.quaternion_order is QuaternionOrder.SCALAR_FIRST
        decoded = rotation_from_quat(data.orientation, scalar_first=scalar_first)
    else:
        if hypothesis.euler_order is None or hypothesis.euler_mode is None:
            raise ValueError("Euler data require Euler order and mode hypotheses.")
        ####
        angle_unit = hypothesis.angle_unit
        if angle_unit is None:
            raise ValueError("Euler data require an angle-unit hypothesis.")
        ####
        sequence = (
            hypothesis.euler_order.upper()
            if hypothesis.euler_mode is EulerMode.INTRINSIC
            else hypothesis.euler_order.lower()
        )
        decoded = rotation_from_euler(
            sequence,
            data.orientation * angle_unit_scale(AngleUnit(angle_unit)),
        )
    ####

    if hypothesis.attitude_direction is AttitudeDirection.WORLD_TO_BODY:
        decoded = decoded.inv()
    ####
    return decoded
####


def decode_rates(data: TrajectoryData, hypothesis: ConventionHypothesis) -> FloatArray:
    if data.rates is None:
        raise ValueError("No angular-rate stream is available for this operation.")
    ####
    positions = {component: index for index, component in enumerate(hypothesis.rate_order)}
    canonical = data.rates[:, [positions["p"], positions["q"], positions["r"]]].copy()
    if hypothesis.rate_unit is RateUnit.DEGREES_PER_SECOND:
        canonical *= pi / 180.0
    ####
    canonical *= float(hypothesis.rate_sign)
    return canonical
####


def observed_interval_increments(attitudes: Rotation, frame: RateFrame) -> Rotation:
    previous = attitudes[:-1]
    following = attitudes[1:]
    if frame is RateFrame.BODY:
        return previous.inv() * following
    ####
    return following * previous.inv()
####


def infer_constant_sample_period(
    attitudes: Rotation,
    rates_rad_s: FloatArray,
    frame: RateFrame,
    config: InferenceConfig,
) -> SamplePeriodEstimate:
    observed = asarray(
        cast(Any, observed_interval_increments(attitudes, frame)).as_rotvec(),
        dtype=float64,
    )
    midpoint_rates = 0.5 * (rates_rad_s[:-1] + rates_rad_s[1:])
    rate_energy = np_sum(midpoint_rates * midpoint_rates, axis=1)
    valid = rate_energy > 1.0e-20
    warnings: list[str] = []

    lower, configured_upper = config.inferred_sample_period_bounds_s
    if not np_any(valid):
        warnings.append(
            "Sample period is unidentifiable because all candidate rates are near zero."
        )
        return SamplePeriodEstimate(lower, tuple(warnings))
    ####

    numerator = float(np_sum(midpoint_rates[valid] * observed[valid]))
    denominator = float(np_sum(rate_energy[valid]))
    estimate = numerator / denominator

    rate_norm = sqrt(rate_energy[valid])
    representative_rate = float(quantile(rate_norm, 0.99))
    no_alias_upper = (
        config.near_alias_step_rad / representative_rate
        if representative_rate > 1.0e-15
        else configured_upper
    )
    upper = max(lower * 1.01, min(configured_upper, no_alias_upper))

    if not np_isfinite(estimate) or estimate <= 0.0:
        warnings.append(
            "The unconstrained sample-period fit was nonpositive; this hypothesis is "
            "directionally inconsistent."
        )
        estimate = lower
    ####
    if estimate > upper:
        warnings.append(
            "The fitted sample period crossed the no-alias bound and was clipped; "
            "timing is ambiguous."
        )
    ####
    estimate = float(clip(estimate, lower, upper))
    return SamplePeriodEstimate(estimate, tuple(warnings))
####


def _period_geometric_risk(
    period_s: float,
    attitudes: Rotation,
    rates_rad_s: FloatArray,
    frame: RateFrame,
    config: InferenceConfig,
) -> float:
    """Evaluate the same raw multiscale geometric risk used for ranking."""

    durations = full(rates_rad_s.shape[0] - 1, period_s, dtype=float64)
    integrated = integrate_interval_increments(
        rates_rad_s,
        durations,
        frame,
        config.integration_substeps,
    )
    scale_risks: list[float] = []
    for window_size in config.window_sizes:
        if window_size > durations.size:
            continue
        ####
        composition = compose_increment_windows(integrated.predicted_increments, window_size, frame)
        observed_windows = observed_window_increments(
            attitudes, composition.starts, window_size, frame
        )
        scaled_errors = rotation_magnitudes(
            composition.window_increments.inv() * observed_windows
        ) / sqrt(float(window_size))
        scale_risks.append(trimmed_mean_square(scaled_errors, config.trim_fraction))
    ####
    return float(mean(scale_risks)) if scale_risks else float("inf")
####


def refine_constant_sample_period(
    attitudes: Rotation,
    rates_rad_s: FloatArray,
    frame: RateFrame,
    config: InferenceConfig,
) -> SamplePeriodEstimate:
    """Seed from the linear estimate and refine every promising group-risk basin."""

    initial_estimate = infer_constant_sample_period(
        attitudes,
        rates_rad_s,
        frame,
        config,
    )
    lower, upper = config.inferred_sample_period_bounds_s
    # A logarithmic grid covers broad timing ranges without spending dozens of
    # integrations on every discrete convention. The linear initializer receives
    # its own local refinement because it is usually the physically relevant basin.
    grid = geomspace(lower, upper, 9, dtype=float64)
    values = asarray(
        [
            _period_geometric_risk(float(period), attitudes, rates_rad_s, frame, config)
            for period in grid
        ],
        dtype=float64,
    )
    candidate_periods = [lower, upper, initial_estimate.period_s]

    def refine(lower_bound: float, upper_bound: float) -> None:
        def objective(period: float) -> float:
            return _period_geometric_risk(float(period), attitudes, rates_rad_s, frame, config)
        ####

        refined = _typed_minimize_scalar(
            objective,
            bounds=(lower_bound, upper_bound),
            method="bounded",
            options={"xatol": max(1.0e-12, (upper - lower) * 1.0e-10)},
        )
        if refined.success:
            candidate_periods.append(float(refined.x))
        ####
    ####

    refine(
        max(lower, initial_estimate.period_s / 1.5),
        min(upper, initial_estimate.period_s * 1.5),
    )
    coarse_index = int(argmin(values))
    if 0 < coarse_index < grid.size - 1:
        refine(float(grid[coarse_index - 1]), float(grid[coarse_index + 1]))
    ####

    best = min(
        candidate_periods,
        key=lambda period: _period_geometric_risk(period, attitudes, rates_rad_s, frame, config),
    )
    warnings = list(initial_estimate.warnings)
    bound_tolerance = max(1.0e-12, (upper - lower) * 1.0e-6)
    if abs(best - lower) <= bound_tolerance or abs(best - upper) <= bound_tolerance:
        warnings.append(
            "The full geometric sample-period fit selected an active search bound; "
            "timing remains ambiguous."
        )
    ####
    return SamplePeriodEstimate(float(best), tuple(warnings))
####


def resolve_interval_durations(
    data: TrajectoryData,
    attitudes: Rotation,
    rates_rad_s: FloatArray,
    frame: RateFrame,
    config: InferenceConfig,
) -> IntervalDurationResolution:
    if data.time_s is not None:
        return IntervalDurationResolution(diff(data.time_s), None, "timestamps", ())
    ####
    if config.sample_period_s is not None:
        intervals = full(data.sample_count - 1, config.sample_period_s, dtype=float64)
        return IntervalDurationResolution(intervals, None, "provided_sample_period", ())
    ####
    if not config.infer_sample_period_when_missing:
        raise ValueError(
            "No timestamps or sample_period_s were supplied and sample-period "
            "inference is disabled."
        )
    ####

    estimate = refine_constant_sample_period(attitudes, rates_rad_s, frame, config)
    intervals = full(data.sample_count - 1, estimate.period_s, dtype=float64)
    return IntervalDurationResolution(
        intervals,
        estimate.period_s,
        "inferred_sample_period",
        estimate.warnings,
    )
####


def integrate_interval_increments(
    rates_rad_s: FloatArray,
    interval_durations_s: FloatArray,
    frame: RateFrame,
    substeps: int,
) -> IntegratedIntervals:
    interval_count = rates_rad_s.shape[0] - 1
    predicted: Rotation = cast(Any, Rotation.identity(num=interval_count))
    path_angles = zeros(interval_count, dtype=float64)

    for substep in range(substeps):
        alpha = (substep + 0.5) / substeps
        interpolated = (1.0 - alpha) * rates_rad_s[:-1] + alpha * rates_rad_s[1:]
        substep_dt = interval_durations_s / substeps
        increments = rotation_from_rotvec(interpolated * substep_dt[:, newaxis])
        path_angles += norm(interpolated, axis=1) * substep_dt
        predicted = predicted * increments if frame is RateFrame.BODY else increments * predicted
    ####

    rate_change_angles = norm(rates_rad_s[1:] - rates_rad_s[:-1], axis=1) * interval_durations_s
    return IntegratedIntervals(
        predicted_increments=predicted,
        path_angles_rad=path_angles,
        rate_change_angles_rad=rate_change_angles,
    )
####


def compose_increment_windows(
    interval_increments: Rotation,
    window_size: int,
    frame: RateFrame,
) -> WindowComposition:
    """Compose disjoint increment windows using cached prefix products.

    The previous implementation multiplied every increment inside every window,
    making a window of length ``ell`` cost ``O(ell)``.  Prefix products reduce the
    full pass to ``O(N)`` and each requested window to ``O(1)`` while preserving
    the noncommutative body/world product order.
    """

    interval_count = len(interval_increments)
    starts = arange(0, interval_count - window_size + 1, dtype=int64)
    if starts.size == 0:
        return WindowComposition(rotation_from_quat(empty((0, 4))), starts)
    ####

    prefix_quaternions = empty((interval_count + 1, 4), dtype=float64)
    prefix_quaternions[0] = (0.0, 0.0, 0.0, 1.0)
    running: Rotation = cast(Any, Rotation.identity())
    for index in range(interval_count):
        increment = interval_increments[index]
        running = running * increment if frame is RateFrame.BODY else increment * running
        prefix_quaternions[index + 1] = asarray(cast(Any, running).as_quat(), dtype=float64)
    ####

    prefix = rotation_from_quat(prefix_quaternions)
    left = prefix[starts]
    right = prefix[starts + window_size]
    if frame is RateFrame.BODY:
        return WindowComposition(left.inv() * right, starts)
    ####
    return WindowComposition(right * left.inv(), starts)
####


def observed_window_increments(
    attitudes: Rotation,
    starts: NDArray[int64],
    window_size: int,
    frame: RateFrame,
) -> Rotation:
    previous = attitudes[starts]
    following = attitudes[starts + window_size]
    if frame is RateFrame.BODY:
        return previous.inv() * following
    ####
    return following * previous.inv()
####


def quaternion_norm_statistics(data: TrajectoryData) -> QuaternionNormStatistics | None:
    if data.representation is not AttitudeRepresentation.QUATERNION:
        return None
    ####
    if data.orientation is None:
        return None
    ####
    norms = norm(data.orientation, axis=1)
    errors = abs(norms - 1.0)
    return QuaternionNormStatistics(
        mean_abs_error=float(mean(errors)),
        max_abs_error=float(amax(errors)),
    )
####
