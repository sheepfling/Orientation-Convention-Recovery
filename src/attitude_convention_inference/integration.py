"""Integrate rate streams and compose observed or predicted rotation windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt as scalar_sqrt
from typing import Any, cast

from numpy import (
    arange,
    array_equal,
    asarray,
    concatenate,
    cumsum,
    diff,
    empty,
    float64,
    int64,
    interp,
    newaxis,
    searchsorted,
    square,
    unique,
    zeros,
)
from numpy.linalg import norm
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from .hypothesis_model import ConventionHypothesis
from .math_utils import (
    FloatArray,
    mean_float,
    rotation_from_quat,
    rotation_from_rotvec,
    rotation_product,
    rotation_take,
)
from .models import RateFrame, RateUnit
from .trajectory import TrajectoryData

__all__ = [
    "IncrementWindowComposer",
    "IntegratedAnchorIntervals",
    "IntegratedIntervals",
    "IntegratedIntervalsCache",
    "ObservedIncrementCache",
    "WindowComposition",
    "compose_increment_windows",
    "integrate_interval_increments",
    "integrate_rates_between_anchors",
    "observed_interval_increments",
    "observed_window_increments",
]


@dataclass(frozen=True, slots=True)
class IntegratedIntervals:
    """Predicted interval increments and interval-level integration diagnostics."""

    predicted_increments: Rotation
    path_angles_rad: FloatArray
    rate_change_angles_rad: FloatArray
####


@dataclass(frozen=True, slots=True)
class IntegratedAnchorIntervals:
    """Full-rate integrations aggregated over consecutive orientation anchors."""

    predicted_increments: Rotation
    path_angles_rad: FloatArray
    rate_change_rms_angle_rad: float
####


@dataclass(frozen=True, slots=True)
class _RateIntegrationCacheKey:
    """Rate convention and frame, the only convention-dependent integration inputs."""

    rate_order: str
    rate_unit: RateUnit | str
    rate_sign: int
    frame: RateFrame
    substeps: int
####


@dataclass(frozen=True, slots=True)
class _IntegratedIntervalsCacheEntry:
    """Cached input snapshots paired with the prediction computed from them."""

    rates_rad_s: FloatArray
    interval_durations_s: FloatArray
    result: IntegratedIntervals
####


def _new_cached_integrations() -> dict[_RateIntegrationCacheKey, _IntegratedIntervalsCacheEntry]:
    return {}
####


@dataclass(slots=True)
class IntegratedIntervalsCache:
    """Reuse predictions within one inference when the resolved durations match."""

    data: TrajectoryData
    _integrations: dict[_RateIntegrationCacheKey, _IntegratedIntervalsCacheEntry] = field(
        default_factory=_new_cached_integrations
    )

    def integrate(
        self,
        hypothesis: ConventionHypothesis,
        rates_rad_s: FloatArray,
        interval_durations_s: FloatArray,
        frame: RateFrame,
        substeps: int,
    ) -> IntegratedIntervals:
        """Reuse a prediction only when its rates and exact durations still match."""

        key = _RateIntegrationCacheKey(
            rate_order=hypothesis.rate_order,
            rate_unit=hypothesis.rate_unit,
            rate_sign=int(hypothesis.rate_sign),
            frame=frame,
            substeps=substeps,
        )
        cached = self._integrations.get(key)
        if (
            cached is not None
            and array_equal(cached.rates_rad_s, rates_rad_s)
            and array_equal(cached.interval_durations_s, interval_durations_s)
        ):
            return cached.result
        ####

        integrated = integrate_interval_increments(
            rates_rad_s,
            interval_durations_s,
            frame,
            substeps,
        )
        self._integrations[key] = _IntegratedIntervalsCacheEntry(
            rates_rad_s=rates_rad_s.copy(),
            interval_durations_s=interval_durations_s.copy(),
            result=integrated,
        )
        return integrated
    ####
####


@dataclass(frozen=True, slots=True)
class WindowComposition:
    """Window-composed rotations and the source interval start indices."""

    window_increments: Rotation
    starts: NDArray[int64]
####


@dataclass(frozen=True, slots=True)
class IncrementWindowComposer:
    """Reusable prefix products for noncommutative increment windows."""

    prefix: Rotation
    frame: RateFrame

    @classmethod
    def from_increments(
        cls,
        interval_increments: Rotation,
        frame: RateFrame,
    ) -> IncrementWindowComposer:
        """Build one normalized scalar-last prefix stream for all window sizes."""

        interval_count = len(interval_increments)
        increment_quaternions = asarray(cast(Any, interval_increments).as_quat(), dtype=float64)
        prefix_quaternions = empty((interval_count + 1, 4), dtype=float64)
        prefix_quaternions[0] = (0.0, 0.0, 0.0, 1.0)
        rx, ry, rz, rw = 0.0, 0.0, 0.0, 1.0
        for index in range(interval_count):
            ix = float(increment_quaternions[index, 0])
            iy = float(increment_quaternions[index, 1])
            iz = float(increment_quaternions[index, 2])
            iw = float(increment_quaternions[index, 3])
            if frame is RateFrame.BODY:
                ax, ay, az, aw = rx, ry, rz, rw
                bx, by, bz, bw = ix, iy, iz, iw
            else:
                ax, ay, az, aw = ix, iy, iz, iw
                bx, by, bz, bw = rx, ry, rz, rw
            ####
            # Scalar-last Hamilton product in Rotation.__mul__ order.
            nx = aw * bx + ax * bw + ay * bz - az * by
            ny = aw * by - ax * bz + ay * bw + az * bx
            nz = aw * bz + ax * by - ay * bx + az * bw
            nw = aw * bw - ax * bx - ay * by - az * bz
            # Keep the prefix on the unit sphere over long trajectories.
            inverse_norm = 1.0 / scalar_sqrt(nx * nx + ny * ny + nz * nz + nw * nw)
            rx, ry, rz, rw = (
                nx * inverse_norm,
                ny * inverse_norm,
                nz * inverse_norm,
                nw * inverse_norm,
            )
            prefix_quaternions[index + 1] = (rx, ry, rz, rw)
        ####
        return cls(prefix=rotation_from_quat(prefix_quaternions), frame=frame)
    ####

    def compose(self, window_size: int) -> WindowComposition:
        """Compose every window of one size from the shared prefix rotations."""

        interval_count = len(self.prefix) - 1
        starts = arange(0, interval_count - window_size + 1, dtype=int64)
        if starts.size == 0:
            return WindowComposition(rotation_from_quat(empty((0, 4))), starts)
        ####
        left = rotation_take(self.prefix, starts)
        right = rotation_take(self.prefix, starts + window_size)
        if self.frame is RateFrame.BODY:
            return WindowComposition(rotation_product(left.inv(), right), starts)
        ####
        return WindowComposition(rotation_product(right, left.inv()), starts)
    ####

    def compose_ranges(
        self,
        start_indices: NDArray[int64],
        end_indices: NDArray[int64],
    ) -> Rotation:
        """Compose variable-length ranges of adjacent increments from the prefix."""

        if start_indices.ndim != 1 or end_indices.ndim != 1:
            raise ValueError("range indices must be one-dimensional")
        ####
        if start_indices.shape != end_indices.shape:
            raise ValueError("range start and end arrays must have the same shape")
        ####
        interval_count = len(self.prefix) - 1
        if (
            (start_indices < 0).any()
            or (end_indices > interval_count).any()
            or (end_indices <= start_indices).any()
        ):
            raise ValueError(
                "each composition range must be non-empty and within the increment stream"
            )
        ####
        left = rotation_take(self.prefix, start_indices)
        right = rotation_take(self.prefix, end_indices)
        if self.frame is RateFrame.BODY:
            return rotation_product(left.inv(), right)
        ####
        return rotation_product(right, left.inv())
    ####
####


def observed_interval_increments(attitudes: Rotation, frame: RateFrame) -> Rotation:
    """Return observed adjacent-sample rotations in the requested rate frame."""

    previous = attitudes[:-1]
    following = attitudes[1:]
    if frame is RateFrame.BODY:
        return previous.inv() * following
    ####
    return following * previous.inv()
####


def integrate_interval_increments(
    rates_rad_s: FloatArray,
    interval_durations_s: FloatArray,
    frame: RateFrame,
    substeps: int,
) -> IntegratedIntervals:
    """Integrate linearly interpolated rates over each sample interval."""

    interval_count = rates_rad_s.shape[0] - 1
    predicted: Rotation = cast(Any, Rotation.identity(num=interval_count))
    path_angles = zeros(interval_count, dtype=float64)

    for substep in range(substeps):
        alpha = (substep + 0.5) / substeps
        interpolated = (1.0 - alpha) * rates_rad_s[:-1] + alpha * rates_rad_s[1:]
        substep_dt = interval_durations_s / substeps
        increments = rotation_from_rotvec(interpolated * substep_dt[:, newaxis])
        path_angles += norm(interpolated, axis=1) * substep_dt
        predicted = (
            rotation_product(predicted, increments)
            if frame is RateFrame.BODY
            else rotation_product(increments, predicted)
        )
    ####

    rate_change_angles = norm(rates_rad_s[1:] - rates_rad_s[:-1], axis=1) * interval_durations_s
    return IntegratedIntervals(
        predicted_increments=predicted,
        path_angles_rad=path_angles,
        rate_change_angles_rad=rate_change_angles,
    )
####


def integrate_rates_between_anchors(
    rates_rad_s: FloatArray,
    rate_times_s: FloatArray,
    anchor_times_s: FloatArray,
    frame: RateFrame,
    substeps: int,
) -> IntegratedAnchorIntervals:
    """Integrate the complete piecewise-linear rate stream between sparse anchors.

    Rate values at anchor times are linearly interpolated, with extrapolation
    forbidden. Inserting anchors into the rate time grid preserves the original
    piecewise-linear signal; each rate interval is integrated and the resulting
    non-commuting rotations are composed over each anchor-to-anchor range.
    """

    if rates_rad_s.ndim != 2 or rates_rad_s.shape[1] != 3 or rates_rad_s.shape[0] < 2:
        raise ValueError("rates_rad_s must have shape (N, 3) with N >= 2")
    ####
    if rate_times_s.ndim != 1 or rate_times_s.size != rates_rad_s.shape[0]:
        raise ValueError("rate_times_s must have one value per rate sample")
    ####
    if anchor_times_s.ndim != 1 or anchor_times_s.size < 2:
        raise ValueError("anchor_times_s must contain at least two values")
    ####
    if substeps < 1:
        raise ValueError("substeps must be at least one")
    ####
    if not (diff(rate_times_s) > 0.0).all() or not (diff(anchor_times_s) > 0.0).all():
        raise ValueError("rate and anchor timestamps must be strictly increasing")
    ####
    if not (rate_times_s[0] <= anchor_times_s[0] and anchor_times_s[-1] <= rate_times_s[-1]):
        raise ValueError("rate_times_s must cover all anchor times; extrapolation is not supported")
    ####

    combined_times = unique(concatenate((rate_times_s, anchor_times_s)))
    anchor_indices = asarray(searchsorted(combined_times, anchor_times_s), dtype=int64)
    if not (combined_times[anchor_indices] == anchor_times_s).all():
        raise RuntimeError("anchor timestamps were not preserved in the integration grid")
    ####
    augmented_rates = asarray(
        tuple(interp(combined_times, rate_times_s, rates_rad_s[:, axis]) for axis in range(3)),
        dtype=float64,
    ).T

    integrated = integrate_interval_increments(
        augmented_rates,
        asarray(diff(combined_times), dtype=float64),
        frame,
        substeps,
    )
    composer = IncrementWindowComposer.from_increments(integrated.predicted_increments, frame)
    start_indices = anchor_indices[:-1]
    end_indices = anchor_indices[1:]
    cumulative_path_angles = concatenate(
        (zeros(1, dtype=float64), cumsum(integrated.path_angles_rad))
    )
    anchor_path_angles = asarray(
        cumulative_path_angles[end_indices] - cumulative_path_angles[start_indices],
        dtype=float64,
    )
    rate_change_angles = norm(rates_rad_s[1:] - rates_rad_s[:-1], axis=1) * diff(rate_times_s)
    rate_change_rms = scalar_sqrt(mean_float(square(rate_change_angles)))
    return IntegratedAnchorIntervals(
        predicted_increments=composer.compose_ranges(start_indices, end_indices),
        path_angles_rad=anchor_path_angles,
        rate_change_rms_angle_rad=rate_change_rms,
    )
####


def compose_increment_windows(
    interval_increments: Rotation,
    window_size: int,
    frame: RateFrame,
) -> WindowComposition:
    """Compose one window size; use a composer to reuse prefixes across sizes."""

    return IncrementWindowComposer.from_increments(interval_increments, frame).compose(window_size)
####


def observed_window_increments(
    attitudes: Rotation,
    starts: NDArray[int64],
    window_size: int,
    frame: RateFrame,
) -> Rotation:
    """Return observed rotations over windows of adjacent sample intervals."""

    previous = rotation_take(attitudes, starts)
    following = rotation_take(attitudes, starts + window_size)
    if frame is RateFrame.BODY:
        return rotation_product(previous.inv(), following)
    ####
    return rotation_product(following, previous.inv())
####


@dataclass(frozen=True, slots=True)
class _ObservedWindowCacheKey:
    frame: RateFrame
    window_size: int
####


def _new_observed_interval_cache() -> dict[RateFrame, Rotation]:
    return {}
####


def _new_observed_window_cache() -> dict[_ObservedWindowCacheKey, WindowComposition]:
    return {}
####


@dataclass(slots=True)
class ObservedIncrementCache:
    """Reuse observed rotations for one decoded-attitude series and rate frame."""

    attitudes: Rotation
    _intervals: dict[RateFrame, Rotation] = field(default_factory=_new_observed_interval_cache)
    _windows: dict[_ObservedWindowCacheKey, WindowComposition] = field(
        default_factory=_new_observed_window_cache
    )

    def interval_increments(self, frame: RateFrame) -> Rotation:
        """Return observed one-step rotations, computing each frame once."""

        increments = self._intervals.get(frame)
        if increments is None:
            increments = observed_interval_increments(self.attitudes, frame)
            self._intervals[frame] = increments
        ####
        return increments
    ####

    def window_composition(self, window_size: int, frame: RateFrame) -> WindowComposition:
        """Return observed window rotations and starts for one size/frame pair."""

        if window_size < 1:
            raise ValueError("window_size must be positive")
        ####
        key = _ObservedWindowCacheKey(frame=frame, window_size=window_size)
        composition = self._windows.get(key)
        if composition is None:
            interval_count = len(self.attitudes) - 1
            starts = arange(0, interval_count - window_size + 1, dtype=int64)
            composition = WindowComposition(
                window_increments=observed_window_increments(
                    self.attitudes,
                    starts,
                    window_size,
                    frame,
                ),
                starts=starts,
            )
            self._windows[key] = composition
        ####
        return composition
    ####
####
