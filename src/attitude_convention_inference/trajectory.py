"""Validated, immutable trajectory samples and input-quality summaries."""

from __future__ import annotations

from dataclasses import dataclass

from numpy import all as np_all
from numpy import amax, array, diff, float64
from numpy import any as np_any
from numpy import isfinite as np_isfinite
from numpy.linalg import norm

from .math_utils import FloatArray, mean_float
from .models import AttitudeRepresentation

__all__ = [
    "QuaternionNormStatistics",
    "SparseOrientationRateData",
    "TrajectoryData",
    "quaternion_norm_statistics",
]


@dataclass(frozen=True, slots=True)
class TrajectoryData:
    """Aligned orientation/rate samples with optional strictly increasing times.

    ``time_s`` values are finite seconds and may use an absolute clock origin. The
    inference calculations use elapsed intervals, not the absolute origin. For
    high-resolution epoch timestamps, subtract the origin while values are still
    integers before converting them to floating-point seconds.
    """

    representation: AttitudeRepresentation
    orientation: FloatArray | None
    rates: FloatArray | None
    time_s: FloatArray | None = None

    def __post_init__(self) -> None:
        orientation = (
            None if self.orientation is None else array(self.orientation, dtype=float64, copy=True)
        )
        rates = None if self.rates is None else array(self.rates, dtype=float64, copy=True)
        time_s = None if self.time_s is None else array(self.time_s, dtype=float64, copy=True)

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

        # Own the buffers before freezing them so callers cannot change validated data.
        for stream in (orientation, rates, time_s):
            if stream is not None:
                stream.setflags(write=False)
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


@dataclass(frozen=True, slots=True)
class SparseOrientationRateData:
    """Sparse orientation anchors paired with a separately sampled rate stream.

    Both time arrays use the same clock and seconds. Rate samples must cover all
    anchor times; inference interpolates the piecewise-linear rate signal at each
    anchor and integrates every intervening rate interval. Extrapolation and
    alignment by row number are not supported.
    """

    representation: AttitudeRepresentation
    orientation_anchors: FloatArray
    orientation_anchor_times_s: FloatArray
    rates: FloatArray
    rate_times_s: FloatArray

    def __post_init__(self) -> None:
        representation = AttitudeRepresentation(self.representation)
        orientation_anchors = array(self.orientation_anchors, dtype=float64, copy=True)
        orientation_anchor_times_s = array(
            self.orientation_anchor_times_s, dtype=float64, copy=True
        )
        rates = array(self.rates, dtype=float64, copy=True)
        rate_times_s = array(self.rate_times_s, dtype=float64, copy=True)

        expected_width = 4 if representation is AttitudeRepresentation.QUATERNION else 3
        if orientation_anchors.ndim != 2 or orientation_anchors.shape[1] != expected_width:
            raise ValueError(
                f"orientation_anchors must have shape (N, {expected_width}) for {representation}."
            )
        ####
        if rates.ndim != 2 or rates.shape[1] != 3:
            raise ValueError("rates must have shape (N, 3).")
        ####
        if orientation_anchors.shape[0] < 3:
            raise ValueError("At least three orientation anchors are required for inference.")
        ####
        if rates.shape[0] < 2:
            raise ValueError("At least two rate samples are required for interpolation.")
        ####
        if (
            orientation_anchor_times_s.ndim != 1
            or orientation_anchor_times_s.size != (orientation_anchors.shape[0])
        ):
            raise ValueError(
                "orientation_anchor_times_s must have one value per orientation anchor."
            )
        ####
        if rate_times_s.ndim != 1 or rate_times_s.size != rates.shape[0]:
            raise ValueError("rate_times_s must have one value per rate sample.")
        ####
        if not np_all(np_isfinite(orientation_anchors)):
            raise ValueError("orientation_anchors must contain only finite values.")
        ####
        if not np_all(np_isfinite(rates)):
            raise ValueError("rates must contain only finite values.")
        ####
        for name, times in (
            ("orientation_anchor_times_s", orientation_anchor_times_s),
            ("rate_times_s", rate_times_s),
        ):
            if not np_all(np_isfinite(times)):
                raise ValueError(f"{name} must contain only finite values.")
            ####
            if not np_all(diff(times) > 0.0):
                raise ValueError(f"{name} must be strictly increasing.")
            ####
        ####
        if (
            orientation_anchor_times_s[0] < rate_times_s[0]
            or orientation_anchor_times_s[-1] > rate_times_s[-1]
        ):
            raise ValueError(
                "rate_times_s must cover all orientation anchor times; "
                "extrapolation is not supported."
            )
        ####
        if representation is AttitudeRepresentation.QUATERNION:
            quaternion_norms = norm(orientation_anchors, axis=1)
            if np_any(quaternion_norms < 1.0e-12):
                raise ValueError("Quaternion orientation anchors must have nonzero norm.")
            ####
        ####

        for stream in (orientation_anchors, orientation_anchor_times_s, rates, rate_times_s):
            stream.setflags(write=False)
        ####
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "orientation_anchors", orientation_anchors)
        object.__setattr__(self, "orientation_anchor_times_s", orientation_anchor_times_s)
        object.__setattr__(self, "rates", rates)
        object.__setattr__(self, "rate_times_s", rate_times_s)
    ####

    @property
    def anchor_count(self) -> int:
        """Number of timestamped orientation observations."""

        return int(self.orientation_anchors.shape[0])
    ####

    @property
    def rate_sample_count(self) -> int:
        """Number of timestamped rate measurements."""

        return int(self.rates.shape[0])
    ####
####


@dataclass(frozen=True, slots=True)
class QuaternionNormStatistics:
    """Quaternion unit-norm error statistics."""

    mean_abs_error: float
    max_abs_error: float
####


def quaternion_norm_statistics(data: TrajectoryData) -> QuaternionNormStatistics | None:
    """Summarize quaternion norm deviation, when an attitude stream is present."""

    if data.representation is not AttitudeRepresentation.QUATERNION:
        return None
    ####
    if data.orientation is None:
        return None
    ####
    norms = norm(data.orientation, axis=1)
    errors = abs(norms - 1.0)
    return QuaternionNormStatistics(
        mean_abs_error=mean_float(errors),
        max_abs_error=float(amax(errors)),
    )
####
