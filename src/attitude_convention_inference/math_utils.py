"""Small numerical and SciPy adapters shared by the inference lanes.

This module intentionally contains no convention policy.  It owns the narrow
boundary between typed NumPy arrays and SciPy rotations, plus reusable scalar
numerical transforms used by scoring and diagnostics.
"""

from __future__ import annotations

from typing import Any, cast

from numpy import amin, asarray, ceil, clip, exp, float64, mean, sort, square
from numpy import sum as np_sum
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

FloatArray = NDArray[float64]

__all__ = [
    "FloatArray",
    "rotation_from_euler",
    "rotation_from_quat",
    "rotation_from_rotvec",
    "rotation_magnitudes",
    "rotation_quaternions",
    "rotation_rotvecs",
    "support_weights",
    "trimmed_mean_square",
]


def rotation_from_quat(
    quaternions: FloatArray,
    *,
    scalar_first: bool = False,
) -> Rotation:
    """Construct a SciPy rotation from a typed quaternion array."""

    return cast(
        Rotation,
        cast(Any, Rotation).from_quat(quaternions, scalar_first=scalar_first),
    )
####


def rotation_from_euler(sequence: str, angles: FloatArray) -> Rotation:
    """Construct a SciPy rotation from radians in the requested Euler sequence."""

    return cast(Rotation, cast(Any, Rotation).from_euler(sequence, angles, degrees=False))
####


def rotation_from_rotvec(rotvecs: FloatArray) -> Rotation:
    """Construct a SciPy rotation from rotation vectors in radians."""

    return cast(Rotation, cast(Any, Rotation).from_rotvec(rotvecs))
####


def rotation_magnitudes(rotation: Rotation) -> FloatArray:
    """Return rotation magnitudes as a typed floating-point array."""

    return asarray(cast(Any, rotation).magnitude(), dtype=float64)
####


def rotation_rotvecs(rotation: Rotation) -> FloatArray:
    """Return rotation vectors as a typed floating-point array."""

    return asarray(cast(Any, rotation).as_rotvec(), dtype=float64)
####


def rotation_quaternions(rotation: Rotation) -> FloatArray:
    """Return scalar-last quaternions as a typed floating-point array."""

    return asarray(cast(Any, rotation).as_quat(), dtype=float64)
####


def trimmed_mean_square(values: FloatArray, trim_fraction: float) -> float:
    """Return a lower-tail trimmed mean of squared values."""

    if values.size == 0:
        return float("inf")
    ####
    keep_count = max(1, int(ceil(values.size * (1.0 - trim_fraction))))
    return float(mean(sort(square(values))[:keep_count]))
####


def support_weights(criteria: FloatArray) -> FloatArray:
    """Convert criterion values into normalized, numerically stable weights."""

    minimum = float(amin(criteria))
    raw: FloatArray = exp(-0.5 * clip(criteria - minimum, 0.0, 1400.0))
    return asarray(raw / float(np_sum(raw)), dtype=float64)
####
