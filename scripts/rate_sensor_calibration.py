"""Rate-channel calibration transforms for independent synthetic validation data."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi, sqrt

from numpy import array, clip, cos, cross, float64, sin
from numpy.typing import NDArray

FloatArray = NDArray[float64]

__all__ = ["RATE_SENSOR_CALIBRATION_PROFILES", "RateSensorCalibration"]


@dataclass(frozen=True, slots=True)
class RateSensorCalibration:
    """Known rate-channel calibration applied in the declared rate frame.

    The transform order is misalignment rotation, bias, axis scale, axis polarity,
    additive measurement noise, then absolute clipping. Values are expressed in
    radians per second except for the misalignment angle, which is in degrees.
    """

    profile_id: str
    misalignment_angle_deg: float = 0.0
    misalignment_axis: tuple[float, float, float] = (0.3, -0.5, 0.8)
    bias_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale_by_axis: tuple[float, float, float] = (1.0, 1.0, 1.0)
    polarity_by_axis: tuple[int, int, int] = (1, 1, 1)
    clip_abs_rad_s: float | None = None

    def __post_init__(self) -> None:
        if not self.profile_id or any(
            not (character.isascii() and (character.isalnum() or character in "-_."))
            for character in self.profile_id
        ):
            raise ValueError("rate calibration profile_id must be a non-empty ASCII slug")
        ####
        if any(
            len(values) != 3
            for values in (
                self.misalignment_axis,
                self.bias_rad_s,
                self.scale_by_axis,
                self.polarity_by_axis,
            )
        ):
            raise ValueError(
                "rate calibration axis, bias, scale, and polarity vectors need 3 values"
            )
        ####
        values = (
            self.misalignment_angle_deg,
            *self.misalignment_axis,
            *self.bias_rad_s,
            *self.scale_by_axis,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("rate calibration parameters must be finite")
        ####
        axis_norm = sqrt(sum(component * component for component in self.misalignment_axis))
        if not isfinite(axis_norm) or axis_norm <= 0.0:
            raise ValueError("rate calibration misalignment axis must have non-zero norm")
        ####
        if any(scale <= 0.0 for scale in self.scale_by_axis):
            raise ValueError("rate calibration axis scales must be positive")
        ####
        if any(polarity not in (-1, 1) for polarity in self.polarity_by_axis):
            raise ValueError("rate calibration axis polarities must be -1 or +1")
        ####
        if self.clip_abs_rad_s is not None and (
            not isfinite(self.clip_abs_rad_s) or self.clip_abs_rad_s <= 0.0
        ):
            raise ValueError("rate calibration clipping limit must be positive and finite")
        ####
    ####

    def measurement_condition_fields(self) -> dict[str, object]:
        """Return named, unit-bearing calibration values for a corpus manifest."""

        return {
            "rate_sensor_calibration_id": self.profile_id,
            "rate_sensor_misalignment_angle_deg": self.misalignment_angle_deg,
            "rate_sensor_misalignment_axis": self.misalignment_axis,
            "rate_sensor_bias_rad_s": self.bias_rad_s,
            "rate_sensor_scale_by_axis": self.scale_by_axis,
            "rate_sensor_polarity_by_axis": self.polarity_by_axis,
            "rate_sensor_clip_abs_rad_s": self.clip_abs_rad_s,
        }
    ####

    def apply_to_rates(
        self,
        rates_rad_s: FloatArray,
        *,
        measurement_noise_rad_s: FloatArray | None = None,
    ) -> FloatArray:
        """Apply the declared deterministic calibration, noise, and clipping order."""

        measured = array(rates_rad_s, dtype=float64, copy=True)
        if measured.ndim != 2 or measured.shape[1] != 3:
            raise ValueError("rate measurements must have shape (sample_count, 3)")
        ####
        if self.misalignment_angle_deg != 0.0:
            axis = array(self.misalignment_axis, dtype=float64)
            axis /= sqrt(float((axis * axis).sum()))
            angle_rad = self.misalignment_angle_deg * pi / 180.0
            projection = (measured * axis).sum(axis=1, keepdims=True)
            measured = (
                measured * cos(angle_rad)
                + cross(axis, measured) * sin(angle_rad)
                + projection * axis * (1.0 - cos(angle_rad))
            )
        ####
        if any(self.bias_rad_s):
            measured += array(self.bias_rad_s, dtype=float64)
        ####
        if self.scale_by_axis != (1.0, 1.0, 1.0):
            measured *= array(self.scale_by_axis, dtype=float64)
        ####
        if self.polarity_by_axis != (1, 1, 1):
            measured *= array(self.polarity_by_axis, dtype=float64)
        ####
        if measurement_noise_rad_s is not None:
            noise = array(measurement_noise_rad_s, dtype=float64)
            if noise.shape != measured.shape:
                raise ValueError("measurement noise must match the rate array shape")
            ####
            measured += noise
        ####
        if self.clip_abs_rad_s is not None:
            measured = clip(measured, -self.clip_abs_rad_s, self.clip_abs_rad_s)
        ####
        return array(measured, dtype=float64)
    ####
####


RATE_SENSOR_CALIBRATION_PROFILES: tuple[RateSensorCalibration, ...] = (
    RateSensorCalibration(profile_id="identity"),
    RateSensorCalibration(profile_id="misalignment-15deg", misalignment_angle_deg=15.0),
    RateSensorCalibration(
        profile_id="bias-p-150mrad-s",
        bias_rad_s=(0.15, 0.0, 0.0),
    ),
    RateSensorCalibration(
        profile_id="axis-scale-1p3-0p8-1p1",
        scale_by_axis=(1.3, 0.8, 1.1),
    ),
    RateSensorCalibration(
        profile_id="polarity-p-flip",
        polarity_by_axis=(-1, 1, 1),
    ),
    RateSensorCalibration(
        profile_id="clip-0p3-rad-s",
        clip_abs_rad_s=0.3,
    ),
    RateSensorCalibration(
        profile_id="polarity-p-flip-clip-0p3-rad-s",
        polarity_by_axis=(-1, 1, 1),
        clip_abs_rad_s=0.3,
    ),
    RateSensorCalibration(
        profile_id="axis-scale-1p3-0p8-1p1-polarity-p-flip",
        scale_by_axis=(1.3, 0.8, 1.1),
        polarity_by_axis=(-1, 1, 1),
    ),
    RateSensorCalibration(
        profile_id="axis-scale-1p3-0p8-1p1-clip-0p3-rad-s",
        scale_by_axis=(1.3, 0.8, 1.1),
        clip_abs_rad_s=0.3,
    ),
    RateSensorCalibration(
        profile_id="axis-scale-1p3-0p8-1p1-polarity-p-flip-clip-0p3-rad-s",
        scale_by_axis=(1.3, 0.8, 1.1),
        polarity_by_axis=(-1, 1, 1),
        clip_abs_rad_s=0.3,
    ),
    RateSensorCalibration(
        profile_id="polarity-q-flip",
        polarity_by_axis=(1, -1, 1),
    ),
    RateSensorCalibration(
        profile_id="polarity-r-flip",
        polarity_by_axis=(1, 1, -1),
    ),
    RateSensorCalibration(
        profile_id="polarity-pq-flip",
        polarity_by_axis=(-1, -1, 1),
    ),
    RateSensorCalibration(
        profile_id="polarity-pr-flip",
        polarity_by_axis=(-1, 1, -1),
    ),
    RateSensorCalibration(
        profile_id="polarity-qr-flip",
        polarity_by_axis=(1, -1, -1),
    ),
    RateSensorCalibration(
        profile_id="polarity-pqr-flip",
        polarity_by_axis=(-1, -1, -1),
    ),
    RateSensorCalibration(
        profile_id="clip-0p15-rad-s",
        clip_abs_rad_s=0.15,
    ),
    RateSensorCalibration(
        profile_id="clip-0p6-rad-s",
        clip_abs_rad_s=0.6,
    ),
    RateSensorCalibration(
        profile_id="polarity-p-flip-clip-0p15-rad-s",
        polarity_by_axis=(-1, 1, 1),
        clip_abs_rad_s=0.15,
    ),
    RateSensorCalibration(
        profile_id="polarity-p-flip-clip-0p6-rad-s",
        polarity_by_axis=(-1, 1, 1),
        clip_abs_rad_s=0.6,
    ),
)
