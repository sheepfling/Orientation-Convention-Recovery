"""Generate independently labeled synthetic logs without the inference package."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import lru_cache
from hashlib import sha256
from itertools import product
from math import ceil, exp, floor, isfinite
from pathlib import Path
from typing import Literal, Protocol

from numpy import (
    arange,
    arcsin,
    arctan2,
    array,
    bool_,
    clip,
    cos,
    cross,
    cumsum,
    float64,
    interp,
    pi,
    sin,
    sqrt,
    stack,
    zeros,
)
from numpy.random import Generator, SeedSequence, default_rng
from numpy.typing import NDArray

if __package__:
    from scripts.rate_sensor_calibration import (
        RATE_SENSOR_CALIBRATION_PROFILES,
        RateSensorCalibration,
    )
    from scripts.validation_noise import (
        sample_time_domain_dropout_mask,
        sample_time_domain_ou_noise,
        sample_time_domain_uniform_values,
        validate_ou_sample_period,
    )
else:
    from rate_sensor_calibration import RATE_SENSOR_CALIBRATION_PROFILES, RateSensorCalibration
    from validation_noise import (
        sample_time_domain_dropout_mask,
        sample_time_domain_ou_noise,
        sample_time_domain_uniform_values,
        validate_ou_sample_period,
    )
####

FloatArray = NDArray[float64]
TruthRateUnit = Literal["radians_per_second", "degrees_per_second"]
TruthAttitudeDirection = Literal["body_to_world", "world_to_body"]
TruthQuaternionOrder = Literal["scalar_first", "scalar_last"]
TruthRateFrame = Literal["body", "world"]
RateNoiseFrame = Literal["truth-rate-frame", "body", "world"]
TruthRateOrder = Literal["pqr", "prq", "qpr", "qrp", "rpq", "rqp"]
TruthRateSign = Literal[-1, 1]


class MotionProfile(StrEnum):
    """Analytic and numerically integrated motion families used in synthetic studies."""

    RICH_THREE_AXIS = "rich-three-axis"
    LOW_EXCITATION = "low-excitation"
    SINGLE_AXIS = "single-axis"
    NEAR_GIMBAL = "near-gimbal"
    SAMPLED_RATE_NONANALYTIC = "sampled-rate-nonanalytic"
    SAMPLED_RATE_PIECEWISE_LINEAR = "sampled-rate-piecewise-linear"
    SAMPLED_RATE_MANEUVER_BURSTS = "sampled-rate-maneuver-bursts"
####


SAMPLED_RATE_PROFILES = (
    MotionProfile.SAMPLED_RATE_NONANALYTIC,
    MotionProfile.SAMPLED_RATE_PIECEWISE_LINEAR,
    MotionProfile.SAMPLED_RATE_MANEUVER_BURSTS,
)


@dataclass(frozen=True, slots=True)
class SyntheticMotion:
    time_s: FloatArray
    euler_xyz_rad: FloatArray
    quaternion_xyzw: FloatArray
    body_rates_rad_s: FloatArray
    world_rates_rad_s: FloatArray
####


class _RateSignal(Protocol):
    def evaluate(self, time_s: FloatArray) -> FloatArray: ...
####


@dataclass(frozen=True, slots=True)
class QuaternionTruthConvention:
    """A raw quaternion/rate encoding convention for independent synthetic labels."""

    attitude_direction: TruthAttitudeDirection
    quaternion_order: TruthQuaternionOrder
    rate_frame: TruthRateFrame
    rate_order: TruthRateOrder
    rate_unit: TruthRateUnit
    rate_sign: TruthRateSign
####


BASE_QUATERNION_TRUTH = QuaternionTruthConvention(
    attitude_direction="body_to_world",
    quaternion_order="scalar_last",
    rate_frame="body",
    rate_order="pqr",
    rate_unit="radians_per_second",
    rate_sign=1,
)
TRUTH_ATTITUDE_DIRECTIONS: tuple[TruthAttitudeDirection, ...] = (
    "body_to_world",
    "world_to_body",
)
TRUTH_QUATERNION_ORDERS: tuple[TruthQuaternionOrder, ...] = ("scalar_first", "scalar_last")
TRUTH_RATE_FRAMES: tuple[TruthRateFrame, ...] = ("body", "world")
TRUTH_RATE_ORDERS: tuple[TruthRateOrder, ...] = ("pqr", "prq", "qpr", "qrp", "rpq", "rqp")
TRUTH_RATE_SIGNS: tuple[TruthRateSign, ...] = (-1, 1)
TRUTH_RATE_UNITS: tuple[TruthRateUnit, ...] = ("radians_per_second", "degrees_per_second")


def all_quaternion_truth_conventions() -> tuple[QuaternionTruthConvention, ...]:
    """Enumerate the full 192-value quaternion/rate convention cross product."""

    return tuple(
        QuaternionTruthConvention(
            attitude_direction=attitude_direction,
            quaternion_order=quaternion_order,
            rate_frame=rate_frame,
            rate_order=rate_order,
            rate_unit=rate_unit,
            rate_sign=rate_sign,
        )
        for attitude_direction, quaternion_order, rate_frame, rate_order, rate_unit, rate_sign in (
            product(
                TRUTH_ATTITUDE_DIRECTIONS,
                TRUTH_QUATERNION_ORDERS,
                TRUTH_RATE_FRAMES,
                TRUTH_RATE_ORDERS,
                TRUTH_RATE_UNITS,
                TRUTH_RATE_SIGNS,
            )
        )
    )
####


@dataclass(frozen=True, slots=True)
class NoiseProfile:
    """Per-axis standard deviations for one seeded measurement-noise condition."""

    name: Literal["none", "low", "moderate", "high"]
    orientation_sigma_rad: float
    rate_sigma_rad_s: float
####


@dataclass(frozen=True, slots=True)
class _PreparedMeasurements:
    """One seeded physical/noise realization shared by its truth encodings."""

    noise_seed: int
    recorded_times_s: FloatArray
    retained_samples: NDArray[bool_]
    orientation_quaternions_xyzw: FloatArray
    body_rate_measurements_rad_s: tuple[FloatArray, ...]
    world_rate_measurements_rad_s: tuple[FloatArray, ...]
    dropped_sample_count: int
####


@dataclass(frozen=True, slots=True)
class _NoisyCaseFactor:
    """Named input dimensions for one synthetic noisy truth encoding."""

    motion_seed: int
    jitter_index: int
    jitter_amplitude_s: float
    clock_rate_index: int
    clock_rate_error_ppm: float
    dropout_index: int
    dropout_probability: float
    orientation_latency_index: int
    orientation_latency_s: float
    rate_latency_index: int
    rate_latency_s: float
    orientation_ar1_index: int
    orientation_ar1_coefficient: float
    rate_ar1_index: int
    rate_ar1_coefficient: float
    calibration_index: int
    calibration: RateSensorCalibration
    truth_convention: QuaternionTruthConvention
    replicate_index: int
####


NOISE_PROFILES: tuple[NoiseProfile, ...] = (
    NoiseProfile("low", 0.0015, 0.012),
    NoiseProfile("moderate", 0.003, 0.025),
    NoiseProfile("high", 0.006, 0.05),
)
NO_NOISE_PROFILE = NoiseProfile("none", 0.0, 0.0)
NOISE_SAMPLE_PERIOD_S = 0.025
SAMPLED_RATE_INTEGRATION_STEP_S = NOISE_SAMPLE_PERIOD_S / 10.0
DEFAULT_NOISE_SEED_START = 20261030
DEFAULT_MOTION_SEED = 730201
RATE_KNOT_PERIOD_S = 0.5
RATE_KNOT_MIN_TIME_S = -60.0
RATE_KNOT_MAX_TIME_S = 3600.0
RATE_KNOT_SIGMA_RAD_S = 0.45
MANEUVER_RATE_PERIOD_S = 8.0
MANEUVER_RATE_BURST_COUNT = 8
MANEUVER_RATE_SLOT_PERIOD_S = MANEUVER_RATE_PERIOD_S / MANEUVER_RATE_BURST_COUNT


def _axis_quaternion(axis: int, angles: FloatArray) -> FloatArray:
    result = zeros((len(angles), 4), dtype=float64)
    result[:, axis] = sin(angles / 2.0)
    result[:, 3] = cos(angles / 2.0)
    return result
####


def _multiply_quaternions(left: FloatArray, right: FloatArray) -> FloatArray:
    vector = (
        left[:, 3:4] * right[:, :3] + right[:, 3:4] * left[:, :3] + cross(left[:, :3], right[:, :3])
    )
    scalar = left[:, 3] * right[:, 3] - (left[:, :3] * right[:, :3]).sum(axis=1)
    return stack((vector[:, 0], vector[:, 1], vector[:, 2], scalar), axis=1)
####


def analytic_motion(
    time_s: FloatArray,
    profile: MotionProfile = MotionProfile.RICH_THREE_AXIS,
) -> SyntheticMotion:
    """Use closed-form ZYX rotations and body rates for a named motion family."""

    if profile in SAMPLED_RATE_PROFILES:
        raise ValueError("Use the corresponding sampled-rate motion generator for this profile.")
    ####
    phi = 0.33 * sin(1.3 * time_s) + 0.09 * cos(2.1 * time_s)
    theta = 0.25 * sin(0.8 * time_s) + 0.06 * cos(1.7 * time_s)
    psi = 0.41 * sin(1.1 * time_s) + 0.07 * cos(0.6 * time_s)
    phi_dot = 0.429 * cos(1.3 * time_s) - 0.189 * sin(2.1 * time_s)
    theta_dot = 0.20 * cos(0.8 * time_s) - 0.102 * sin(1.7 * time_s)
    psi_dot = 0.451 * cos(1.1 * time_s) - 0.042 * sin(0.6 * time_s)
    if profile is MotionProfile.LOW_EXCITATION:
        phi *= 0.08
        theta *= 0.08
        psi *= 0.08
        phi_dot *= 0.08
        theta_dot *= 0.08
        psi_dot *= 0.08
    ####
    if profile is MotionProfile.SINGLE_AXIS:
        phi = zeros(len(time_s), dtype=float64)
        theta = zeros(len(time_s), dtype=float64)
        phi_dot = zeros(len(time_s), dtype=float64)
        theta_dot = zeros(len(time_s), dtype=float64)
    ####
    if profile is MotionProfile.NEAR_GIMBAL:
        theta = pi / 2.0 - 0.035 + 0.02 * sin(0.8 * time_s) + 0.006 * cos(1.7 * time_s)
        theta_dot = 0.016 * cos(0.8 * time_s) - 0.0102 * sin(1.7 * time_s)
    ####
    body_rates = stack(
        (
            phi_dot - psi_dot * sin(theta),
            theta_dot * cos(phi) + psi_dot * sin(phi) * cos(theta),
            -theta_dot * sin(phi) + psi_dot * cos(phi) * cos(theta),
        ),
        axis=1,
    )
    quaternion = _multiply_quaternions(
        _multiply_quaternions(_axis_quaternion(2, psi), _axis_quaternion(1, theta)),
        _axis_quaternion(0, phi),
    )
    twice_cross = 2.0 * cross(quaternion[:, :3], body_rates)
    world_rates = (
        body_rates + quaternion[:, 3:4] * twice_cross + cross(quaternion[:, :3], twice_cross)
    )
    return SyntheticMotion(
        time_s=time_s,
        euler_xyz_rad=stack((phi, theta, psi), axis=1),
        quaternion_xyzw=quaternion,
        body_rates_rad_s=body_rates,
        world_rates_rad_s=world_rates,
    )
####


@dataclass(frozen=True, slots=True)
class SampledRateSignal:
    frequencies_rad_s: FloatArray
    amplitudes_rad_s: FloatArray
    phases_rad: FloatArray
    offsets_rad_s: FloatArray

    def evaluate(self, time_s: FloatArray) -> FloatArray:
        times = array(time_s, dtype=float64)
        oscillations = sin(
            times[:, None, None] * self.frequencies_rad_s[None, None, :]
            + self.phases_rad[None, :, :]
        )
        return self.offsets_rad_s[None, :] + (oscillations * self.amplitudes_rad_s[None, :, :]).sum(
            axis=2
        )
    ####
####


@dataclass(frozen=True, slots=True)
class PiecewiseLinearRateSignal:
    """Seeded three-axis rates linearly interpolated between fixed-time knots."""

    knot_times_s: FloatArray
    knot_rates_rad_s: FloatArray

    def __post_init__(self) -> None:
        knot_times = array(self.knot_times_s, dtype=float64, copy=True)
        knot_rates = array(self.knot_rates_rad_s, dtype=float64, copy=True)
        if knot_times.ndim != 1 or knot_times.size < 2:
            raise ValueError("knot_times_s must be one-dimensional with at least two knots.")
        ####
        if knot_rates.shape != (knot_times.size, 3):
            raise ValueError("knot_rates_rad_s must have shape (knot_count, 3).")
        ####
        if not all(isfinite(float(value)) for value in knot_times) or not all(
            isfinite(float(value)) for value in knot_rates.flat
        ):
            raise ValueError("Rate knots must contain only finite values.")
        ####
        if not all(float(interval) > 0.0 for interval in knot_times[1:] - knot_times[:-1]):
            raise ValueError("knot_times_s must be strictly increasing.")
        ####
        knot_times.setflags(write=False)
        knot_rates.setflags(write=False)
        object.__setattr__(self, "knot_times_s", knot_times)
        object.__setattr__(self, "knot_rates_rad_s", knot_rates)
    ####

    def evaluate(self, time_s: FloatArray) -> FloatArray:
        times = array(time_s, dtype=float64)
        if times.ndim != 1 or not all(isfinite(float(value)) for value in times):
            raise ValueError("time_s must be a one-dimensional array of finite values.")
        ####
        if times.size and (
            float(times.min()) < float(self.knot_times_s[0])
            or float(times.max()) > float(self.knot_times_s[-1])
        ):
            raise ValueError(
                "time_s must stay within the piecewise-linear rate signal's supported range "
                f"[{self.knot_times_s[0]}, {self.knot_times_s[-1]}]."
            )
        ####
        return stack(
            tuple(
                interp(times, self.knot_times_s, self.knot_rates_rad_s[:, axis])
                for axis in range(3)
            ),
            axis=1,
        )
    ####
####


@dataclass(frozen=True, slots=True)
class ManeuverBurstRateSignal:
    """Periodic smooth body-rate bursts separated by zero-rate rest intervals."""

    start_times_s: FloatArray
    durations_s: FloatArray
    amplitudes_rad_s: FloatArray
    repetition_period_s: float

    def __post_init__(self) -> None:
        start_times = array(self.start_times_s, dtype=float64, copy=True)
        durations = array(self.durations_s, dtype=float64, copy=True)
        amplitudes = array(self.amplitudes_rad_s, dtype=float64, copy=True)
        if start_times.ndim != 2 or start_times.shape[0] < 1 or start_times.shape[1] != 3:
            raise ValueError("start_times_s must have shape (burst_count, 3).")
        ####
        if durations.shape != start_times.shape:
            raise ValueError("durations_s must match the per-axis burst start times.")
        ####
        if amplitudes.shape != start_times.shape:
            raise ValueError("amplitudes_rad_s must have shape (burst_count, 3).")
        ####
        if not all(isfinite(float(value)) for value in start_times.flat):
            raise ValueError("start_times_s must contain only finite values.")
        ####
        if not all(isfinite(float(value)) and value > 0.0 for value in durations.flat):
            raise ValueError("durations_s must contain positive finite values.")
        ####
        if not all(isfinite(float(value)) for value in amplitudes.flat):
            raise ValueError("amplitudes_rad_s must contain only finite values.")
        ####
        if not isfinite(self.repetition_period_s) or self.repetition_period_s <= 0.0:
            raise ValueError("repetition_period_s must be positive and finite.")
        ####
        start_intervals = start_times[1:, :] - start_times[:-1, :]
        if not all(float(interval) > 0.0 for interval in start_intervals.flat):
            raise ValueError("start_times_s must be strictly increasing.")
        ####
        if any(float(start) < 0.0 for start in start_times.flat) or any(
            float(end) > self.repetition_period_s for end in (start_times + durations).flat
        ):
            raise ValueError("Bursts must fit inside the repetition period.")
        ####
        gaps = start_times[1:, :] - (start_times[:-1, :] + durations[:-1, :])
        if any(float(gap) < 0.0 for gap in gaps.flat):
            raise ValueError("Per-axis maneuver bursts must not overlap.")
        ####
        start_times.setflags(write=False)
        durations.setflags(write=False)
        amplitudes.setflags(write=False)
        object.__setattr__(self, "start_times_s", start_times)
        object.__setattr__(self, "durations_s", durations)
        object.__setattr__(self, "amplitudes_rad_s", amplitudes)
    ####

    def evaluate(self, time_s: FloatArray) -> FloatArray:
        times = array(time_s, dtype=float64)
        if times.ndim != 1 or not all(isfinite(float(value)) for value in times):
            raise ValueError("time_s must be a one-dimensional array of finite values.")
        ####
        phase = (
            (times % self.repetition_period_s)[:, None, None] - self.start_times_s[None, :, :]
        ) / self.durations_s[None, :, :]
        active = (phase >= 0.0) & (phase <= 1.0)
        envelope = sin(pi * clip(phase, 0.0, 1.0)) ** 2
        envelope[~active] = 0.0
        return array(
            (envelope * self.amplitudes_rad_s[None, :, :]).sum(axis=1),
            dtype=float64,
        )
    ####
####


def sampled_rate_signal(seed: int) -> SampledRateSignal:
    random = default_rng(seed)
    harmonic_count = 7
    return SampledRateSignal(
        frequencies_rad_s=random.uniform(0.35, 2.8, harmonic_count),
        amplitudes_rad_s=random.normal(0.0, 0.22, (3, harmonic_count)),
        phases_rad=random.uniform(-pi, pi, (3, harmonic_count)),
        offsets_rad_s=random.normal(0.0, 0.08, 3),
    )
####


@lru_cache(maxsize=16)
def piecewise_linear_rate_signal(seed: int) -> PiecewiseLinearRateSignal:
    """Build a query-independent seeded rate signal from fixed 0.5 s knots."""

    if seed < 0:
        raise ValueError("seed must be non-negative.")
    ####
    knot_times_s = arange(
        RATE_KNOT_MIN_TIME_S,
        RATE_KNOT_MAX_TIME_S + RATE_KNOT_PERIOD_S / 2.0,
        RATE_KNOT_PERIOD_S,
        dtype=float64,
    )
    knot_rates_rad_s = default_rng(seed).normal(
        0.0,
        RATE_KNOT_SIGMA_RAD_S,
        (len(knot_times_s), 3),
    )
    return PiecewiseLinearRateSignal(knot_times_s, knot_rates_rad_s)
####


@lru_cache(maxsize=16)
def maneuver_burst_rate_signal(seed: int) -> ManeuverBurstRateSignal:
    """Build seeded smooth 3-axis rate pulses with inter-burst rest intervals."""

    if seed < 0:
        raise ValueError("seed must be non-negative.")
    ####
    random = default_rng(seed)
    axis_delays_s = array((0.0, 0.06, 0.12), dtype=float64)
    slot_starts_s = (
        arange(MANEUVER_RATE_BURST_COUNT, dtype=float64)[:, None] * MANEUVER_RATE_SLOT_PERIOD_S
    )
    start_times_s = (
        slot_starts_s
        + axis_delays_s[None, :]
        + random.uniform(0.02, 0.04, (MANEUVER_RATE_BURST_COUNT, 3))
    )
    durations_s = random.uniform(0.72, 0.78, (MANEUVER_RATE_BURST_COUNT, 3))
    amplitudes_rad_s = random.normal(0.0, 0.3, (MANEUVER_RATE_BURST_COUNT, 3))
    return ManeuverBurstRateSignal(
        start_times_s=start_times_s,
        durations_s=durations_s,
        amplitudes_rad_s=amplitudes_rad_s,
        repetition_period_s=MANEUVER_RATE_PERIOD_S,
    )
####


def _trapezoidal_rate_increment(
    first_rate_rad_s: FloatArray,
    second_rate_rad_s: FloatArray,
    duration_s: float,
) -> FloatArray:
    """Return the quaternion increment from endpoint-averaged body rates."""

    rotation_vector = 0.5 * (first_rate_rad_s + second_rate_rad_s) * duration_s
    angle = sqrt(float((rotation_vector * rotation_vector).sum()))
    vector_scale = 0.5 if angle <= 1.0e-15 else sin(0.5 * angle) / angle
    return array(
        [
            rotation_vector[0] * vector_scale,
            rotation_vector[1] * vector_scale,
            rotation_vector[2] * vector_scale,
            cos(0.5 * angle),
        ],
        dtype=float64,
    )
####


def _fixed_grid_rate_quaternions(
    time_s: FloatArray,
    rate_signal: _RateSignal,
    *,
    reference_time_s: float,
    integration_step_s: float,
) -> FloatArray:
    """Evaluate one absolute-time trajectory on a query-independent fixed grid."""

    scaled_offsets = (time_s - reference_time_s) / integration_step_s
    base_indices: list[int] = []
    for scaled_offset in scaled_offsets:
        offset = float(scaled_offset)
        nearest_index = round(offset)
        if abs(offset - nearest_index) <= 1.0e-10:
            base_index = nearest_index
        elif offset > 0.0:
            base_index = floor(offset)
        else:
            base_index = ceil(offset)
        ####
        base_indices.append(int(base_index))
    ####

    negative_steps = max(0, -min(base_indices))
    positive_steps = max(0, max(base_indices))
    grid_indices = arange(-negative_steps, positive_steps + 1, dtype=float64)
    grid_times = reference_time_s + grid_indices * integration_step_s
    grid_rates = rate_signal.evaluate(grid_times)
    grid_quaternions = zeros((len(grid_times), 4), dtype=float64)
    anchor_index = negative_steps
    grid_quaternions[anchor_index, 3] = 1.0

    for step in range(1, positive_steps + 1):
        previous_index = anchor_index + step - 1
        current_index = anchor_index + step
        increment = _trapezoidal_rate_increment(
            grid_rates[previous_index],
            grid_rates[current_index],
            integration_step_s,
        )
        grid_quaternions[current_index] = _multiply_quaternions(
            grid_quaternions[previous_index : previous_index + 1],
            increment[None, :],
        )[0]
    ####
    for step in range(1, negative_steps + 1):
        later_index = anchor_index - step + 1
        earlier_index = anchor_index - step
        increment = _trapezoidal_rate_increment(
            grid_rates[later_index],
            grid_rates[earlier_index],
            -integration_step_s,
        )
        grid_quaternions[earlier_index] = _multiply_quaternions(
            grid_quaternions[later_index : later_index + 1],
            increment[None, :],
        )[0]
    ####

    query_rates = rate_signal.evaluate(time_s)
    query_quaternions = zeros((len(time_s), 4), dtype=float64)
    for sample_index, (base_index, query_time) in enumerate(zip(base_indices, time_s, strict=True)):
        grid_index = base_index + anchor_index
        grid_time = float(grid_times[grid_index])
        duration = float(query_time) - grid_time
        if abs(duration) <= integration_step_s * 1.0e-10:
            duration = 0.0
        ####
        increment = _trapezoidal_rate_increment(
            grid_rates[grid_index],
            query_rates[sample_index],
            duration,
        )
        query_quaternions[sample_index] = _multiply_quaternions(
            grid_quaternions[grid_index : grid_index + 1],
            increment[None, :],
        )[0]
    ####
    return query_quaternions
####


def _validated_rate_motion_times(
    time_s: FloatArray,
    *,
    seed: int,
    reference_time_s: float | None,
    integration_step_s: float | None,
) -> FloatArray:
    times = array(time_s, dtype=float64)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("time_s must be a one-dimensional array with at least two samples.")
    ####
    if not all(isfinite(float(interval)) and interval > 0.0 for interval in times[1:] - times[:-1]):
        raise ValueError("time_s must be finite and strictly increasing.")
    ####
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    ####
    if reference_time_s is not None and integration_step_s is None:
        raise ValueError("reference_time_s requires integration_step_s.")
    ####
    if reference_time_s is not None and not isfinite(reference_time_s):
        raise ValueError("reference_time_s must be finite.")
    ####
    if integration_step_s is not None and (
        not isfinite(integration_step_s) or integration_step_s <= 0.0
    ):
        raise ValueError("integration_step_s must be positive and finite.")
    ####
    return times
####


def _motion_from_rate_signal(
    times: FloatArray,
    rate_signal: _RateSignal,
    *,
    reference_time_s: float | None,
    integration_step_s: float | None,
) -> SyntheticMotion:
    body_rates = rate_signal.evaluate(times)

    if integration_step_s is None:
        quaternions = zeros((times.size, 4), dtype=float64)
        quaternions[0, 3] = 1.0
        for index, duration in enumerate(times[1:] - times[:-1], start=1):
            increment = _trapezoidal_rate_increment(
                body_rates[index - 1],
                body_rates[index],
                float(duration),
            )
            quaternions[index] = _multiply_quaternions(
                quaternions[index - 1 : index], increment[None, :]
            )[0]
        ####
    else:
        reference = times[0] if reference_time_s is None else reference_time_s
        quaternions = _fixed_grid_rate_quaternions(
            times,
            rate_signal,
            reference_time_s=reference,
            integration_step_s=integration_step_s,
        )
    ####

    x, y, z, w = quaternions.T
    euler_xyz = stack(
        (
            arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            arcsin(clip(2.0 * (w * y - z * x), -1.0, 1.0)),
            arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        ),
        axis=1,
    )
    twice_cross = 2.0 * cross(quaternions[:, :3], body_rates)
    world_rates = (
        body_rates + quaternions[:, 3:4] * twice_cross + cross(quaternions[:, :3], twice_cross)
    )
    return SyntheticMotion(
        time_s=times,
        euler_xyz_rad=euler_xyz,
        quaternion_xyzw=quaternions,
        body_rates_rad_s=body_rates,
        world_rates_rad_s=world_rates,
    )
####


def sampled_rate_motion(
    time_s: FloatArray,
    *,
    seed: int = DEFAULT_MOTION_SEED,
    reference_time_s: float | None = None,
    integration_step_s: float | None = None,
) -> SyntheticMotion:
    """Integrate a seeded random Fourier-mixture body-rate trajectory."""

    times = _validated_rate_motion_times(
        time_s,
        seed=seed,
        reference_time_s=reference_time_s,
        integration_step_s=integration_step_s,
    )
    return _motion_from_rate_signal(
        times,
        sampled_rate_signal(seed),
        reference_time_s=reference_time_s,
        integration_step_s=integration_step_s,
    )
####


def piecewise_linear_rate_motion(
    time_s: FloatArray,
    *,
    seed: int = DEFAULT_MOTION_SEED,
    reference_time_s: float | None = None,
    integration_step_s: float | None = None,
) -> SyntheticMotion:
    """Integrate seeded random body rates linearly interpolated between fixed knots."""

    times = _validated_rate_motion_times(
        time_s,
        seed=seed,
        reference_time_s=reference_time_s,
        integration_step_s=integration_step_s,
    )
    return _motion_from_rate_signal(
        times,
        piecewise_linear_rate_signal(seed),
        reference_time_s=reference_time_s,
        integration_step_s=integration_step_s,
    )
####


def maneuver_burst_rate_motion(
    time_s: FloatArray,
    *,
    seed: int = DEFAULT_MOTION_SEED,
    reference_time_s: float | None = None,
    integration_step_s: float | None = None,
) -> SyntheticMotion:
    """Integrate a seeded, periodic sequence of smooth body-rate maneuvers."""

    times = _validated_rate_motion_times(
        time_s,
        seed=seed,
        reference_time_s=reference_time_s,
        integration_step_s=integration_step_s,
    )
    return _motion_from_rate_signal(
        times,
        maneuver_burst_rate_signal(seed),
        reference_time_s=reference_time_s,
        integration_step_s=integration_step_s,
    )
####


def _motion_for_profile(
    time_s: FloatArray,
    profile: MotionProfile,
    motion_seed: int,
) -> SyntheticMotion:
    if profile is MotionProfile.SAMPLED_RATE_NONANALYTIC:
        return sampled_rate_motion(
            time_s,
            seed=motion_seed,
            reference_time_s=0.0,
            integration_step_s=SAMPLED_RATE_INTEGRATION_STEP_S,
        )
    ####
    if profile is MotionProfile.SAMPLED_RATE_PIECEWISE_LINEAR:
        return piecewise_linear_rate_motion(
            time_s,
            seed=motion_seed,
            reference_time_s=0.0,
            integration_step_s=SAMPLED_RATE_INTEGRATION_STEP_S,
        )
    ####
    if profile is MotionProfile.SAMPLED_RATE_MANEUVER_BURSTS:
        return maneuver_burst_rate_motion(
            time_s,
            seed=motion_seed,
            reference_time_s=0.0,
            integration_step_s=SAMPLED_RATE_INTEGRATION_STEP_S,
        )
    ####
    return analytic_motion(time_s, profile)
####


def _motion_label_source(profile: MotionProfile, motion_seed: int) -> str:
    integration_description = (
        "independent fixed-grid trapezoidal quaternion integration and seeded acquisition "
        "perturbations: scripts/generate_validation_corpus.py"
    )
    if profile is MotionProfile.SAMPLED_RATE_NONANALYTIC:
        return (
            f"seeded random Fourier rates (motion seed {motion_seed}) with "
            f"{integration_description}"
        )
    ####
    if profile is MotionProfile.SAMPLED_RATE_PIECEWISE_LINEAR:
        return (
            f"seeded piecewise-linear rate knots (motion seed {motion_seed}) with "
            f"{integration_description}"
        )
    ####
    if profile is MotionProfile.SAMPLED_RATE_MANEUVER_BURSTS:
        return (
            f"seeded smooth maneuver-burst rates (motion seed {motion_seed}) with "
            f"{integration_description}"
        )
    ####
    return "analytic ZYX generator with seeded acquisition perturbations: " + (
        "scripts/generate_validation_corpus.py"
    )
####


def _noisy_quaternions(
    quaternions_xyzw: FloatArray,
    *,
    sigma_rad: float,
    random: Generator,
    ar1_coefficient: float = 0.0,
    sample_period_s: float = NOISE_SAMPLE_PERIOD_S,
    correlation_time_s: float | None = None,
) -> FloatArray:
    """Apply seeded isotropic rotation-vector noise with optional temporal correlation."""

    if correlation_time_s is None:
        rotation_vectors = correlated_gaussian_noise(
            random,
            sigma=sigma_rad,
            sample_count=len(quaternions_xyzw),
            dimension_count=3,
            ar1_coefficient=ar1_coefficient,
        )
    else:
        rotation_vectors = sample_time_domain_ou_noise(
            random,
            len(quaternions_xyzw),
            sample_period_s,
            sigma_rad,
            correlation_time_s,
        )
    ####
    angles = sqrt((rotation_vectors * rotation_vectors).sum(axis=1))
    factors = zeros(len(angles), dtype=float64)
    nonzero = angles > 0.0
    factors[nonzero] = sin(angles[nonzero] / 2.0) / angles[nonzero]
    factors[~nonzero] = 0.5
    noise_quaternions = stack(
        (
            rotation_vectors[:, 0] * factors,
            rotation_vectors[:, 1] * factors,
            rotation_vectors[:, 2] * factors,
            cos(angles / 2.0),
        ),
        axis=1,
    )
    return _multiply_quaternions(noise_quaternions, quaternions_xyzw)
####


def _rotate_rate_vectors_between_frames(
    quaternion_xyzw: FloatArray,
    rates_rad_s: FloatArray,
    *,
    from_frame: TruthRateFrame,
    to_frame: TruthRateFrame,
) -> FloatArray:
    """Rotate vectors between body and world coordinates using body-to-world attitudes."""

    if from_frame == to_frame:
        return rates_rad_s.copy()
    ####
    direction = 1.0 if from_frame == "body" else -1.0
    rotation_vector = direction * quaternion_xyzw[:, :3]
    twice_cross = 2.0 * cross(rotation_vector, rates_rad_s)
    return rates_rad_s + quaternion_xyzw[:, 3:4] * twice_cross + cross(rotation_vector, twice_cross)
####


def correlated_gaussian_noise(
    random: Generator,
    *,
    sigma: float,
    sample_count: int,
    dimension_count: int,
    ar1_coefficient: float,
) -> FloatArray:
    """Generate stationary per-axis Gaussian AR(1) noise with marginal std ``sigma``."""

    if not isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigma must be finite and non-negative.")
    ####
    if sample_count < 1 or dimension_count < 1:
        raise ValueError("sample_count and dimension_count must be positive.")
    ####
    if not isfinite(ar1_coefficient) or abs(ar1_coefficient) >= 1.0:
        raise ValueError("ar1_coefficient must be strictly between -1 and 1.")
    ####
    innovations = random.normal(0.0, sigma, (sample_count, dimension_count)).astype(float64)
    if ar1_coefficient == 0.0:
        return innovations
    ####
    correlated = zeros(innovations.shape, dtype=float64)
    correlated[0] = innovations[0]
    innovation_scale = float(sqrt(1.0 - ar1_coefficient**2))
    for sample_index in range(1, sample_count):
        correlated[sample_index] = (
            ar1_coefficient * correlated[sample_index - 1]
            + innovation_scale * innovations[sample_index]
        )
    ####
    return correlated
####


def _retained_sample_mask(
    sample_count: int,
    sample_period_s: float,
    dropout_probability: float,
    dropout_seed: SeedSequence,
    sample_counts: tuple[int, ...],
) -> NDArray[bool_]:
    """Draw paired Bernoulli dropouts while retaining configured duration endpoints."""

    retained = sample_time_domain_dropout_mask(
        default_rng(dropout_seed),
        sample_count,
        sample_period_s,
        dropout_probability,
    )
    protected_endpoints = {0}
    protected_endpoints.update(count - 1 for count in sample_counts if count <= sample_count)
    retained[list(protected_endpoints)] = True
    return retained
####


def _write_csv(path: Path, time_s: FloatArray, orientation: FloatArray, rates: FloatArray) -> None:
    orientation_names = tuple(f"a{index}" for index in range(orientation.shape[1]))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("time_s", *orientation_names, "r0", "r1", "r2"))
        for timestamp, attitude, rate in zip(time_s, orientation, rates, strict=True):
            writer.writerow((timestamp, *attitude, *rate))
        ####
    ####
####


def _encode_measurements_for_truth(
    quaternions_xyzw: FloatArray,
    frame_rates_pqr_rad_s: FloatArray,
    truth: QuaternionTruthConvention,
) -> tuple[FloatArray, FloatArray]:
    """Encode physical attitudes and rates as raw channels for one truth label."""

    encoded_quaternions = quaternions_xyzw.copy()
    if truth.attitude_direction == "world_to_body":
        encoded_quaternions[:, :3] *= -1.0
    ####
    if truth.quaternion_order == "scalar_first":
        encoded_quaternions = encoded_quaternions[:, (3, 0, 1, 2)]
    ####

    rate_axis_positions = {component: index for index, component in enumerate("pqr")}
    ordered_rates = frame_rates_pqr_rad_s[
        :, [rate_axis_positions[component] for component in truth.rate_order]
    ]
    encoded_rates = ordered_rates * truth.rate_sign
    if truth.rate_unit == "degrees_per_second":
        encoded_rates *= 180.0 / pi
    ####
    return encoded_quaternions, encoded_rates
####


def _truth_convention_token(truth: QuaternionTruthConvention) -> str:
    direction = "b2w" if truth.attitude_direction == "body_to_world" else "w2b"
    quaternion_order = "sf" if truth.quaternion_order == "scalar_first" else "sl"
    rate_frame = "b" if truth.rate_frame == "body" else "w"
    rate_sign = "plus" if truth.rate_sign == 1 else "minus"
    rate_unit = "rad" if truth.rate_unit == "radians_per_second" else "deg"
    return f"{direction}-{quaternion_order}-{rate_frame}-{truth.rate_order}-{rate_sign}-{rate_unit}"
####


def generate_corpus(
    outdir: Path,
    *,
    noise_profiles: tuple[NoiseProfile, ...] = (NOISE_PROFILES[0],),
    noise_replicates: int = 1,
    noise_seed_start: int = DEFAULT_NOISE_SEED_START,
    motion_seed: int | None = None,
    motion_seeds: tuple[int, ...] = (),
    noise_sample_counts: tuple[int, ...] = (161,),
    sample_period_s: float = NOISE_SAMPLE_PERIOD_S,
    truth_rate_units: tuple[TruthRateUnit, ...] = ("radians_per_second",),
    truth_convention_sweep: bool = False,
    motion_profiles: tuple[MotionProfile, ...] = (MotionProfile.RICH_THREE_AXIS,),
    timestamp_jitter_amplitudes_s: tuple[float, ...] = (0.0,),
    timestamp_clock_rate_errors_ppm: tuple[float, ...] = (0.0,),
    dropout_probabilities: tuple[float, ...] = (0.0,),
    orientation_latencies_s: tuple[float, ...] = (0.0,),
    rate_latencies_s: tuple[float, ...] = (0.0,),
    orientation_noise_correlation_time_s: float | None = None,
    rate_noise_correlation_time_s: float | None = None,
    rate_noise_frame: RateNoiseFrame = "truth-rate-frame",
    rate_sensor_calibrations: tuple[RateSensorCalibration, ...] = (
        RATE_SENSOR_CALIBRATION_PROFILES[0],
    ),
    orientation_noise_ar1_coefficients: tuple[float, ...] = (0.0,),
    rate_noise_ar1_coefficients: tuple[float, ...] = (0.0,),
) -> Path:
    """Write labeled synthetic cases across motion and acquisition conditions."""

    if not noise_profiles:
        raise ValueError("At least one noise profile is required.")
    ####
    if not motion_profiles:
        raise ValueError("At least one motion profile is required.")
    ####
    if not rate_sensor_calibrations:
        raise ValueError("At least one rate-sensor calibration profile is required.")
    ####
    calibration_ids = tuple(calibration.profile_id for calibration in rate_sensor_calibrations)
    if len(calibration_ids) != len(set(calibration_ids)):
        raise ValueError("rate-sensor calibration profile IDs must be unique.")
    ####
    if rate_noise_frame not in ("truth-rate-frame", "body", "world"):
        raise ValueError("rate_noise_frame must be truth-rate-frame, body, or world.")
    ####
    if not isfinite(sample_period_s) or sample_period_s <= 0.0:
        raise ValueError("sample_period_s must be positive and finite.")
    ####
    for name, time_constant_s in (
        ("orientation", orientation_noise_correlation_time_s),
        ("rate", rate_noise_correlation_time_s),
    ):
        if time_constant_s is not None and (
            not isfinite(time_constant_s) or time_constant_s <= 0.0
        ):
            raise ValueError(f"{name} noise correlation time must be positive and finite.")
        ####
    ####
    if orientation_noise_correlation_time_s is not None:
        if orientation_noise_ar1_coefficients != (0.0,):
            raise ValueError(
                "Specify orientation OU correlation time or AR(1) coefficients, not both."
            )
        ####
        orientation_noise_ar1_coefficients = (
            exp(-sample_period_s / orientation_noise_correlation_time_s),
        )
    ####
    if rate_noise_correlation_time_s is not None:
        if rate_noise_ar1_coefficients != (0.0,):
            raise ValueError("Specify rate OU correlation time or AR(1) coefficients, not both.")
        ####
        rate_noise_ar1_coefficients = (exp(-sample_period_s / rate_noise_correlation_time_s),)
    ####
    if (
        orientation_noise_correlation_time_s is not None
        or rate_noise_correlation_time_s is not None
    ):
        validate_ou_sample_period(sample_period_s)
    ####
    if motion_seed is not None and motion_seeds:
        raise ValueError("Use either motion_seed or motion_seeds, not both.")
    ####
    if motion_seed is not None and motion_seed < 0:
        raise ValueError("motion_seed must be non-negative.")
    ####
    if any(seed < 0 for seed in motion_seeds):
        raise ValueError("motion_seeds must contain non-negative seeds.")
    ####
    if len(motion_seeds) != len(set(motion_seeds)):
        raise ValueError("motion_seeds must be unique.")
    ####
    if motion_seeds and not any(profile in SAMPLED_RATE_PROFILES for profile in motion_profiles):
        raise ValueError("motion_seeds requires at least one sampled-rate motion profile.")
    ####
    resolved_motion_seed = DEFAULT_MOTION_SEED if motion_seed is None else motion_seed
    sampled_motion_seeds = motion_seeds or (resolved_motion_seed,)
    supported_rate_units: tuple[TruthRateUnit, ...] = (
        "radians_per_second",
        "degrees_per_second",
    )
    if not truth_rate_units or any(unit not in supported_rate_units for unit in truth_rate_units):
        raise ValueError(
            "truth_rate_units must contain radians_per_second and/or degrees_per_second."
        )
    ####
    if len(truth_rate_units) != len(set(truth_rate_units)):
        raise ValueError("truth_rate_units must be unique.")
    ####
    if truth_convention_sweep and truth_rate_units != ("radians_per_second",):
        raise ValueError(
            "truth_rate_units cannot be customized with truth_convention_sweep; "
            "the full sweep includes both rate units."
        )
    ####
    truth_conventions = (
        all_quaternion_truth_conventions()
        if truth_convention_sweep
        else tuple(replace(BASE_QUATERNION_TRUTH, rate_unit=unit) for unit in truth_rate_units)
    )
    noisy_rate_unit_search = (
        supported_rate_units
        if truth_convention_sweep or "degrees_per_second" in truth_rate_units
        else ("radians_per_second",)
    )
    noisy_rate_sign_search = TRUTH_RATE_SIGNS if truth_convention_sweep else (1,)
    search_space: dict[str, object] = {}
    if len(noisy_rate_unit_search) > 1:
        search_space["rate_units"] = list(noisy_rate_unit_search)
    ####
    if len(noisy_rate_sign_search) > 1:
        search_space["rate_signs"] = list(noisy_rate_sign_search)
    ####
    config: dict[str, object] = {"search_space": search_space} if search_space else {}
    if noise_replicates < 1:
        raise ValueError("noise_replicates must be positive.")
    ####
    if noise_seed_start < 0:
        raise ValueError("noise_seed_start must be non-negative.")
    ####
    if not noise_sample_counts or any(count < 21 for count in noise_sample_counts):
        raise ValueError("Each noise sample count must be at least 21.")
    ####
    if len(noise_sample_counts) != len(set(noise_sample_counts)):
        raise ValueError("noise_sample_counts must be unique.")
    ####
    factor_values = {
        "timestamp_jitter_amplitudes_s": timestamp_jitter_amplitudes_s,
        "dropout_probabilities": dropout_probabilities,
        "orientation_latencies_s": orientation_latencies_s,
        "rate_latencies_s": rate_latencies_s,
    }
    for name, values in factor_values.items():
        if not values or any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError(f"{name} must contain finite non-negative values.")
        ####
        if len(values) != len(set(values)):
            raise ValueError(f"{name} must be unique.")
        ####
    ####
    if not timestamp_clock_rate_errors_ppm or any(
        not isfinite(value) or value <= -1_000_000.0 for value in timestamp_clock_rate_errors_ppm
    ):
        raise ValueError(
            "timestamp_clock_rate_errors_ppm must contain finite values greater than -1,000,000."
        )
    ####
    if len(timestamp_clock_rate_errors_ppm) != len(set(timestamp_clock_rate_errors_ppm)):
        raise ValueError("timestamp_clock_rate_errors_ppm values must be unique.")
    ####
    if any(probability >= 1.0 for probability in dropout_probabilities):
        raise ValueError("dropout_probabilities must be less than 1.")
    ####
    if any(amplitude >= sample_period_s / 2.0 for amplitude in timestamp_jitter_amplitudes_s):
        raise ValueError("timestamp jitter amplitudes must be less than half the sample period.")
    ####
    correlation_values = {
        "orientation_noise_ar1_coefficients": orientation_noise_ar1_coefficients,
        "rate_noise_ar1_coefficients": rate_noise_ar1_coefficients,
    }
    for name, values in correlation_values.items():
        if not values or any(not isfinite(value) or abs(value) >= 1.0 for value in values):
            raise ValueError(f"{name} must be strictly between -1 and 1.")
        ####
        if len(values) != len(set(values)):
            raise ValueError(f"{name} must be unique.")
        ####
    ####
    for sample_count in noise_sample_counts:
        for dropout_probability in dropout_probabilities:
            for replicate_index in range(noise_replicates):
                noise_seed = noise_seed_start + replicate_index
                dropout_seed = SeedSequence(noise_seed).spawn(4)[3]
                retained = _retained_sample_mask(
                    sample_count,
                    sample_period_s,
                    dropout_probability,
                    dropout_seed,
                    noise_sample_counts,
                )
                if int(retained.sum()) < 21:
                    raise ValueError(
                        "A seeded dropout realization left fewer than 21 samples for "
                        f"n={sample_count}, p={dropout_probability}, seed={noise_seed}."
                    )
                ####
            ####
        ####
    ####
    outdir.mkdir(parents=True, exist_ok=False)
    regular = analytic_motion(arange(101, dtype=float64) * 0.025)
    jitter = 0.025 + 0.003 * sin(arange(120, dtype=float64) * 0.7)
    jittered_time = array([0.0, *cumsum(jitter)], dtype=float64)
    keep = arange(len(jittered_time)) % 13 != 7
    jittered = analytic_motion(jittered_time[keep])
    euler = analytic_motion(arange(91, dtype=float64) * 0.03)
    _write_csv(
        outdir / "quaternion_body.csv",
        regular.time_s,
        regular.quaternion_xyzw,
        regular.body_rates_rad_s,
    )
    inverted = jittered.quaternion_xyzw.copy()
    inverted[:, :3] *= -1.0
    inverted[::11] *= -1.0
    world_rqp = jittered.world_rates_rad_s[:, (2, 1, 0)]
    _write_csv(
        outdir / "quaternion_world_jittered.csv",
        jittered.time_s,
        inverted[:, (3, 0, 1, 2)],
        world_rqp,
    )
    _write_csv(
        outdir / "euler_degrees.csv",
        euler.time_s,
        euler.euler_xyz_rad * (180.0 / pi),
        euler.body_rates_rad_s * (180.0 / pi),
    )

    noisy_cases: list[dict[str, object]] = []
    for motion_profile in motion_profiles:
        profile_motion_seeds = (
            sampled_motion_seeds
            if motion_profile in SAMPLED_RATE_PROFILES
            else (resolved_motion_seed,)
        )
        for sample_count in noise_sample_counts:
            noise_times = arange(sample_count, dtype=float64) * sample_period_s
            for profile in noise_profiles:
                condition_factors = (
                    _NoisyCaseFactor(
                        motion_seed=motion_seed,
                        jitter_index=jitter_index,
                        jitter_amplitude_s=jitter_amplitude_s,
                        clock_rate_index=clock_rate_index,
                        clock_rate_error_ppm=clock_rate_error_ppm,
                        dropout_index=dropout_index,
                        dropout_probability=dropout_probability,
                        orientation_latency_index=orientation_latency_index,
                        orientation_latency_s=orientation_latency_s,
                        rate_latency_index=rate_latency_index,
                        rate_latency_s=rate_latency_s,
                        orientation_ar1_index=orientation_ar1_index,
                        orientation_ar1_coefficient=orientation_ar1_coefficient,
                        rate_ar1_index=rate_ar1_index,
                        rate_ar1_coefficient=rate_ar1_coefficient,
                        calibration_index=calibration_index,
                        calibration=calibration,
                        truth_convention=truth_convention,
                        replicate_index=replicate_index,
                    )
                    for (
                        motion_seed,
                        (jitter_index, jitter_amplitude_s),
                        (clock_rate_index, clock_rate_error_ppm),
                        (dropout_index, dropout_probability),
                        (orientation_latency_index, orientation_latency_s),
                        (rate_latency_index, rate_latency_s),
                        (orientation_ar1_index, orientation_ar1_coefficient),
                        (rate_ar1_index, rate_ar1_coefficient),
                    ) in product(
                        profile_motion_seeds,
                        enumerate(timestamp_jitter_amplitudes_s),
                        enumerate(timestamp_clock_rate_errors_ppm),
                        enumerate(dropout_probabilities),
                        enumerate(orientation_latencies_s),
                        enumerate(rate_latencies_s),
                        enumerate(orientation_noise_ar1_coefficients),
                        enumerate(rate_noise_ar1_coefficients),
                    )
                    for calibration_index, calibration in enumerate(rate_sensor_calibrations)
                    for truth_convention in truth_conventions
                    for replicate_index in range(noise_replicates)
                )
                prepared_condition_key: tuple[int, ...] | None = None
                prepared_measurements: tuple[_PreparedMeasurements, ...] = ()
                for factor in condition_factors:
                    motion_seed = factor.motion_seed
                    jitter_index = factor.jitter_index
                    jitter_amplitude_s = factor.jitter_amplitude_s
                    clock_rate_index = factor.clock_rate_index
                    clock_rate_error_ppm = factor.clock_rate_error_ppm
                    dropout_index = factor.dropout_index
                    dropout_probability = factor.dropout_probability
                    orientation_latency_index = factor.orientation_latency_index
                    orientation_latency_s = factor.orientation_latency_s
                    rate_latency_index = factor.rate_latency_index
                    rate_latency_s = factor.rate_latency_s
                    orientation_ar1_index = factor.orientation_ar1_index
                    orientation_ar1_coefficient = factor.orientation_ar1_coefficient
                    rate_ar1_index = factor.rate_ar1_index
                    rate_ar1_coefficient = factor.rate_ar1_coefficient
                    calibration_index = factor.calibration_index
                    calibration = factor.calibration
                    truth_convention = factor.truth_convention
                    replicate_index = factor.replicate_index
                    condition_key = (
                        motion_seed,
                        jitter_index,
                        clock_rate_index,
                        dropout_index,
                        orientation_latency_index,
                        rate_latency_index,
                        orientation_ar1_index,
                        rate_ar1_index,
                    )
                    if condition_key != prepared_condition_key:
                        prepared_condition_key = condition_key
                        prepared_for_condition: list[_PreparedMeasurements] = []
                        for prepared_replicate_index in range(noise_replicates):
                            prepared_noise_seed = noise_seed_start + prepared_replicate_index
                            (
                                orientation_seed,
                                rate_seed,
                                jitter_seed,
                                dropout_seed,
                            ) = SeedSequence(prepared_noise_seed).spawn(4)
                            orientation_random = default_rng(orientation_seed)
                            rate_random = default_rng(rate_seed)
                            jitter_random = default_rng(jitter_seed)
                            timestamp_jitter = sample_time_domain_uniform_values(
                                jitter_random,
                                sample_count,
                                sample_period_s,
                                -jitter_amplitude_s,
                                jitter_amplitude_s,
                            )
                            timestamp_jitter[0] = 0.0
                            observed_times = noise_times + timestamp_jitter
                            orientation_motion = _motion_for_profile(
                                observed_times - orientation_latency_s,
                                motion_profile,
                                motion_seed,
                            )
                            rate_motion = _motion_for_profile(
                                observed_times - rate_latency_s,
                                motion_profile,
                                motion_seed,
                            )
                            noisy_quaternions = _noisy_quaternions(
                                orientation_motion.quaternion_xyzw,
                                sigma_rad=profile.orientation_sigma_rad,
                                random=orientation_random,
                                ar1_coefficient=orientation_ar1_coefficient,
                                sample_period_s=sample_period_s,
                                correlation_time_s=orientation_noise_correlation_time_s,
                            )
                            if rate_noise_correlation_time_s is None:
                                rate_noise = correlated_gaussian_noise(
                                    rate_random,
                                    sigma=profile.rate_sigma_rad_s,
                                    sample_count=sample_count,
                                    dimension_count=3,
                                    ar1_coefficient=rate_ar1_coefficient,
                                )
                            else:
                                rate_noise = sample_time_domain_ou_noise(
                                    rate_random,
                                    sample_count,
                                    sample_period_s,
                                    profile.rate_sigma_rad_s,
                                    rate_noise_correlation_time_s,
                                )
                            ####
                            if rate_noise_frame == "truth-rate-frame":
                                body_rate_noise = rate_noise
                                world_rate_noise = rate_noise
                            elif rate_noise_frame == "body":
                                body_rate_noise = rate_noise
                                world_rate_noise = _rotate_rate_vectors_between_frames(
                                    rate_motion.quaternion_xyzw,
                                    rate_noise,
                                    from_frame="body",
                                    to_frame="world",
                                )
                            else:
                                body_rate_noise = _rotate_rate_vectors_between_frames(
                                    rate_motion.quaternion_xyzw,
                                    rate_noise,
                                    from_frame="world",
                                    to_frame="body",
                                )
                                world_rate_noise = rate_noise
                            ####
                            body_rate_measurements = tuple(
                                selected_calibration.apply_to_rates(
                                    rate_motion.body_rates_rad_s,
                                    measurement_noise_rad_s=body_rate_noise,
                                )
                                for selected_calibration in rate_sensor_calibrations
                            )
                            world_rate_measurements = tuple(
                                selected_calibration.apply_to_rates(
                                    rate_motion.world_rates_rad_s,
                                    measurement_noise_rad_s=world_rate_noise,
                                )
                                for selected_calibration in rate_sensor_calibrations
                            )
                            timestamp_scale = 1.0 + clock_rate_error_ppm * 1.0e-6
                            recorded_times = observed_times * timestamp_scale
                            retained = _retained_sample_mask(
                                sample_count,
                                sample_period_s,
                                dropout_probability,
                                dropout_seed,
                                noise_sample_counts,
                            )
                            if int(retained.sum()) < 21:
                                raise ValueError(
                                    "Dropout realization left fewer than 21 samples for "
                                    f"{motion_profile.value}, {profile.name}, "
                                    f"n={sample_count}, p={dropout_probability}, "
                                    f"seed={prepared_noise_seed}."
                                )
                            ####
                            prepared_for_condition.append(
                                _PreparedMeasurements(
                                    noise_seed=prepared_noise_seed,
                                    recorded_times_s=recorded_times,
                                    retained_samples=retained,
                                    orientation_quaternions_xyzw=noisy_quaternions,
                                    body_rate_measurements_rad_s=body_rate_measurements,
                                    world_rate_measurements_rad_s=world_rate_measurements,
                                    dropped_sample_count=int((~retained).sum()),
                                )
                            )
                        ####
                        prepared_measurements = tuple(prepared_for_condition)
                    ####
                    prepared = prepared_measurements[replicate_index]
                    noise_seed = prepared.noise_seed
                    noisy_quaternions = prepared.orientation_quaternions_xyzw
                    frame_rate_measurements = (
                        prepared.body_rate_measurements_rad_s
                        if truth_convention.rate_frame == "body"
                        else prepared.world_rate_measurements_rad_s
                    )
                    encoded_quaternions, encoded_rates = _encode_measurements_for_truth(
                        noisy_quaternions,
                        frame_rate_measurements[calibration_index],
                        truth_convention,
                    )
                    recorded_times = prepared.recorded_times_s
                    retained = prepared.retained_samples
                    actual_dropped_count = prepared.dropped_sample_count
                    jitter_token = f"j{jitter_index:02d}"
                    clock_rate_token = f"tc{clock_rate_index:02d}"
                    dropout_token = f"d{dropout_index:02d}"
                    orientation_latency_token = f"ol{orientation_latency_index:02d}"
                    rate_latency_token = f"rl{rate_latency_index:02d}"
                    orientation_ar1_token = f"oa{orientation_ar1_index:02d}"
                    rate_ar1_token = f"ra{rate_ar1_index:02d}"
                    rate_unit_token = (
                        "urad" if truth_convention.rate_unit == "radians_per_second" else "udeg"
                    )
                    condition_token = (
                        f"{jitter_token}-{dropout_token}-"
                        f"{orientation_latency_token}-{rate_latency_token}-"
                        f"{orientation_ar1_token}-{rate_ar1_token}-{rate_unit_token}"
                    )
                    condition_token += f"-cal-{calibration.profile_id}"
                    if len(timestamp_clock_rate_errors_ppm) > 1 or clock_rate_error_ppm != 0.0:
                        condition_token += f"-{clock_rate_token}"
                    ####
                    if truth_convention_sweep:
                        condition_token += f"-{_truth_convention_token(truth_convention)}"
                    ####
                    case_id = (
                        f"{motion_profile.value}-m{motion_seed}-noisy-{profile.name}-"
                        f"{condition_token}-n{sample_count}-"
                        f"{replicate_index + 1:03d}"
                    )
                    # Keep generated paths short on Windows; the manifest retains full details.
                    case_digest = sha256(case_id.encode("utf-8")).hexdigest()[:32]
                    csv_name = f"quaternion_noisy_{case_digest}.csv"
                    _write_csv(
                        outdir / csv_name,
                        recorded_times[retained],
                        encoded_quaternions[retained],
                        encoded_rates[retained],
                    )
                    noisy_cases.append(
                        {
                            "case_id": case_id,
                            "csv": csv_name,
                            "label_source": _motion_label_source(motion_profile, motion_seed),
                            "measurement_conditions": {
                                "motion_profile_id": motion_profile.value,
                                **(
                                    {"motion_seed": motion_seed}
                                    if motion_profile in SAMPLED_RATE_PROFILES
                                    else {}
                                ),
                                "timestamp_jitter_amplitude_s": jitter_amplitude_s,
                                "timestamp_clock_rate_error_ppm": clock_rate_error_ppm,
                                "dropout_probability": dropout_probability,
                                "dropped_sample_count": actual_dropped_count,
                                "orientation_latency_s": orientation_latency_s,
                                "rate_latency_s": rate_latency_s,
                                **calibration.measurement_condition_fields(),
                                "orientation_noise_sigma_rad": profile.orientation_sigma_rad,
                                "rate_noise_sigma_rad_s": profile.rate_sigma_rad_s,
                                "orientation_noise_ar1_coefficient": orientation_ar1_coefficient,
                                "rate_noise_ar1_coefficient": rate_ar1_coefficient,
                                "orientation_noise_correlation_time_s": (
                                    orientation_noise_correlation_time_s
                                ),
                                "rate_noise_correlation_time_s": rate_noise_correlation_time_s,
                                "rate_noise_frame": rate_noise_frame,
                                "noise_seed": noise_seed,
                                "sample_count": sample_count,
                                "observed_sample_count": int(retained.sum()),
                                "sample_period_s": sample_period_s,
                            },
                            "representation": "quaternion",
                            "orientation_columns": ["a0", "a1", "a2", "a3"],
                            "rate_columns": ["r0", "r1", "r2"],
                            "time_column": "time_s",
                            "truth": {
                                "attitude_direction": truth_convention.attitude_direction,
                                "quaternion_order": truth_convention.quaternion_order,
                                "rate_frame": truth_convention.rate_frame,
                                "rate_order": truth_convention.rate_order,
                                "rate_unit": truth_convention.rate_unit,
                                "rate_sign": truth_convention.rate_sign,
                            },
                            **({"config": config} if config else {}),
                        }
                    )
                ####
            ####
        ####
    ####

    manifest = {
        "schema_version": 5,
        "cases": [
            {
                "case_id": "analytic-quaternion-body",
                "csv": "quaternion_body.csv",
                "label_source": "analytic ZYX generator: scripts/generate_validation_corpus.py",
                "measurement_conditions": {
                    "motion_profile_id": MotionProfile.RICH_THREE_AXIS.value,
                },
                "representation": "quaternion",
                "orientation_columns": ["a0", "a1", "a2", "a3"],
                "rate_columns": ["r0", "r1", "r2"],
                "time_column": "time_s",
                "truth": {
                    "attitude_direction": "body_to_world",
                    "quaternion_order": "scalar_last",
                    "rate_frame": "body",
                    "rate_order": "pqr",
                    "rate_unit": "radians_per_second",
                },
            },
            {
                "case_id": "analytic-quaternion-world-jittered",
                "csv": "quaternion_world_jittered.csv",
                "label_source": "analytic ZYX generator: scripts/generate_validation_corpus.py",
                "measurement_conditions": {
                    "motion_profile_id": MotionProfile.RICH_THREE_AXIS.value,
                    "timestamp_jitter_amplitude_s": 0.003,
                    "dropped_sample_count": int((~keep).sum()),
                },
                "representation": "quaternion",
                "orientation_columns": ["a0", "a1", "a2", "a3"],
                "rate_columns": ["r0", "r1", "r2"],
                "time_column": "time_s",
                "truth": {
                    "attitude_direction": "world_to_body",
                    "quaternion_order": "scalar_first",
                    "rate_frame": "world",
                    "rate_order": "rqp",
                    "rate_unit": "radians_per_second",
                },
            },
            {
                "case_id": "analytic-euler-degrees",
                "csv": "euler_degrees.csv",
                "label_source": "analytic ZYX generator: scripts/generate_validation_corpus.py",
                "measurement_conditions": {
                    "motion_profile_id": MotionProfile.RICH_THREE_AXIS.value,
                },
                "representation": "euler",
                "orientation_columns": ["a0", "a1", "a2"],
                "rate_columns": ["r0", "r1", "r2"],
                "time_column": "time_s",
                "truth": {
                    "attitude_direction": "body_to_world",
                    "euler_order": "xyz",
                    "euler_mode": "extrinsic",
                    "angle_unit": "degrees",
                    "rate_frame": "body",
                    "rate_order": "pqr",
                    "rate_unit": "degrees_per_second",
                },
                "config": {
                    "search_space": {
                        "attitude_directions": ["body_to_world"],
                        "rate_frames": ["body"],
                        "rate_orders": ["pqr", "rqp"],
                        "rate_units": ["radians_per_second", "degrees_per_second"],
                        "angle_units": ["radians", "degrees"],
                    }
                },
            },
            *noisy_cases,
        ],
    }
    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
####


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir", type=Path, default=Path(".validation-corpus"), help="new output directory"
    )
    parser.add_argument(
        "--noise-profile",
        choices=("none", "low", "moderate", "high", "all"),
        default="low",
        help="measurement-noise profile to include; none generates clean controls",
    )
    parser.add_argument(
        "--noise-replicates",
        type=int,
        default=1,
        help="seeded realizations per noise profile",
    )
    parser.add_argument(
        "--noise-seed-start",
        type=int,
        default=DEFAULT_NOISE_SEED_START,
        help="first seeded realization; subsequent replicates use consecutive seeds",
    )
    motion_seed_group = parser.add_mutually_exclusive_group()
    motion_seed_group.add_argument(
        "--motion-seed",
        type=int,
        default=None,
        help="seed for a sampled-rate motion profile",
    )
    motion_seed_group.add_argument(
        "--motion-seeds",
        type=int,
        nargs="+",
        metavar="SEED",
        help="paired sampled-rate motion seeds; noise/acquisition seeds are reused across motions",
    )
    parser.add_argument(
        "--noise-sample-counts",
        type=int,
        nargs="+",
        default=(161,),
        metavar="N",
        help="sample counts for noisy cases; seed-paired across these durations",
    )
    parser.add_argument(
        "--sample-period-s",
        type=float,
        default=NOISE_SAMPLE_PERIOD_S,
        help="uniform sample period for noisy cases (default: 0.025 s)",
    )
    parser.add_argument(
        "--truth-rate-units",
        choices=("radians_per_second", "degrees_per_second"),
        nargs="+",
        default=("radians_per_second",),
        help="units used to write noisy angular-rate columns",
    )
    parser.add_argument(
        "--truth-convention-sweep",
        action="store_true",
        help="generate the full 192-convention quaternion/rate truth sweep",
    )
    parser.add_argument(
        "--motion-profile",
        choices=(*tuple(profile.value for profile in MotionProfile), "all"),
        default=MotionProfile.RICH_THREE_AXIS.value,
        help=("motion family for noisy cases (analytic, Fourier-rate, piecewise-rate, or bursts)"),
    )
    parser.add_argument(
        "--timestamp-jitter-amplitudes-s",
        type=float,
        nargs="+",
        default=(0.0,),
        metavar="SECONDS",
        help="maximum absolute timestamp jitter; each value is a separate condition",
    )
    parser.add_argument(
        "--timestamp-clock-rate-errors-ppm",
        type=float,
        nargs="+",
        default=(0.0,),
        metavar="PPM",
        help=(
            "timestamp-clock scale errors in parts per million; positive values mean logged "
            "elapsed time runs faster than physical elapsed time"
        ),
    )
    parser.add_argument(
        "--dropout-probabilities",
        type=float,
        nargs="+",
        default=(0.0,),
        metavar="PROBABILITY",
        help="whole-row sample dropout probabilities in [0, 1)",
    )
    parser.add_argument(
        "--orientation-latencies-s",
        type=float,
        nargs="+",
        default=(0.0,),
        metavar="SECONDS",
        help="orientation channel delays relative to the row timestamp",
    )
    parser.add_argument(
        "--rate-latencies-s",
        type=float,
        nargs="+",
        default=(0.0,),
        metavar="SECONDS",
        help="angular-rate channel delays relative to the row timestamp",
    )
    parser.add_argument(
        "--orientation-noise-correlation-time-s",
        type=float,
        help=(
            "stationary OU time constant for orientation noise; cannot be combined with "
            "non-default orientation AR(1) coefficients"
        ),
    )
    parser.add_argument(
        "--rate-noise-correlation-time-s",
        type=float,
        help=(
            "stationary OU time constant for rate noise; cannot be combined with "
            "non-default rate AR(1) coefficients"
        ),
    )
    parser.add_argument(
        "--rate-noise-frame",
        choices=("truth-rate-frame", "body", "world"),
        default="truth-rate-frame",
        help=(
            "coordinate frame for rate-noise vectors; truth-rate-frame preserves the historical "
            "per-convention injection, while body/world select one shared physical vector frame"
        ),
    )
    parser.add_argument(
        "--rate-sensor-calibrations",
        choices=tuple(profile.profile_id for profile in RATE_SENSOR_CALIBRATION_PROFILES),
        nargs="+",
        default=("identity",),
        metavar="PROFILE",
        help=(
            "known synthetic rate-channel calibrations to apply before convention encoding; "
            "choices include identity, misalignment, bias, and axis scale"
        ),
    )
    parser.add_argument(
        "--orientation-noise-ar1-coefficients",
        type=float,
        nargs="+",
        default=(0.0,),
        metavar="RHO",
        help="sample-lag AR(1) coefficients for orientation noise, strictly between -1 and 1",
    )
    parser.add_argument(
        "--rate-noise-ar1-coefficients",
        type=float,
        nargs="+",
        default=(0.0,),
        metavar="RHO",
        help="sample-lag AR(1) coefficients for rate noise, strictly between -1 and 1",
    )
    args = parser.parse_args()
    if args.noise_profile == "all":
        profiles = NOISE_PROFILES
    elif args.noise_profile == "none":
        profiles = (NO_NOISE_PROFILE,)
    else:
        profiles = tuple(
            profile for profile in NOISE_PROFILES if profile.name == args.noise_profile
        )
    ####
    motion_profiles = (
        tuple(MotionProfile)
        if args.motion_profile == "all"
        else tuple(profile for profile in MotionProfile if profile.value == args.motion_profile)
    )
    print(
        generate_corpus(
            args.outdir,
            noise_profiles=profiles,
            noise_replicates=args.noise_replicates,
            noise_seed_start=args.noise_seed_start,
            motion_seed=args.motion_seed,
            motion_seeds=tuple(args.motion_seeds or ()),
            noise_sample_counts=tuple(args.noise_sample_counts),
            sample_period_s=args.sample_period_s,
            truth_rate_units=tuple(args.truth_rate_units),
            truth_convention_sweep=args.truth_convention_sweep,
            motion_profiles=motion_profiles,
            timestamp_jitter_amplitudes_s=tuple(args.timestamp_jitter_amplitudes_s),
            timestamp_clock_rate_errors_ppm=tuple(args.timestamp_clock_rate_errors_ppm),
            dropout_probabilities=tuple(args.dropout_probabilities),
            orientation_latencies_s=tuple(args.orientation_latencies_s),
            rate_latencies_s=tuple(args.rate_latencies_s),
            orientation_noise_correlation_time_s=args.orientation_noise_correlation_time_s,
            rate_noise_correlation_time_s=args.rate_noise_correlation_time_s,
            rate_noise_frame=args.rate_noise_frame,
            rate_sensor_calibrations=tuple(
                profile
                for profile in RATE_SENSOR_CALIBRATION_PROFILES
                if profile.profile_id in args.rate_sensor_calibrations
            ),
            orientation_noise_ar1_coefficients=tuple(args.orientation_noise_ar1_coefficients),
            rate_noise_ar1_coefficients=tuple(args.rate_noise_ar1_coefficients),
        )
    )
####


if __name__ == "__main__":
    main()
####
