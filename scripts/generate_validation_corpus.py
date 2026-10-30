"""Generate analytically labeled synthetic logs without using the inference package."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from math import isfinite
from pathlib import Path
from typing import Literal

from numpy import arange, array, bool_, cos, cross, cumsum, float64, pi, sin, sqrt, stack, zeros
from numpy.random import Generator, SeedSequence, default_rng
from numpy.typing import NDArray

FloatArray = NDArray[float64]


class MotionProfile(StrEnum):
    """Independent analytic motion families used by the sensitivity harness."""

    RICH_THREE_AXIS = "rich-three-axis"
    LOW_EXCITATION = "low-excitation"
    SINGLE_AXIS = "single-axis"
    NEAR_GIMBAL = "near-gimbal"
####


@dataclass(frozen=True, slots=True)
class AnalyticMotion:
    time_s: FloatArray
    euler_xyz_rad: FloatArray
    quaternion_xyzw: FloatArray
    body_rates_rad_s: FloatArray
    world_rates_rad_s: FloatArray
####


@dataclass(frozen=True, slots=True)
class NoiseProfile:
    """Per-axis standard deviations for one seeded measurement-noise condition."""

    name: Literal["low", "moderate", "high"]
    orientation_sigma_rad: float
    rate_sigma_rad_s: float
####


NOISE_PROFILES: tuple[NoiseProfile, ...] = (
    NoiseProfile("low", 0.0015, 0.012),
    NoiseProfile("moderate", 0.003, 0.025),
    NoiseProfile("high", 0.006, 0.05),
)
NOISE_SAMPLE_PERIOD_S = 0.025


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
) -> AnalyticMotion:
    """Use closed-form ZYX rotations and body rates for a named motion family."""

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
    return AnalyticMotion(
        time_s=time_s,
        euler_xyz_rad=stack((phi, theta, psi), axis=1),
        quaternion_xyzw=quaternion,
        body_rates_rad_s=body_rates,
        world_rates_rad_s=world_rates,
    )
####


def _noisy_quaternions(
    quaternions_xyzw: FloatArray,
    *,
    sigma_rad: float,
    random: Generator,
    ar1_coefficient: float = 0.0,
) -> FloatArray:
    """Apply seeded isotropic rotation-vector noise with optional temporal AR(1)."""

    rotation_vectors = correlated_gaussian_noise(
        random,
        sigma=sigma_rad,
        sample_count=len(quaternions_xyzw),
        dimension_count=3,
        ar1_coefficient=ar1_coefficient,
    )
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
    dropout_probability: float,
    dropout_seed: SeedSequence,
    sample_counts: tuple[int, ...],
) -> NDArray[bool_]:
    """Draw paired Bernoulli dropouts while retaining configured duration endpoints."""

    retained = default_rng(dropout_seed).random(sample_count) >= dropout_probability
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


def generate_corpus(
    outdir: Path,
    *,
    noise_profiles: tuple[NoiseProfile, ...] = (NOISE_PROFILES[0],),
    noise_replicates: int = 1,
    noise_sample_counts: tuple[int, ...] = (161,),
    motion_profiles: tuple[MotionProfile, ...] = (MotionProfile.RICH_THREE_AXIS,),
    timestamp_jitter_amplitudes_s: tuple[float, ...] = (0.0,),
    dropout_probabilities: tuple[float, ...] = (0.0,),
    orientation_latencies_s: tuple[float, ...] = (0.0,),
    rate_latencies_s: tuple[float, ...] = (0.0,),
    orientation_noise_ar1_coefficients: tuple[float, ...] = (0.0,),
    rate_noise_ar1_coefficients: tuple[float, ...] = (0.0,),
) -> Path:
    """Write labeled synthetic cases across seeded acquisition conditions."""

    if not noise_profiles:
        raise ValueError("At least one noise profile is required.")
    ####
    if not motion_profiles:
        raise ValueError("At least one motion profile is required.")
    ####
    if noise_replicates < 1:
        raise ValueError("noise_replicates must be positive.")
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
    if any(probability >= 1.0 for probability in dropout_probabilities):
        raise ValueError("dropout_probabilities must be less than 1.")
    ####
    if any(amplitude >= NOISE_SAMPLE_PERIOD_S / 2.0 for amplitude in timestamp_jitter_amplitudes_s):
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
                noise_seed = 20261030 + replicate_index
                dropout_seed = SeedSequence(noise_seed).spawn(4)[3]
                retained = _retained_sample_mask(
                    sample_count,
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
        for sample_count in noise_sample_counts:
            noise_times = arange(sample_count, dtype=float64) * NOISE_SAMPLE_PERIOD_S
            for profile in noise_profiles:
                condition_factors = product(
                    enumerate(timestamp_jitter_amplitudes_s),
                    enumerate(dropout_probabilities),
                    enumerate(orientation_latencies_s),
                    enumerate(rate_latencies_s),
                    enumerate(orientation_noise_ar1_coefficients),
                    enumerate(rate_noise_ar1_coefficients),
                    range(noise_replicates),
                )
                for (
                    (jitter_index, jitter_amplitude_s),
                    (dropout_index, dropout_probability),
                    (orientation_latency_index, orientation_latency_s),
                    (rate_latency_index, rate_latency_s),
                    (orientation_ar1_index, orientation_ar1_coefficient),
                    (rate_ar1_index, rate_ar1_coefficient),
                    replicate_index,
                ) in condition_factors:
                    noise_seed = 20261030 + replicate_index
                    (
                        orientation_seed,
                        rate_seed,
                        jitter_seed,
                        dropout_seed,
                    ) = SeedSequence(noise_seed).spawn(4)
                    orientation_random = default_rng(orientation_seed)
                    rate_random = default_rng(rate_seed)
                    jitter_random = default_rng(jitter_seed)
                    timestamp_jitter = jitter_random.uniform(
                        -jitter_amplitude_s, jitter_amplitude_s, sample_count
                    ).astype(float64)
                    timestamp_jitter[0] = 0.0
                    observed_times = noise_times + timestamp_jitter
                    orientation_motion = analytic_motion(
                        observed_times - orientation_latency_s, motion_profile
                    )
                    rate_motion = analytic_motion(observed_times - rate_latency_s, motion_profile)
                    noisy_quaternions = _noisy_quaternions(
                        orientation_motion.quaternion_xyzw,
                        sigma_rad=profile.orientation_sigma_rad,
                        random=orientation_random,
                        ar1_coefficient=orientation_ar1_coefficient,
                    )
                    noisy_rates = rate_motion.body_rates_rad_s + (
                        correlated_gaussian_noise(
                            rate_random,
                            sigma=profile.rate_sigma_rad_s,
                            sample_count=sample_count,
                            dimension_count=3,
                            ar1_coefficient=rate_ar1_coefficient,
                        )
                    )
                    retained = _retained_sample_mask(
                        sample_count,
                        dropout_probability,
                        dropout_seed,
                        noise_sample_counts,
                    )
                    if int(retained.sum()) < 21:
                        raise ValueError(
                            "Dropout realization left fewer than 21 samples for "
                            f"{motion_profile.value}, {profile.name}, "
                            f"n={sample_count}, p={dropout_probability}, "
                            f"seed={noise_seed}."
                        )
                    ####
                    actual_dropped_count = int((~retained).sum())
                    jitter_token = f"j{jitter_index:02d}"
                    dropout_token = f"d{dropout_index:02d}"
                    orientation_latency_token = f"ol{orientation_latency_index:02d}"
                    rate_latency_token = f"rl{rate_latency_index:02d}"
                    orientation_ar1_token = f"oa{orientation_ar1_index:02d}"
                    rate_ar1_token = f"ra{rate_ar1_index:02d}"
                    condition_token = (
                        f"{jitter_token}-{dropout_token}-"
                        f"{orientation_latency_token}-{rate_latency_token}-"
                        f"{orientation_ar1_token}-{rate_ar1_token}"
                    )
                    case_id = (
                        f"analytic-noisy-{motion_profile.value}-{profile.name}-"
                        f"{condition_token}-n{sample_count}-"
                        f"{replicate_index + 1:03d}"
                    )
                    csv_name = (
                        f"quaternion_noisy_{motion_profile.value}_{profile.name}_"
                        f"{condition_token}_n{sample_count}_"
                        f"{replicate_index + 1:03d}.csv"
                    )
                    _write_csv(
                        outdir / csv_name,
                        observed_times[retained],
                        noisy_quaternions[retained],
                        noisy_rates[retained],
                    )
                    noisy_cases.append(
                        {
                            "case_id": case_id,
                            "csv": csv_name,
                            "label_source": (
                                "analytic ZYX generator with seeded acquisition perturbations: "
                                "scripts/generate_validation_corpus.py"
                            ),
                            "measurement_conditions": {
                                "motion_profile_id": motion_profile.value,
                                "timestamp_jitter_amplitude_s": jitter_amplitude_s,
                                "dropout_probability": dropout_probability,
                                "dropped_sample_count": actual_dropped_count,
                                "orientation_latency_s": orientation_latency_s,
                                "rate_latency_s": rate_latency_s,
                                "orientation_noise_sigma_rad": profile.orientation_sigma_rad,
                                "rate_noise_sigma_rad_s": profile.rate_sigma_rad_s,
                                "orientation_noise_ar1_coefficient": orientation_ar1_coefficient,
                                "rate_noise_ar1_coefficient": rate_ar1_coefficient,
                                "noise_seed": noise_seed,
                                "sample_count": sample_count,
                                "observed_sample_count": int(retained.sum()),
                                "sample_period_s": NOISE_SAMPLE_PERIOD_S,
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
                        }
                    )
                ####
            ####
        ####
    ####

    manifest = {
        "schema_version": 1,
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
        choices=("low", "moderate", "high", "all"),
        default="low",
        help="measurement-noise profile to include",
    )
    parser.add_argument(
        "--noise-replicates",
        type=int,
        default=1,
        help="seeded realizations per noise profile",
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
        "--motion-profile",
        choices=(*tuple(profile.value for profile in MotionProfile), "all"),
        default=MotionProfile.RICH_THREE_AXIS.value,
        help="analytic motion family to include in noisy cases",
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
    profiles = (
        NOISE_PROFILES
        if args.noise_profile == "all"
        else tuple(profile for profile in NOISE_PROFILES if profile.name == args.noise_profile)
    )
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
            noise_sample_counts=tuple(args.noise_sample_counts),
            motion_profiles=motion_profiles,
            timestamp_jitter_amplitudes_s=tuple(args.timestamp_jitter_amplitudes_s),
            dropout_probabilities=tuple(args.dropout_probabilities),
            orientation_latencies_s=tuple(args.orientation_latencies_s),
            rate_latencies_s=tuple(args.rate_latencies_s),
            orientation_noise_ar1_coefficients=tuple(args.orientation_noise_ar1_coefficients),
            rate_noise_ar1_coefficients=tuple(args.rate_noise_ar1_coefficients),
        )
    )
####


if __name__ == "__main__":
    main()
####
