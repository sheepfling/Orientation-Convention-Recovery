"""Generate analytically labeled synthetic logs without using the inference package."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from numpy import arange, array, cos, cross, cumsum, float64, pi, sin, sqrt, stack, zeros
from numpy.random import Generator, default_rng
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
) -> FloatArray:
    """Apply seeded, isotropic small-rotation measurement noise independently per sample."""

    rotation_vectors = random.normal(0.0, sigma_rad, (len(quaternions_xyzw), 3)).astype(float64)
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


def _write_csv(
    path: Path, motion: AnalyticMotion, orientation: FloatArray, rates: FloatArray
) -> None:
    orientation_names = tuple(f"a{index}" for index in range(orientation.shape[1]))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("time_s", *orientation_names, "r0", "r1", "r2"))
        for time_s, attitude, rate in zip(motion.time_s, orientation, rates, strict=True):
            writer.writerow((time_s, *attitude, *rate))
        ####
    ####
####


def generate_corpus(
    outdir: Path,
    *,
    noise_profiles: tuple[NoiseProfile, ...] = (NOISE_PROFILES[0],),
    noise_replicates: int = 1,
    motion_profiles: tuple[MotionProfile, ...] = (MotionProfile.RICH_THREE_AXIS,),
) -> Path:
    """Write analytic convention cases across optional noise and motion profiles."""

    if not noise_profiles:
        raise ValueError("At least one noise profile is required.")
    ####
    if not motion_profiles:
        raise ValueError("At least one motion profile is required.")
    ####
    if noise_replicates < 1:
        raise ValueError("noise_replicates must be positive.")
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
        regular,
        regular.quaternion_xyzw,
        regular.body_rates_rad_s,
    )
    inverted = jittered.quaternion_xyzw.copy()
    inverted[:, :3] *= -1.0
    inverted[::11] *= -1.0
    world_rqp = jittered.world_rates_rad_s[:, (2, 1, 0)]
    _write_csv(
        outdir / "quaternion_world_jittered.csv",
        jittered,
        inverted[:, (3, 0, 1, 2)],
        world_rqp,
    )
    _write_csv(
        outdir / "euler_degrees.csv",
        euler,
        euler.euler_xyz_rad * (180.0 / pi),
        euler.body_rates_rad_s * (180.0 / pi),
    )

    noisy_cases: list[dict[str, object]] = []
    noise_times = arange(161, dtype=float64) * 0.025
    for motion_profile in motion_profiles:
        for profile in noise_profiles:
            for replicate_index in range(noise_replicates):
                noise_seed = 20261030 + replicate_index
                noise_random = default_rng(noise_seed)
                noisy = analytic_motion(noise_times, motion_profile)
                noisy_quaternions = _noisy_quaternions(
                    noisy.quaternion_xyzw,
                    sigma_rad=profile.orientation_sigma_rad,
                    random=noise_random,
                )
                noisy_rates = noisy.body_rates_rad_s + noise_random.normal(
                    0.0, profile.rate_sigma_rad_s, noisy.body_rates_rad_s.shape
                )
                case_id = (
                    f"analytic-noisy-{motion_profile.value}-{profile.name}-"
                    f"{replicate_index + 1:03d}"
                )
                csv_name = (
                    f"quaternion_noisy_{motion_profile.value}_{profile.name}_"
                    f"{replicate_index + 1:03d}.csv"
                )
                _write_csv(outdir / csv_name, noisy, noisy_quaternions, noisy_rates)
                noisy_cases.append(
                    {
                        "case_id": case_id,
                        "csv": csv_name,
                        "label_source": (
                            "analytic ZYX generator with seeded measurement noise: "
                            "scripts/generate_validation_corpus.py"
                        ),
                        "measurement_conditions": {
                            "motion_profile_id": motion_profile.value,
                            "orientation_noise_sigma_rad": profile.orientation_sigma_rad,
                            "rate_noise_sigma_rad_s": profile.rate_sigma_rad_s,
                            "noise_seed": noise_seed,
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
        "--motion-profile",
        choices=(*tuple(profile.value for profile in MotionProfile), "all"),
        default=MotionProfile.RICH_THREE_AXIS.value,
        help="analytic motion family to include in noisy cases",
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
            motion_profiles=motion_profiles,
        )
    )
####


if __name__ == "__main__":
    main()
####
