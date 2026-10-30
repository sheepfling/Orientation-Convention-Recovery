"""Generate analytically labeled synthetic logs without using the inference package."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from numpy import arange, array, cos, cross, cumsum, float64, pi, sin, sqrt, stack, zeros
from numpy.random import Generator, default_rng
from numpy.typing import NDArray

FloatArray = NDArray[float64]


@dataclass(frozen=True, slots=True)
class AnalyticMotion:
    time_s: FloatArray
    euler_xyz_rad: FloatArray
    quaternion_xyzw: FloatArray
    body_rates_rad_s: FloatArray
    world_rates_rad_s: FloatArray
####


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


def analytic_motion(time_s: FloatArray) -> AnalyticMotion:
    """Use Rz(psi) Ry(theta) Rx(phi) and its closed-form body angular velocity."""

    phi = 0.33 * sin(1.3 * time_s) + 0.09 * cos(2.1 * time_s)
    theta = 0.25 * sin(0.8 * time_s) + 0.06 * cos(1.7 * time_s)
    psi = 0.41 * sin(1.1 * time_s) + 0.07 * cos(0.6 * time_s)
    phi_dot = 0.429 * cos(1.3 * time_s) - 0.189 * sin(2.1 * time_s)
    theta_dot = 0.20 * cos(0.8 * time_s) - 0.102 * sin(1.7 * time_s)
    psi_dot = 0.451 * cos(1.1 * time_s) - 0.042 * sin(0.6 * time_s)
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


def generate_corpus(outdir: Path) -> Path:
    """Write independently calculated clean, jittered, noisy, and Euler cases."""

    outdir.mkdir(parents=True, exist_ok=False)
    regular = analytic_motion(arange(101, dtype=float64) * 0.025)
    jitter = 0.025 + 0.003 * sin(arange(120, dtype=float64) * 0.7)
    jittered_time = array([0.0, *cumsum(jitter)], dtype=float64)
    keep = arange(len(jittered_time)) % 13 != 7
    jittered = analytic_motion(jittered_time[keep])
    euler = analytic_motion(arange(91, dtype=float64) * 0.03)
    noisy = analytic_motion(arange(161, dtype=float64) * 0.025)
    noise_seed = 20261030
    attitude_noise_sigma_rad = 0.0015
    rate_noise_sigma_rad_s = 0.012
    noise_random = default_rng(noise_seed)
    noisy_quaternions = _noisy_quaternions(
        noisy.quaternion_xyzw,
        sigma_rad=attitude_noise_sigma_rad,
        random=noise_random,
    )
    noisy_rates = noisy.body_rates_rad_s + noise_random.normal(
        0.0, rate_noise_sigma_rad_s, noisy.body_rates_rad_s.shape
    )

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
    _write_csv(outdir / "quaternion_body_noisy.csv", noisy, noisy_quaternions, noisy_rates)

    manifest = {
        "schema_version": 1,
        "cases": [
            {
                "case_id": "analytic-quaternion-body",
                "csv": "quaternion_body.csv",
                "label_source": "analytic ZYX generator: scripts/generate_validation_corpus.py",
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
            {
                "case_id": "analytic-quaternion-body-noisy",
                "csv": "quaternion_body_noisy.csv",
                "label_source": (
                    "analytic ZYX generator with seeded measurement noise: "
                    "scripts/generate_validation_corpus.py"
                ),
                "measurement_conditions": {
                    "orientation_noise_sigma_rad": attitude_noise_sigma_rad,
                    "rate_noise_sigma_rad_s": rate_noise_sigma_rad_s,
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
            },
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
    args = parser.parse_args()
    print(generate_corpus(args.outdir))
####


if __name__ == "__main__":
    main()
####
