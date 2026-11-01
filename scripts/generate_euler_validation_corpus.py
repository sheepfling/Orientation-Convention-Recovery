"""Generate seeded, independently labeled Euler-convention validation cases."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
from math import exp, isfinite, pi
from pathlib import Path
from typing import Literal

from numpy import absolute, arange, array, cos, float64, matmul, sin, stack, swapaxes, zeros
from numpy.random import Generator, SeedSequence, default_rng
from numpy.typing import NDArray

if __package__:
    from scripts.validation_noise import (
        sample_time_domain_ou_noise,
        validate_ou_sample_period,
    )
else:
    from validation_noise import sample_time_domain_ou_noise, validate_ou_sample_period
####

__all__ = [
    "ALL_EULER_CONVENTIONS",
    "EulerAngleSamples",
    "EulerConvention",
    "EulerCorpusConfig",
    "EulerPhysicalKinematics",
    "EulerRotationKinematics",
    "encode_euler_angles",
    "euler_rotation_kinematics",
    "generate_euler_angle_samples",
    "generate_euler_corpus",
    "physical_euler_kinematics",
    "sample_time_domain_ou_noise",
]

FloatArray = NDArray[float64]
AttitudeDirectionName = Literal["body_to_world", "world_to_body"]
EulerModeName = Literal["intrinsic", "extrinsic"]
AngleUnitName = Literal["radians", "degrees", "turns", "grads"]

EULER_ORDERS: tuple[str, ...] = (
    "xyz",
    "xzy",
    "yxz",
    "yzx",
    "zxy",
    "zyx",
    "xyx",
    "xzx",
    "yxy",
    "yzy",
    "zxz",
    "zyz",
)
EULER_MODES: tuple[EulerModeName, ...] = ("intrinsic", "extrinsic")
ANGLE_UNITS: tuple[AngleUnitName, ...] = ("radians", "degrees", "turns", "grads")
ATTITUDE_DIRECTIONS: tuple[AttitudeDirectionName, ...] = (
    "body_to_world",
    "world_to_body",
)
ANGLE_UNIT_RADIANS: dict[AngleUnitName, float] = {
    "radians": 1.0,
    "degrees": pi / 180.0,
    "turns": 2.0 * pi,
    "grads": pi / 200.0,
}
DEFAULT_MOTION_SEEDS: tuple[int, ...] = (730301, 730302, 730303)
DEFAULT_NOISE_SEED = 20301031
SAMPLE_PERIOD_S = 0.025
ANGLE_SIGNAL_HARMONICS = 6


@dataclass(frozen=True, slots=True)
class EulerAngleSamples:
    """Seeded Euler coordinates and their analytic time derivatives."""

    angles_rad: FloatArray
    angle_rates_rad_s: FloatArray
####


@dataclass(frozen=True, slots=True)
class EulerRotationKinematics:
    """An Euler-composed rotation matrix and its time derivative."""

    rotation_matrices: NDArray[float64]
    rotation_matrix_derivatives_s: NDArray[float64]
####


@dataclass(frozen=True, slots=True)
class EulerPhysicalKinematics:
    """Body-to-world rotations and body rates for one labeled convention."""

    rotation_body_to_world_matrices: NDArray[float64]
    body_rates_rad_s: FloatArray
####


@dataclass(frozen=True, slots=True)
class EulerConvention:
    """One known Euler attitude encoding paired with fixed body-rate labels."""

    attitude_direction: AttitudeDirectionName
    euler_order: str
    euler_mode: EulerModeName
    angle_unit: AngleUnitName
    rate_frame: Literal["body"] = "body"
    rate_order: Literal["pqr"] = "pqr"
    rate_unit: Literal["radians_per_second"] = "radians_per_second"
    rate_sign: Literal[1] = 1
####


ALL_EULER_CONVENTIONS: tuple[EulerConvention, ...] = tuple(
    EulerConvention(
        attitude_direction=direction,
        euler_order=order,
        euler_mode=mode,
        angle_unit=unit,
    )
    for direction, order, mode, unit in product(
        ATTITUDE_DIRECTIONS,
        EULER_ORDERS,
        EULER_MODES,
        ANGLE_UNITS,
    )
)


@dataclass(frozen=True, slots=True)
class EulerCorpusConfig:
    """Named settings for one reproducible, paired Euler corpus."""

    motion_seeds: tuple[int, ...] = DEFAULT_MOTION_SEEDS
    noise_seed: int = DEFAULT_NOISE_SEED
    sample_count: int = 161
    sample_period_s: float = SAMPLE_PERIOD_S
    orientation_noise_sigma_rad: float = 0.0015
    rate_noise_sigma_rad_s: float = 0.012
    orientation_noise_correlation_time_s: float | None = None
    rate_noise_correlation_time_s: float | None = None
####


def generate_euler_angle_samples(time_s: FloatArray, seed: int) -> EulerAngleSamples:
    """Generate rich, bounded Fourier Euler paths without inference/SciPy helpers."""

    if time_s.ndim != 1 or time_s.size < 2:
        raise ValueError("time_s must be one-dimensional with at least two samples.")
    ####
    if not all(isfinite(float(timestamp)) for timestamp in time_s):
        raise ValueError("time_s must contain finite values.")
    ####
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    ####
    random = default_rng(seed)
    frequencies = random.uniform(0.35, 2.8, (3, ANGLE_SIGNAL_HARMONICS))
    amplitudes = random.normal(0.0, 1.0, (3, ANGLE_SIGNAL_HARMONICS))
    amplitude_budgets = array((0.9, 0.45, 0.9), dtype=float64)
    amplitude_totals = absolute(amplitudes).sum(axis=1)
    amplitudes = amplitudes / amplitude_totals[:, None] * amplitude_budgets[:, None]
    phases = random.uniform(-pi, pi, (3, ANGLE_SIGNAL_HARMONICS))
    offsets = array((0.22, 0.72, -0.31), dtype=float64) + random.normal(0.0, 0.015, 3)
    phase = time_s[:, None, None] * frequencies[None, :, :] + phases[None, :, :]
    angles_rad = offsets[None, :] + (sin(phase) * amplitudes[None, :, :]).sum(axis=2)
    angle_rates_rad_s = (cos(phase) * (frequencies * amplitudes)[None, :, :]).sum(axis=2)
    return EulerAngleSamples(
        angles_rad=array(angles_rad, dtype=float64),
        angle_rates_rad_s=array(angle_rates_rad_s, dtype=float64),
    )
####


def _axis_rotation_kinematics(
    axis: str,
    angle_rad: FloatArray,
    angle_rate_rad_s: FloatArray,
) -> EulerRotationKinematics:
    cosine = cos(angle_rad)
    sine = sin(angle_rad)
    cosine_rate = -sine * angle_rate_rad_s
    sine_rate = cosine * angle_rate_rad_s
    rotations = zeros((angle_rad.size, 3, 3), dtype=float64)
    derivatives = zeros((angle_rad.size, 3, 3), dtype=float64)

    if axis == "x":
        rotations[:, 0, 0] = 1.0
        rotations[:, 1, 1] = cosine
        rotations[:, 1, 2] = -sine
        rotations[:, 2, 1] = sine
        rotations[:, 2, 2] = cosine
        derivatives[:, 1, 1] = cosine_rate
        derivatives[:, 1, 2] = -sine_rate
        derivatives[:, 2, 1] = sine_rate
        derivatives[:, 2, 2] = cosine_rate
    elif axis == "y":
        rotations[:, 0, 0] = cosine
        rotations[:, 0, 2] = sine
        rotations[:, 1, 1] = 1.0
        rotations[:, 2, 0] = -sine
        rotations[:, 2, 2] = cosine
        derivatives[:, 0, 0] = cosine_rate
        derivatives[:, 0, 2] = sine_rate
        derivatives[:, 2, 0] = -sine_rate
        derivatives[:, 2, 2] = cosine_rate
    elif axis == "z":
        rotations[:, 0, 0] = cosine
        rotations[:, 0, 1] = -sine
        rotations[:, 1, 0] = sine
        rotations[:, 1, 1] = cosine
        rotations[:, 2, 2] = 1.0
        derivatives[:, 0, 0] = cosine_rate
        derivatives[:, 0, 1] = -sine_rate
        derivatives[:, 1, 0] = sine_rate
        derivatives[:, 1, 1] = cosine_rate
    else:
        raise ValueError(f"Unsupported Euler axis: {axis}")
    ####
    return EulerRotationKinematics(rotations, derivatives)
####


def euler_rotation_kinematics(
    angles_rad: FloatArray,
    angle_rates_rad_s: FloatArray,
    order: str,
    mode: EulerModeName,
) -> EulerRotationKinematics:
    """Compose Euler matrices and their analytic derivatives by axis products."""

    if order not in EULER_ORDERS:
        raise ValueError(f"Unsupported Euler order: {order}")
    ####
    if mode not in EULER_MODES:
        raise ValueError(f"Unsupported Euler mode: {mode}")
    ####
    if angles_rad.ndim != 2 or angles_rad.shape[1] != 3:
        raise ValueError("angles_rad must have shape (sample_count, 3).")
    ####
    if angle_rates_rad_s.shape != angles_rad.shape:
        raise ValueError("Angle values and rates must have identical shapes.")
    ####
    axis_kinematics = tuple(
        _axis_rotation_kinematics(order[index], angles_rad[:, index], angle_rates_rad_s[:, index])
        for index in range(3)
    )
    first, second, third = axis_kinematics
    first_rotation = first.rotation_matrices
    second_rotation = second.rotation_matrices
    third_rotation = third.rotation_matrices
    first_derivative = first.rotation_matrix_derivatives_s
    second_derivative = second.rotation_matrix_derivatives_s
    third_derivative = third.rotation_matrix_derivatives_s

    if mode == "extrinsic":
        rotation_matrices = matmul(matmul(third_rotation, second_rotation), first_rotation)
        derivatives = (
            matmul(matmul(third_derivative, second_rotation), first_rotation)
            + matmul(matmul(third_rotation, second_derivative), first_rotation)
            + matmul(matmul(third_rotation, second_rotation), first_derivative)
        )
    else:
        rotation_matrices = matmul(matmul(first_rotation, second_rotation), third_rotation)
        derivatives = (
            matmul(matmul(first_derivative, second_rotation), third_rotation)
            + matmul(matmul(first_rotation, second_derivative), third_rotation)
            + matmul(matmul(first_rotation, second_rotation), third_derivative)
        )
    ####
    return EulerRotationKinematics(rotation_matrices, derivatives)
####


def physical_euler_kinematics(
    angle_samples: EulerAngleSamples,
    convention: EulerConvention,
) -> EulerPhysicalKinematics:
    """Derive body-frame rates from independently composed SO(3) matrices."""

    raw_kinematics = euler_rotation_kinematics(
        angle_samples.angles_rad,
        angle_samples.angle_rates_rad_s,
        convention.euler_order,
        convention.euler_mode,
    )
    if convention.attitude_direction == "body_to_world":
        rotation_body_to_world = raw_kinematics.rotation_matrices
        derivative_body_to_world = raw_kinematics.rotation_matrix_derivatives_s
    else:
        rotation_body_to_world = swapaxes(raw_kinematics.rotation_matrices, 1, 2)
        derivative_body_to_world = swapaxes(raw_kinematics.rotation_matrix_derivatives_s, 1, 2)
    ####
    body_angular_velocity_matrix = matmul(
        swapaxes(rotation_body_to_world, 1, 2),
        derivative_body_to_world,
    )
    skew_body_angular_velocity = 0.5 * (
        body_angular_velocity_matrix - swapaxes(body_angular_velocity_matrix, 1, 2)
    )
    body_rates_rad_s = stack(
        (
            skew_body_angular_velocity[:, 2, 1],
            skew_body_angular_velocity[:, 0, 2],
            skew_body_angular_velocity[:, 1, 0],
        ),
        axis=1,
    )
    return EulerPhysicalKinematics(
        rotation_body_to_world_matrices=rotation_body_to_world,
        body_rates_rad_s=array(body_rates_rad_s, dtype=float64),
    )
####


def encode_euler_angles(angles_rad: FloatArray, convention: EulerConvention) -> FloatArray:
    """Store raw Euler angles in the convention's declared angle unit."""

    return array(angles_rad / ANGLE_UNIT_RADIANS[convention.angle_unit], dtype=float64)
####


def _validate_config(config: EulerCorpusConfig) -> None:
    if not config.motion_seeds:
        raise ValueError("At least one motion seed is required.")
    ####
    if any(seed < 0 for seed in config.motion_seeds):
        raise ValueError("Motion seeds must be non-negative.")
    ####
    if len(config.motion_seeds) != len(set(config.motion_seeds)):
        raise ValueError("Motion seeds must be unique.")
    ####
    if config.noise_seed < 0:
        raise ValueError("The noise seed must be non-negative.")
    ####
    if config.sample_count < 21:
        raise ValueError("At least 21 samples are required by the validation corpus.")
    ####
    if not isfinite(config.sample_period_s) or config.sample_period_s <= 0.0:
        raise ValueError("The sample period must be positive and finite.")
    ####
    if not isfinite(config.orientation_noise_sigma_rad) or config.orientation_noise_sigma_rad < 0.0:
        raise ValueError("Orientation noise must be finite and non-negative.")
    ####
    if not isfinite(config.rate_noise_sigma_rad_s) or config.rate_noise_sigma_rad_s < 0.0:
        raise ValueError("Rate noise must be finite and non-negative.")
    ####
    for name, time_constant_s in (
        ("orientation", config.orientation_noise_correlation_time_s),
        ("rate", config.rate_noise_correlation_time_s),
    ):
        if time_constant_s is not None and (
            not isfinite(time_constant_s) or time_constant_s <= 0.0
        ):
            raise ValueError(
                f"{name.capitalize()} noise correlation time must be positive and finite."
            )
        ####
    ####
    if (
        config.orientation_noise_correlation_time_s is not None
        or config.rate_noise_correlation_time_s is not None
    ):
        validate_ou_sample_period(config.sample_period_s)
    ####
####


def _sample_noise(
    random: Generator,
    sample_count: int,
    sample_period_s: float,
    sigma: float,
    correlation_time_s: float | None,
) -> FloatArray:
    if correlation_time_s is None:
        return array(random.normal(0.0, sigma, (sample_count, 3)), dtype=float64)
    ####
    return sample_time_domain_ou_noise(
        random,
        sample_count,
        sample_period_s,
        sigma,
        correlation_time_s,
    )
####


def _write_csv(path: Path, time_s: FloatArray, euler: FloatArray, rates: FloatArray) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("time_s", "a0", "a1", "a2", "r0", "r1", "r2"))
        for timestamp, angles, angular_rate in zip(time_s, euler, rates, strict=True):
            writer.writerow((timestamp, *angles, *angular_rate))
        ####
    ####
####


def _validation_config() -> dict[str, object]:
    return {
        "search_space": {
            "attitude_directions": list(ATTITUDE_DIRECTIONS),
            "euler_orders": list(EULER_ORDERS),
            "euler_modes": list(EULER_MODES),
            "angle_units": list(ANGLE_UNITS),
            "rate_frames": ["body"],
            "rate_orders": ["pqr"],
            "rate_units": ["radians_per_second"],
            "rate_signs": [1],
        }
    }
####


def _case_id(convention: EulerConvention, motion_seed: int, config: EulerCorpusConfig) -> str:
    base = (
        f"euler-{convention.euler_order}-{convention.euler_mode}-{convention.angle_unit}-"
        f"{convention.attitude_direction}-m{motion_seed}-n{config.sample_count}-"
        f"noise{config.noise_seed}"
    )
    if (
        config.orientation_noise_correlation_time_s is None
        and config.rate_noise_correlation_time_s is None
    ):
        return base
    ####
    orientation_time_constant = (
        "white"
        if config.orientation_noise_correlation_time_s is None
        else format(config.orientation_noise_correlation_time_s, "g")
    )
    rate_time_constant = (
        "white"
        if config.rate_noise_correlation_time_s is None
        else format(config.rate_noise_correlation_time_s, "g")
    )
    return f"{base}-ou{orientation_time_constant}-{rate_time_constant}"
####


def generate_euler_corpus(outdir: Path, config: EulerCorpusConfig) -> Path:
    """Write a full Euler convention matrix from a native random-Fourier signal."""

    _validate_config(config)
    outdir.mkdir(parents=True, exist_ok=False)

    time_s = arange(config.sample_count, dtype=float64) * config.sample_period_s
    if (
        config.orientation_noise_correlation_time_s is None
        and config.rate_noise_correlation_time_s is None
    ):
        noise = default_rng(config.noise_seed)
        orientation_noise_random = noise
        rate_noise_random = noise
    else:
        orientation_noise_seed, rate_noise_seed = SeedSequence(config.noise_seed).spawn(2)
        orientation_noise_random = default_rng(orientation_noise_seed)
        rate_noise_random = default_rng(rate_noise_seed)
    ####
    angle_noise_rad = _sample_noise(
        orientation_noise_random,
        config.sample_count,
        config.sample_period_s,
        config.orientation_noise_sigma_rad,
        config.orientation_noise_correlation_time_s,
    )
    rate_noise = _sample_noise(
        rate_noise_random,
        config.sample_count,
        config.sample_period_s,
        config.rate_noise_sigma_rad_s,
        config.rate_noise_correlation_time_s,
    )
    orientation_ar1_coefficient = (
        0.0
        if config.orientation_noise_correlation_time_s is None
        else exp(-config.sample_period_s / config.orientation_noise_correlation_time_s)
    )
    rate_ar1_coefficient = (
        0.0
        if config.rate_noise_correlation_time_s is None
        else exp(-config.sample_period_s / config.rate_noise_correlation_time_s)
    )
    orientation_noise_description = (
        "independent per-sample Gaussian noise"
        if config.orientation_noise_correlation_time_s is None
        else (
            "stationary OU noise with "
            f"{config.orientation_noise_correlation_time_s:g} s correlation time"
        )
    )
    rate_noise_description = (
        "independent per-sample Gaussian noise"
        if config.rate_noise_correlation_time_s is None
        else (
            f"stationary OU noise with {config.rate_noise_correlation_time_s:g} s correlation time"
        )
    )
    validation_config = _validation_config()
    cases: list[dict[str, object]] = []

    for motion_seed in config.motion_seeds:
        angle_samples = generate_euler_angle_samples(time_s, motion_seed)
        noisy_euler_angles_rad = angle_samples.angles_rad + angle_noise_rad

        for convention in ALL_EULER_CONVENTIONS:
            case_id = _case_id(convention, motion_seed, config)
            file_digest = sha256(case_id.encode("utf-8")).hexdigest()[:32]
            csv_name = f"euler_noisy_{file_digest}.csv"
            physical_kinematics = physical_euler_kinematics(angle_samples, convention)
            euler_measurements = encode_euler_angles(noisy_euler_angles_rad, convention)
            noisy_body_rates = physical_kinematics.body_rates_rad_s + rate_noise
            _write_csv(outdir / csv_name, time_s, euler_measurements, noisy_body_rates)
            cases.append(
                {
                    "case_id": case_id,
                    "csv": csv_name,
                    "label_source": (
                        "Seeded random-Fourier Euler angles with analytic rotation-matrix "
                        "derivatives and body-rate extraction in "
                        "scripts.generate_euler_validation_corpus.py; no inference or SciPy "
                        "rotation routines are used. Orientation noise: "
                        f"{orientation_noise_description}; rate noise: {rate_noise_description}."
                    ),
                    "measurement_conditions": {
                        "motion_profile_id": "euler-random-fourier-native-rich",
                        "motion_seed": motion_seed,
                        "orientation_noise_sigma_rad": config.orientation_noise_sigma_rad,
                        "rate_noise_sigma_rad_s": config.rate_noise_sigma_rad_s,
                        "orientation_noise_ar1_coefficient": orientation_ar1_coefficient,
                        "rate_noise_ar1_coefficient": rate_ar1_coefficient,
                        "orientation_noise_correlation_time_s": (
                            config.orientation_noise_correlation_time_s
                        ),
                        "rate_noise_correlation_time_s": config.rate_noise_correlation_time_s,
                        "noise_seed": config.noise_seed,
                        "sample_count": config.sample_count,
                        "observed_sample_count": config.sample_count,
                        "sample_period_s": config.sample_period_s,
                    },
                    "representation": "euler",
                    "orientation_columns": ["a0", "a1", "a2"],
                    "rate_columns": ["r0", "r1", "r2"],
                    "time_column": "time_s",
                    "truth": asdict(convention),
                    "config": validation_config,
                }
            )
        ####
    ####

    manifest = {"schema_version": 1, "cases": cases}
    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
####


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(".validation-corpus-euler"),
        help="new output directory for the Euler convention corpus",
    )
    parser.add_argument(
        "--motion-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_MOTION_SEEDS),
        help="independent rate-process seeds (default: three fixed seeds)",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=DEFAULT_NOISE_SEED,
        help="seed shared across motion processes for paired measurement noise",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=161,
        help="number of samples per trajectory",
    )
    parser.add_argument(
        "--sample-period-s",
        type=float,
        default=SAMPLE_PERIOD_S,
        help="uniform sample period in seconds",
    )
    parser.add_argument(
        "--orientation-noise-sigma-rad",
        type=float,
        default=0.0015,
        help="per-axis standard deviation of rotation-vector measurement noise",
    )
    parser.add_argument(
        "--rate-noise-sigma-rad-s",
        type=float,
        default=0.012,
        help="per-axis standard deviation of body-rate measurement noise",
    )
    parser.add_argument(
        "--orientation-noise-correlation-time-s",
        type=float,
        help="stationary OU time constant for orientation noise; omit for independent samples",
    )
    parser.add_argument(
        "--rate-noise-correlation-time-s",
        type=float,
        help="stationary OU time constant for rate noise; omit for independent samples",
    )
    args = parser.parse_args()
    config = EulerCorpusConfig(
        motion_seeds=tuple(args.motion_seeds),
        noise_seed=args.noise_seed,
        sample_count=args.sample_count,
        sample_period_s=args.sample_period_s,
        orientation_noise_sigma_rad=args.orientation_noise_sigma_rad,
        rate_noise_sigma_rad_s=args.rate_noise_sigma_rad_s,
        orientation_noise_correlation_time_s=args.orientation_noise_correlation_time_s,
        rate_noise_correlation_time_s=args.rate_noise_correlation_time_s,
    )
    print(generate_euler_corpus(args.outdir, config))
####


if __name__ == "__main__":
    main()
####
