from __future__ import annotations

import json
import sys
from collections.abc import Callable
from math import exp
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pytest
from numpy.random import default_rng
from numpy.typing import NDArray
from scipy.integrate import (
    solve_ivp as scipy_solve_ivp,  # pyright: ignore[reportUnknownVariableType]
)
from scipy.spatial.transform import Rotation as SciPyRotation

from attitude_convention_inference import InferenceDecisionStatus
from attitude_convention_inference.statistics import estimate_noise_process
from scripts.generate_validation_corpus import (
    DEFAULT_MOTION_SEED,
    DEFAULT_NOISE_SEED_START,
    MANEUVER_RATE_PERIOD_S,
    NO_NOISE_PROFILE,
    NOISE_PROFILES,
    RATE_SENSOR_CALIBRATION_PROFILES,
    SAMPLED_RATE_INTEGRATION_STEP_S,
    SAMPLED_RATE_PROFILES,
    MotionProfile,
    NoiseProfile,
    RateNoiseFrame,
    RateSensorCalibration,
    SyntheticMotion,
    all_quaternion_truth_conventions,
    analytic_motion,
    correlated_gaussian_noise,
    generate_corpus,
    maneuver_burst_rate_motion,
    maneuver_burst_rate_signal,
    piecewise_linear_rate_motion,
    piecewise_linear_rate_signal,
    sampled_rate_motion,
    sampled_rate_signal,
)
from scripts.generate_validation_corpus import main as generate_corpus_main
from scripts.validate_corpus import ValidationManifest, validate_corpus

FloatArray = NDArray[np.float64]


class _OdeSolution(Protocol):
    success: bool
    y: FloatArray
####


class _Dop853Solver(Protocol):
    def __call__(
        self,
        function: Callable[[float, FloatArray], FloatArray],
        time_span_s: tuple[float, float],
        initial_state: FloatArray,
        *,
        method: str,
        t_eval: FloatArray,
        rtol: float,
        atol: float,
    ) -> _OdeSolution: ...
####


_dop853_solver = cast(_Dop853Solver, scipy_solve_ivp)


class _RateSignalForTest(Protocol):
    def evaluate(self, time_s: FloatArray) -> FloatArray: ...
####


def _rate_signal_for_profile(profile: MotionProfile, seed: int) -> _RateSignalForTest:
    if profile is MotionProfile.SAMPLED_RATE_NONANALYTIC:
        return sampled_rate_signal(seed)
    ####
    if profile is MotionProfile.SAMPLED_RATE_PIECEWISE_LINEAR:
        return piecewise_linear_rate_signal(seed)
    ####
    if profile is MotionProfile.SAMPLED_RATE_MANEUVER_BURSTS:
        return maneuver_burst_rate_signal(seed)
    ####
    raise ValueError(f"{profile.value} is not a sampled-rate motion profile.")
####


def _rate_motion_for_profile(
    profile: MotionProfile,
    time_s: FloatArray,
    *,
    seed: int,
    reference_time_s: float | None = None,
    integration_step_s: float | None = None,
) -> SyntheticMotion:
    if profile is MotionProfile.SAMPLED_RATE_NONANALYTIC:
        return sampled_rate_motion(
            time_s,
            seed=seed,
            reference_time_s=reference_time_s,
            integration_step_s=integration_step_s,
        )
    ####
    if profile is MotionProfile.SAMPLED_RATE_PIECEWISE_LINEAR:
        return piecewise_linear_rate_motion(
            time_s,
            seed=seed,
            reference_time_s=reference_time_s,
            integration_step_s=integration_step_s,
        )
    ####
    if profile is MotionProfile.SAMPLED_RATE_MANEUVER_BURSTS:
        return maneuver_burst_rate_motion(
            time_s,
            seed=seed,
            reference_time_s=reference_time_s,
            integration_step_s=integration_step_s,
        )
    ####
    raise ValueError(f"{profile.value} is not a sampled-rate motion profile.")
####


@pytest.mark.parametrize(
    "profile",
    tuple(profile for profile in MotionProfile if profile not in SAMPLED_RATE_PROFILES),
)
def test_analytic_rates_match_finite_difference_rotations(profile: MotionProfile) -> None:
    time_s = np.asarray([0.1, 0.7, 1.3, 2.1], dtype=np.float64)
    epsilon = 1.0e-6
    scipy_rotation = cast(Any, SciPyRotation)
    earlier = scipy_rotation.from_quat(analytic_motion(time_s - epsilon, profile).quaternion_xyzw)
    later = scipy_rotation.from_quat(analytic_motion(time_s + epsilon, profile).quaternion_xyzw)
    center = scipy_rotation.from_quat(analytic_motion(time_s, profile).quaternion_xyzw)
    body_rates = np.asarray((earlier.inv() * later).as_rotvec(), dtype=np.float64) / (2.0 * epsilon)
    world_rates = np.asarray(center.apply(body_rates), dtype=np.float64)
    analytic = analytic_motion(time_s, profile)

    np.testing.assert_allclose(analytic.body_rates_rad_s, body_rates, atol=1.0e-9)
    np.testing.assert_allclose(analytic.world_rates_rad_s, world_rates, atol=1.0e-9)
####


def test_sampled_rate_nonanalytic_corpus_recovers_its_seeded_truth(tmp_path: Path) -> None:
    time_s = np.arange(101, dtype=np.float64) * 0.025
    motion = sampled_rate_motion(time_s, seed=1947)
    repeated_motion = sampled_rate_motion(time_s, seed=1947)
    different_motion = sampled_rate_motion(time_s, seed=1948)
    scipy_rotation = cast(Any, SciPyRotation)
    rotations = scipy_rotation.from_quat(motion.quaternion_xyzw)

    np.testing.assert_array_equal(motion.body_rates_rad_s, repeated_motion.body_rates_rad_s)
    np.testing.assert_array_equal(motion.quaternion_xyzw, repeated_motion.quaternion_xyzw)
    assert not np.array_equal(motion.body_rates_rad_s, different_motion.body_rates_rad_s)
    np.testing.assert_allclose(
        motion.euler_xyz_rad,
        np.asarray(rotations.as_euler("xyz", degrees=False), dtype=np.float64),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(rotations.apply(motion.body_rates_rad_s), dtype=np.float64),
        motion.world_rates_rad_s,
    )

    manifest = generate_corpus(
        tmp_path / "sampled-rate-nonanalytic",
        noise_profiles=(NoiseProfile("low", 0.0, 0.0),),
        noise_sample_counts=(101,),
        motion_profiles=(MotionProfile.SAMPLED_RATE_NONANALYTIC,),
        motion_seed=1947,
    )
    assessment = validate_corpus(manifest)
    sampled_case = next(
        case
        for case in assessment.cases
        if case.measurement_conditions.motion_profile_id
        == MotionProfile.SAMPLED_RATE_NONANALYTIC.value
    )

    assert "motion seed 1947" in sampled_case.label_source
    assert "independent fixed-grid trapezoidal quaternion integration" in sampled_case.label_source
    assert sampled_case.outcome != "evidence_miss"
    assert sampled_case.operational_correct
    assert sampled_case.truth_class_id in sampled_case.evidence_contender_class_ids
####


def test_piecewise_linear_rate_signal_is_seeded_and_rejects_extrapolation() -> None:
    first = piecewise_linear_rate_signal(2901)
    repeated = piecewise_linear_rate_signal(2901)
    different = piecewise_linear_rate_signal(2902)
    query_times = np.asarray((-0.25, 0.0, 0.2, 0.5, 0.75), dtype=np.float64)

    np.testing.assert_array_equal(first.evaluate(query_times), repeated.evaluate(query_times))
    assert not np.array_equal(first.evaluate(query_times), different.evaluate(query_times))
    knot_indices = np.searchsorted(first.knot_times_s, np.asarray((0.0, 0.5)))
    np.testing.assert_array_equal(
        first.evaluate(np.asarray((0.0, 0.5), dtype=np.float64)),
        first.knot_rates_rad_s[knot_indices],
    )
    with pytest.raises(ValueError, match="read-only"):
        first.knot_rates_rad_s[0, 0] = 0.0
    ####
    with pytest.raises(ValueError, match="supported range"):
        first.evaluate(np.asarray((3600.5,), dtype=np.float64))
    ####
####


def test_maneuver_burst_signal_is_seeded_periodic_and_has_rest_intervals() -> None:
    signal = maneuver_burst_rate_signal(3901)
    repeated = maneuver_burst_rate_signal(3901)
    different = maneuver_burst_rate_signal(3902)
    query_times = np.asarray((0.0, 0.1, 0.8, 2.7, 7.9, 8.0), dtype=np.float64)

    np.testing.assert_array_equal(signal.evaluate(query_times), repeated.evaluate(query_times))
    assert not np.array_equal(signal.evaluate(query_times), different.evaluate(query_times))
    np.testing.assert_allclose(
        signal.evaluate(query_times + MANEUVER_RATE_PERIOD_S),
        signal.evaluate(query_times),
        atol=1.0e-14,
    )
    np.testing.assert_array_equal(signal.evaluate(np.asarray((0.0, 8.0))), np.zeros((2, 3)))
    for axis in range(3):
        rest_time_s = float(
            (
                signal.start_times_s[0, axis]
                + signal.durations_s[0, axis]
                + signal.start_times_s[1, axis]
            )
            / 2.0
        )
        assert signal.evaluate(np.asarray((rest_time_s,)))[0, axis] == 0.0
    ####
    staggered_time_s = float((signal.start_times_s[0, 0] + signal.start_times_s[0, 1]) / 2.0)
    staggered_rate = signal.evaluate(np.asarray((staggered_time_s,)))[0]
    assert staggered_rate[0] != 0.0
    np.testing.assert_array_equal(staggered_rate[1:], np.zeros(2))
    with pytest.raises(ValueError, match="read-only"):
        signal.amplitudes_rad_s[0, 0] = 0.0
    ####
####


def test_sampled_rate_latency_mismatch_never_becomes_a_false_recommendation(
    tmp_path: Path,
) -> None:
    manifest = generate_corpus(
        tmp_path / "sampled-rate-latency-mismatch",
        noise_profiles=(NoiseProfile("low", 0.0, 0.0),),
        noise_sample_counts=(161,),
        motion_profiles=(MotionProfile.SAMPLED_RATE_NONANALYTIC,),
        motion_seed=2948,
        orientation_latencies_s=(0.0, 0.1),
        rate_latencies_s=(0.0, 0.1),
    )
    assessment = validate_corpus(manifest)
    sampled_cases = tuple(
        case
        for case in assessment.cases
        if case.measurement_conditions.motion_profile_id
        == MotionProfile.SAMPLED_RATE_NONANALYTIC.value
    )

    assert len(sampled_cases) == 4
    for case in sampled_cases:
        assert case.data_winner_correct
        assert case.operational_correct
        assert case.truth_class_id in case.evidence_contender_class_ids
        assert case.recommendation_correct in (None, True)
    ####
    matched_delay_cases = tuple(
        case
        for case in sampled_cases
        if case.measurement_conditions.orientation_latency_s
        == case.measurement_conditions.rate_latency_s
    )
    assert len(matched_delay_cases) == 2
    assert all(case.recommendation_correct is True for case in matched_delay_cases)
####


@pytest.mark.parametrize("motion_profile", SAMPLED_RATE_PROFILES)
def test_sampled_rate_motion_seeds_are_paired_and_summarized_separately(
    tmp_path: Path,
    motion_profile: MotionProfile,
) -> None:
    motion_seeds = (1947, 1948)
    manifest = generate_corpus(
        tmp_path / f"paired-motion-seed-corpus-{motion_profile.value}",
        noise_profiles=(NOISE_PROFILES[0],),
        noise_replicates=2,
        noise_seed_start=3100,
        noise_sample_counts=(41,),
        motion_profiles=(motion_profile,),
        motion_seeds=motion_seeds,
    )
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    noisy_cases = [
        case
        for case in manifest_payload["cases"]
        if case["measurement_conditions"].get("motion_seed") is not None
    ]

    assert len(noisy_cases) == len(motion_seeds) * 2
    assert len({case["case_id"] for case in noisy_cases}) == len(noisy_cases)
    assert len({case["csv"] for case in noisy_cases}) == len(noisy_cases)
    assert {case["measurement_conditions"]["motion_seed"] for case in noisy_cases} == set(
        motion_seeds
    )
    assert {case["measurement_conditions"]["noise_seed"] for case in noisy_cases} == {3100, 3101}
    assert all(len(Path(case["csv"]).name) <= 64 for case in noisy_cases)
    csv_by_seed_pair = {
        (
            case["measurement_conditions"]["motion_seed"],
            case["measurement_conditions"]["noise_seed"],
        ): (manifest.parent / case["csv"]).read_bytes()
        for case in noisy_cases
    }
    for noise_seed in (3100, 3101):
        assert csv_by_seed_pair[(1947, noise_seed)] != csv_by_seed_pair[(1948, noise_seed)]
    ####
    for motion_seed in motion_seeds:
        assert csv_by_seed_pair[(motion_seed, 3100)] != csv_by_seed_pair[(motion_seed, 3101)]
    ####

    assessment = validate_corpus(manifest)
    motion_summaries = tuple(
        summary
        for summary in assessment.condition_summaries
        if summary.measurement_conditions.motion_profile_id == motion_profile.value
    )

    assert len(motion_summaries) == len(motion_seeds)
    assert {summary.measurement_conditions.motion_seed for summary in motion_summaries} == set(
        motion_seeds
    )
    assert all(summary.case_count == 2 for summary in motion_summaries)
    assert all(summary.measurement_conditions.noise_seed is None for summary in motion_summaries)
####


def test_generated_csv_names_fit_long_output_directories(tmp_path: Path) -> None:
    output_directory = tmp_path / f"synthetic-validation-corpus-{'x' * 80}"
    manifest = generate_corpus(
        output_directory,
        noise_profiles=(NoiseProfile("low", 0.0, 0.0),),
        noise_sample_counts=(21,),
        motion_profiles=(MotionProfile.SAMPLED_RATE_NONANALYTIC,),
        motion_seed=2949,
    )
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    noisy_cases = tuple(
        case
        for case in manifest_payload["cases"]
        if case["measurement_conditions"].get("motion_seed") is not None
    )

    assert len(noisy_cases) == 1
    csv_name = noisy_cases[0]["csv"]
    assert len(Path(csv_name).name) <= 64
    assert (output_directory / csv_name).is_file()
####


@pytest.mark.parametrize("seed", (1947, 1948, 730201))
@pytest.mark.parametrize("motion_profile", SAMPLED_RATE_PROFILES)
def test_rate_motion_attitudes_converge_to_independent_dop853_solution(
    seed: int,
    motion_profile: MotionProfile,
) -> None:
    rate_signal = _rate_signal_for_profile(motion_profile, seed)
    step_sizes_s = (0.05, 0.025, 0.0125)
    maximum_errors_rad: list[float] = []
    scipy_rotation = cast(Any, SciPyRotation)

    for step_size_s in step_sizes_s:
        time_s = np.arange(0.0, 4.0 + step_size_s / 2.0, step_size_s, dtype=np.float64)

        def body_to_world_derivative(
            time: float,
            flattened_attitude: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            angular_rate = rate_signal.evaluate(np.asarray([time], dtype=np.float64))[0]
            x_rate, y_rate, z_rate = angular_rate
            skew_rate = np.asarray(
                (
                    (0.0, -z_rate, y_rate),
                    (z_rate, 0.0, -x_rate),
                    (-y_rate, x_rate, 0.0),
                ),
                dtype=np.float64,
            )
            attitude = flattened_attitude.reshape((3, 3))
            return (attitude @ skew_rate).reshape(-1)
        ####

        reference = _dop853_solver(
            body_to_world_derivative,
            (float(time_s[0]), float(time_s[-1])),
            np.eye(3, dtype=np.float64).reshape(-1),
            method="DOP853",
            t_eval=time_s,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        assert reference.success
        reference_matrices = reference.y.T.reshape((-1, 3, 3))
        sampled_motion = _rate_motion_for_profile(motion_profile, time_s, seed=seed)
        sampled_matrices = scipy_rotation.from_quat(sampled_motion.quaternion_xyzw).as_matrix()
        relative_rotations = np.swapaxes(sampled_matrices, 1, 2) @ reference_matrices
        errors_rad = np.asarray(
            scipy_rotation.from_matrix(relative_rotations).magnitude(),
            dtype=np.float64,
        )
        maximum_errors_rad.append(float(np.max(errors_rad)))
    ####

    assert maximum_errors_rad[0] > maximum_errors_rad[1] > maximum_errors_rad[2]
    assert maximum_errors_rad[0] / maximum_errors_rad[1] > 2.5
    assert maximum_errors_rad[1] / maximum_errors_rad[2] > 2.5
    assert maximum_errors_rad[1] < 2.0e-4
####


@pytest.mark.parametrize("motion_profile", SAMPLED_RATE_PROFILES)
def test_fixed_grid_rate_trajectory_is_query_independent(
    motion_profile: MotionProfile,
) -> None:
    sparse_times = np.asarray((-0.077, 0.0, 0.101, 0.254), dtype=np.float64)
    dense_times = np.asarray(
        (-0.15, -0.1, -0.077, -0.05, 0.0, 0.025, 0.101, 0.15, 0.254, 0.3),
        dtype=np.float64,
    )
    sparse = _rate_motion_for_profile(
        motion_profile,
        sparse_times,
        seed=1947,
        reference_time_s=0.0,
        integration_step_s=SAMPLED_RATE_INTEGRATION_STEP_S,
    )
    dense = _rate_motion_for_profile(
        motion_profile,
        dense_times,
        seed=1947,
        reference_time_s=0.0,
        integration_step_s=SAMPLED_RATE_INTEGRATION_STEP_S,
    )
    dense_indices = np.searchsorted(dense_times, sparse_times)

    np.testing.assert_array_equal(sparse.quaternion_xyzw, dense.quaternion_xyzw[dense_indices])
    np.testing.assert_array_equal(sparse.body_rates_rad_s, dense.body_rates_rad_s[dense_indices])
    np.testing.assert_array_equal(sparse.world_rates_rad_s, dense.world_rates_rad_s[dense_indices])
    np.testing.assert_array_equal(sparse.quaternion_xyzw[1], np.asarray((0.0, 0.0, 0.0, 1.0)))
####


@pytest.mark.parametrize("seed", (1947, 1948, 730201))
@pytest.mark.parametrize("motion_profile", SAMPLED_RATE_PROFILES)
def test_fixed_grid_rate_trajectory_matches_dop853_before_and_after_reference(
    seed: int,
    motion_profile: MotionProfile,
) -> None:
    rate_signal = _rate_signal_for_profile(motion_profile, seed)
    base_times = np.arange(161, dtype=np.float64) * 0.025
    time_s = np.unique(
        np.concatenate((base_times, base_times - 0.013, base_times - 0.027, base_times + 0.0021))
    )
    scipy_rotation = cast(Any, SciPyRotation)

    def body_to_world_derivative(
        time: float,
        flattened_attitude: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        angular_rate = rate_signal.evaluate(np.asarray([time], dtype=np.float64))[0]
        x_rate, y_rate, z_rate = angular_rate
        skew_rate = np.asarray(
            (
                (0.0, -z_rate, y_rate),
                (z_rate, 0.0, -x_rate),
                (-y_rate, x_rate, 0.0),
            ),
            dtype=np.float64,
        )
        attitude = flattened_attitude.reshape((3, 3))
        return (attitude @ skew_rate).reshape(-1)
    ####

    identity = np.eye(3, dtype=np.float64).reshape(-1)
    negative_times = time_s[time_s < 0.0][::-1]
    negative_reference = _dop853_solver(
        body_to_world_derivative,
        (0.0, float(negative_times[-1])),
        identity,
        method="DOP853",
        t_eval=negative_times,
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    positive_times = time_s[time_s >= 0.0]
    positive_reference = _dop853_solver(
        body_to_world_derivative,
        (0.0, float(positive_times[-1])),
        identity,
        method="DOP853",
        t_eval=positive_times,
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    assert negative_reference.success
    assert positive_reference.success
    reference_matrices = np.concatenate(
        (
            negative_reference.y.T.reshape((-1, 3, 3))[::-1],
            positive_reference.y.T.reshape((-1, 3, 3)),
        )
    )

    fixed_grid_motion = _rate_motion_for_profile(
        motion_profile,
        time_s,
        seed=seed,
        reference_time_s=0.0,
        integration_step_s=SAMPLED_RATE_INTEGRATION_STEP_S,
    )
    sampled_matrices = scipy_rotation.from_quat(fixed_grid_motion.quaternion_xyzw).as_matrix()
    relative_rotations = np.swapaxes(sampled_matrices, 1, 2) @ reference_matrices
    errors_rad = np.asarray(scipy_rotation.from_matrix(relative_rotations).magnitude())

    assert float(np.max(errors_rad)) < 1.0e-5
####


def test_ar1_noise_zero_correlation_preserves_white_noise_sequence() -> None:
    seed = 723
    actual = correlated_gaussian_noise(
        default_rng(seed),
        sigma=0.7,
        sample_count=41,
        dimension_count=3,
        ar1_coefficient=0.0,
    )
    expected = default_rng(seed).normal(0.0, 0.7, (41, 3)).astype(np.float64)

    np.testing.assert_array_equal(actual, expected)
####


def test_ar1_noise_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="sigma must be finite and non-negative"):
        correlated_gaussian_noise(
            default_rng(1),
            sigma=-0.1,
            sample_count=3,
            dimension_count=3,
            ar1_coefficient=0.0,
        )
    ####
    with pytest.raises(ValueError, match="sample_count and dimension_count must be positive"):
        correlated_gaussian_noise(
            default_rng(1),
            sigma=0.1,
            sample_count=0,
            dimension_count=3,
            ar1_coefficient=0.0,
        )
    ####
    with pytest.raises(ValueError, match="ar1_coefficient must be strictly between -1 and 1"):
        correlated_gaussian_noise(
            default_rng(1),
            sigma=0.1,
            sample_count=3,
            dimension_count=3,
            ar1_coefficient=1.0,
        )
    ####
####


@pytest.mark.parametrize("coefficient", (0.75, -0.65))
def test_ar1_noise_has_configured_lag_one_correlation(coefficient: float) -> None:
    samples = correlated_gaussian_noise(
        default_rng(81),
        sigma=0.7,
        sample_count=30_000,
        dimension_count=3,
        ar1_coefficient=coefficient,
    )

    assert float(np.std(samples[:, 0])) == pytest.approx(0.7, abs=0.02)
    assert float(np.corrcoef(samples[:-1, 0], samples[1:, 0])[0, 1]) == pytest.approx(
        coefficient, abs=0.02
    )
####


def test_sampled_rate_ou_noise_is_paired_at_common_times_across_cadences(
    tmp_path: Path,
) -> None:
    sampled_rows: list[NDArray[np.float64]] = []
    cadences = (
        ("10hz", 0.1, 41),
        ("40hz", 0.025, 161),
        ("80hz", 0.0125, 321),
        ("100hz", 0.01, 401),
    )
    for cadence_name, sample_period_s, sample_count in cadences:
        manifest_path = generate_corpus(
            tmp_path / f"sampled-rate-ou-{cadence_name}",
            noise_seed_start=90173,
            motion_seeds=(1947,),
            noise_sample_counts=(sample_count,),
            sample_period_s=sample_period_s,
            motion_profiles=(MotionProfile.SAMPLED_RATE_NONANALYTIC,),
            orientation_noise_correlation_time_s=0.05,
            rate_noise_correlation_time_s=0.1,
        )
        manifest = ValidationManifest.model_validate(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        noisy_case = next(
            case for case in manifest.cases if case.measurement_conditions.noise_seed == 90173
        )
        conditions = noisy_case.measurement_conditions
        assert conditions.sample_period_s == sample_period_s
        assert conditions.sample_count == sample_count
        assert conditions.orientation_noise_correlation_time_s == 0.05
        assert conditions.rate_noise_correlation_time_s == 0.1
        assert conditions.rate_noise_frame == "truth-rate-frame"
        assert conditions.orientation_noise_ar1_coefficient == pytest.approx(
            exp(-sample_period_s / 0.05)
        )
        assert conditions.rate_noise_ar1_coefficient == pytest.approx(exp(-sample_period_s / 0.1))
        sampled_rows.append(
            np.loadtxt(
                manifest_path.parent / noisy_case.csv,
                delimiter=",",
                skiprows=1,
                dtype=np.float64,
            )
        )
    ####

    np.testing.assert_allclose(sampled_rows[0], sampled_rows[1][::4], rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(sampled_rows[0], sampled_rows[2][::8], rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(sampled_rows[0], sampled_rows[3][::10], rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(sampled_rows[1], sampled_rows[2][::2], rtol=0.0, atol=2.0e-12)
####


def test_corpus_ou_noise_requires_a_shared_grid_and_exclusive_correlation_input(
    tmp_path: Path,
) -> None:
    invalid_period_output = tmp_path / "invalid-ou-period"
    with pytest.raises(ValueError, match=r"multiple of 0\.5 ms"):
        generate_corpus(
            invalid_period_output,
            noise_sample_counts=(41,),
            sample_period_s=0.00325,
            orientation_noise_correlation_time_s=0.05,
        )
    ####
    assert not invalid_period_output.exists()

    conflicting_model_output = tmp_path / "conflicting-ou-model"
    with pytest.raises(ValueError, match="Specify orientation OU correlation time"):
        generate_corpus(
            conflicting_model_output,
            noise_sample_counts=(41,),
            orientation_noise_correlation_time_s=0.05,
            orientation_noise_ar1_coefficients=(0.6,),
        )
    ####
    assert not conflicting_model_output.exists()
####


def test_interval_difference_autocorrelation_is_not_measurement_ar1_coefficient() -> None:
    measurement_noise = correlated_gaussian_noise(
        default_rng(902),
        sigma=0.7,
        sample_count=20_000,
        dimension_count=3,
        ar1_coefficient=0.8,
    )

    interval_errors = np.diff(measurement_noise, axis=0)
    estimate = estimate_noise_process(interval_errors, trim_fraction=0.02)

    assert estimate.lag1_autocorrelation == pytest.approx(-0.1, abs=0.02)
    assert estimate.lag1_autocorrelation != pytest.approx(0.8, abs=0.1)
####


def test_generated_analytic_corpus_reports_noisy_ambiguity_reproducibly(tmp_path: Path) -> None:
    manifest = generate_corpus(tmp_path / "analytic-corpus")
    repeated_manifest = generate_corpus(tmp_path / "repeated-corpus")

    for first_path in sorted(manifest.parent.iterdir()):
        filename = first_path.name
        assert (manifest.parent / filename).read_bytes() == (
            repeated_manifest.parent / filename
        ).read_bytes()
    ####

    assessment = validate_corpus(manifest)

    assert assessment.summary.case_count == 4
    assert assessment.summary.resolved_correct == 3
    assert assessment.summary.ambiguous_contains_truth == 1
    assert assessment.summary.evidence_miss == 0
    assert assessment.environment.python
    assert assessment.environment.numpy
    assert assessment.environment.scipy
    assert assessment.environment.package
    dropped_sample_count = assessment.cases[1].measurement_conditions.dropped_sample_count
    assert dropped_sample_count is not None and dropped_sample_count > 0
    noisy_case = assessment.cases[3]
    assert noisy_case.measurement_conditions.noise_seed == 20261030
    assert noisy_case.measurement_conditions.sample_count == 161
    assert noisy_case.measurement_conditions.sample_period_s == 0.025
    assert noisy_case.measurement_conditions.duration_s == pytest.approx(4.0)
    assert noisy_case.measurement_conditions.orientation_noise_sigma_rad == 0.0015
    assert noisy_case.measurement_conditions.rate_noise_sigma_rad_s == 0.012
    assert noisy_case.outcome == "ambiguous_contains_truth"
    assert noisy_case.truth_class_id in noisy_case.evidence_contender_class_ids
    assert noisy_case.operational_correct
    low_noise_summary = next(
        summary
        for summary in assessment.condition_summaries
        if summary.measurement_conditions.orientation_noise_sigma_rad == 0.0015
    )
    assert low_noise_summary.case_count == 1
    assert low_noise_summary.measurement_conditions.motion_profile_id == (
        MotionProfile.RICH_THREE_AXIS.value
    )
    assert low_noise_summary.ambiguous_contains_truth == 1
    assert low_noise_summary.operational_miss == 0
    assert low_noise_summary.indeterminate_confidence == 1
    assert low_noise_summary.evidence_miss_rate == 0.0
    assert low_noise_summary.operational_miss_rate == 0.0
    assert low_noise_summary.indeterminate_confidence_rate == 1.0
####


def test_noise_replicates_are_aggregated_by_measurement_condition(tmp_path: Path) -> None:
    manifest = generate_corpus(
        tmp_path / "replicated-noise-corpus",
        noise_profiles=(NOISE_PROFILES[0],),
        noise_replicates=2,
        motion_profiles=(MotionProfile.RICH_THREE_AXIS, MotionProfile.SINGLE_AXIS),
    )

    assessment = validate_corpus(manifest)

    low_noise_summaries = tuple(
        summary
        for summary in assessment.condition_summaries
        if summary.measurement_conditions.orientation_noise_sigma_rad == 0.0015
    )
    assert len(low_noise_summaries) == 2
    assert {
        summary.measurement_conditions.motion_profile_id for summary in low_noise_summaries
    } == {MotionProfile.RICH_THREE_AXIS.value, MotionProfile.SINGLE_AXIS.value}
    assert all(summary.case_count == 2 for summary in low_noise_summaries)
    assert all(summary.measurement_conditions.noise_seed is None for summary in low_noise_summaries)
    assert all(
        summary.evidence_miss_rate == summary.evidence_miss / 2 for summary in low_noise_summaries
    )
####


def test_corpus_generator_supports_independent_seed_ranges(tmp_path: Path) -> None:
    manifest = generate_corpus(
        tmp_path / "independent-seed-corpus",
        noise_replicates=3,
        noise_seed_start=20301030,
        noise_sample_counts=(41,),
    )
    payload: Any = json.loads(manifest.read_text(encoding="utf-8"))
    seeded_cases = tuple(
        case for case in payload["cases"] if "noise_seed" in case["measurement_conditions"]
    )
    seeds = {case["measurement_conditions"]["noise_seed"] for case in seeded_cases}

    assert DEFAULT_NOISE_SEED_START == 20261030
    assert seeds == {20301030, 20301031, 20301032}
    assert len(seeded_cases) == 3
####


def test_corpus_cli_supports_a_clean_none_noise_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outdir = tmp_path / "zero-noise-corpus"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_validation_corpus",
            "--outdir",
            str(outdir),
            "--noise-profile",
            "none",
            "--rate-sensor-calibrations",
            "identity",
            "bias-p-150mrad-s",
            "--motion-profile",
            MotionProfile.SAMPLED_RATE_MANEUVER_BURSTS.value,
            "--noise-sample-counts",
            "21",
        ],
    )

    generate_corpus_main()

    payload: Any = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    noisy_cases = tuple(
        case
        for case in payload["cases"]
        if case["measurement_conditions"].get("noise_seed") is not None
    )
    assert len(noisy_cases) == 2
    assert {
        case["measurement_conditions"]["rate_sensor_calibration_id"] for case in noisy_cases
    } == {"identity", "bias-p-150mrad-s"}
    assert all(
        case["measurement_conditions"]["orientation_noise_sigma_rad"] == 0.0 for case in noisy_cases
    )
    assert all(
        case["measurement_conditions"]["rate_noise_sigma_rad_s"] == 0.0 for case in noisy_cases
    )
    assert all(
        case["label_source"].startswith("seeded smooth maneuver-burst rates")
        for case in noisy_cases
    )
####


def test_convention_sweep_prepares_motion_once_per_replicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    motion_generation_count = 0

    def track_motion_generation(
        time_s: FloatArray,
        profile: MotionProfile,
        motion_seed: int,
    ) -> SyntheticMotion:
        nonlocal motion_generation_count
        motion_generation_count += 1
        return _rate_motion_for_profile(
            profile,
            time_s,
            seed=motion_seed,
            reference_time_s=0.0,
            integration_step_s=SAMPLED_RATE_INTEGRATION_STEP_S,
        )
    ####

    monkeypatch.setattr(
        "scripts.generate_validation_corpus._motion_for_profile",
        track_motion_generation,
    )
    manifest = generate_corpus(
        tmp_path / "prepared-convention-sweep",
        noise_profiles=(NO_NOISE_PROFILE,),
        noise_replicates=2,
        motion_seeds=(751101, 751102),
        noise_sample_counts=(21,),
        truth_convention_sweep=True,
        motion_profiles=(MotionProfile.SAMPLED_RATE_NONANALYTIC,),
    )

    payload: Any = json.loads(manifest.read_text(encoding="utf-8"))
    convention_cases = tuple(
        case for case in payload["cases"] if "motion_seed" in case["measurement_conditions"]
    )
    assert len(convention_cases) == 2 * 2 * 192
    assert motion_generation_count == 2 * 2 * 2
####


@pytest.mark.parametrize(
    "motion_profile",
    (
        MotionProfile.RICH_THREE_AXIS,
        MotionProfile.SAMPLED_RATE_NONANALYTIC,
        MotionProfile.SAMPLED_RATE_PIECEWISE_LINEAR,
        MotionProfile.SAMPLED_RATE_MANEUVER_BURSTS,
    ),
)
def test_truth_convention_sweep_encodes_all_quaternion_rate_conventions(
    tmp_path: Path,
    motion_profile: MotionProfile,
) -> None:
    orientation_latency_s = 0.013 if motion_profile in SAMPLED_RATE_PROFILES else 0.0
    rate_latency_s = 0.027 if motion_profile in SAMPLED_RATE_PROFILES else 0.0
    manifest_path = generate_corpus(
        tmp_path / f"all-truth-conventions-{motion_profile.value}",
        noise_profiles=(NoiseProfile("low", 0.0, 0.0),),
        noise_sample_counts=(41,),
        orientation_latencies_s=(orientation_latency_s,),
        rate_latencies_s=(rate_latency_s,),
        truth_convention_sweep=True,
        motion_profiles=(motion_profile,),
    )
    payload: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    noisy_cases = tuple(
        case
        for case in payload["cases"]
        if case["measurement_conditions"].get("noise_seed") is not None
    )
    sample_times = np.arange(41, dtype=np.float64) * 0.025
    orientation_times = sample_times - orientation_latency_s
    rate_times = sample_times - rate_latency_s
    if motion_profile in SAMPLED_RATE_PROFILES:
        expected_orientation_motion = _rate_motion_for_profile(
            motion_profile,
            orientation_times,
            seed=DEFAULT_MOTION_SEED,
            reference_time_s=0.0,
            integration_step_s=SAMPLED_RATE_INTEGRATION_STEP_S,
        )
        expected_rate_motion = _rate_motion_for_profile(
            motion_profile,
            rate_times,
            seed=DEFAULT_MOTION_SEED,
            reference_time_s=0.0,
            integration_step_s=SAMPLED_RATE_INTEGRATION_STEP_S,
        )
    else:
        expected_orientation_motion = analytic_motion(sample_times, motion_profile)
        expected_rate_motion = expected_orientation_motion
    ####
    scipy_rotation = cast(Any, SciPyRotation)
    expected_rotations = scipy_rotation.from_quat(
        expected_orientation_motion.quaternion_xyzw
    ).as_matrix()

    assert len(all_quaternion_truth_conventions()) == 192
    assert len(noisy_cases) == 192
    assert len({json.dumps(case["truth"], sort_keys=True) for case in noisy_cases}) == 192
    for case in noisy_cases:
        truth = case["truth"]
        search_space = case["config"]["search_space"]
        assert set(search_space["rate_units"]) == {
            "radians_per_second",
            "degrees_per_second",
        }
        assert set(search_space["rate_signs"]) == {-1, 1}
        data = np.loadtxt(
            manifest_path.parent / case["csv"],
            delimiter=",",
            skiprows=1,
            dtype=np.float64,
        )
        encoded_quaternions = data[:, 1:5]
        if truth["quaternion_order"] == "scalar_first":
            encoded_quaternions = encoded_quaternions[:, (1, 2, 3, 0)]
        ####
        if truth["attitude_direction"] == "world_to_body":
            encoded_quaternions[:, :3] *= -1.0
        ####
        decoded_rotations = scipy_rotation.from_quat(encoded_quaternions).as_matrix()
        np.testing.assert_allclose(decoded_rotations, expected_rotations, atol=1.0e-12)

        encoded_rates = data[:, 5:8]
        rate_positions = {axis: index for index, axis in enumerate(truth["rate_order"])}
        decoded_rates = encoded_rates[:, [rate_positions[axis] for axis in "pqr"]]
        if truth["rate_unit"] == "degrees_per_second":
            decoded_rates *= np.pi / 180.0
        ####
        decoded_rates *= truth["rate_sign"]
        expected_rates = (
            expected_rate_motion.body_rates_rad_s
            if truth["rate_frame"] == "body"
            else expected_rate_motion.world_rates_rad_s
        )
        np.testing.assert_allclose(decoded_rates, expected_rates, atol=1.0e-12)
    ####
####


def test_rate_sensor_calibration_applies_named_operations_in_documented_order() -> None:
    calibration = RateSensorCalibration(
        profile_id="operation-order-test",
        misalignment_angle_deg=90.0,
        misalignment_axis=(0.0, 0.0, 1.0),
        bias_rad_s=(1.0, 0.0, 0.0),
        scale_by_axis=(2.0, 1.0, 1.0),
        polarity_by_axis=(-1, 1, 1),
        clip_abs_rad_s=0.5,
    )

    measured = calibration.apply_to_rates(
        np.asarray(((1.0, 0.0, 0.0),), dtype=np.float64),
        measurement_noise_rad_s=np.asarray(((0.1, 0.2, 0.3),), dtype=np.float64),
    )

    np.testing.assert_allclose(
        measured,
        np.asarray(((-0.5, 0.5, 0.3),), dtype=np.float64),
        atol=1.0e-15,
    )
####


def test_rate_sensor_calibration_profiles_cover_scale_polarity_clipping_factorial() -> None:
    unit_scale = (1.0, 1.0, 1.0)
    axis_scale = (1.3, 0.8, 1.1)
    normal_polarity = (1, 1, 1)
    inverted_polarity = (-1, 1, 1)
    profile_by_settings = {
        (
            profile.scale_by_axis,
            profile.polarity_by_axis,
            profile.clip_abs_rad_s,
        ): profile.profile_id
        for profile in RATE_SENSOR_CALIBRATION_PROFILES
        if profile.misalignment_angle_deg == 0.0 and profile.bias_rad_s == (0.0, 0.0, 0.0)
    }

    expected_first_factorial = {
        (unit_scale, normal_polarity, None): "identity",
        (unit_scale, inverted_polarity, None): "polarity-p-flip",
        (unit_scale, normal_polarity, 0.3): "clip-0p3-rad-s",
        (unit_scale, inverted_polarity, 0.3): "polarity-p-flip-clip-0p3-rad-s",
        (axis_scale, normal_polarity, None): "axis-scale-1p3-0p8-1p1",
        (axis_scale, inverted_polarity, None): "axis-scale-1p3-0p8-1p1-polarity-p-flip",
        (axis_scale, normal_polarity, 0.3): "axis-scale-1p3-0p8-1p1-clip-0p3-rad-s",
        (axis_scale, inverted_polarity, 0.3): (
            "axis-scale-1p3-0p8-1p1-polarity-p-flip-clip-0p3-rad-s"
        ),
    }
    assert all(
        profile_by_settings[settings] == profile_id
        for settings, profile_id in expected_first_factorial.items()
    )
####


def test_rate_sensor_calibration_profiles_cover_polarity_and_clipping_levels() -> None:
    axis_polarities = {
        profile.polarity_by_axis
        for profile in RATE_SENSOR_CALIBRATION_PROFILES
        if profile.misalignment_angle_deg == 0.0
        and profile.bias_rad_s == (0.0, 0.0, 0.0)
        and profile.scale_by_axis == (1.0, 1.0, 1.0)
        and profile.clip_abs_rad_s is None
    }
    assert axis_polarities == {
        (1, 1, 1),
        (-1, 1, 1),
        (1, -1, 1),
        (1, 1, -1),
        (-1, -1, 1),
        (-1, 1, -1),
        (1, -1, -1),
        (-1, -1, -1),
    }

    profile_by_id = {profile.profile_id: profile for profile in RATE_SENSOR_CALIBRATION_PROFILES}
    for clip_limit in (0.15, 0.3, 0.6):
        clip_slug = str(clip_limit).replace(".", "p")
        assert profile_by_id[f"clip-{clip_slug}-rad-s"].clip_abs_rad_s == clip_limit
        assert profile_by_id[f"polarity-p-flip-clip-{clip_slug}-rad-s"].clip_abs_rad_s == clip_limit
    ####
####


def test_rate_sensor_calibration_is_explicit_in_generated_manifest(tmp_path: Path) -> None:
    calibrations = (
        RATE_SENSOR_CALIBRATION_PROFILES[0],
        RATE_SENSOR_CALIBRATION_PROFILES[2],
    )
    manifest_path = generate_corpus(
        tmp_path / "calibration-metadata",
        noise_profiles=(NO_NOISE_PROFILE,),
        noise_sample_counts=(21,),
        rate_sensor_calibrations=calibrations,
    )

    manifest = ValidationManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    calibrated_cases = tuple(
        case for case in manifest.cases if case.measurement_conditions.noise_seed is not None
    )

    assert manifest.schema_version == 5
    assert len(calibrated_cases) == 2
    assert {
        case.measurement_conditions.rate_sensor_calibration_id for case in calibrated_cases
    } == {"identity", "bias-p-150mrad-s"}
    bias_case = next(
        case
        for case in calibrated_cases
        if case.measurement_conditions.rate_sensor_calibration_id == "bias-p-150mrad-s"
    )
    assert bias_case.measurement_conditions.rate_sensor_bias_rad_s == (0.15, 0.0, 0.0)
    assert bias_case.measurement_conditions.rate_sensor_scale_by_axis == (1.0, 1.0, 1.0)
    assert bias_case.measurement_conditions.rate_sensor_polarity_by_axis == (1, 1, 1)
####


@pytest.mark.parametrize("rate_noise_frame", ("body", "world"))
def test_canonical_rate_noise_preserves_body_world_coordinate_pairing(
    tmp_path: Path,
    rate_noise_frame: RateNoiseFrame,
) -> None:
    manifest_path = generate_corpus(
        tmp_path / f"canonical-rate-noise-{rate_noise_frame}",
        noise_profiles=(NoiseProfile("low", 0.0, 0.025),),
        noise_seed_start=71931,
        motion_seed=751204,
        noise_sample_counts=(41,),
        truth_convention_sweep=True,
        motion_profiles=(MotionProfile.SAMPLED_RATE_NONANALYTIC,),
        rate_noise_frame=rate_noise_frame,
    )
    payload: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    convention_cases = tuple(
        case
        for case in payload["cases"]
        if case["measurement_conditions"].get("noise_seed") is not None
        and case["truth"]["attitude_direction"] == "body_to_world"
        and case["truth"]["quaternion_order"] == "scalar_last"
        and case["truth"]["rate_order"] == "pqr"
        and case["truth"]["rate_unit"] == "radians_per_second"
        and case["truth"]["rate_sign"] == 1
    )
    cases_by_frame = {case["truth"]["rate_frame"]: case for case in convention_cases}

    assert set(cases_by_frame) == {"body", "world"}
    assert all(
        case["measurement_conditions"]["rate_noise_frame"] == rate_noise_frame
        for case in cases_by_frame.values()
    )
    body_case = cases_by_frame["body"]
    world_case = cases_by_frame["world"]
    body_data = np.loadtxt(
        manifest_path.parent / body_case["csv"],
        delimiter=",",
        skiprows=1,
        dtype=np.float64,
    )
    world_data = np.loadtxt(
        manifest_path.parent / world_case["csv"],
        delimiter=",",
        skiprows=1,
        dtype=np.float64,
    )
    scipy_rotation = cast(Any, SciPyRotation)
    expected_world_rates = scipy_rotation.from_quat(body_data[:, 1:5]).apply(body_data[:, 5:8])

    np.testing.assert_allclose(world_data[:, 1:5], body_data[:, 1:5], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(world_data[:, 5:8], expected_world_rates, rtol=0.0, atol=2.0e-12)
####


def test_paired_noise_replicates_cover_multiple_recording_durations(tmp_path: Path) -> None:
    manifest = generate_corpus(
        tmp_path / "duration-corpus",
        noise_profiles=(NOISE_PROFILES[0],),
        noise_replicates=2,
        noise_sample_counts=(161, 401),
        motion_profiles=(MotionProfile.RICH_THREE_AXIS,),
    )

    assessment = validate_corpus(manifest)
    duration_summaries = tuple(
        summary
        for summary in assessment.condition_summaries
        if summary.measurement_conditions.orientation_noise_sigma_rad == 0.0015
    )
    assert len(duration_summaries) == 2
    assert {summary.measurement_conditions.sample_count for summary in duration_summaries} == {
        161,
        401,
    }
    assert {summary.measurement_conditions.duration_s for summary in duration_summaries} == {
        4.0,
        10.0,
    }
    assert all(summary.case_count == 2 for summary in duration_summaries)

    manifest_payload: Any = json.loads(manifest.read_text(encoding="utf-8"))
    noisy_csv_by_count = {
        case["measurement_conditions"]["sample_count"]: case["csv"]
        for case in manifest_payload["cases"]
        if case["measurement_conditions"].get("noise_seed") == 20261030
    }
    short_path = manifest.parent / noisy_csv_by_count[161]
    long_path = manifest.parent / noisy_csv_by_count[401]
    short_lines = short_path.read_text(encoding="utf-8").splitlines()
    long_lines = long_path.read_text(encoding="utf-8").splitlines()
    assert short_lines == long_lines[: len(short_lines)]
####


def test_timestamp_clock_scale_errors_are_applied_and_grouped(tmp_path: Path) -> None:
    clock_scale_errors_ppm = (
        -50_000.0,
        -10_000.0,
        -1_000.0,
        0.0,
        1_000.0,
        10_000.0,
        50_000.0,
    )
    manifest_path = generate_corpus(
        tmp_path / "timestamp-clock-scale",
        noise_profiles=(NO_NOISE_PROFILE,),
        noise_sample_counts=(161, 401),
        timestamp_clock_rate_errors_ppm=clock_scale_errors_ppm,
    )
    manifest: Any = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 5
    sampled_cases = tuple(
        case for case in manifest["cases"] if "sample_count" in case["measurement_conditions"]
    )
    assert len(sampled_cases) == 14

    errors_ppm: set[float] = set()
    for case in sampled_cases:
        conditions = case["measurement_conditions"]
        error_ppm = float(conditions["timestamp_clock_rate_error_ppm"])
        errors_ppm.add(error_ppm)
        rows = np.loadtxt(manifest_path.parent / case["csv"], delimiter=",", skiprows=1)
        expected_period_s = 0.025 * (1.0 + error_ppm * 1.0e-6)
        np.testing.assert_allclose(np.diff(rows[:, 0]), expected_period_s, atol=1.0e-12)
    ####
    assert errors_ppm == set(clock_scale_errors_ppm)

    assessment = validate_corpus(manifest_path)
    grouped_cases = tuple(
        summary
        for summary in assessment.condition_summaries
        if summary.measurement_conditions.sample_count in (161, 401)
    )
    assert len(grouped_cases) == 14
    assert {
        summary.measurement_conditions.timestamp_clock_rate_error_ppm for summary in grouped_cases
    } == errors_ppm

    assessed_scale_cases = tuple(
        case
        for case in assessment.cases
        if case.measurement_conditions.timestamp_clock_rate_error_ppm is not None
    )
    assert len(assessed_scale_cases) == 14
    assert assessment.summary.evidence_miss == 0
    assert assessment.summary.operational_miss == 0
    assert assessment.summary.recommendation_miss == 0
    for case in assessed_scale_cases:
        error_ppm = case.measurement_conditions.timestamp_clock_rate_error_ppm
        assert error_ppm is not None
        assert case.data_winning_class_ids == (case.truth_class_id,)
        assert case.evidence_contender_class_ids == (case.truth_class_id,)
        assert case.operational_correct
        if abs(error_ppm) <= 1_000.0:
            assert case.evidence_sufficient
            assert case.decision.status is InferenceDecisionStatus.RECOMMENDATION
            assert case.recommendation_correct is True
        else:
            assert not case.evidence_sufficient
            assert case.decision.status is InferenceDecisionStatus.ABSTENTION
            assert case.recommendation_correct is None
        ####
    ####
####


def test_paired_acquisition_perturbations_are_recorded_and_reproducible(tmp_path: Path) -> None:
    def generate(outdir: Path) -> Path:
        return generate_corpus(
            outdir,
            noise_profiles=(NOISE_PROFILES[0],),
            noise_replicates=2,
            noise_sample_counts=(161, 401),
            motion_profiles=(MotionProfile.RICH_THREE_AXIS,),
            timestamp_jitter_amplitudes_s=(0.002,),
            dropout_probabilities=(0.15,),
            orientation_latencies_s=(0.01,),
            rate_latencies_s=(0.025,),
            orientation_noise_ar1_coefficients=(0.6,),
            rate_noise_ar1_coefficients=(-0.4,),
        )
    ####

    manifest = generate(tmp_path / "acquisition-corpus")
    repeated_manifest = generate(tmp_path / "repeated-acquisition-corpus")

    for first_path in sorted(manifest.parent.iterdir()):
        assert (manifest.parent / first_path.name).read_bytes() == (
            repeated_manifest.parent / first_path.name
        ).read_bytes()
    ####

    assessment = validate_corpus(manifest)
    assert assessment.summary.case_count == 7
    noisy_conditions = tuple(
        condition
        for condition in assessment.condition_summaries
        if condition.measurement_conditions.dropout_probability is not None
    )
    assert len(noisy_conditions) == 2
    for condition in noisy_conditions:
        cases = [
            case
            for case in assessment.cases
            if case.measurement_conditions.sample_count
            == condition.measurement_conditions.sample_count
        ]
        assert condition.case_count == 2
        assert condition.measurement_conditions.dropout_probability == 0.15
        assert condition.measurement_conditions.timestamp_jitter_amplitude_s == 0.002
        assert condition.measurement_conditions.orientation_latency_s == 0.01
        assert condition.measurement_conditions.rate_latency_s == 0.025
        assert condition.measurement_conditions.orientation_noise_ar1_coefficient == 0.6
        assert condition.measurement_conditions.rate_noise_ar1_coefficient == -0.4
        assert condition.dropped_sample_count_min == min(
            case.measurement_conditions.dropped_sample_count or 0 for case in cases
        )
        assert condition.dropped_sample_count_max == max(
            case.measurement_conditions.dropped_sample_count or 0 for case in cases
        )
        assert condition.observed_sample_count_min == min(case.sample_count for case in cases)
        assert condition.observed_sample_count_max == max(case.sample_count for case in cases)
    ####

    payload: Any = json.loads(manifest.read_text(encoding="utf-8"))
    cases_by_seed_and_count = {
        (
            case["measurement_conditions"]["noise_seed"],
            case["measurement_conditions"]["sample_count"],
        ): case
        for case in payload["cases"]
        if "noise_seed" in case["measurement_conditions"]
    }
    for noise_seed in (20261030, 20261031):
        short_case = cases_by_seed_and_count[(noise_seed, 161)]
        long_case = cases_by_seed_and_count[(noise_seed, 401)]
        short_lines = (manifest.parent / short_case["csv"]).read_text(encoding="utf-8").splitlines()
        long_lines = (manifest.parent / long_case["csv"]).read_text(encoding="utf-8").splitlines()
        assert short_lines == long_lines[: len(short_lines)]
        for case in (short_case, long_case):
            timestamps = np.loadtxt(
                manifest.parent / case["csv"], delimiter=",", skiprows=1, usecols=(0,)
            )
            assert np.all(np.diff(timestamps) > 0.0)
        ####
        conditions = short_case["measurement_conditions"]
        assert conditions["orientation_noise_ar1_coefficient"] == 0.6
        assert conditions["rate_noise_ar1_coefficient"] == -0.4
        assert (
            conditions["observed_sample_count"] + conditions["dropped_sample_count"]
            == conditions["sample_count"]
        )
    ####
####


def test_recommendation_can_coexist_with_indeterminate_confidence_label(
    tmp_path: Path,
) -> None:
    manifest = generate_corpus(
        tmp_path / "dropout-recommendation-corpus",
        noise_profiles=(NOISE_PROFILES[0],),
        noise_replicates=1,
        noise_seed_start=20701030,
        motion_profiles=(MotionProfile.RICH_THREE_AXIS,),
        dropout_probabilities=(0.35,),
    )

    assessment = validate_corpus(manifest)
    noisy_cases = [
        case for case in assessment.cases if case.measurement_conditions.noise_seed is not None
    ]

    assert len(noisy_cases) == 1
    case = noisy_cases[0]
    assert case.evidence_sufficient
    assert case.decision.status is InferenceDecisionStatus.RECOMMENDATION
    assert case.recommendation_correct is True
    assert case.confidence_label == "indeterminate"
####


def test_ar1_conditions_are_separate_in_validation_summary(tmp_path: Path) -> None:
    manifest = generate_corpus(
        tmp_path / "ar1-factor-corpus",
        noise_sample_counts=(41,),
        orientation_noise_ar1_coefficients=(0.0, 0.7),
        rate_noise_ar1_coefficients=(-0.5, 0.0),
    )

    assessment = validate_corpus(manifest)
    summaries = [
        summary
        for summary in assessment.condition_summaries
        if summary.measurement_conditions.sample_count == 41
    ]

    assert len(summaries) == 4
    assert {
        (
            summary.measurement_conditions.orientation_noise_ar1_coefficient,
            summary.measurement_conditions.rate_noise_ar1_coefficient,
        )
        for summary in summaries
    } == {(0.0, -0.5), (0.0, 0.0), (0.7, -0.5), (0.7, 0.0)}
    assert all(summary.case_count == 1 for summary in summaries)
####


@pytest.mark.parametrize(
    "motion_profile",
    (
        MotionProfile.RICH_THREE_AXIS,
        MotionProfile.SAMPLED_RATE_NONANALYTIC,
        MotionProfile.SAMPLED_RATE_PIECEWISE_LINEAR,
        MotionProfile.SAMPLED_RATE_MANEUVER_BURSTS,
    ),
)
def test_channel_latency_is_relative_to_row_timestamps(
    tmp_path: Path,
    motion_profile: MotionProfile,
) -> None:
    zero_noise = NoiseProfile("low", 0.0, 0.0)
    manifest = generate_corpus(
        tmp_path / "latency-corpus",
        noise_profiles=(zero_noise,),
        noise_sample_counts=(41,),
        orientation_latencies_s=(0.01,),
        rate_latencies_s=(0.025,),
        motion_profiles=(motion_profile,),
        motion_seed=1947,
    )
    payload: Any = json.loads(manifest.read_text(encoding="utf-8"))
    noisy_case = next(
        case for case in payload["cases"] if "noise_seed" in case["measurement_conditions"]
    )
    values = np.loadtxt(manifest.parent / noisy_case["csv"], delimiter=",", skiprows=1)
    conditions = noisy_case["measurement_conditions"]
    orientation_times = values[:, 0] - conditions["orientation_latency_s"]
    rate_times = values[:, 0] - conditions["rate_latency_s"]
    if motion_profile in SAMPLED_RATE_PROFILES:
        expected_orientation = _rate_motion_for_profile(
            motion_profile,
            orientation_times,
            seed=1947,
            reference_time_s=0.0,
            integration_step_s=SAMPLED_RATE_INTEGRATION_STEP_S,
        ).quaternion_xyzw
        expected_rates = _rate_motion_for_profile(
            motion_profile,
            rate_times,
            seed=1947,
            reference_time_s=0.0,
            integration_step_s=SAMPLED_RATE_INTEGRATION_STEP_S,
        ).body_rates_rad_s
    else:
        expected_orientation = analytic_motion(orientation_times, motion_profile).quaternion_xyzw
        expected_rates = analytic_motion(rate_times, motion_profile).body_rates_rad_s
    ####

    np.testing.assert_allclose(values[:, 1:5], expected_orientation, atol=1.0e-14)
    np.testing.assert_allclose(values[:, 5:8], expected_rates, atol=1.0e-14)
####


def test_distinct_acquisition_values_generate_distinct_case_files(tmp_path: Path) -> None:
    manifest = generate_corpus(
        tmp_path / "distinct-factor-corpus",
        noise_sample_counts=(41,),
        timestamp_jitter_amplitudes_s=(0.001111111, 0.001111112),
    )
    payload: Any = json.loads(manifest.read_text(encoding="utf-8"))
    noisy_cases = [
        case for case in payload["cases"] if "noise_seed" in case["measurement_conditions"]
    ]

    assert len(noisy_cases) == 2
    assert len({case["case_id"] for case in noisy_cases}) == 2
    assert len({case["csv"] for case in noisy_cases}) == 2
####


def test_impossible_dropout_realization_fails_before_creating_output(tmp_path: Path) -> None:
    outdir = tmp_path / "too-sparse-corpus"

    with pytest.raises(ValueError, match="fewer than 21 samples"):
        generate_corpus(
            outdir,
            noise_sample_counts=(41,),
            dropout_probabilities=(0.95,),
        )
    ####

    assert not outdir.exists()
####


@pytest.mark.parametrize("coefficient", (-1.0, 1.0, float("nan"), float("inf")))
def test_invalid_ar1_coefficients_fail_before_creating_output(
    tmp_path: Path,
    coefficient: float,
) -> None:
    outdir = tmp_path / "invalid-ar1-corpus"

    with pytest.raises(ValueError, match="strictly between -1 and 1"):
        generate_corpus(outdir, orientation_noise_ar1_coefficients=(coefficient,))
    ####

    assert not outdir.exists()
####


@pytest.mark.parametrize(
    ("motion_seed", "motion_seeds", "message"),
    (
        (None, (-1,), "non-negative"),
        (None, (1947, 1947), "unique"),
        (1947, (1948,), "either motion_seed or motion_seeds"),
        (None, (1947,), "requires at least one sampled-rate motion profile"),
    ),
)
def test_invalid_motion_seed_options_fail_before_creating_output(
    tmp_path: Path,
    motion_seed: int | None,
    motion_seeds: tuple[int, ...],
    message: str,
) -> None:
    outdir = tmp_path / "invalid-motion-seeds-corpus"

    with pytest.raises(ValueError, match=message):
        generate_corpus(
            outdir,
            motion_seed=motion_seed,
            motion_seeds=motion_seeds,
        )
    ####

    assert not outdir.exists()
####
