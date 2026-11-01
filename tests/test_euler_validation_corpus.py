from __future__ import annotations

import json
from math import exp, expm1, sqrt
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as SciPyRotation

from scripts.generate_euler_validation_corpus import (
    ALL_EULER_CONVENTIONS,
    EulerCorpusConfig,
    euler_rotation_kinematics,
    generate_euler_angle_samples,
    generate_euler_corpus,
    physical_euler_kinematics,
    sample_time_domain_ou_noise,
)
from scripts.validate_corpus import ValidationManifest


def test_manual_euler_axis_products_match_scipy_for_all_conventions() -> None:
    scipy_rotation = cast(Any, SciPyRotation)
    angles_rad = np.asarray(((0.37, -0.41, 0.59),), dtype=np.float64)
    angle_rates_rad_s = np.zeros_like(angles_rad)

    assert len(ALL_EULER_CONVENTIONS) == 192
    for convention in ALL_EULER_CONVENTIONS:
        raw_orientation = euler_rotation_kinematics(
            angles_rad,
            angle_rates_rad_s,
            convention.euler_order,
            convention.euler_mode,
        ).rotation_matrices
        sequence = (
            convention.euler_order.upper()
            if convention.euler_mode == "intrinsic"
            else convention.euler_order.lower()
        )
        reference = scipy_rotation.from_euler(sequence, angles_rad[0], degrees=False)
        if convention.attitude_direction == "world_to_body":
            raw_orientation = np.swapaxes(raw_orientation, 1, 2)
            reference = reference.inv()
        ####
        np.testing.assert_allclose(
            raw_orientation[0],
            reference.as_matrix(),
            atol=1.0e-12,
        )
    ####
####


def test_analytic_body_rates_match_finite_difference_rotation_matrices() -> None:
    time_s = np.asarray((0.2, 0.7, 1.1), dtype=np.float64)
    epsilon_s = 1.0e-5
    angle_samples = generate_euler_angle_samples(time_s, seed=7812)
    earlier_samples = generate_euler_angle_samples(time_s - epsilon_s, seed=7812)
    later_samples = generate_euler_angle_samples(time_s + epsilon_s, seed=7812)
    scipy_rotation = cast(Any, SciPyRotation)

    for convention in ALL_EULER_CONVENTIONS:
        expected_rates = physical_euler_kinematics(angle_samples, convention).body_rates_rad_s
        earlier_rotation = physical_euler_kinematics(
            earlier_samples,
            convention,
        ).rotation_body_to_world_matrices
        later_rotation = physical_euler_kinematics(
            later_samples,
            convention,
        ).rotation_body_to_world_matrices
        relative_rotations = np.swapaxes(earlier_rotation, 1, 2) @ later_rotation
        observed_rates = scipy_rotation.from_matrix(relative_rotations).as_rotvec() / (
            2.0 * epsilon_s
        )
        np.testing.assert_allclose(expected_rates, observed_rates, atol=2.0e-8)
    ####
####


def test_time_domain_ou_noise_is_paired_at_common_times_across_cadences() -> None:
    time_constant_s = 0.05
    noise_by_sample_period = {
        sample_period_s: sample_time_domain_ou_noise(
            np.random.default_rng(43017),
            sample_count,
            sample_period_s,
            sigma=0.003,
            correlation_time_s=time_constant_s,
        )
        for sample_period_s, sample_count in (
            (0.1, 41),
            (0.025, 161),
            (0.0125, 321),
            (0.01, 401),
        )
    }

    np.testing.assert_array_equal(noise_by_sample_period[0.1], noise_by_sample_period[0.025][::4])
    np.testing.assert_array_equal(noise_by_sample_period[0.1], noise_by_sample_period[0.0125][::8])
    np.testing.assert_array_equal(noise_by_sample_period[0.1], noise_by_sample_period[0.01][::10])
    np.testing.assert_array_equal(
        noise_by_sample_period[0.025], noise_by_sample_period[0.0125][::2]
    )
    base_grid_noise = sample_time_domain_ou_noise(
        np.random.default_rng(43017),
        sample_count=20_001,
        sample_period_s=0.001,
        sigma=0.003,
        correlation_time_s=time_constant_s,
    )
    observed_lag_one = np.corrcoef(base_grid_noise[:-1, 0], base_grid_noise[1:, 0])[0, 1]
    assert observed_lag_one == pytest.approx(exp(-0.001 / time_constant_s), abs=0.02)
    assert np.std(base_grid_noise[:, 0]) == pytest.approx(0.003, abs=0.0003)
    half_millisecond_noise = sample_time_domain_ou_noise(
        np.random.default_rng(43017),
        sample_count=40_001,
        sample_period_s=0.0005,
        sigma=0.003,
        correlation_time_s=time_constant_s,
    )
    observed_half_millisecond_lag = np.corrcoef(
        half_millisecond_noise[:-1, 0], half_millisecond_noise[1:, 0]
    )[0, 1]
    assert observed_half_millisecond_lag == pytest.approx(exp(-0.0005 / time_constant_s), abs=0.02)
    assert np.std(half_millisecond_noise[:, 0]) == pytest.approx(0.003, abs=0.0003)
####


def test_time_domain_ou_noise_preserves_legacy_integer_millisecond_values() -> None:
    sample_count = 401
    sigma = 0.003
    correlation_time_s = 0.05
    random = np.random.default_rng(43017)
    step_correlation = exp(-0.001 / correlation_time_s)
    innovation_sigma = sigma * sqrt(-expm1(-2.0 * 0.001 / correlation_time_s))
    expected = np.zeros((sample_count, 3), dtype=np.float64)
    expected[0] = random.normal(0.0, sigma, 3)
    innovations = random.normal(0.0, innovation_sigma, (sample_count - 1, 3))
    for sample_index in range(1, sample_count):
        expected[sample_index] = (
            step_correlation * expected[sample_index - 1] + innovations[sample_index - 1]
        )
    ####
    actual = sample_time_domain_ou_noise(
        np.random.default_rng(43017),
        sample_count,
        sample_period_s=0.001,
        sigma=sigma,
        correlation_time_s=correlation_time_s,
    )

    np.testing.assert_array_equal(actual, expected)
####


def test_generated_euler_manifest_has_full_typed_truth_and_search_space(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "euler-corpus"
    manifest_path = generate_euler_corpus(
        output_directory,
        EulerCorpusConfig(motion_seeds=(730301,), sample_count=41),
    )
    payload: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = ValidationManifest.model_validate(payload)

    assert len(manifest.cases) == 192
    assert len({case.case_id for case in manifest.cases}) == 192
    assert len({case.csv for case in manifest.cases}) == 192
    assert all(case.representation.value == "euler" for case in manifest.cases)
    assert all(case.truth.euler_order is not None for case in manifest.cases)
    assert all(case.truth.euler_mode is not None for case in manifest.cases)
    assert all(case.truth.angle_unit is not None for case in manifest.cases)
    assert all(str(case.truth.rate_frame) == "body" for case in manifest.cases)
    assert all(case.truth.rate_order == "pqr" for case in manifest.cases)
    assert all(case.truth.rate_sign == 1 for case in manifest.cases)
    assert all(
        case.truth.euler_order in case.config.search_space.euler_orders for case in manifest.cases
    )
    assert all(
        case.truth.euler_mode in case.config.search_space.euler_modes for case in manifest.cases
    )
    assert all(
        case.truth.angle_unit in case.config.search_space.angle_units for case in manifest.cases
    )
    assert all((output_directory / case.csv).is_file() for case in manifest.cases)
    assert all(len(case.csv.name) <= 64 for case in manifest.cases)
####


def test_generated_ou_manifest_records_time_constants_and_sampled_ar1_coefficients(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "ou-euler-corpus"
    orientation_time_constant_s = 0.05
    rate_time_constant_s = 0.10
    manifest_path = generate_euler_corpus(
        output_directory,
        EulerCorpusConfig(
            motion_seeds=(730301,),
            sample_count=41,
            sample_period_s=0.1,
            orientation_noise_correlation_time_s=orientation_time_constant_s,
            rate_noise_correlation_time_s=rate_time_constant_s,
        ),
    )
    payload: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = ValidationManifest.model_validate(payload)
    conditions = manifest.cases[0].measurement_conditions

    assert conditions.orientation_noise_correlation_time_s == orientation_time_constant_s
    assert conditions.rate_noise_correlation_time_s == rate_time_constant_s
    assert conditions.orientation_noise_ar1_coefficient == pytest.approx(
        exp(-0.1 / orientation_time_constant_s)
    )
    assert conditions.rate_noise_ar1_coefficient == pytest.approx(exp(-0.1 / rate_time_constant_s))
    assert "stationary OU noise" in manifest.cases[0].label_source
####


def test_ou_noise_rejects_sample_periods_off_the_half_millisecond_grid() -> None:
    with pytest.raises(ValueError, match=r"multiple of 0\.5 ms"):
        generate_euler_corpus(
            Path("unused-output-path"),
            EulerCorpusConfig(
                motion_seeds=(730301,),
                sample_count=41,
                sample_period_s=0.00325,
                orientation_noise_correlation_time_s=0.05,
            ),
        )
    ####
####
