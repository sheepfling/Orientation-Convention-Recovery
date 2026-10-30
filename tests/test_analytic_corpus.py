from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.random import default_rng

from attitude_convention_inference.math_utils import rotation_from_quat, rotation_rotvecs
from attitude_convention_inference.statistics import estimate_noise_process
from scripts.generate_validation_corpus import (
    NOISE_PROFILES,
    MotionProfile,
    NoiseProfile,
    analytic_motion,
    correlated_gaussian_noise,
    generate_corpus,
)
from scripts.validate_corpus import validate_corpus


@pytest.mark.parametrize("profile", tuple(MotionProfile))
def test_analytic_rates_match_finite_difference_rotations(profile: MotionProfile) -> None:
    time_s = np.asarray([0.1, 0.7, 1.3, 2.1], dtype=np.float64)
    epsilon = 1.0e-6
    earlier = rotation_from_quat(analytic_motion(time_s - epsilon, profile).quaternion_xyzw)
    later = rotation_from_quat(analytic_motion(time_s + epsilon, profile).quaternion_xyzw)
    center = rotation_from_quat(analytic_motion(time_s, profile).quaternion_xyzw)
    body_rates = rotation_rotvecs(earlier.inv() * later) / (2.0 * epsilon)
    world_rates = np.asarray(cast(Any, center).apply(body_rates), dtype=np.float64)
    analytic = analytic_motion(time_s, profile)

    np.testing.assert_allclose(analytic.body_rates_rad_s, body_rates, atol=1.0e-9)
    np.testing.assert_allclose(analytic.world_rates_rad_s, world_rates, atol=1.0e-9)
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


def test_channel_latency_is_relative_to_row_timestamps(tmp_path: Path) -> None:
    zero_noise = NoiseProfile("low", 0.0, 0.0)
    manifest = generate_corpus(
        tmp_path / "latency-corpus",
        noise_profiles=(zero_noise,),
        noise_sample_counts=(41,),
        orientation_latencies_s=(0.01,),
        rate_latencies_s=(0.025,),
    )
    payload: Any = json.loads(manifest.read_text(encoding="utf-8"))
    noisy_case = next(
        case for case in payload["cases"] if "noise_seed" in case["measurement_conditions"]
    )
    values = np.loadtxt(manifest.parent / noisy_case["csv"], delimiter=",", skiprows=1)
    conditions = noisy_case["measurement_conditions"]
    expected_orientation = analytic_motion(
        values[:, 0] - conditions["orientation_latency_s"]
    ).quaternion_xyzw
    expected_rates = analytic_motion(values[:, 0] - conditions["rate_latency_s"]).body_rates_rad_s

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
