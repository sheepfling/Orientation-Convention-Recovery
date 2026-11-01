from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.random import default_rng
from numpy.typing import NDArray

from attitude_convention_inference.hypothesis_model import ConventionHypothesis
from attitude_convention_inference.io import load_csv_trajectory
from attitude_convention_inference.math_utils import mean_float
from attitude_convention_inference.model_common import AttitudeRepresentation
from attitude_convention_inference.statistics import log_likelihood_advantage_by_prefix
from scripts.compare_correlated_evidence import compare_manifest
from scripts.correlated_evidence import (
    IntervalResidualNoiseModel,
    covariance_adjusted_log_likelihood_ratio,
    paired_log_likelihood_ratios_by_case,
)
from scripts.generate_validation_corpus import correlated_gaussian_noise, generate_corpus
from scripts.validate_correlated_residual_model import (
    COUNTERFACTUAL_KINDS,
    counterfactual_hypotheses,
)
from scripts.validate_correlated_residual_model import (
    validate_manifest as validate_residual_model,
)

FloatArray = NDArray[np.float64]


def test_interval_covariance_matches_simulated_difference_and_trapezoid_noise() -> None:
    model = IntervalResidualNoiseModel(
        orientation_sigma_rad=0.02,
        orientation_ar1_coefficient=0.6,
        rate_sigma_rad_s=0.03,
        rate_ar1_coefficient=-0.2,
        sample_period_s=0.1,
    )
    orientation_noise = correlated_gaussian_noise(
        default_rng(101),
        sigma=model.orientation_sigma_rad,
        sample_count=100_000,
        dimension_count=3,
        ar1_coefficient=model.orientation_ar1_coefficient,
    )
    rate_noise = correlated_gaussian_noise(
        default_rng(202),
        sigma=model.rate_sigma_rad_s,
        sample_count=100_000,
        dimension_count=3,
        ar1_coefficient=model.rate_ar1_coefficient,
    )
    half_period = model.sample_period_s / 2.0
    interval_errors = np.diff(orientation_noise, axis=0) - half_period * (
        rate_noise[:-1] + rate_noise[1:]
    )
    centered = interval_errors - np.mean(interval_errors, axis=0)

    for lag in (0, 1, 2):
        empirical_covariance = mean_float(
            np.asarray(centered[: len(centered) - lag or None] * centered[lag:], dtype=np.float64)
        )
        assert empirical_covariance == pytest.approx(model.lag_covariance_rad2(lag), rel=0.025)
    ####

    covariance = model.covariance_matrix(8)
    np.testing.assert_allclose(covariance, covariance.T, atol=0.0)
    np.testing.assert_allclose(np.diag(covariance), model.lag_covariance_rad2(0))
####


def test_covariance_adjusted_likelihood_reduces_to_iid_reference() -> None:
    rng = default_rng(73)
    winner = np.asarray(rng.normal(0.0, 0.002, (23, 3)), dtype=np.float64)
    competitors = (
        np.asarray(rng.normal(0.0, 0.003, (23, 3)), dtype=np.float64),
        np.asarray(rng.normal(0.0, 0.005, (23, 3)), dtype=np.float64),
    )
    sigma = 0.0025
    covariance_cholesky = np.eye(23, dtype=np.float64) * sigma

    colored_advantage = covariance_adjusted_log_likelihood_ratio(
        winner,
        competitors,
        covariance_cholesky,
    )
    iid_advantage = log_likelihood_advantage_by_prefix(winner, competitors, sigma)[-1]

    assert float(colored_advantage) == pytest.approx(float(iid_advantage), abs=1.0e-9)
####


def test_covariance_adjusted_likelihood_includes_temporal_cross_terms() -> None:
    rng = default_rng(117)
    covariance = np.asarray(
        (
            (1.0, 0.35, 0.10),
            (0.35, 1.2, 0.25),
            (0.10, 0.25, 0.9),
        ),
        dtype=np.float64,
    )
    winner = np.asarray(rng.normal(0.0, 0.3, (3, 3)), dtype=np.float64)
    competitors = (
        np.asarray(rng.normal(0.0, 0.5, (3, 3)), dtype=np.float64),
        np.asarray(rng.normal(0.0, 0.7, (3, 3)), dtype=np.float64),
    )
    factor = np.asarray(np.linalg.cholesky(covariance), dtype=np.float64)

    actual = covariance_adjusted_log_likelihood_ratio(winner, competitors, factor)
    winner_quadratic = float(np.sum(winner * np.linalg.solve(covariance, winner)))
    competitor_quadratics = tuple(
        float(np.sum(errors * np.linalg.solve(covariance, errors))) for errors in competitors
    )
    expected = 0.5 * (min(competitor_quadratics) - winner_quadratic)

    assert actual == pytest.approx(expected, abs=1.0e-12)
####


def test_batched_pairwise_likelihoods_match_single_case_references() -> None:
    model = IntervalResidualNoiseModel(
        orientation_sigma_rad=0.015,
        orientation_ar1_coefficient=0.4,
        rate_sigma_rad_s=0.025,
        rate_ar1_coefficient=0.2,
        sample_period_s=0.04,
    )
    random = default_rng(991)
    winner = np.asarray(random.normal(0.0, 0.002, (5, 17, 3)), dtype=np.float64)
    competitor = np.asarray(random.normal(0.0, 0.003, (5, 17, 3)), dtype=np.float64)
    covariance = model.covariance_matrix(17)
    marginal_variance = model.lag_covariance_rad2(0)
    covariance_factor = np.asarray(np.linalg.cholesky(covariance), dtype=np.float64)

    actual = paired_log_likelihood_ratios_by_case(
        winner,
        competitor,
        marginal_variance_rad2=marginal_variance,
        covariance_cholesky=covariance_factor,
    )

    expected_iid = tuple(
        log_likelihood_advantage_by_prefix(
            winner[index],
            (competitor[index],),
            float(np.sqrt(marginal_variance)),
        )[-1]
        for index in range(len(winner))
    )
    expected_colored = tuple(
        covariance_adjusted_log_likelihood_ratio(
            winner[index],
            (competitor[index],),
            covariance_factor,
        )
        for index in range(len(winner))
    )
    np.testing.assert_allclose(actual.marginal_iid_by_case, expected_iid, atol=1.0e-10)
    np.testing.assert_allclose(
        actual.temporal_covariance_by_case,
        expected_colored,
        atol=1.0e-10,
    )
####


def test_pairwise_likelihoods_include_distinct_covariance_normalizers() -> None:
    truth_model = IntervalResidualNoiseModel(
        orientation_sigma_rad=0.015,
        orientation_ar1_coefficient=0.4,
        rate_sigma_rad_s=0.025,
        rate_ar1_coefficient=0.2,
        sample_period_s=0.04,
    )
    alternative_model = IntervalResidualNoiseModel(
        orientation_sigma_rad=truth_model.orientation_sigma_rad,
        orientation_ar1_coefficient=truth_model.orientation_ar1_coefficient,
        rate_sigma_rad_s=truth_model.rate_sigma_rad_s * np.pi / 180.0,
        rate_ar1_coefficient=truth_model.rate_ar1_coefficient,
        sample_period_s=truth_model.sample_period_s,
    )
    random = default_rng(884)
    truth: FloatArray = np.asarray(random.normal(0.0, 0.003, (4, 9, 3)), dtype=np.float64)
    alternative: FloatArray = np.asarray(random.normal(0.0, 0.004, (4, 9, 3)), dtype=np.float64)
    truth_covariance = truth_model.covariance_matrix(9)
    alternative_covariance = alternative_model.covariance_matrix(9)
    truth_factor = np.asarray(np.linalg.cholesky(truth_covariance), dtype=np.float64)
    alternative_factor = np.asarray(np.linalg.cholesky(alternative_covariance), dtype=np.float64)

    actual = paired_log_likelihood_ratios_by_case(
        truth,
        alternative,
        marginal_variance_rad2=truth_model.lag_covariance_rad2(0),
        covariance_cholesky=truth_factor,
        alternative_marginal_variance_rad2=alternative_model.lag_covariance_rad2(0),
        alternative_covariance_cholesky=alternative_factor,
    )
    expected_iid: list[float] = []
    expected_colored: list[float] = []
    truth_logdet = float(np.linalg.slogdet(truth_covariance)[1])
    alternative_logdet = float(np.linalg.slogdet(alternative_covariance)[1])
    for index in range(truth.shape[0]):
        truth_case: FloatArray = truth[index]
        alternative_case: FloatArray = alternative[index]
        truth_sum_squares = float(np.sum(np.square(truth_case)))
        alternative_sum_squares = float(np.sum(np.square(alternative_case)))
        expected_iid.append(
            0.5
            * (
                3
                * truth_case.shape[0]
                * np.log(
                    alternative_model.lag_covariance_rad2(0) / truth_model.lag_covariance_rad2(0)
                )
                + alternative_sum_squares / alternative_model.lag_covariance_rad2(0)
                - truth_sum_squares / truth_model.lag_covariance_rad2(0)
            )
        )
        truth_quadratic = float(np.sum(truth_case * np.linalg.solve(truth_covariance, truth_case)))
        alternative_quadratic = float(
            np.sum(alternative_case * np.linalg.solve(alternative_covariance, alternative_case))
        )
        expected_colored.append(
            0.5
            * (3 * (alternative_logdet - truth_logdet) + alternative_quadratic - truth_quadratic)
        )
    ####

    np.testing.assert_allclose(actual.marginal_iid_by_case, expected_iid, atol=1.0e-10)
    np.testing.assert_allclose(actual.temporal_covariance_by_case, expected_colored, atol=1.0e-9)
####


def test_manifest_comparison_runs_on_a_seeded_synthetic_case(tmp_path: Path) -> None:
    manifest_path = generate_corpus(tmp_path / "corpus", noise_replicates=4)

    report = compare_manifest(manifest_path, max_cases_per_condition=1)
    covariance_report = validate_residual_model(manifest_path, max_cases_per_condition=4)
    held_out_report = validate_residual_model(
        manifest_path,
        max_cases_per_condition=4,
        seed_partition="odd",
    )

    assert len(report.cases) == 1
    assert report.cases[0].data_winner_correct
    assert report.cases[0].known_noise_colored_log_likelihood_ratio > 0.0
    assert report.condition_summaries[0].case_count == 1
    assert any(
        "known" in assumption and "manifest" in assumption for assumption in report.assumptions
    )
    covariance_by_candidate = {
        summary.candidate_kind: summary for summary in covariance_report.condition_summaries
    }
    assert set(covariance_by_candidate) == {"truth", *COUNTERFACTUAL_KINDS}
    assert all(summary.case_count == 4 for summary in covariance_by_candidate.values())
    assert covariance_by_candidate["truth"].lag_covariance_diagnostics[0].lag == 0
    assert covariance_by_candidate["truth"].centered_whitened_sum_squares_per_dof > 0.0
    truth_zero_lag = covariance_by_candidate["truth"].lag_covariance_diagnostics[0]
    np.testing.assert_allclose(
        truth_zero_lag.empirical_covariance_rad2,
        np.trace(np.asarray(truth_zero_lag.empirical_covariance_block_rad2)) / 3.0,
    )
    assert truth_zero_lag.covariance_block_error_as_fraction_of_zero_lag_variance == pytest.approx(
        np.sqrt(3.0)
        * covariance_by_candidate["truth"].zero_lag_cross_axis_covariance_relative_error
    )
    assert covariance_by_candidate["rate_unit_flip"].covariance_mode == "rate_unit_adjusted"
    expected_unit_adjusted_rate_sigma = (
        covariance_by_candidate["truth"].truth_noise_model.rate_sigma_rad_s * np.pi / 180.0
    )
    assert covariance_by_candidate["rate_unit_flip"].candidate_noise_model.rate_sigma_rad_s == (
        pytest.approx(expected_unit_adjusted_rate_sigma)
    )
    assert (
        covariance_by_candidate["attitude_direction_flip"].uncentered_whitened_sum_squares_per_dof
        > covariance_by_candidate["truth"].uncentered_whitened_sum_squares_per_dof
    )
    pairwise = covariance_report.pairwise_likelihood_summaries
    assert {summary.alternative_kind for summary in pairwise} == set(COUNTERFACTUAL_KINDS)
    assert all(summary.case_count == 4 for summary in pairwise)
    assert all(
        summary.iid_truth_win_count + summary.iid_alternative_win_count + summary.iid_tie_count == 4
        for summary in pairwise
    )
    assert all(
        summary.colored_truth_win_count
        + summary.colored_alternative_win_count
        + summary.colored_tie_count
        == 4
        for summary in pairwise
    )
    covariance_differences = covariance_report.candidate_covariance_differences
    assert {summary.alternative_kind for summary in covariance_differences} == set(
        COUNTERFACTUAL_KINDS
    )
    assert all(summary.case_count == 4 for summary in covariance_differences)
    assert all(
        summary.maximum_block_difference_fraction_of_truth_zero_lag
        == pytest.approx(
            max(
                lag_difference.frobenius_difference_fraction_of_truth_zero_lag
                for lag_difference in summary.lag_differences
            )
        )
        for summary in covariance_differences
    )
    assert all(summary.case_count == 2 for summary in held_out_report.condition_summaries)
    assert held_out_report.seed_partition == "odd"
    assert held_out_report.selected_case_count == 2
    assert held_out_report.selected_noise_seeds == (20261031, 20261033)
    assert held_out_report.covariance_holdout is None
    covariance_holdout = covariance_report.covariance_holdout
    assert covariance_holdout is not None
    assert covariance_holdout.training_noise_seeds == (20261030, 20261032)
    assert covariance_holdout.heldout_noise_seeds == (20261031, 20261033)
    assert len(covariance_holdout.summaries) == 1 + len(COUNTERFACTUAL_KINDS)
    assert all(summary.training_case_count == 2 for summary in covariance_holdout.summaries)
    assert all(summary.heldout_case_count == 2 for summary in covariance_holdout.summaries)
    assert all(
        len(summary.lag_diagnostics) == 6
        and summary.fitted_candidate_rms_error_fraction_of_truth_c0 >= 0.0
        and summary.scalar_reference_rms_error_fraction_of_truth_c0 >= 0.0
        for summary in covariance_holdout.summaries
    )
####


def test_generated_rate_unit_counterfactual_is_symmetric_in_both_units(tmp_path: Path) -> None:
    manifest_path = generate_corpus(
        tmp_path / "paired-rate-units",
        noise_replicates=2,
        noise_seed_start=8300,
        noise_sample_counts=(41,),
        truth_rate_units=("radians_per_second", "degrees_per_second"),
    )
    manifest_payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    noisy_cases = [
        case
        for case in manifest_payload["cases"]
        if case["measurement_conditions"].get("noise_seed") == 8300
    ]
    cases_by_unit = {case["truth"]["rate_unit"]: case for case in noisy_cases}
    radians_case = cases_by_unit["radians_per_second"]
    degrees_case = cases_by_unit["degrees_per_second"]

    def load_rates(case: dict[str, Any]) -> FloatArray:
        csv_path = manifest_path.parent / str(case["csv"])
        data = load_csv_trajectory(
            csv_path,
            AttitudeRepresentation.QUATERNION,
            ("a0", "a1", "a2", "a3"),
            ("r0", "r1", "r2"),
            "time_s",
        )
        assert data.rates is not None
        return data.rates
    ####

    radians_rates = load_rates(radians_case)
    degrees_rates = load_rates(degrees_case)
    np.testing.assert_allclose(degrees_rates, radians_rates * (180.0 / np.pi), rtol=1e-12)

    residual_report = validate_residual_model(manifest_path, max_cases_per_condition=2)
    assert residual_report.covariance_holdout is not None
    assert residual_report.covariance_holdout.summaries == ()
    unit_flip_covariance_summaries = {
        summary.truth_hypothesis_id: summary
        for summary in residual_report.condition_summaries
        if summary.candidate_kind == "rate_unit_flip"
    }
    truth_ids_by_unit = {
        unit: ConventionHypothesis.model_validate(case["truth"]).identifier()
        for unit, case in cases_by_unit.items()
    }
    assert set(unit_flip_covariance_summaries) == set(truth_ids_by_unit.values())
    for truth_unit, truth_id in truth_ids_by_unit.items():
        summary = unit_flip_covariance_summaries[truth_id]
        expected_scale = np.pi / 180.0 if truth_unit == "radians_per_second" else 180.0 / np.pi
        assert summary.candidate_noise_model.rate_sigma_rad_s == pytest.approx(
            summary.truth_noise_model.rate_sigma_rad_s * expected_scale
        )
    ####
    unit_flip_summaries = {
        summary.condition.truth_rate_unit: summary
        for summary in residual_report.pairwise_likelihood_summaries
        if summary.alternative_kind == "rate_unit_flip"
    }
    assert set(unit_flip_summaries) == {"radians_per_second", "degrees_per_second"}
    for truth_unit, summary in unit_flip_summaries.items():
        assert summary.covariance_mode == "rate_unit_adjusted"
        assert summary.case_count == 2
        assert summary.colored_truth_win_count == 2
        assert summary.condition.truth_rate_unit.value == truth_unit
    ####

    selected_case_ids = {radians_case["case_id"], degrees_case["case_id"]}
    manifest_payload["cases"] = [
        case for case in manifest_payload["cases"] if case["case_id"] in selected_case_ids
    ]
    paired_manifest = manifest_path.parent / "paired-rate-units-manifest.json"
    paired_manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    comparison = compare_manifest(paired_manifest, max_cases_per_condition=2)
    assert len(comparison.condition_summaries) == 2
    compared_truth_ids = {
        summary.condition.truth_hypothesis_id for summary in comparison.condition_summaries
    }
    assert compared_truth_ids == {
        ConventionHypothesis.model_validate(radians_case["truth"]).identifier(),
        ConventionHypothesis.model_validate(degrees_case["truth"]).identifier(),
    }
    assert all(case_comparison.data_winner_correct for case_comparison in comparison.cases)
####


def test_counterfactuals_each_change_exactly_one_convention_field() -> None:
    truth = ConventionHypothesis.model_validate(
        {
            "attitude_direction": "body_to_world",
            "quaternion_order": "scalar_first",
            "rate_frame": "body",
            "rate_order": "pqr",
            "rate_unit": "radians_per_second",
            "rate_sign": 1,
        }
    )
    expected_changed_fields = {
        "attitude_direction_flip": "attitude_direction",
        "quaternion_storage_order_flip": "quaternion_order",
        "rate_frame_flip": "rate_frame",
        "rate_order_reversal": "rate_order",
        "rate_sign_flip": "rate_sign",
        "rate_unit_flip": "rate_unit",
    }
    candidates = dict(counterfactual_hypotheses(truth))

    assert set(candidates) == set(COUNTERFACTUAL_KINDS)
    for candidate_kind, candidate in candidates.items():
        changed_fields = {
            field
            for field, truth_value in truth.model_dump().items()
            if candidate.model_dump()[field] != truth_value
        }
        assert changed_fields == {expected_changed_fields[candidate_kind]}
    ####
####


@pytest.mark.parametrize(
    "values",
    (
        {"orientation_sigma_rad": -0.1},
        {"orientation_ar1_coefficient": 1.0},
        {"rate_ar1_coefficient": float("nan")},
        {"sample_period_s": 0.0},
        {"rate_sigma_rad_s": 0.0, "orientation_sigma_rad": 0.0},
    ),
)
def test_interval_noise_model_rejects_invalid_parameters(values: dict[str, float]) -> None:
    parameters = {
        "orientation_sigma_rad": 0.01,
        "orientation_ar1_coefficient": 0.0,
        "rate_sigma_rad_s": 0.02,
        "rate_ar1_coefficient": 0.0,
        "sample_period_s": 0.025,
    }
    parameters.update(values)

    with pytest.raises(ValueError):
        IntervalResidualNoiseModel(**parameters)
    ####
####
