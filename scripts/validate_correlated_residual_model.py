"""Validate the first-order residual covariance on fully scored synthetic trajectories."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from functools import cache
from math import isclose, pi, sqrt
from pathlib import Path
from typing import Literal

from numpy import asarray, eye, float64, mean, quantile, square, stack, zeros
from numpy import sum as np_sum
from numpy.linalg import cholesky as np_cholesky
from numpy.linalg import cond as np_cond
from numpy.linalg import norm as np_norm
from numpy.typing import NDArray

from attitude_convention_inference.core import TrajectoryData
from attitude_convention_inference.hypothesis_model import ConventionHypothesis
from attitude_convention_inference.io import load_csv_trajectory
from attitude_convention_inference.math_utils import mean_float
from attitude_convention_inference.model_common import (
    AttitudeDirection,
    QuaternionOrder,
    RateFrame,
    RateUnit,
)
from attitude_convention_inference.scoring import score_hypothesis
from scripts.correlated_evidence import (
    IntervalResidualNoiseModel,
    paired_log_likelihood_ratios_by_case,
    solve_lower_triangular,
)
from scripts.validate_corpus import ValidationCase, ValidationManifest

FloatArray = NDArray[float64]
CounterfactualKind = Literal[
    "attitude_direction_flip",
    "quaternion_storage_order_flip",
    "rate_frame_flip",
    "rate_order_reversal",
    "rate_sign_flip",
    "rate_unit_flip",
]
CandidateCovarianceMode = Literal["truth_shared", "rate_unit_adjusted"]
SeedPartition = Literal["all", "even", "odd"]
CovarianceBlock = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
ResidualCandidateKind = Literal[
    "truth",
    "attitude_direction_flip",
    "quaternion_storage_order_flip",
    "rate_frame_flip",
    "rate_order_reversal",
    "rate_sign_flip",
    "rate_unit_flip",
]
COUNTERFACTUAL_KINDS: tuple[CounterfactualKind, ...] = (
    "attitude_direction_flip",
    "quaternion_storage_order_flip",
    "rate_frame_flip",
    "rate_order_reversal",
    "rate_sign_flip",
    "rate_unit_flip",
)
DIAGNOSTIC_LAGS = (0, 1, 2, 4, 8, 16)


@dataclass(frozen=True, slots=True)
class _ResidualCondition:
    """Sampling, motion, and measurement-noise values that define one report cell."""

    motion_profile_id: str | None
    sample_count: int
    truth_hypothesis_id: str
    truth_rate_unit: RateUnit
    truth_noise_model: IntervalResidualNoiseModel
####


@dataclass(frozen=True, slots=True)
class LagCovarianceDiagnostic:
    """Empirical and modeled covariance for one temporal lag.

    The normalized error is signed and measured relative to the modeled lag-zero
    variance, so it remains interpretable when the covariance at this lag is small.
    The matrix block retains cross-axis covariance hidden by the scalar trace.
    """

    lag: int
    empirical_covariance_rad2: float
    model_covariance_rad2: float
    empirical_covariance_block_rad2: CovarianceBlock
    error_as_fraction_of_zero_lag_variance: float
    covariance_block_error_as_fraction_of_zero_lag_variance: float
####


@dataclass(frozen=True, slots=True)
class ResidualConditionSummary:
    """Candidate residual diagnostics against its stated covariance model.

    The rate-unit counterfactual scales rate-noise standard deviation by the
    unit-conversion ratio. Other counterfactuals currently share the truth model.
    """

    candidate_kind: ResidualCandidateKind
    covariance_mode: CandidateCovarianceMode
    motion_profile_id: str | None
    sample_count: int
    truth_hypothesis_id: str
    truth_noise_model: IntervalResidualNoiseModel
    candidate_noise_model: IntervalResidualNoiseModel
    case_count: int
    lag_covariance_diagnostics: tuple[LagCovarianceDiagnostic, ...]
    zero_lag_cross_axis_covariance_relative_error: float
    centered_whitened_sum_squares_per_dof: float
    uncentered_whitened_sum_squares_per_dof: float
    replicate_mean_residual_rms_in_sigma: float
    covariance_condition_number: float
####


@dataclass(frozen=True, slots=True)
class ResidualModelValidationReport:
    """Synthetic-only residual covariance results and their scope assumptions."""

    manifest: str
    max_cases_per_condition: int
    seed_partition: SeedPartition
    selected_case_count: int
    selected_noise_seeds: tuple[int, ...]
    assumptions: tuple[str, ...]
    condition_summaries: tuple[ResidualConditionSummary, ...]
    candidate_covariance_differences: tuple[CandidateCovarianceDifferenceSummary, ...]
    covariance_holdout: CovarianceHoldoutReport | None
    pairwise_likelihood_summaries: tuple[PairwiseLikelihoodSummary, ...]
####


@dataclass(frozen=True, slots=True)
class CovarianceLagDifference:
    """Empirical candidate-vs-truth covariance-block difference at one lag."""

    lag: int
    frobenius_difference_fraction_of_truth_zero_lag: float
####


@dataclass(frozen=True, slots=True)
class CandidateCovarianceDifferenceSummary:
    """Descriptive, paired covariance comparison for one alternative and condition."""

    alternative_kind: CounterfactualKind
    covariance_mode: CandidateCovarianceMode
    condition: _ResidualCondition
    case_count: int
    lag_differences: tuple[CovarianceLagDifference, ...]
    maximum_block_difference_fraction_of_truth_zero_lag: float
    root_mean_square_block_difference_fraction_of_truth_zero_lag: float
####


@dataclass(frozen=True, slots=True)
class CovarianceHoldoutLagDiagnostic:
    """Held-out empirical block error for fitted and scalar covariance models."""

    lag: int
    fitted_candidate_block_error_fraction_of_truth_c0: float
    scalar_reference_block_error_fraction_of_truth_c0: float
####


@dataclass(frozen=True, slots=True)
class CandidateCovarianceHoldoutSummary:
    """Held-out covariance prediction comparison for one candidate and condition."""

    candidate_kind: ResidualCandidateKind
    covariance_mode: CandidateCovarianceMode
    condition: _ResidualCondition
    training_case_count: int
    heldout_case_count: int
    lag_diagnostics: tuple[CovarianceHoldoutLagDiagnostic, ...]
    fitted_candidate_rms_error_fraction_of_truth_c0: float
    scalar_reference_rms_error_fraction_of_truth_c0: float
####


@dataclass(frozen=True, slots=True)
class CovarianceHoldoutReport:
    """Even-seed covariance fitting evaluated on the disjoint odd-seed cohort."""

    training_noise_seeds: tuple[int, ...]
    heldout_noise_seeds: tuple[int, ...]
    summaries: tuple[CandidateCovarianceHoldoutSummary, ...]
####


@dataclass(frozen=True, slots=True)
class PairwiseLikelihoodSummary:
    """Truth-vs-counterfactual differences under stated reference covariance models."""

    alternative_kind: CounterfactualKind
    covariance_mode: CandidateCovarianceMode
    condition: _ResidualCondition
    case_count: int
    mean_reference_iid_log_likelihood_ratio: float
    mean_reference_colored_log_likelihood_ratio: float
    median_reference_colored_log_likelihood_ratio: float
    reference_colored_log_likelihood_ratio_5th_percentile: float
    reference_colored_log_likelihood_ratio_95th_percentile: float
    iid_truth_win_count: int
    iid_alternative_win_count: int
    iid_tie_count: int
    colored_truth_win_count: int
    colored_alternative_win_count: int
    colored_tie_count: int
####


@cache
def _factorized_covariance(
    noise_model: IntervalResidualNoiseModel,
    interval_count: int,
) -> tuple[FloatArray, FloatArray, float]:
    covariance = noise_model.covariance_matrix(interval_count)
    covariance_cholesky = asarray(np_cholesky(covariance), dtype=float64)
    return covariance, covariance_cholesky, float(np_cond(covariance))
####


def _noise_model(case: ValidationCase) -> IntervalResidualNoiseModel:
    conditions = case.measurement_conditions
    if conditions.sample_period_s is None or conditions.sample_count is None:
        raise ValueError(f"{case.case_id}: a known sample period/count is required.")
    ####
    unsupported_factors = (
        conditions.timestamp_jitter_amplitude_s is None
        or conditions.timestamp_jitter_amplitude_s != 0.0,
        conditions.dropout_probability != 0.0,
        conditions.orientation_latency_s is None or conditions.orientation_latency_s != 0.0,
        conditions.rate_latency_s is None or conditions.rate_latency_s != 0.0,
    )
    if any(unsupported_factors):
        raise ValueError(
            f"{case.case_id}: residual covariance validation requires synchronized, "
            "uniform, no-dropout measurements."
        )
    ####
    if (
        conditions.orientation_noise_sigma_rad is None
        or conditions.orientation_noise_ar1_coefficient is None
        or conditions.rate_noise_sigma_rad_s is None
        or conditions.rate_noise_ar1_coefficient is None
    ):
        message = f"{case.case_id}: known noise parameters are required for covariance validation."
        raise ValueError(message)
    ####
    return IntervalResidualNoiseModel(
        orientation_sigma_rad=conditions.orientation_noise_sigma_rad,
        orientation_ar1_coefficient=conditions.orientation_noise_ar1_coefficient,
        rate_sigma_rad_s=conditions.rate_noise_sigma_rad_s,
        rate_ar1_coefficient=conditions.rate_noise_ar1_coefficient,
        sample_period_s=conditions.sample_period_s,
    )
####


def _condition(case: ValidationCase) -> _ResidualCondition:
    sample_count = case.measurement_conditions.sample_count
    if sample_count is None:
        raise ValueError(f"{case.case_id}: sample_count is required.")
    ####
    return _ResidualCondition(
        motion_profile_id=case.measurement_conditions.motion_profile_id,
        sample_count=sample_count,
        truth_hypothesis_id=case.truth.identifier(),
        truth_rate_unit=RateUnit(case.truth.rate_unit),
        truth_noise_model=_noise_model(case),
    )
####


def _rate_unit_scale_to_radians(unit: RateUnit) -> float:
    return 1.0 if unit is RateUnit.RADIANS_PER_SECOND else pi / 180.0
####


def _candidate_noise_model(
    condition: _ResidualCondition,
    candidate_kind: ResidualCandidateKind,
) -> tuple[IntervalResidualNoiseModel, CandidateCovarianceMode]:
    if candidate_kind != "rate_unit_flip":
        return condition.truth_noise_model, "truth_shared"
    ####
    alternative_unit = (
        RateUnit.DEGREES_PER_SECOND
        if condition.truth_rate_unit is RateUnit.RADIANS_PER_SECOND
        else RateUnit.RADIANS_PER_SECOND
    )
    unit_conversion_ratio = _rate_unit_scale_to_radians(
        alternative_unit
    ) / _rate_unit_scale_to_radians(condition.truth_rate_unit)
    return (
        replace(
            condition.truth_noise_model,
            rate_sigma_rad_s=condition.truth_noise_model.rate_sigma_rad_s * unit_conversion_ratio,
        ),
        "rate_unit_adjusted",
    )
####


def _select_cases(
    cases: Sequence[ValidationCase],
    max_cases_per_condition: int,
    seed_partition: SeedPartition,
) -> tuple[ValidationCase, ...]:
    selected: list[ValidationCase] = []
    counts: dict[_ResidualCondition, int] = defaultdict(int)
    for case in cases:
        noise_seed = case.measurement_conditions.noise_seed
        if noise_seed is None:
            continue
        ####
        if seed_partition == "even" and noise_seed % 2 != 0:
            continue
        ####
        if seed_partition == "odd" and noise_seed % 2 == 0:
            continue
        ####
        condition = _condition(case)
        if counts[condition] >= max_cases_per_condition:
            continue
        ####
        selected.append(case)
        counts[condition] += 1
    ####
    if not selected:
        raise ValueError("The manifest contains no eligible seeded noisy cases.")
    ####
    return tuple(selected)
####


def _load_case_data(manifest_path: Path, case: ValidationCase) -> TrajectoryData:
    if case.representation.value != "quaternion":
        raise ValueError(f"{case.case_id}: this validation requires quaternion data.")
    ####
    if case.orientation_columns is None or case.rate_columns is None or case.time_column is None:
        raise ValueError(
            f"{case.case_id}: explicit orientation, rate, and time columns are required."
        )
    ####
    if len(case.rate_columns) != 3:
        raise ValueError(f"{case.case_id}: exactly three rate columns are required.")
    ####
    csv_path = case.csv if case.csv.is_absolute() else manifest_path.parent / case.csv
    data = load_csv_trajectory(
        csv_path,
        case.representation,
        case.orientation_columns,
        case.rate_columns,
        case.time_column,
    )
    period = case.measurement_conditions.sample_period_s
    expected_count = case.measurement_conditions.sample_count
    if period is None or data.time_s is None or data.sample_count != expected_count:
        raise ValueError(f"{case.case_id}: CSV sampling metadata is incomplete or inconsistent.")
    ####
    if not all(
        isclose(float(data.time_s[index + 1] - data.time_s[index]), period, abs_tol=1.0e-9)
        for index in range(data.sample_count - 1)
    ):
        raise ValueError(f"{case.case_id}: CSV timestamps are not uniformly spaced as declared.")
    ####
    return data
####


def counterfactual_hypotheses(
    hypothesis: ConventionHypothesis,
) -> tuple[tuple[CounterfactualKind, ConventionHypothesis], ...]:
    if hypothesis.quaternion_order is None:
        raise ValueError("Convention counterfactuals require a quaternion hypothesis.")
    ####
    flipped_direction = (
        AttitudeDirection.WORLD_TO_BODY
        if hypothesis.attitude_direction is AttitudeDirection.BODY_TO_WORLD
        else AttitudeDirection.BODY_TO_WORLD
    )
    flipped_quaternion_order = (
        QuaternionOrder.SCALAR_LAST
        if hypothesis.quaternion_order is QuaternionOrder.SCALAR_FIRST
        else QuaternionOrder.SCALAR_FIRST
    )
    flipped_rate_frame = (
        RateFrame.WORLD if hypothesis.rate_frame is RateFrame.BODY else RateFrame.BODY
    )
    flipped_rate_unit = (
        RateUnit.DEGREES_PER_SECOND
        if hypothesis.rate_unit is RateUnit.RADIANS_PER_SECOND
        else RateUnit.RADIANS_PER_SECOND
    )
    return (
        (
            "attitude_direction_flip",
            hypothesis.model_copy(update={"attitude_direction": flipped_direction}),
        ),
        (
            "quaternion_storage_order_flip",
            hypothesis.model_copy(update={"quaternion_order": flipped_quaternion_order}),
        ),
        ("rate_frame_flip", hypothesis.model_copy(update={"rate_frame": flipped_rate_frame})),
        (
            "rate_order_reversal",
            hypothesis.model_copy(update={"rate_order": hypothesis.rate_order[::-1]}),
        ),
        ("rate_sign_flip", hypothesis.model_copy(update={"rate_sign": -hypothesis.rate_sign})),
        ("rate_unit_flip", hypothesis.model_copy(update={"rate_unit": flipped_rate_unit})),
    )
####


def _empirical_lag_covariance_block(centered_residuals: FloatArray, lag: int) -> FloatArray:
    case_count, interval_count, axis_count = centered_residuals.shape
    if case_count < 2 or axis_count != 3:
        raise ValueError("Centered residuals must contain at least two cases and three axes.")
    ####
    if lag < 0 or lag >= interval_count:
        raise ValueError("lag must be within the residual interval count.")
    ####
    left = centered_residuals[:, : interval_count - lag or None, :]
    right = centered_residuals[:, lag:, :]
    correction = case_count / (case_count - 1)
    covariance_block = zeros((3, 3), dtype=float64)
    for row_axis in range(3):
        for column_axis in range(3):
            covariance_block[row_axis, column_axis] = float(
                mean(left[:, :, row_axis] * right[:, :, column_axis]) * correction
            )
        ####
    ####
    return covariance_block
####


def _covariance_block_tuple(covariance_block: FloatArray) -> CovarianceBlock:
    return (
        (
            float(covariance_block[0, 0]),
            float(covariance_block[0, 1]),
            float(covariance_block[0, 2]),
        ),
        (
            float(covariance_block[1, 0]),
            float(covariance_block[1, 1]),
            float(covariance_block[1, 2]),
        ),
        (
            float(covariance_block[2, 0]),
            float(covariance_block[2, 1]),
            float(covariance_block[2, 2]),
        ),
    )
####


def _summarize_condition(
    condition: _ResidualCondition,
    candidate_kind: ResidualCandidateKind,
    residual_records: Sequence[FloatArray],
) -> ResidualConditionSummary:
    residuals = stack(residual_records)
    centered = residuals - mean(residuals, axis=0, keepdims=True)
    case_count, interval_count, _ = centered.shape
    if case_count < 2:
        raise ValueError("At least two independent cases are required per condition.")
    ####
    candidate_noise_model, covariance_mode = _candidate_noise_model(condition, candidate_kind)
    zero_lag_variance = candidate_noise_model.lag_covariance_rad2(0)
    empirical_axis_covariance = _empirical_lag_covariance_block(centered, 0)
    lag_diagnostics: list[LagCovarianceDiagnostic] = []
    for lag in DIAGNOSTIC_LAGS:
        empirical_covariance_block = (
            empirical_axis_covariance
            if lag == 0
            else _empirical_lag_covariance_block(centered, lag)
        )
        empirical_covariance = float(
            (
                empirical_covariance_block[0, 0]
                + empirical_covariance_block[1, 1]
                + empirical_covariance_block[2, 2]
            )
            / 3.0
        )
        model_covariance = candidate_noise_model.lag_covariance_rad2(lag)
        lag_diagnostics.append(
            LagCovarianceDiagnostic(
                lag=lag,
                empirical_covariance_rad2=empirical_covariance,
                model_covariance_rad2=model_covariance,
                empirical_covariance_block_rad2=_covariance_block_tuple(empirical_covariance_block),
                error_as_fraction_of_zero_lag_variance=(
                    (empirical_covariance - model_covariance) / zero_lag_variance
                ),
                covariance_block_error_as_fraction_of_zero_lag_variance=float(
                    np_norm(empirical_covariance_block - eye(3, dtype=float64) * model_covariance)
                    / zero_lag_variance
                ),
            )
        )
    ####
    _, factor, condition_number = _factorized_covariance(
        candidate_noise_model,
        interval_count,
    )
    raw_matrix = residuals.transpose(1, 0, 2).reshape(interval_count, -1)
    centered_matrix = centered.transpose(1, 0, 2).reshape(interval_count, -1)
    raw_whitened = solve_lower_triangular(factor, raw_matrix)
    centered_whitened = solve_lower_triangular(factor, centered_matrix)
    raw_sum_squares_per_dof = float(mean(square(raw_whitened)))
    centered_sum_squares_per_dof = float(
        mean(square(centered_whitened)) * case_count / (case_count - 1)
    )
    replicate_mean_residual = mean(residuals, axis=0)
    replicate_mean_residual_rms_in_sigma = sqrt(
        mean_float(asarray(np_sum(square(replicate_mean_residual), axis=1), dtype=float64))
        / zero_lag_variance
    )
    expected_axis_covariance = eye(3, dtype=float64) * zero_lag_variance
    cross_axis_relative_error = float(
        np_norm(empirical_axis_covariance - expected_axis_covariance)
        / np_norm(expected_axis_covariance)
    )
    return ResidualConditionSummary(
        candidate_kind=candidate_kind,
        covariance_mode=covariance_mode,
        motion_profile_id=condition.motion_profile_id,
        sample_count=condition.sample_count,
        truth_hypothesis_id=condition.truth_hypothesis_id,
        truth_noise_model=condition.truth_noise_model,
        candidate_noise_model=candidate_noise_model,
        case_count=case_count,
        lag_covariance_diagnostics=tuple(lag_diagnostics),
        zero_lag_cross_axis_covariance_relative_error=cross_axis_relative_error,
        centered_whitened_sum_squares_per_dof=centered_sum_squares_per_dof,
        uncentered_whitened_sum_squares_per_dof=raw_sum_squares_per_dof,
        replicate_mean_residual_rms_in_sigma=replicate_mean_residual_rms_in_sigma,
        covariance_condition_number=condition_number,
    )
####


def _candidate_covariance_difference_summaries(
    residual_groups: dict[tuple[_ResidualCondition, ResidualCandidateKind], list[FloatArray]],
) -> tuple[CandidateCovarianceDifferenceSummary, ...]:
    conditions = {condition for condition, _ in residual_groups}
    summaries: list[CandidateCovarianceDifferenceSummary] = []
    for condition in sorted(
        conditions,
        key=lambda item: (
            item.motion_profile_id or "",
            item.sample_count,
            item.truth_noise_model.orientation_sigma_rad,
            item.truth_noise_model.orientation_ar1_coefficient,
            item.truth_noise_model.rate_sigma_rad_s,
            item.truth_noise_model.rate_ar1_coefficient,
            item.truth_hypothesis_id,
        ),
    ):
        truth_records = residual_groups[(condition, "truth")]
        truth_residuals = stack(truth_records)
        truth_centered = truth_residuals - mean(truth_residuals, axis=0, keepdims=True)
        truth_covariance_blocks = {
            lag: _empirical_lag_covariance_block(truth_centered, lag) for lag in DIAGNOSTIC_LAGS
        }
        truth_zero_lag_variance = condition.truth_noise_model.lag_covariance_rad2(0)
        for alternative_kind in COUNTERFACTUAL_KINDS:
            alternative_records = residual_groups[(condition, alternative_kind)]
            if len(alternative_records) != len(truth_records):
                raise RuntimeError(f"Truth and {alternative_kind} residual records are not paired.")
            ####
            alternative_residuals = stack(alternative_records)
            alternative_centered = alternative_residuals - mean(
                alternative_residuals,
                axis=0,
                keepdims=True,
            )
            _, covariance_mode = _candidate_noise_model(
                condition,
                alternative_kind,
            )
            lag_differences = tuple(
                CovarianceLagDifference(
                    lag=lag,
                    frobenius_difference_fraction_of_truth_zero_lag=float(
                        np_norm(
                            _empirical_lag_covariance_block(alternative_centered, lag)
                            - truth_covariance_blocks[lag]
                        )
                        / truth_zero_lag_variance
                    ),
                )
                for lag in DIAGNOSTIC_LAGS
            )
            values = tuple(
                item.frobenius_difference_fraction_of_truth_zero_lag for item in lag_differences
            )
            summaries.append(
                CandidateCovarianceDifferenceSummary(
                    alternative_kind=alternative_kind,
                    covariance_mode=covariance_mode,
                    condition=condition,
                    case_count=len(truth_records),
                    lag_differences=lag_differences,
                    maximum_block_difference_fraction_of_truth_zero_lag=max(values),
                    root_mean_square_block_difference_fraction_of_truth_zero_lag=sqrt(
                        sum(value**2 for value in values) / len(values)
                    ),
                )
            )
        ####
    ####
    return tuple(summaries)
####


def _candidate_covariance_holdout_summaries(
    training_groups: dict[tuple[_ResidualCondition, ResidualCandidateKind], list[FloatArray]],
    heldout_groups: dict[tuple[_ResidualCondition, ResidualCandidateKind], list[FloatArray]],
) -> tuple[CandidateCovarianceHoldoutSummary, ...]:
    summaries: list[CandidateCovarianceHoldoutSummary] = []
    group_keys = sorted(
        training_groups.keys() & heldout_groups.keys(),
        key=lambda item: (
            item[0].motion_profile_id or "",
            item[0].sample_count,
            item[0].truth_noise_model.orientation_sigma_rad,
            item[0].truth_noise_model.orientation_ar1_coefficient,
            item[0].truth_noise_model.rate_sigma_rad_s,
            item[0].truth_noise_model.rate_ar1_coefficient,
            item[0].truth_hypothesis_id,
            item[1],
        ),
    )
    for condition, candidate_kind in group_keys:
        training_records = training_groups[(condition, candidate_kind)]
        heldout_records = heldout_groups[(condition, candidate_kind)]
        if len(training_records) < 2 or len(heldout_records) < 2:
            continue
        ####
        training_residuals = stack(training_records)
        training_centered = training_residuals - mean(
            training_residuals,
            axis=0,
            keepdims=True,
        )
        heldout_residuals = stack(heldout_records)
        heldout_centered = heldout_residuals - mean(
            heldout_residuals,
            axis=0,
            keepdims=True,
        )
        candidate_noise_model, covariance_mode = _candidate_noise_model(
            condition,
            candidate_kind,
        )
        truth_zero_lag_variance = condition.truth_noise_model.lag_covariance_rad2(0)
        lag_diagnostics: list[CovarianceHoldoutLagDiagnostic] = []
        for lag in DIAGNOSTIC_LAGS:
            fitted_candidate_block = _empirical_lag_covariance_block(training_centered, lag)
            heldout_block = _empirical_lag_covariance_block(heldout_centered, lag)
            scalar_reference_variance = candidate_noise_model.lag_covariance_rad2(lag)
            scalar_reference_block = eye(3, dtype=float64) * scalar_reference_variance
            lag_diagnostics.append(
                CovarianceHoldoutLagDiagnostic(
                    lag=lag,
                    fitted_candidate_block_error_fraction_of_truth_c0=float(
                        np_norm(heldout_block - fitted_candidate_block) / truth_zero_lag_variance
                    ),
                    scalar_reference_block_error_fraction_of_truth_c0=float(
                        np_norm(heldout_block - scalar_reference_block) / truth_zero_lag_variance
                    ),
                )
            )
        ####
        fitted_errors = tuple(
            item.fitted_candidate_block_error_fraction_of_truth_c0 for item in lag_diagnostics
        )
        scalar_errors = tuple(
            item.scalar_reference_block_error_fraction_of_truth_c0 for item in lag_diagnostics
        )
        summaries.append(
            CandidateCovarianceHoldoutSummary(
                candidate_kind=candidate_kind,
                covariance_mode=covariance_mode,
                condition=condition,
                training_case_count=len(training_records),
                heldout_case_count=len(heldout_records),
                lag_diagnostics=tuple(lag_diagnostics),
                fitted_candidate_rms_error_fraction_of_truth_c0=sqrt(
                    sum(value**2 for value in fitted_errors) / len(fitted_errors)
                ),
                scalar_reference_rms_error_fraction_of_truth_c0=sqrt(
                    sum(value**2 for value in scalar_errors) / len(scalar_errors)
                ),
            )
        )
    ####
    return tuple(summaries)
####


def _pairwise_likelihood_summaries(
    residual_groups: dict[tuple[_ResidualCondition, ResidualCandidateKind], list[FloatArray]],
) -> tuple[PairwiseLikelihoodSummary, ...]:
    conditions = {condition for condition, _ in residual_groups}
    summaries: list[PairwiseLikelihoodSummary] = []
    for condition in sorted(
        conditions,
        key=lambda item: (
            item.motion_profile_id or "",
            item.sample_count,
            item.truth_noise_model.orientation_sigma_rad,
            item.truth_noise_model.orientation_ar1_coefficient,
            item.truth_noise_model.rate_sigma_rad_s,
            item.truth_noise_model.rate_ar1_coefficient,
            item.truth_hypothesis_id,
        ),
    ):
        truth_records = residual_groups[(condition, "truth")]
        case_count = len(truth_records)
        truth_covariance, truth_factor, _ = _factorized_covariance(
            condition.truth_noise_model,
            truth_records[0].shape[0],
        )
        truth_marginal_variance = float(truth_covariance[0, 0])
        for alternative_kind in COUNTERFACTUAL_KINDS:
            alternative_records = residual_groups[(condition, alternative_kind)]
            if len(truth_records) != len(alternative_records):
                raise RuntimeError(f"Truth and {alternative_kind} residual records are not paired.")
            ####
            alternative_model, covariance_mode = _candidate_noise_model(
                condition,
                alternative_kind,
            )
            alternative_covariance, alternative_factor, _ = _factorized_covariance(
                alternative_model,
                truth_records[0].shape[0],
            )
            scores = paired_log_likelihood_ratios_by_case(
                stack(truth_records),
                stack(alternative_records),
                marginal_variance_rad2=truth_marginal_variance,
                covariance_cholesky=truth_factor,
                alternative_marginal_variance_rad2=float(alternative_covariance[0, 0]),
                alternative_covariance_cholesky=alternative_factor,
            )
            iid_advantages = scores.marginal_iid_by_case
            colored_advantages = scores.temporal_covariance_by_case
            iid_truth_wins = int(np_sum(iid_advantages > 0.0))
            iid_alternative_wins = int(np_sum(iid_advantages < 0.0))
            colored_truth_wins = int(np_sum(colored_advantages > 0.0))
            colored_alternative_wins = int(np_sum(colored_advantages < 0.0))
            summaries.append(
                PairwiseLikelihoodSummary(
                    alternative_kind=alternative_kind,
                    covariance_mode=covariance_mode,
                    condition=condition,
                    case_count=case_count,
                    mean_reference_iid_log_likelihood_ratio=float(mean(iid_advantages)),
                    mean_reference_colored_log_likelihood_ratio=float(mean(colored_advantages)),
                    median_reference_colored_log_likelihood_ratio=float(
                        quantile(colored_advantages, 0.5)
                    ),
                    reference_colored_log_likelihood_ratio_5th_percentile=float(
                        quantile(colored_advantages, 0.05)
                    ),
                    reference_colored_log_likelihood_ratio_95th_percentile=float(
                        quantile(colored_advantages, 0.95)
                    ),
                    iid_truth_win_count=iid_truth_wins,
                    iid_alternative_win_count=iid_alternative_wins,
                    iid_tie_count=case_count - iid_truth_wins - iid_alternative_wins,
                    colored_truth_win_count=colored_truth_wins,
                    colored_alternative_win_count=colored_alternative_wins,
                    colored_tie_count=(case_count - colored_truth_wins - colored_alternative_wins),
                )
            )
        ####
    ####
    return tuple(summaries)
####


def validate_manifest(
    manifest_path: Path,
    *,
    max_cases_per_condition: int = 200,
    seed_partition: SeedPartition = "all",
) -> ResidualModelValidationReport:
    """Compare truth and one-field counterfactuals on seeded synthetic data."""

    if max_cases_per_condition < 2:
        raise ValueError("max_cases_per_condition must be at least two to estimate covariance.")
    ####
    if seed_partition not in ("all", "even", "odd"):
        raise ValueError("seed_partition must be 'all', 'even', or 'odd'.")
    ####
    manifest = ValidationManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    selected = _select_cases(manifest.cases, max_cases_per_condition, seed_partition)
    residual_groups: dict[tuple[_ResidualCondition, ResidualCandidateKind], list[FloatArray]] = (
        defaultdict(list)
    )
    training_groups: dict[tuple[_ResidualCondition, ResidualCandidateKind], list[FloatArray]] = (
        defaultdict(list)
    )
    heldout_groups: dict[tuple[_ResidualCondition, ResidualCandidateKind], list[FloatArray]] = (
        defaultdict(list)
    )
    for case in selected:
        data = _load_case_data(manifest_path, case)
        candidates: tuple[tuple[ResidualCandidateKind, ConventionHypothesis], ...] = (
            ("truth", case.truth),
            *counterfactual_hypotheses(case.truth),
        )
        condition = _condition(case)
        noise_seed = case.measurement_conditions.noise_seed
        if noise_seed is None:
            raise RuntimeError(f"{case.case_id}: selected noisy case has no noise seed.")
        ####
        for candidate_kind, hypothesis in candidates:
            scored = score_hypothesis(
                data,
                hypothesis,
                case.config,
                retain_interval_details=True,
            )
            if scored.interval_error_rotvecs is None:
                raise RuntimeError(f"{case.case_id}: scorer omitted requested interval residuals.")
            ####
            residual_groups[(condition, candidate_kind)].append(scored.interval_error_rotvecs)
            if seed_partition == "all":
                parity_groups = training_groups if noise_seed % 2 == 0 else heldout_groups
                parity_groups[(condition, candidate_kind)].append(scored.interval_error_rotvecs)
            ####
        ####
    ####
    summaries = tuple(
        _summarize_condition(condition, candidate_kind, records)
        for (condition, candidate_kind), records in sorted(
            residual_groups.items(),
            key=lambda item: (
                item[0][0].motion_profile_id or "",
                item[0][0].sample_count,
                item[0][0].truth_noise_model.orientation_ar1_coefficient,
                item[0][0].truth_noise_model.rate_ar1_coefficient,
                item[0][0].truth_hypothesis_id,
                item[0][1],
            ),
        )
    )
    selected_noise_seeds = tuple(
        sorted(
            {
                case.measurement_conditions.noise_seed
                for case in selected
                if case.measurement_conditions.noise_seed is not None
            }
        )
    )
    covariance_holdout = (
        CovarianceHoldoutReport(
            training_noise_seeds=tuple(seed for seed in selected_noise_seeds if seed % 2 == 0),
            heldout_noise_seeds=tuple(seed for seed in selected_noise_seeds if seed % 2 != 0),
            summaries=_candidate_covariance_holdout_summaries(training_groups, heldout_groups),
        )
        if seed_partition == "all"
        else None
    )
    return ResidualModelValidationReport(
        manifest=str(manifest_path),
        max_cases_per_condition=max_cases_per_condition,
        seed_partition=seed_partition,
        selected_case_count=len(selected),
        selected_noise_seeds=selected_noise_seeds,
        assumptions=(
            "Only seeded, uniform, synchronized quaternion trajectories without dropout or "
            "latency are included.",
            "Seed partitions select cases by noise_seed parity; model noise parameters come from "
            "the manifest and are known to this synthetic evaluator, not estimated.",
            "Each trajectory is scored under its labeled convention and six counterfactuals, "
            "each changing exactly one of attitude direction, quaternion storage order, rate "
            "frame, rate order, rate sign, or rate unit.",
            "Pairwise reference scores use the labeled truth's stated scalar covariance model; "
            "the rate-unit alternative adjusts rate-noise scale, while other alternatives share "
            "the truth covariance as a first-order approximation.",
            "Candidate-vs-truth 3x3 lag-block differences are estimated from the same selected "
            "replicates and are descriptive, in-sample diagnostics rather than held-out results.",
            "When all seeds are selected, candidate covariance blocks are fitted on even noise "
            "seeds and evaluated on independent odd noise seeds; conditions with fewer than two "
            "cases in either partition are omitted from this holdout summary.",
            "The held-out covariance comparison evaluates empirical second-order block transfer; "
            "it is not a held-out likelihood or a calibrated inference score.",
            "The first-order covariance remains an approximation for transformed orientation "
            "conventions; these scores are diagnostics, not calibrated package likelihoods.",
            "Residual covariance is estimated after centering independent replicates at each "
            "interval; the replicate-mean residual is reported separately.",
            "Whitened sum-of-squares diagnostics are also reported without centering; this exposes "
            "any nonzero mean residual in the full scoring pipeline.",
            "The comparison validates second-order residual statistics, not calibrated likelihood "
            "ratios, confidence labels, or recommendation thresholds.",
        ),
        condition_summaries=summaries,
        candidate_covariance_differences=_candidate_covariance_difference_summaries(
            residual_groups
        ),
        covariance_holdout=covariance_holdout,
        pairwise_likelihood_summaries=_pairwise_likelihood_summaries(residual_groups),
    )
####


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="synthetic validation manifest")
    parser.add_argument("--output", type=Path, required=True, help="new JSON output path")
    parser.add_argument(
        "--max-cases-per-condition",
        type=int,
        default=200,
        help="bounded seeded trials per motion/noise condition",
    )
    parser.add_argument(
        "--seed-partition",
        choices=("all", "even", "odd"),
        default="all",
        help="select all seeds or one parity for disjoint held-out synthetic checks",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    ####
    report = validate_manifest(
        args.manifest,
        max_cases_per_condition=args.max_cases_per_condition,
        seed_partition=args.seed_partition,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    case_count = sum(
        item.case_count for item in report.condition_summaries if item.candidate_kind == "truth"
    )
    print(f"Wrote covariance diagnostics for {case_count} cases to {args.output}")
####


if __name__ == "__main__":
    main()
####
