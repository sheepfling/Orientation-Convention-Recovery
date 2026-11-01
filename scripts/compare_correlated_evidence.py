"""Compare package evidence with known-noise covariance references on synthetic data."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import cache
from math import isclose, sqrt
from pathlib import Path

from numpy import asarray, float64
from numpy.linalg import cholesky, cond
from numpy.typing import NDArray

from attitude_convention_inference import infer_conventions
from attitude_convention_inference.core import TrajectoryData
from attitude_convention_inference.io import load_csv_trajectory
from attitude_convention_inference.models import CandidateResult
from attitude_convention_inference.scoring import score_hypothesis
from attitude_convention_inference.statistics import log_likelihood_advantage_by_prefix
from scripts.correlated_evidence import (
    IntervalResidualNoiseModel,
    covariance_adjusted_log_likelihood_ratio,
)
from scripts.validate_corpus import ValidationCase, ValidationManifest

FloatArray = NDArray[float64]


@dataclass(frozen=True, slots=True)
class _StudyCondition:
    motion_profile_id: str | None
    sample_count: int
    truth_hypothesis_id: str
    noise_model: IntervalResidualNoiseModel
####


@dataclass(frozen=True, slots=True)
class CaseEvidenceComparison:
    case_id: str
    condition: _StudyCondition
    package_iid_log_likelihood_ratio: float
    known_noise_iid_marginal_log_likelihood_ratio: float
    known_noise_colored_log_likelihood_ratio: float
    package_evidence_sufficient: bool
    data_winner_correct: bool
    recommendation_correct: bool | None
####


@dataclass(frozen=True, slots=True)
class ConditionEvidenceSummary:
    condition: _StudyCondition
    case_count: int
    mean_package_iid_log_likelihood_ratio: float
    mean_known_noise_iid_marginal_log_likelihood_ratio: float
    mean_known_noise_colored_log_likelihood_ratio: float
    covariance_condition_number: float
    package_recommendation_count: int
    package_recommendation_miss_count: int
####


@dataclass(frozen=True, slots=True)
class EvidenceComparisonReport:
    manifest: str
    max_cases_per_condition: int
    assumptions: tuple[str, ...]
    condition_summaries: tuple[ConditionEvidenceSummary, ...]
    cases: tuple[CaseEvidenceComparison, ...]
####


@cache
def _factorized_covariance(
    condition: _StudyCondition,
    interval_count: int,
) -> tuple[FloatArray, FloatArray, float]:
    covariance = condition.noise_model.covariance_matrix(interval_count)
    covariance_cholesky = asarray(cholesky(covariance), dtype=float64)
    return covariance, covariance_cholesky, float(cond(covariance))
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
            f"{case.case_id}: covariance reference requires synchronized, uniform, "
            "no-dropout measurements."
        )
    ####
    if (
        conditions.orientation_noise_sigma_rad is None
        or conditions.orientation_noise_ar1_coefficient is None
        or conditions.rate_noise_sigma_rad_s is None
        or conditions.rate_noise_ar1_coefficient is None
    ):
        message = f"{case.case_id}: known noise parameters are required for covariance analysis."
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


def _case_condition(case: ValidationCase) -> _StudyCondition:
    sample_count = case.measurement_conditions.sample_count
    if sample_count is None:
        raise ValueError(f"{case.case_id}: sample_count is required.")
    ####
    return _StudyCondition(
        motion_profile_id=case.measurement_conditions.motion_profile_id,
        sample_count=sample_count,
        truth_hypothesis_id=case.truth.identifier(),
        noise_model=_noise_model(case),
    )
####


def _select_cases(
    cases: Sequence[ValidationCase],
    max_cases_per_condition: int,
) -> tuple[ValidationCase, ...]:
    selected: list[ValidationCase] = []
    condition_counts: dict[_StudyCondition, int] = defaultdict(int)
    for case in cases:
        if case.measurement_conditions.noise_seed is None:
            continue
        ####
        key = _case_condition(case)
        if condition_counts[key] >= max_cases_per_condition:
            continue
        ####
        selected.append(case)
        condition_counts[key] += 1
    ####
    if not selected:
        raise ValueError("The manifest contains no eligible seeded noisy cases.")
    ####
    return tuple(selected)
####


def _class_representatives(results: Sequence[CandidateResult]) -> tuple[CandidateResult, ...]:
    representatives: dict[str, CandidateResult] = {}
    for result in sorted(results, key=lambda item: item.data_rank):
        representatives.setdefault(result.equivalence_class_id, result)
    ####
    return tuple(representatives.values())
####


def _load_case_data(manifest_path: Path, case: ValidationCase) -> TrajectoryData:
    if case.representation.value != "quaternion":
        raise ValueError(f"{case.case_id}: this reference currently requires quaternion data.")
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
    conditions = case.measurement_conditions
    sample_period_s = conditions.sample_period_s
    if sample_period_s is None:
        raise ValueError(f"{case.case_id}: sample_period_s is required.")
    ####
    if data.time_s is None or data.sample_count != conditions.sample_count:
        raise ValueError(f"{case.case_id}: CSV sampling metadata is incomplete or inconsistent.")
    ####
    if not all(
        isclose(float(data.time_s[index + 1] - data.time_s[index]), sample_period_s, abs_tol=1.0e-9)
        for index in range(data.sample_count - 1)
    ):
        raise ValueError(f"{case.case_id}: CSV timestamps are not uniformly spaced as declared.")
    ####
    return data
####


def _compare_case(manifest_path: Path, case: ValidationCase) -> CaseEvidenceComparison:
    data = _load_case_data(manifest_path, case)
    report = infer_conventions(data, config=case.config)
    representatives = _class_representatives(report.rankings)
    if len(representatives) < 2:
        raise ValueError(f"{case.case_id}: at least two evidence classes are required.")
    ####
    residuals_by_class: dict[str, FloatArray] = {}
    for result in representatives:
        scored = score_hypothesis(
            data,
            result.hypothesis,
            case.config,
            retain_interval_details=True,
        )
        if scored.interval_error_rotvecs is None:
            raise RuntimeError("Scoring did not retain requested interval residual vectors.")
        ####
        residuals_by_class[result.equivalence_class_id] = scored.interval_error_rotvecs
    ####
    winner = representatives[0]
    winner_errors = residuals_by_class[winner.equivalence_class_id]
    competitor_errors = tuple(
        residuals_by_class[result.equivalence_class_id] for result in representatives[1:]
    )
    iid_recomputed = float(
        log_likelihood_advantage_by_prefix(
            winner_errors,
            competitor_errors,
            report.diagnostics.estimated_noise_sigma_rad,
        )[-1]
    )
    if not isclose(
        iid_recomputed,
        report.diagnostics.evidence_log_likelihood_ratio,
        rel_tol=1.0e-8,
        abs_tol=1.0e-6,
    ):
        raise RuntimeError(
            f"{case.case_id}: recomputed i.i.d. evidence does not match the package report."
        )
    ####
    condition = _case_condition(case)
    covariance, covariance_factor, _ = _factorized_covariance(condition, winner_errors.shape[0])
    known_noise_iid_sigma = sqrt(float(covariance[0, 0]))
    known_noise_iid = float(
        log_likelihood_advantage_by_prefix(
            winner_errors,
            competitor_errors,
            known_noise_iid_sigma,
        )[-1]
    )
    known_noise_colored = covariance_adjusted_log_likelihood_ratio(
        winner_errors,
        competitor_errors,
        covariance_factor,
    )
    truth_class = next(
        item.equivalence_class_id
        for item in report.rankings
        if item.hypothesis.identifier() == case.truth.identifier()
    )
    return CaseEvidenceComparison(
        case_id=case.case_id,
        condition=condition,
        package_iid_log_likelihood_ratio=report.diagnostics.evidence_log_likelihood_ratio,
        known_noise_iid_marginal_log_likelihood_ratio=known_noise_iid,
        known_noise_colored_log_likelihood_ratio=known_noise_colored,
        package_evidence_sufficient=report.diagnostics.evidence_sufficient,
        data_winner_correct=winner.equivalence_class_id == truth_class,
        recommendation_correct=report.decision.recommended_class_id == truth_class
        if report.decision.recommended_class_id is not None
        else None,
    )
####


def _summarize_cases(
    comparisons: Sequence[CaseEvidenceComparison],
) -> tuple[ConditionEvidenceSummary, ...]:
    grouped: dict[_StudyCondition, list[CaseEvidenceComparison]] = defaultdict(list)
    for comparison in comparisons:
        grouped[comparison.condition].append(comparison)
    ####
    summaries: list[ConditionEvidenceSummary] = []
    for condition, cases in sorted(
        grouped.items(),
        key=lambda item: (
            item[0].truth_hypothesis_id,
            item[0].noise_model.orientation_sigma_rad,
            item[0].noise_model.orientation_ar1_coefficient,
            item[0].noise_model.rate_ar1_coefficient,
        ),
    ):
        count = len(cases)
        interval_count = condition.sample_count - 1
        _, _, covariance_condition_number = _factorized_covariance(condition, interval_count)
        summaries.append(
            ConditionEvidenceSummary(
                condition=condition,
                case_count=count,
                mean_package_iid_log_likelihood_ratio=sum(
                    item.package_iid_log_likelihood_ratio for item in cases
                )
                / count,
                mean_known_noise_iid_marginal_log_likelihood_ratio=sum(
                    item.known_noise_iid_marginal_log_likelihood_ratio for item in cases
                )
                / count,
                mean_known_noise_colored_log_likelihood_ratio=sum(
                    item.known_noise_colored_log_likelihood_ratio for item in cases
                )
                / count,
                covariance_condition_number=covariance_condition_number,
                package_recommendation_count=sum(
                    item.recommendation_correct is not None for item in cases
                ),
                package_recommendation_miss_count=sum(
                    item.recommendation_correct is False for item in cases
                ),
            )
        )
    ####
    return tuple(summaries)
####


def compare_manifest(
    manifest_path: Path,
    *,
    max_cases_per_condition: int = 20,
) -> EvidenceComparisonReport:
    """Run known-noise covariance references on a bounded manifest subset."""

    if max_cases_per_condition < 1:
        raise ValueError("max_cases_per_condition must be positive.")
    ####
    manifest = ValidationManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    selected_cases = _select_cases(manifest.cases, max_cases_per_condition)
    comparisons = tuple(_compare_case(manifest_path, case) for case in selected_cases)
    return EvidenceComparisonReport(
        manifest=str(manifest_path),
        max_cases_per_condition=max_cases_per_condition,
        assumptions=(
            "Noise parameters are known from the synthetic manifest, not estimated from the cases.",
            "The covariance is a first-order small-angle approximation for isotropic orientation "
            "noise differences plus trapezoid-integrated rate noise.",
            "The colored reference uses the labeled truth's scalar covariance for all ranked "
            "classes; transformed-candidate covariance differences are not modeled here.",
            "The reference comparison uses the full-recording residual likelihood only; it does "
            "not evaluate prefix stability or change package decisions.",
        ),
        condition_summaries=_summarize_cases(comparisons),
        cases=comparisons,
    )
####


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="synthetic validation manifest")
    parser.add_argument("--output", type=Path, required=True, help="new JSON output path")
    parser.add_argument(
        "--max-cases-per-condition",
        type=int,
        default=20,
        help="bounded seeded trials to compare per noise condition",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    ####
    comparison = compare_manifest(
        args.manifest,
        max_cases_per_condition=args.max_cases_per_condition,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(comparison), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(comparison.cases)} case comparisons to {args.output}")
####


if __name__ == "__main__":
    main()
####
