"""Compare seeded synthetic OU assessments across nested sample cadences."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from math import isclose, isfinite
from pathlib import Path
from typing import TypeAlias, cast

from attitude_convention_inference import InferenceDecisionStatus, RateFrame
from scripts.compare_corpus_assessments import (
    AssessmentComparisonReport,
    AssessmentOutcomeCounts,
    compare_assessments,
    load_assessment,
)
from scripts.validate_corpus import (
    CaseAssessment,
    CorpusAssessment,
    MeasurementConditions,
    ValidationCase,
    ValidationManifest,
)
from scripts.validation_noise import validate_ou_sample_period

ConditionScalar: TypeAlias = tuple[int | float, ...] | str | int | float | bool | None

__all__ = [
    "CadenceAssessmentComparisonReport",
    "CadenceDescriptor",
    "CadenceRiskMarginComparison",
    "CadenceRiskMarginGroupComparison",
    "CadenceRiskMarginReplicationComparison",
    "RateSensorCalibrationDescriptor",
    "RiskMarginDistribution",
    "compare_cadence_assessments",
    "main",
]

_CADENCE_DERIVED_CONDITIONS = frozenset(
    {
        "sample_count",
        "observed_sample_count",
        "dropped_sample_count",
        "sample_period_s",
        "orientation_noise_ar1_coefficient",
        "rate_noise_ar1_coefficient",
    }
)


@dataclass(frozen=True, slots=True)
class CadenceDescriptor:
    """Scheduled sample grid for the seeded noisy cases in one assessment."""

    sample_count: int
    sample_period_s: float
    duration_s: float
####


@dataclass(frozen=True, slots=True)
class RateSensorCalibrationDescriptor:
    """Named calibration metadata carried by cadence-stratified comparisons."""

    calibration_id: str | None
    misalignment_angle_deg: float | None
    misalignment_axis: tuple[float, float, float] | None
    bias_rad_s: tuple[float, float, float] | None
    scale_by_axis: tuple[float, float, float] | None
    polarity_by_axis: tuple[int, int, int] | None
    clip_abs_rad_s: float | None

    def ordering_key(self) -> str:
        """Return a deterministic key that handles absent and present values."""

        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
    ####
####


@dataclass(frozen=True, slots=True)
class RiskMarginDistribution:
    """Distribution of nearest-alternative log geometric-risk deltas."""

    observed_case_count: int
    tenth_percentile: float | None
    median: float | None
    ninetieth_percentile: float | None
####


@dataclass(frozen=True, slots=True)
class CadenceRiskMarginComparison:
    """Paired change in the nearest alternative's relative geometric fit."""

    paired_case_count: int
    baseline_outcomes: AssessmentOutcomeCounts
    candidate_outcomes: AssessmentOutcomeCounts
    baseline: RiskMarginDistribution
    candidate: RiskMarginDistribution
    candidate_minus_baseline: RiskMarginDistribution
    baseline_unique_contender_count: int
    candidate_unique_contender_count: int
    unique_to_multiple_count: int
    multiple_to_unique_count: int
    nearest_alternative_class_changed_count: int
####


@dataclass(frozen=True, slots=True)
class CadenceRiskMarginGroupComparison:
    """Risk-margin comparison for one motion seed, truth frame, and calibration."""

    motion_profile_id: str | None
    motion_seed: int | None
    truth_rate_frame: str | None
    rate_sensor_calibration: RateSensorCalibrationDescriptor
    comparison: CadenceRiskMarginComparison
####


@dataclass(frozen=True, slots=True)
class CadenceRiskMarginReplicationComparison:
    """Risk margins for one seeded motion/noise, acquisition, and calibration stratum."""

    motion_profile_id: str | None
    motion_seed: int | None
    noise_seed: int | None
    truth_rate_frame: str | None
    timestamp_jitter_amplitude_s: float | None
    dropout_probability: float | None
    orientation_latency_s: float | None
    rate_latency_s: float | None
    rate_sensor_calibration: RateSensorCalibrationDescriptor
    comparison: CadenceRiskMarginComparison
####


@dataclass(frozen=True, slots=True)
class CadenceAssessmentComparisonReport:
    """Paired case outcomes plus the nested cadence design that produced them."""

    schema_version: int
    baseline_cadence: CadenceDescriptor
    candidate_cadence: CadenceDescriptor
    shared_timestamp_stride_ratio: int
    inference_configurations_match: bool
    shared_noisy_input_case_count: int
    shared_common_timestamp_row_count: int
    maximum_shared_input_absolute_difference: float
    outcome_comparison: AssessmentComparisonReport
    risk_margin_comparison: CadenceRiskMarginComparison
    risk_margin_by_motion_seed_and_truth_frame: tuple[CadenceRiskMarginGroupComparison, ...]
    risk_margin_by_motion_noise_seed_and_truth_frame: tuple[
        CadenceRiskMarginReplicationComparison, ...
    ]
####


@dataclass(frozen=True, slots=True)
class _CadenceIndependentCaseIdentity:
    """Case identity with cadence-derived conditions intentionally omitted."""

    stable_case_id: str
    label_source: str
    source_provenance_json: str | None
    representation: str
    truth_class_id: str
    stable_conditions: tuple[tuple[str, ConditionScalar], ...]
####


@dataclass(frozen=True, slots=True)
class _PairedRiskCase:
    baseline: CaseAssessment
    candidate: CaseAssessment
    motion_profile_id: str | None
    motion_seed: int | None
    noise_seed: int | None
    truth_rate_frame: str | None
####


@dataclass(frozen=True, slots=True)
class _RiskMarginReplicationKey:
    """Named condition dimensions defining one replication-level comparison."""

    motion_profile_id: str | None
    motion_seed: int | None
    noise_seed: int | None
    truth_rate_frame: str | None
    timestamp_jitter_amplitude_s: float | None
    dropout_probability: float | None
    orientation_latency_s: float | None
    rate_latency_s: float | None
    rate_sensor_calibration: RateSensorCalibrationDescriptor

    def ordering_key(self) -> tuple[str, int, int, str, float, float, float, float, str]:
        """Return a deterministic ordering with explicit sentinels for absent values."""

        return (
            self.motion_profile_id or "",
            -1 if self.motion_seed is None else self.motion_seed,
            -1 if self.noise_seed is None else self.noise_seed,
            self.truth_rate_frame or "",
            -1.0
            if self.timestamp_jitter_amplitude_s is None
            else self.timestamp_jitter_amplitude_s,
            -1.0 if self.dropout_probability is None else self.dropout_probability,
            -1.0 if self.orientation_latency_s is None else self.orientation_latency_s,
            -1.0 if self.rate_latency_s is None else self.rate_latency_s,
            self.rate_sensor_calibration.ordering_key(),
        )
    ####
####


@dataclass(frozen=True, slots=True)
class _MeasurementRow:
    """One recorded timestamp and its orientation/rate measurement tuple."""

    timestamp_s: float
    values: tuple[float, ...]
####


def _cadence_descriptor(
    assessment: CorpusAssessment,
    side: str,
) -> CadenceDescriptor:
    seeded_cases = tuple(
        case for case in assessment.cases if case.measurement_conditions.noise_seed is not None
    )
    if not seeded_cases:
        raise ValueError(f"{side} assessment contains no seeded noisy cases")
    ####

    grids: set[tuple[int | None, float | None]] = {
        (
            case.measurement_conditions.sample_count,
            case.measurement_conditions.sample_period_s,
        )
        for case in seeded_cases
    }
    if len(grids) != 1:
        raise ValueError(f"{side} assessment must contain exactly one noisy sample grid")
    ####
    sample_count, sample_period_s = next(iter(grids))
    if sample_count is None or sample_period_s is None or sample_count < 2:
        raise ValueError(f"{side} assessment has invalid noisy sample-grid metadata")
    ####
    duration_s = (sample_count - 1) * sample_period_s
    for case in seeded_cases:
        conditions = case.measurement_conditions
        if (
            conditions.observed_sample_count is None
            or conditions.dropped_sample_count is None
            or case.sample_count != conditions.observed_sample_count
            or conditions.observed_sample_count + conditions.dropped_sample_count != sample_count
        ):
            raise ValueError(
                f"{side} {case.case_id}: observed and dropped row counts must match metadata"
            )
        ####
        if (
            conditions.orientation_noise_correlation_time_s is None
            or conditions.rate_noise_correlation_time_s is None
        ):
            raise ValueError(f"{side} {case.case_id}: both channels must use time-domain OU noise")
        ####
    ####

    return CadenceDescriptor(
        sample_count=sample_count,
        sample_period_s=sample_period_s,
        duration_s=duration_s,
    )
####


def _cadence_stride(sample_period_s: float, side: str) -> int:
    try:
        return validate_ou_sample_period(sample_period_s)
    except ValueError as error:
        raise ValueError(
            f"{side} cadence is not supported by the shared OU grid: {error}"
        ) from error
    ####
####


def _validate_nested_cadences(
    baseline: CadenceDescriptor,
    candidate: CadenceDescriptor,
) -> int:
    baseline_stride = _cadence_stride(baseline.sample_period_s, "baseline")
    candidate_stride = _cadence_stride(candidate.sample_period_s, "candidate")
    if baseline_stride == candidate_stride:
        raise ValueError("baseline and candidate assessments must use different cadences")
    ####
    larger_stride = max(baseline_stride, candidate_stride)
    smaller_stride = min(baseline_stride, candidate_stride)
    if larger_stride % smaller_stride != 0:
        raise ValueError("cadences must be nested on the shared OU grid")
    ####
    if not isclose(
        baseline.duration_s,
        candidate.duration_s,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError("cadence assessments must cover the same duration")
    ####
    return larger_stride // smaller_stride
####


def _case_identity(case: CaseAssessment) -> _CadenceIndependentCaseIdentity:
    conditions = case.measurement_conditions
    stable_case_id = case.case_id
    if conditions.noise_seed is not None and conditions.sample_count is not None:
        stable_case_id = stable_case_id.replace(
            f"-n{conditions.sample_count}-",
            "-n*-",
        )
    ####
    stable_conditions = tuple(
        (
            field_name,
            cast(ConditionScalar, getattr(conditions, field_name)),
        )
        for field_name in MeasurementConditions.model_fields
        if field_name not in _CADENCE_DERIVED_CONDITIONS
    )
    return _CadenceIndependentCaseIdentity(
        stable_case_id=stable_case_id,
        label_source=case.label_source,
        source_provenance_json=(
            case.source_provenance.model_dump_json() if case.source_provenance is not None else None
        ),
        representation=case.representation.value,
        truth_class_id=case.truth_class_id,
        stable_conditions=stable_conditions,
    )
####


def _normalized_assessment(
    assessment: CorpusAssessment,
    side: str,
) -> CorpusAssessment:
    cases_by_identity: dict[_CadenceIndependentCaseIdentity, CaseAssessment] = {}
    normalized_cases: list[CaseAssessment] = []
    for case in assessment.cases:
        identity = _case_identity(case)
        if identity in cases_by_identity:
            raise ValueError(f"{side} assessment has duplicate cadence-independent case identities")
        ####
        cases_by_identity[identity] = case
        identity_json = json.dumps(asdict(identity), sort_keys=True, separators=(",", ":"))
        normalized_id = "cadence-pair-" + hashlib.sha256(identity_json.encode("utf-8")).hexdigest()
        normalized_cases.append(case.model_copy(update={"case_id": normalized_id}))
    ####
    return assessment.model_copy(update={"cases": tuple(normalized_cases)})
####


def _validate_matching_identities(
    baseline: CorpusAssessment,
    candidate: CorpusAssessment,
) -> None:
    baseline_identities = {_case_identity(case) for case in baseline.cases}
    candidate_identities = {_case_identity(case) for case in candidate.cases}
    if baseline_identities != candidate_identities:
        baseline_only = len(baseline_identities - candidate_identities)
        candidate_only = len(candidate_identities - baseline_identities)
        raise ValueError(
            "cadence assessments must contain the same seeded motion/noise/truth cases; "
            f"baseline-only={baseline_only}, candidate-only={candidate_only}"
        )
    ####
####


def _manifest_case_index(
    assessment: CorpusAssessment,
    manifest: ValidationManifest,
    side: str,
) -> dict[str, ValidationCase]:
    manifest_cases: dict[str, ValidationCase] = {}
    for case in manifest.cases:
        if case.case_id in manifest_cases:
            raise ValueError(f"{side} manifest contains duplicate case_id {case.case_id!r}")
        ####
        manifest_cases[case.case_id] = case
    ####
    assessment_cases = {case.case_id: case for case in assessment.cases}
    if manifest_cases.keys() != assessment_cases.keys():
        raise ValueError(f"{side} assessment and manifest case IDs do not match exactly")
    ####
    for case_id, assessed_case in assessment_cases.items():
        manifest_case = manifest_cases[case_id]
        if manifest_case.representation != assessed_case.representation:
            raise ValueError(f"{side} {case_id}: assessment representation differs from manifest")
        ####
        if manifest_case.label_source != assessed_case.label_source:
            raise ValueError(f"{side} {case_id}: assessment label source differs from manifest")
        ####
        if manifest_case.source_provenance != assessed_case.source_provenance:
            raise ValueError(f"{side} {case_id}: assessment provenance differs from manifest")
        ####
        if manifest_case.measurement_conditions != assessed_case.measurement_conditions:
            raise ValueError(f"{side} {case_id}: assessment conditions differ from manifest")
        ####
    ####
    return manifest_cases
####


def _validate_matching_manifest_configs(
    baseline: CorpusAssessment,
    candidate: CorpusAssessment,
    baseline_manifest_cases: dict[str, ValidationCase],
    candidate_manifest_cases: dict[str, ValidationCase],
) -> None:
    candidate_cases_by_identity = {_case_identity(case): case for case in candidate.cases}
    for baseline_case in baseline.cases:
        identity = _case_identity(baseline_case)
        candidate_case = candidate_cases_by_identity[identity]
        baseline_manifest_case = baseline_manifest_cases[baseline_case.case_id]
        candidate_manifest_case = candidate_manifest_cases[candidate_case.case_id]
        if baseline_manifest_case.truth != candidate_manifest_case.truth:
            raise ValueError(
                "cadence-paired cases must use the same independently labeled truth: "
                f"{baseline_case.case_id!r} and {candidate_case.case_id!r}"
            )
        ####
        baseline_config = baseline_manifest_case.config
        candidate_config = candidate_manifest_case.config
        if baseline_config != candidate_config:
            raise ValueError(
                "cadence-paired cases must use the same inference configuration: "
                f"{baseline_case.case_id!r} and {candidate_case.case_id!r}"
            )
        ####
    ####
####


def _linear_quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ####
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (ordered[upper_index] - ordered[lower_index])
####


def _risk_margin_distribution(values: list[float]) -> RiskMarginDistribution:
    return RiskMarginDistribution(
        observed_case_count=len(values),
        tenth_percentile=_linear_quantile(values, 0.1),
        median=_linear_quantile(values, 0.5),
        ninetieth_percentile=_linear_quantile(values, 0.9),
    )
####


def _outcome_counts(cases: tuple[CaseAssessment, ...]) -> AssessmentOutcomeCounts:
    recommendations = tuple(
        case for case in cases if case.decision.status is InferenceDecisionStatus.RECOMMENDATION
    )
    evidence_misses = tuple(
        case for case in cases if case.truth_class_id not in case.evidence_contender_class_ids
    )
    return AssessmentOutcomeCounts(
        case_count=len(cases),
        recommendation_count=len(recommendations),
        correct_recommendation_count=sum(
            case.recommendation_correct is True for case in recommendations
        ),
        abstention_count=sum(
            case.decision.status is InferenceDecisionStatus.ABSTENTION for case in cases
        ),
        data_winner_miss_count=sum(not case.data_winner_correct for case in cases),
        operational_miss_count=sum(not case.operational_correct for case in cases),
        evidence_miss_count=len(evidence_misses),
        evidence_insufficient_count=sum(not case.evidence_sufficient for case in cases),
        incorrect_evidence_sufficient_count=sum(
            case.evidence_sufficient for case in evidence_misses
        ),
        recommendation_miss_count=sum(case.recommendation_correct is False for case in cases),
    )
####


def _risk_margin_comparison(
    pairs: tuple[_PairedRiskCase, ...],
) -> CadenceRiskMarginComparison:
    baseline_values = [
        case.baseline.nearest_alternative_log_risk_delta
        for case in pairs
        if case.baseline.nearest_alternative_log_risk_delta is not None
    ]
    candidate_values = [
        case.candidate.nearest_alternative_log_risk_delta
        for case in pairs
        if case.candidate.nearest_alternative_log_risk_delta is not None
    ]
    paired_changes = [
        case.candidate.nearest_alternative_log_risk_delta
        - case.baseline.nearest_alternative_log_risk_delta
        for case in pairs
        if case.baseline.nearest_alternative_log_risk_delta is not None
        and case.candidate.nearest_alternative_log_risk_delta is not None
    ]
    return CadenceRiskMarginComparison(
        paired_case_count=len(pairs),
        baseline_outcomes=_outcome_counts(tuple(case.baseline for case in pairs)),
        candidate_outcomes=_outcome_counts(tuple(case.candidate for case in pairs)),
        baseline=_risk_margin_distribution(baseline_values),
        candidate=_risk_margin_distribution(candidate_values),
        candidate_minus_baseline=_risk_margin_distribution(paired_changes),
        baseline_unique_contender_count=sum(
            len(case.baseline.evidence_contender_class_ids) == 1 for case in pairs
        ),
        candidate_unique_contender_count=sum(
            len(case.candidate.evidence_contender_class_ids) == 1 for case in pairs
        ),
        unique_to_multiple_count=sum(
            len(case.baseline.evidence_contender_class_ids) == 1
            and len(case.candidate.evidence_contender_class_ids) > 1
            for case in pairs
        ),
        multiple_to_unique_count=sum(
            len(case.baseline.evidence_contender_class_ids) > 1
            and len(case.candidate.evidence_contender_class_ids) == 1
            for case in pairs
        ),
        nearest_alternative_class_changed_count=sum(
            case.baseline.nearest_alternative_class_id is not None
            and case.candidate.nearest_alternative_class_id is not None
            and case.baseline.nearest_alternative_class_id
            != case.candidate.nearest_alternative_class_id
            for case in pairs
        ),
    )
####


def _risk_margin_groups(
    pairs: tuple[_PairedRiskCase, ...],
) -> tuple[CadenceRiskMarginGroupComparison, ...]:
    grouped: dict[
        tuple[str | None, int | None, str | None, RateSensorCalibrationDescriptor],
        list[_PairedRiskCase],
    ] = defaultdict(list)
    for case in pairs:
        conditions = case.baseline.measurement_conditions
        calibration = RateSensorCalibrationDescriptor(
            calibration_id=conditions.rate_sensor_calibration_id,
            misalignment_angle_deg=conditions.rate_sensor_misalignment_angle_deg,
            misalignment_axis=conditions.rate_sensor_misalignment_axis,
            bias_rad_s=conditions.rate_sensor_bias_rad_s,
            scale_by_axis=conditions.rate_sensor_scale_by_axis,
            polarity_by_axis=conditions.rate_sensor_polarity_by_axis,
            clip_abs_rad_s=conditions.rate_sensor_clip_abs_rad_s,
        )
        grouped[
            (case.motion_profile_id, case.motion_seed, case.truth_rate_frame, calibration)
        ].append(case)
    ####
    return tuple(
        CadenceRiskMarginGroupComparison(
            motion_profile_id=motion_profile_id,
            motion_seed=motion_seed,
            truth_rate_frame=truth_rate_frame,
            rate_sensor_calibration=calibration,
            comparison=_risk_margin_comparison(tuple(group_cases)),
        )
        for (motion_profile_id, motion_seed, truth_rate_frame, calibration), group_cases in sorted(
            grouped.items(),
            key=lambda item: (
                item[0][0] or "",
                -1 if item[0][1] is None else item[0][1],
                item[0][2] or "",
                item[0][3].ordering_key(),
            ),
        )
    )
####


def _risk_margin_replications(
    pairs: tuple[_PairedRiskCase, ...],
) -> tuple[CadenceRiskMarginReplicationComparison, ...]:
    grouped: dict[_RiskMarginReplicationKey, list[_PairedRiskCase]] = defaultdict(list)
    for case in pairs:
        conditions = case.baseline.measurement_conditions
        key = _RiskMarginReplicationKey(
            motion_profile_id=case.motion_profile_id,
            motion_seed=case.motion_seed,
            noise_seed=case.noise_seed,
            truth_rate_frame=case.truth_rate_frame,
            timestamp_jitter_amplitude_s=conditions.timestamp_jitter_amplitude_s,
            dropout_probability=conditions.dropout_probability,
            orientation_latency_s=conditions.orientation_latency_s,
            rate_latency_s=conditions.rate_latency_s,
            rate_sensor_calibration=RateSensorCalibrationDescriptor(
                calibration_id=conditions.rate_sensor_calibration_id,
                misalignment_angle_deg=conditions.rate_sensor_misalignment_angle_deg,
                misalignment_axis=conditions.rate_sensor_misalignment_axis,
                bias_rad_s=conditions.rate_sensor_bias_rad_s,
                scale_by_axis=conditions.rate_sensor_scale_by_axis,
                polarity_by_axis=conditions.rate_sensor_polarity_by_axis,
                clip_abs_rad_s=conditions.rate_sensor_clip_abs_rad_s,
            ),
        )
        grouped[key].append(case)
    ####
    return tuple(
        CadenceRiskMarginReplicationComparison(
            motion_profile_id=key.motion_profile_id,
            motion_seed=key.motion_seed,
            noise_seed=key.noise_seed,
            truth_rate_frame=key.truth_rate_frame,
            timestamp_jitter_amplitude_s=key.timestamp_jitter_amplitude_s,
            dropout_probability=key.dropout_probability,
            orientation_latency_s=key.orientation_latency_s,
            rate_latency_s=key.rate_latency_s,
            rate_sensor_calibration=key.rate_sensor_calibration,
            comparison=_risk_margin_comparison(tuple(group_cases)),
        )
        for key, group_cases in sorted(
            grouped.items(),
            key=lambda item: item[0].ordering_key(),
        )
    )
####


def _read_measurement_rows(
    case: ValidationCase,
    manifest_root: Path,
    side: str,
) -> tuple[tuple[str, ...], dict[int, _MeasurementRow]]:
    conditions = case.measurement_conditions
    sample_period_s = conditions.sample_period_s
    sample_count = conditions.sample_count
    if sample_period_s is None or sample_count is None:
        raise ValueError(f"{side} {case.case_id}: source manifest lacks sample-grid metadata")
    ####
    if case.csv.is_absolute():
        raise ValueError(f"{side} {case.case_id}: cadence comparison requires relative CSV paths")
    ####
    root = manifest_root.resolve()
    csv_path = (root / case.csv).resolve()
    if not csv_path.is_relative_to(root):
        raise ValueError(f"{side} {case.case_id}: CSV path escapes its manifest directory")
    ####
    if case.time_column is None or case.orientation_columns is None or case.rate_columns is None:
        raise ValueError(
            f"{side} {case.case_id}: manifest must name time, orientation, and rate columns"
        )
    ####
    jitter_amplitude_s = conditions.timestamp_jitter_amplitude_s
    if jitter_amplitude_s is None:
        raise ValueError(f"{side} {case.case_id}: timestamp-jitter amplitude must be known")
    ####
    clock_scale = 1.0 + (conditions.timestamp_clock_rate_error_ppm or 0.0) * 1.0e-6
    if not isfinite(clock_scale) or clock_scale <= 0.0:
        raise ValueError(
            f"{side} {case.case_id}: timestamp clock scale must be positive and finite"
        )
    ####
    measurement_columns = (*case.orientation_columns, *case.rate_columns)
    required_columns = (case.time_column, *measurement_columns)
    if len(required_columns) != len(set(required_columns)):
        raise ValueError(f"{side} {case.case_id}: measurement column names must be unique")
    ####

    rows_by_time_ns: dict[int, _MeasurementRow] = {}
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)
        if header is None or len(header) != len(set(header)):
            raise ValueError(f"{side} {case.case_id}: CSV header is empty or has duplicate columns")
        ####
        missing = set(required_columns) - set(header)
        if missing:
            raise ValueError(f"{side} {case.case_id}: CSV is missing columns {sorted(missing)!r}")
        ####
        time_index = header.index(case.time_column)
        measurement_indices = tuple(header.index(name) for name in measurement_columns)
        previous_sample_index = -1
        expected_count = conditions.observed_sample_count
        if expected_count is None:
            raise ValueError(f"{side} {case.case_id}: observed row count must be known")
        ####
        observed_count = 0
        for row_index, row in enumerate(reader):
            if len(row) != len(header):
                raise ValueError(f"{side} {case.case_id}: CSV row {row_index + 2} has wrong width")
            ####
            try:
                timestamp_s = float(row[time_index])
                values = tuple(float(row[index]) for index in measurement_indices)
            except ValueError as error:
                raise ValueError(
                    f"{side} {case.case_id}: CSV row {row_index + 2} is not numeric"
                ) from error
            ####
            if not isfinite(timestamp_s) or any(not isfinite(value) for value in values):
                raise ValueError(f"{side} {case.case_id}: CSV contains non-finite measurements")
            ####
            nominal_timestamp_s = timestamp_s / clock_scale
            sample_index = round(nominal_timestamp_s / sample_period_s)
            if not 0 <= sample_index < sample_count:
                raise ValueError(f"{side} {case.case_id}: timestamp lies outside its sample grid")
            ####
            expected_timestamp_s = sample_index * sample_period_s
            if abs(nominal_timestamp_s - expected_timestamp_s) > jitter_amplitude_s + 1.0e-12:
                raise ValueError(
                    f"{side} {case.case_id}: timestamp exceeds its declared jitter bound"
                )
            ####
            if sample_index <= previous_sample_index:
                raise ValueError(
                    f"{side} {case.case_id}: timestamps do not follow the increasing sample grid"
                )
            ####
            previous_sample_index = sample_index
            timestamp_ns = round(timestamp_s * 1_000_000_000)
            if timestamp_ns in rows_by_time_ns:
                raise ValueError(f"{side} {case.case_id}: CSV has duplicate timestamps")
            ####
            rows_by_time_ns[timestamp_ns] = _MeasurementRow(
                timestamp_s=timestamp_s,
                values=values,
            )
            observed_count += 1
        ####
    ####
    if observed_count != expected_count:
        raise ValueError(
            f"{side} {case.case_id}: CSV has {observed_count} rows, expected {expected_count}"
        )
    ####
    return measurement_columns, rows_by_time_ns
####


def _verify_shared_measurements(
    baseline: CorpusAssessment,
    candidate: CorpusAssessment,
    baseline_manifest_cases: dict[str, ValidationCase],
    candidate_manifest_cases: dict[str, ValidationCase],
    baseline_manifest_root: Path,
    candidate_manifest_root: Path,
) -> tuple[int, int, float]:
    candidate_by_identity = {_case_identity(case): case for case in candidate.cases}
    shared_case_count = 0
    shared_row_count = 0
    maximum_difference = 0.0
    for baseline_case in baseline.cases:
        if baseline_case.measurement_conditions.noise_seed is None:
            continue
        ####
        identity = _case_identity(baseline_case)
        candidate_case = candidate_by_identity[identity]
        baseline_manifest_case = baseline_manifest_cases[baseline_case.case_id]
        candidate_manifest_case = candidate_manifest_cases[candidate_case.case_id]
        baseline_columns, baseline_rows = _read_measurement_rows(
            baseline_manifest_case,
            baseline_manifest_root,
            "baseline",
        )
        candidate_columns, candidate_rows = _read_measurement_rows(
            candidate_manifest_case,
            candidate_manifest_root,
            "candidate",
        )
        if baseline_columns != candidate_columns:
            raise ValueError(f"{baseline_case.case_id}: measurement column layouts differ")
        ####
        common_times = baseline_rows.keys() & candidate_rows.keys()
        if len(common_times) != min(len(baseline_rows), len(candidate_rows)):
            raise ValueError(f"{baseline_case.case_id}: one sample grid is not nested in the other")
        ####
        for timestamp_ns in common_times:
            baseline_values = baseline_rows[timestamp_ns]
            candidate_values = candidate_rows[timestamp_ns]
            if baseline_values.timestamp_s != candidate_values.timestamp_s:
                raise ValueError(
                    f"{baseline_case.case_id}: common-time timestamps differ at {timestamp_ns} ns"
                )
            ####
            for baseline_value, candidate_value in zip(
                baseline_values.values,
                candidate_values.values,
                strict=True,
            ):
                difference = abs(baseline_value - candidate_value)
                maximum_difference = max(maximum_difference, difference)
                if difference > 2.0e-12:
                    raise ValueError(
                        f"{baseline_case.case_id}: common-time measurement values differ "
                        f"by {difference:.6g} at {timestamp_ns} ns"
                    )
                ####
            ####
        ####
        shared_case_count += 1
        shared_row_count += len(common_times)
    ####
    return shared_case_count, shared_row_count, maximum_difference
####


def compare_cadence_assessments(
    baseline: CorpusAssessment,
    candidate: CorpusAssessment,
    baseline_manifest: ValidationManifest,
    candidate_manifest: ValidationManifest,
    baseline_manifest_root: Path,
    candidate_manifest_root: Path,
    *,
    baseline_path: str | None = None,
    candidate_path: str | None = None,
) -> CadenceAssessmentComparisonReport:
    """Compare seeded OU cases on nested cadences with verified shared-time inputs."""

    if baseline.environment != candidate.environment:
        raise ValueError("cadence assessments must use the same runtime and implementation")
    ####
    baseline_cadence = _cadence_descriptor(baseline, "baseline")
    candidate_cadence = _cadence_descriptor(candidate, "candidate")
    stride_ratio = _validate_nested_cadences(baseline_cadence, candidate_cadence)
    _validate_matching_identities(baseline, candidate)
    baseline_manifest_cases = _manifest_case_index(baseline, baseline_manifest, "baseline")
    candidate_manifest_cases = _manifest_case_index(candidate, candidate_manifest, "candidate")
    _validate_matching_manifest_configs(
        baseline,
        candidate,
        baseline_manifest_cases,
        candidate_manifest_cases,
    )
    candidate_by_identity = {_case_identity(case): case for case in candidate.cases}
    risk_pairs = tuple(
        _PairedRiskCase(
            baseline=baseline_case,
            candidate=candidate_by_identity[_case_identity(baseline_case)],
            motion_profile_id=baseline_case.measurement_conditions.motion_profile_id,
            motion_seed=baseline_case.measurement_conditions.motion_seed,
            noise_seed=baseline_case.measurement_conditions.noise_seed,
            truth_rate_frame=RateFrame(
                baseline_manifest_cases[baseline_case.case_id].truth.rate_frame
            ).value,
        )
        for baseline_case in baseline.cases
        if baseline_case.measurement_conditions.noise_seed is not None
    )
    (
        shared_input_case_count,
        shared_timestamp_row_count,
        maximum_input_difference,
    ) = _verify_shared_measurements(
        baseline,
        candidate,
        baseline_manifest_cases,
        candidate_manifest_cases,
        baseline_manifest_root,
        candidate_manifest_root,
    )
    normalized_baseline = _normalized_assessment(baseline, "baseline")
    normalized_candidate = _normalized_assessment(candidate, "candidate")
    outcome_comparison = compare_assessments(
        normalized_baseline,
        normalized_candidate,
        baseline_path=baseline_path,
        candidate_path=candidate_path,
    )
    return CadenceAssessmentComparisonReport(
        schema_version=6,
        baseline_cadence=baseline_cadence,
        candidate_cadence=candidate_cadence,
        shared_timestamp_stride_ratio=stride_ratio,
        inference_configurations_match=True,
        shared_noisy_input_case_count=shared_input_case_count,
        shared_common_timestamp_row_count=shared_timestamp_row_count,
        maximum_shared_input_absolute_difference=maximum_input_difference,
        outcome_comparison=outcome_comparison,
        risk_margin_comparison=_risk_margin_comparison(risk_pairs),
        risk_margin_by_motion_seed_and_truth_frame=_risk_margin_groups(risk_pairs),
        risk_margin_by_motion_noise_seed_and_truth_frame=_risk_margin_replications(risk_pairs),
    )
####


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="baseline assessment JSON")
    parser.add_argument("candidate", type=Path, help="candidate assessment JSON")
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new JSON report path")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    ####
    try:
        report = compare_cadence_assessments(
            load_assessment(args.baseline),
            load_assessment(args.candidate),
            ValidationManifest.model_validate(
                json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
            ),
            ValidationManifest.model_validate(
                json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
            ),
            args.baseline_manifest.parent,
            args.candidate_manifest.parent,
            baseline_path=str(args.baseline),
            candidate_path=str(args.candidate),
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    ####
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    print(
        f"Compared {report.outcome_comparison.paired_case_count} cadence-paired cases "
        f"(stride ratio {report.shared_timestamp_stride_ratio}) to {args.output}"
    )
####


if __name__ == "__main__":
    main()
####
