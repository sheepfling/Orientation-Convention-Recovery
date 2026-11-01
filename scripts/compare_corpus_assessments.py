"""Compare two validation assessments that contain the same labeled cases."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import isclose
from pathlib import Path
from typing import TypeAlias, cast

from attitude_convention_inference import InferenceDecisionStatus
from scripts.validate_corpus import (
    CaseAssessment,
    CorpusAssessment,
    MeasurementConditions,
)

ConditionValue: TypeAlias = tuple[int | float, ...] | str | int | float | bool | None
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

__all__ = [
    "AssessmentChangeCounts",
    "AssessmentComparisonReport",
    "AssessmentOutcomeCounts",
    "AssessmentRuntime",
    "ConditionDifference",
    "MotionSeedComparison",
    "PairedAssessmentMetrics",
    "compare_assessments",
    "load_assessment",
    "main",
]


@dataclass(frozen=True, slots=True)
class AssessmentOutcomeCounts:
    """Outcome counts for one side of a paired assessment comparison."""

    case_count: int
    recommendation_count: int
    correct_recommendation_count: int
    abstention_count: int
    data_winner_miss_count: int
    operational_miss_count: int
    evidence_miss_count: int
    evidence_insufficient_count: int
    incorrect_evidence_sufficient_count: int
    recommendation_miss_count: int
####


@dataclass(frozen=True, slots=True)
class AssessmentChangeCounts:
    """Paired changes in winners, evidence sets, and public decisions."""

    data_winner_set_change_count: int
    operational_class_change_count: int
    evidence_set_change_count: int
    evidence_set_expansion_count: int
    evidence_set_contraction_count: int
    evidence_set_replacement_count: int
    decision_status_change_count: int
    decision_reason_change_count: int
    recommended_class_change_count: int
####


@dataclass(frozen=True, slots=True)
class PairedAssessmentMetrics:
    """Named outcome and paired-change summaries for a defined case cohort."""

    case_count: int
    baseline: AssessmentOutcomeCounts
    candidate: AssessmentOutcomeCounts
    changes: AssessmentChangeCounts
####


@dataclass(frozen=True, slots=True)
class MotionSeedComparison:
    """Paired metrics for one motion profile and seed."""

    motion_profile_id: str | None
    motion_seed: int | None
    metrics: PairedAssessmentMetrics
####


@dataclass(frozen=True, slots=True)
class ConditionDifference:
    """A measurement-condition field that differs among paired case IDs."""

    field_name: str
    changed_case_count: int
    baseline_values: tuple[ConditionValue, ...]
    candidate_values: tuple[ConditionValue, ...]
####


@dataclass(frozen=True, slots=True)
class AssessmentRuntime:
    """Runtime and package versions recorded by one assessment."""

    python: str
    platform: str
    numpy: str
    scipy: str
    package: str
    implementation_sha256: str
####


@dataclass(frozen=True, slots=True)
class AssessmentComparisonReport:
    """Machine-readable summary of two exactly paired validation assessments."""

    schema_version: int
    baseline_path: str | None
    candidate_path: str | None
    baseline_environment: AssessmentRuntime
    candidate_environment: AssessmentRuntime
    baseline_implementation_sha256: str
    candidate_implementation_sha256: str
    implementation_sha256_matches: bool
    paired_case_count: int
    seeded_noisy_case_count: int
    all_cases: PairedAssessmentMetrics
    seeded_noisy_cases: PairedAssessmentMetrics
    by_motion_seed: tuple[MotionSeedComparison, ...]
    measurement_condition_differences: tuple[ConditionDifference, ...]
####


@dataclass(frozen=True, slots=True)
class _AssessmentPair:
    baseline: CaseAssessment
    candidate: CaseAssessment
####


def _case_index(assessment: CorpusAssessment, side: str) -> dict[str, CaseAssessment]:
    cases_by_id: dict[str, CaseAssessment] = {}
    for case in assessment.cases:
        if case.case_id in cases_by_id:
            raise ValueError(f"{side} assessment contains duplicate case_id {case.case_id!r}")
        ####
        cases_by_id[case.case_id] = case
    ####
    return cases_by_id
####


def _pair_cases(
    baseline: CorpusAssessment,
    candidate: CorpusAssessment,
) -> tuple[_AssessmentPair, ...]:
    baseline_by_id = _case_index(baseline, "baseline")
    candidate_by_id = _case_index(candidate, "candidate")
    baseline_ids = set(baseline_by_id)
    candidate_ids = set(candidate_by_id)
    if baseline_ids != candidate_ids:
        baseline_only = sorted(baseline_ids - candidate_ids)
        candidate_only = sorted(candidate_ids - baseline_ids)
        raise ValueError(
            "assessment case IDs must match exactly; "
            f"baseline-only={baseline_only[:5]!r}, candidate-only={candidate_only[:5]!r}"
        )
    ####

    pairs: list[_AssessmentPair] = []
    for case_id in sorted(baseline_ids):
        baseline_case = baseline_by_id[case_id]
        candidate_case = candidate_by_id[case_id]
        if baseline_case.truth_class_id != candidate_case.truth_class_id:
            raise ValueError(f"{case_id}: independently labeled truth class differs")
        ####
        if baseline_case.representation != candidate_case.representation:
            raise ValueError(f"{case_id}: trajectory representation differs")
        ####
        if baseline_case.label_source != candidate_case.label_source:
            raise ValueError(f"{case_id}: truth-label source differs")
        ####
        if baseline_case.source_provenance != candidate_case.source_provenance:
            raise ValueError(f"{case_id}: source provenance differs")
        ####
        pairs.append(_AssessmentPair(baseline_case, candidate_case))
    ####
    return tuple(pairs)
####


def _outcome_counts(cases: Sequence[CaseAssessment]) -> AssessmentOutcomeCounts:
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
        recommendation_miss_count=sum(
            case.recommendation_correct is False for case in recommendations
        ),
    )
####


def _change_counts(pairs: Sequence[_AssessmentPair]) -> AssessmentChangeCounts:
    data_winner_set_change_count = 0
    operational_class_change_count = 0
    evidence_set_change_count = 0
    evidence_set_expansion_count = 0
    evidence_set_contraction_count = 0
    evidence_set_replacement_count = 0
    decision_status_change_count = 0
    decision_reason_change_count = 0
    recommended_class_change_count = 0

    for pair in pairs:
        baseline = pair.baseline
        candidate = pair.candidate
        baseline_winners = set(baseline.data_winning_class_ids)
        candidate_winners = set(candidate.data_winning_class_ids)
        data_winner_set_change_count += baseline_winners != candidate_winners
        operational_class_change_count += (
            baseline.operational_class_id != candidate.operational_class_id
        )

        baseline_evidence = set(baseline.evidence_contender_class_ids)
        candidate_evidence = set(candidate.evidence_contender_class_ids)
        if baseline_evidence != candidate_evidence:
            evidence_set_change_count += 1
            if baseline_evidence < candidate_evidence:
                evidence_set_expansion_count += 1
            elif candidate_evidence < baseline_evidence:
                evidence_set_contraction_count += 1
            else:
                evidence_set_replacement_count += 1
            ####
        ####

        decision_status_change_count += baseline.decision.status != candidate.decision.status
        decision_reason_change_count += baseline.decision.reason != candidate.decision.reason
        recommended_class_change_count += (
            baseline.decision.recommended_class_id != candidate.decision.recommended_class_id
        )
    ####

    return AssessmentChangeCounts(
        data_winner_set_change_count=data_winner_set_change_count,
        operational_class_change_count=operational_class_change_count,
        evidence_set_change_count=evidence_set_change_count,
        evidence_set_expansion_count=evidence_set_expansion_count,
        evidence_set_contraction_count=evidence_set_contraction_count,
        evidence_set_replacement_count=evidence_set_replacement_count,
        decision_status_change_count=decision_status_change_count,
        decision_reason_change_count=decision_reason_change_count,
        recommended_class_change_count=recommended_class_change_count,
    )
####


def _paired_metrics(pairs: Sequence[_AssessmentPair]) -> PairedAssessmentMetrics:
    return PairedAssessmentMetrics(
        case_count=len(pairs),
        baseline=_outcome_counts(tuple(pair.baseline for pair in pairs)),
        candidate=_outcome_counts(tuple(pair.candidate for pair in pairs)),
        changes=_change_counts(pairs),
    )
####


def _runtime_description(assessment: CorpusAssessment) -> AssessmentRuntime:
    environment = assessment.environment
    return AssessmentRuntime(
        python=environment.python,
        platform=environment.platform,
        numpy=environment.numpy,
        scipy=environment.scipy,
        package=environment.package,
        implementation_sha256=environment.implementation_sha256,
    )
####


def _condition_scalar(value: object) -> ConditionValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    ####
    if isinstance(value, tuple):
        values = cast(tuple[object, ...], value)
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in values):
            return cast(tuple[int | float, ...], values)
        ####
    ####
    raise TypeError("measurement condition is not a JSON-compatible value")
####


def _condition_differences(
    pairs: Sequence[_AssessmentPair],
) -> tuple[ConditionDifference, ...]:
    differences: list[ConditionDifference] = []
    for field_name in MeasurementConditions.model_fields:
        baseline_values: list[ConditionValue] = []
        candidate_values: list[ConditionValue] = []
        changed_case_count = 0
        for pair in pairs:
            baseline_value = _condition_scalar(
                getattr(pair.baseline.measurement_conditions, field_name)
            )
            candidate_value = _condition_scalar(
                getattr(pair.candidate.measurement_conditions, field_name)
            )
            if baseline_value == candidate_value:
                continue
            ####
            changed_case_count += 1
            if baseline_value not in baseline_values:
                baseline_values.append(baseline_value)
            ####
            if candidate_value not in candidate_values:
                candidate_values.append(candidate_value)
            ####
        ####
        if changed_case_count:
            differences.append(
                ConditionDifference(
                    field_name=field_name,
                    changed_case_count=changed_case_count,
                    baseline_values=tuple(baseline_values),
                    candidate_values=tuple(candidate_values),
                )
            )
        ####
    ####
    return tuple(differences)
####


def compare_assessments(
    baseline: CorpusAssessment,
    candidate: CorpusAssessment,
    *,
    baseline_path: str | None = None,
    candidate_path: str | None = None,
) -> AssessmentComparisonReport:
    """Compare assessments only when they contain the same independently labeled cases."""

    pairs = _pair_cases(baseline, candidate)
    noisy_pairs = tuple(
        pair
        for pair in pairs
        if pair.baseline.measurement_conditions.noise_seed is not None
        or pair.candidate.measurement_conditions.noise_seed is not None
    )

    pairs_by_motion_seed: dict[tuple[str | None, int | None], list[_AssessmentPair]] = defaultdict(
        list
    )
    for pair in noisy_pairs:
        conditions = pair.baseline.measurement_conditions
        pairs_by_motion_seed[(conditions.motion_profile_id, conditions.motion_seed)].append(pair)
    ####
    by_motion_seed = tuple(
        MotionSeedComparison(
            motion_profile_id=motion_profile_id,
            motion_seed=motion_seed,
            metrics=_paired_metrics(seed_pairs),
        )
        for (motion_profile_id, motion_seed), seed_pairs in sorted(
            pairs_by_motion_seed.items(),
            key=lambda item: (item[0][0] or "", -1 if item[0][1] is None else item[0][1]),
        )
    )

    baseline_hash = baseline.environment.implementation_sha256
    candidate_hash = candidate.environment.implementation_sha256
    return AssessmentComparisonReport(
        schema_version=2,
        baseline_path=baseline_path,
        candidate_path=candidate_path,
        baseline_environment=_runtime_description(baseline),
        candidate_environment=_runtime_description(candidate),
        baseline_implementation_sha256=baseline_hash,
        candidate_implementation_sha256=candidate_hash,
        implementation_sha256_matches=baseline_hash == candidate_hash,
        paired_case_count=len(pairs),
        seeded_noisy_case_count=len(noisy_pairs),
        all_cases=_paired_metrics(pairs),
        seeded_noisy_cases=_paired_metrics(noisy_pairs),
        by_motion_seed=by_motion_seed,
        measurement_condition_differences=_condition_differences(pairs),
    )
####


def load_assessment(path: Path) -> CorpusAssessment:
    """Load a corpus assessment while checking serialized derived durations."""

    payload = cast(JsonValue, json.loads(path.read_text(encoding="utf-8")))
    return CorpusAssessment.model_validate(_normalize_serialized_duration(payload))
####


def _normalize_serialized_duration(payload: JsonValue) -> JsonValue:
    """Verify and remove the computed duration included in serialized assessments."""

    if not isinstance(payload, dict):
        return payload
    ####
    normalized_payload = dict(payload)
    for section in ("cases", "condition_summaries"):
        entries = normalized_payload.get(section)
        if not isinstance(entries, list):
            continue
        ####
        normalized_entries: list[JsonValue] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                normalized_entries.append(entry)
                continue
            ####
            normalized_entry = dict(entry)
            conditions = normalized_entry.get("measurement_conditions")
            if isinstance(conditions, dict) and "duration_s" in conditions:
                normalized_conditions = dict(conditions)
                reported_duration = normalized_conditions.pop("duration_s")
                sample_count = normalized_conditions.get("sample_count")
                sample_period_s = normalized_conditions.get("sample_period_s")
                if sample_count is None or sample_period_s is None:
                    expected_duration: float | None = None
                elif (
                    isinstance(sample_count, int)
                    and not isinstance(sample_count, bool)
                    and isinstance(sample_period_s, (int, float))
                    and not isinstance(sample_period_s, bool)
                ):
                    expected_duration = (sample_count - 1) * sample_period_s
                else:
                    raise ValueError(
                        f"{section}[{index}]: cannot verify serialized duration_s "
                        "against invalid sample_count/sample_period_s values"
                    )
                ####
                duration_matches = (reported_duration is None and expected_duration is None) or (
                    isinstance(reported_duration, (int, float))
                    and not isinstance(reported_duration, bool)
                    and expected_duration is not None
                    and isclose(
                        reported_duration,
                        expected_duration,
                        rel_tol=1.0e-12,
                        abs_tol=1.0e-12,
                    )
                )
                if not duration_matches:
                    raise ValueError(
                        f"{section}[{index}]: serialized duration_s does not match "
                        "sample_count and sample_period_s"
                    )
                ####
                normalized_entry["measurement_conditions"] = normalized_conditions
            ####
            normalized_entries.append(normalized_entry)
        ####
        normalized_payload[section] = normalized_entries
    ####
    return normalized_payload
####


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="baseline assessment JSON")
    parser.add_argument("candidate", type=Path, help="candidate assessment JSON")
    parser.add_argument("--output", type=Path, required=True, help="new JSON report path")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    ####
    try:
        report = compare_assessments(
            load_assessment(args.baseline),
            load_assessment(args.candidate),
            baseline_path=str(args.baseline),
            candidate_path=str(args.candidate),
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    ####
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    print(
        f"Compared {report.paired_case_count} paired cases "
        f"({report.seeded_noisy_case_count} seeded noisy) to {args.output}"
    )
####


if __name__ == "__main__":
    main()
####
