from __future__ import annotations

import json
from pathlib import Path

import pytest

from attitude_convention_inference import (
    AttitudeRepresentation,
    InferenceDecision,
    InferenceDecisionStatus,
)
from scripts.compare_corpus_assessments import compare_assessments, load_assessment
from scripts.validate_corpus import (
    CaseAssessment,
    CorpusAssessment,
    EvaluationEnvironment,
    InferenceDiagnostics,
    MeasurementConditions,
    ValidationSummary,
)

IMPLEMENTATION_HASH = "a" * 64


def _case(
    *,
    case_id: str,
    motion_seed: int,
    noise_seed: int,
    orientation_latency_s: float,
    evidence_contenders: tuple[str, ...] = ("truth",),
    data_winners: tuple[str, ...] = ("truth",),
    operational_class_id: str = "truth",
) -> CaseAssessment:
    decision = InferenceDecision.from_evidence(
        evidence_sufficient=True,
        evidence_contender_class_ids=evidence_contenders,
    )
    recommendation_correct = (
        decision.recommended_class_id == "truth"
        if decision.status is InferenceDecisionStatus.RECOMMENDATION
        else None
    )
    case_outcome = (
        "resolved_correct" if recommendation_correct is True else "ambiguous_contains_truth"
    )
    return CaseAssessment(
        case_id=case_id,
        label_source="test truth",
        representation=AttitudeRepresentation.QUATERNION,
        measurement_conditions=MeasurementConditions(
            motion_profile_id="piecewise-test",
            motion_seed=motion_seed,
            noise_seed=noise_seed,
            orientation_latency_s=orientation_latency_s,
            sample_count=21,
            sample_period_s=0.1,
        ),
        sample_count=21,
        hypothesis_count=192,
        truth_class_id="truth",
        evidence_contender_class_ids=evidence_contenders,
        data_winning_class_ids=data_winners,
        data_winner_correct="truth" in data_winners,
        operational_class_id=operational_class_id,
        operational_correct=operational_class_id == "truth",
        decision=decision,
        recommendation_correct=recommendation_correct,
        outcome=case_outcome,
        confidence_label="moderate",
        evidence_sufficient=True,
        inference_diagnostics=InferenceDiagnostics(
            confidence=0.5,
            estimated_noise_sigma_rad=0.01,
            noise_residual_count=20,
            noise_lag1_autocorrelation=0.0,
            evidence_intervals_used=10,
            evidence_fraction=0.5,
            evidence_log_likelihood_ratio=2.0,
            warnings=(),
        ),
    )
####


def _assessment(cases: tuple[CaseAssessment, ...], *, implementation_hash: str) -> CorpusAssessment:
    recommendation_count = sum(
        case.decision.status is InferenceDecisionStatus.RECOMMENDATION for case in cases
    )
    abstention_count = len(cases) - recommendation_count
    summary = ValidationSummary(
        case_count=len(cases),
        resolved_correct=sum(case.outcome == "resolved_correct" for case in cases),
        ambiguous_contains_truth=sum(case.outcome == "ambiguous_contains_truth" for case in cases),
        evidence_miss=sum(case.outcome == "evidence_miss" for case in cases),
        data_winner_miss=sum(not case.data_winner_correct for case in cases),
        operational_miss=sum(not case.operational_correct for case in cases),
        evidence_insufficient=sum(not case.evidence_sufficient for case in cases),
        incorrect_evidence_sufficient=0,
        recommendation_count=recommendation_count,
        abstention_count=abstention_count,
        recommendation_miss=sum(case.recommendation_correct is False for case in cases),
    )
    return CorpusAssessment(
        environment=EvaluationEnvironment(
            python="3.12",
            platform="test",
            numpy="2.0",
            scipy="1.13",
            package="0.0.1a0",
            implementation_sha256=implementation_hash,
        ),
        summary=summary,
        cases=cases,
        condition_summaries=(),
    )
####


def test_compares_exact_pairs_and_reports_condition_and_motion_seed_changes() -> None:
    baseline_cases = (
        _case(case_id="case-a", motion_seed=10, noise_seed=100, orientation_latency_s=0.0),
        _case(case_id="case-b", motion_seed=11, noise_seed=101, orientation_latency_s=0.0),
    )
    candidate_cases = (
        _case(
            case_id="case-a",
            motion_seed=10,
            noise_seed=100,
            orientation_latency_s=0.013,
            evidence_contenders=("truth", "other"),
        ),
        _case(
            case_id="case-b",
            motion_seed=11,
            noise_seed=101,
            orientation_latency_s=0.013,
            data_winners=("truth", "other"),
        ),
    )
    report = compare_assessments(
        _assessment(baseline_cases, implementation_hash=IMPLEMENTATION_HASH),
        _assessment(candidate_cases, implementation_hash="b" * 64),
    )

    assert report.paired_case_count == 2
    assert report.seeded_noisy_case_count == 2
    assert not report.implementation_sha256_matches
    assert report.seeded_noisy_cases.baseline.recommendation_count == 2
    assert report.seeded_noisy_cases.candidate.recommendation_count == 1
    assert report.seeded_noisy_cases.candidate.correct_recommendation_count == 1
    assert report.seeded_noisy_cases.candidate.recommendation_miss_count == 0
    assert report.seeded_noisy_cases.changes.data_winner_set_change_count == 1
    assert report.seeded_noisy_cases.changes.evidence_set_expansion_count == 1
    assert report.seeded_noisy_cases.changes.decision_status_change_count == 1
    assert len(report.by_motion_seed) == 2
    assert len(report.measurement_condition_differences) == 1
    difference = report.measurement_condition_differences[0]
    assert difference.field_name == "orientation_latency_s"
    assert difference.changed_case_count == 2
    assert difference.baseline_values == (0.0,)
    assert difference.candidate_values == (0.013,)
####


def test_reports_named_three_axis_calibration_changes() -> None:
    baseline_case = _case(
        case_id="case-a", motion_seed=10, noise_seed=100, orientation_latency_s=0.0
    )
    candidate_case = baseline_case.model_copy(
        update={
            "measurement_conditions": baseline_case.measurement_conditions.model_copy(
                update={"rate_sensor_bias_rad_s": (0.15, -0.105, 0.06)}
            )
        }
    )

    report = compare_assessments(
        _assessment((baseline_case,), implementation_hash=IMPLEMENTATION_HASH),
        _assessment((candidate_case,), implementation_hash=IMPLEMENTATION_HASH),
    )

    difference = next(
        item
        for item in report.measurement_condition_differences
        if item.field_name == "rate_sensor_bias_rad_s"
    )
    assert difference.changed_case_count == 1
    assert difference.baseline_values == (None,)
    assert difference.candidate_values == ((0.15, -0.105, 0.06),)
####


def test_rejects_assessments_with_different_case_id_sets() -> None:
    baseline = _assessment(
        (_case(case_id="case-a", motion_seed=10, noise_seed=100, orientation_latency_s=0.0),),
        implementation_hash=IMPLEMENTATION_HASH,
    )
    candidate = _assessment(
        (_case(case_id="case-b", motion_seed=10, noise_seed=100, orientation_latency_s=0.013),),
        implementation_hash=IMPLEMENTATION_HASH,
    )

    with pytest.raises(ValueError, match="case IDs must match exactly"):
        compare_assessments(baseline, candidate)
    ####
####


def test_rejects_truth_label_changes_for_a_paired_case() -> None:
    baseline_case = _case(
        case_id="case-a", motion_seed=10, noise_seed=100, orientation_latency_s=0.0
    )
    candidate_case = baseline_case.model_copy(update={"truth_class_id": "other-truth"})
    baseline = _assessment((baseline_case,), implementation_hash=IMPLEMENTATION_HASH)
    candidate = _assessment((candidate_case,), implementation_hash=IMPLEMENTATION_HASH)

    with pytest.raises(ValueError, match="independently labeled truth class differs"):
        compare_assessments(baseline, candidate)
    ####
####


def test_rejects_duplicate_case_ids() -> None:
    duplicate_case = _case(
        case_id="case-a", motion_seed=10, noise_seed=100, orientation_latency_s=0.0
    )
    baseline = _assessment(
        (duplicate_case, duplicate_case),
        implementation_hash=IMPLEMENTATION_HASH,
    )
    candidate = _assessment((duplicate_case,), implementation_hash=IMPLEMENTATION_HASH)

    with pytest.raises(ValueError, match="duplicate case_id"):
        compare_assessments(baseline, candidate)
    ####
####


def test_reads_serialized_assessment_after_verifying_computed_durations(
    tmp_path: Path,
) -> None:
    assessment = _assessment(
        (_case(case_id="case-a", motion_seed=10, noise_seed=100, orientation_latency_s=0.0),),
        implementation_hash=IMPLEMENTATION_HASH,
    )
    assessment_path = tmp_path / "assessment.json"
    assessment_path.write_text(assessment.model_dump_json(), encoding="utf-8")

    loaded = load_assessment(assessment_path)

    assert loaded.cases[0].measurement_conditions.duration_s == pytest.approx(2.0)
####


def test_rejects_inconsistent_serialized_duration(tmp_path: Path) -> None:
    assessment = _assessment(
        (_case(case_id="case-a", motion_seed=10, noise_seed=100, orientation_latency_s=0.0),),
        implementation_hash=IMPLEMENTATION_HASH,
    )
    payload = json.loads(assessment.model_dump_json())
    payload["cases"][0]["measurement_conditions"]["duration_s"] = 99.0
    assessment_path = tmp_path / "assessment.json"
    assessment_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="serialized duration_s does not match"):
        load_assessment(assessment_path)
    ####
####
