from __future__ import annotations

import csv
from math import exp
from pathlib import Path

import pytest

from attitude_convention_inference import (
    AttitudeRepresentation,
    ConventionHypothesis,
    InferenceConfig,
    InferenceDecision,
    InferenceDecisionStatus,
)
from scripts.compare_cadence_assessments import (
    CadenceAssessmentComparisonReport,
    compare_cadence_assessments,
)
from scripts.generate_validation_corpus import NOISE_PROFILES, MotionProfile, generate_corpus
from scripts.validate_corpus import (
    CaseAssessment,
    CorpusAssessment,
    EvaluationEnvironment,
    InferenceDiagnostics,
    MeasurementConditions,
    ValidationCase,
    ValidationManifest,
    ValidationSummary,
    validate_corpus,
)

IMPLEMENTATION_HASH = "a" * 64
TRUTH = ConventionHypothesis(
    attitude_direction="body_to_world",
    rate_frame="body",
    rate_order="pqr",
    rate_unit="radians_per_second",
    quaternion_order="scalar_last",
)


def _case(
    *,
    case_id: str,
    sample_count: int,
    sample_period_s: float,
    evidence_contenders: tuple[str, ...] = ("truth",),
    orientation_noise_correlation_time_s: float | None = 0.05,
    timestamp_jitter_amplitude_s: float = 0.0,
    dropout_probability: float | None = 0.0,
    noise_seed: int = 100,
    nearest_alternative_log_risk_delta: float | None = 2.5,
    nearest_alternative_class_id: str | None = "other",
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
    return CaseAssessment(
        case_id=case_id,
        label_source="seeded OU truth",
        representation=AttitudeRepresentation.QUATERNION,
        measurement_conditions=MeasurementConditions(
            motion_profile_id="piecewise-test",
            motion_seed=10,
            timestamp_jitter_amplitude_s=timestamp_jitter_amplitude_s,
            dropout_probability=dropout_probability,
            dropped_sample_count=0,
            orientation_noise_sigma_rad=0.006,
            rate_noise_sigma_rad_s=0.05,
            orientation_noise_ar1_coefficient=exp(-sample_period_s / 0.05),
            rate_noise_ar1_coefficient=exp(-sample_period_s / 0.1),
            orientation_noise_correlation_time_s=orientation_noise_correlation_time_s,
            rate_noise_correlation_time_s=0.1,
            noise_seed=noise_seed,
            sample_count=sample_count,
            observed_sample_count=sample_count,
            sample_period_s=sample_period_s,
        ),
        sample_count=sample_count,
        hypothesis_count=192,
        truth_class_id=TRUTH.identifier(),
        evidence_contender_class_ids=evidence_contenders,
        nearest_alternative_class_id=nearest_alternative_class_id,
        nearest_alternative_log_risk_delta=nearest_alternative_log_risk_delta,
        data_winning_class_ids=("truth",),
        data_winner_correct=True,
        operational_class_id="truth",
        operational_correct=True,
        decision=decision,
        recommendation_correct=recommendation_correct,
        outcome=(
            "resolved_correct" if recommendation_correct is True else "ambiguous_contains_truth"
        ),
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


def _assessment(
    case: CaseAssessment,
    *,
    implementation_hash: str = IMPLEMENTATION_HASH,
) -> CorpusAssessment:
    recommended = case.decision.status is InferenceDecisionStatus.RECOMMENDATION
    return CorpusAssessment(
        environment=EvaluationEnvironment(
            python="3.12",
            platform="test",
            numpy="2.0",
            scipy="1.13",
            package="0.0.1a0",
            implementation_sha256=implementation_hash,
        ),
        summary=ValidationSummary(
            case_count=1,
            resolved_correct=int(recommended),
            ambiguous_contains_truth=int(not recommended),
            evidence_miss=0,
            data_winner_miss=0,
            operational_miss=0,
            evidence_insufficient=0,
            incorrect_evidence_sufficient=0,
            recommendation_count=int(recommended),
            abstention_count=int(not recommended),
            recommendation_miss=0,
        ),
        cases=(case,),
        condition_summaries=(),
    )
####


def _manifest(
    assessment: CorpusAssessment,
    *,
    config: InferenceConfig | None = None,
    truth: ConventionHypothesis = TRUTH,
) -> ValidationManifest:
    return ValidationManifest(
        schema_version=5,
        cases=tuple(
            ValidationCase(
                case_id=case.case_id,
                csv=Path(f"{case.case_id}.csv"),
                label_source=case.label_source,
                representation=case.representation,
                truth=truth,
                config=config or InferenceConfig(),
                measurement_conditions=case.measurement_conditions,
                orientation_columns=("a0", "a1", "a2", "a3"),
                rate_columns=("r0", "r1", "r2"),
                time_column="time_s",
                source_provenance=case.source_provenance,
            )
            for case in assessment.cases
        ),
    )
####


def _write_measurements(
    manifest: ValidationManifest,
    root: Path,
    *,
    offset: float = 0.0,
    timestamp_offset_s: float = 0.0,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for case in manifest.cases:
        conditions = case.measurement_conditions
        if conditions.sample_count is None or conditions.sample_period_s is None:
            continue
        ####
        if (
            case.time_column is None
            or case.orientation_columns is None
            or case.rate_columns is None
        ):
            raise AssertionError("test fixture case has no measurement columns")
        ####
        columns = (case.time_column, *case.orientation_columns, *case.rate_columns)
        with (root / case.csv).open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(columns)
            for sample_index in range(conditions.sample_count):
                time_s = sample_index * conditions.sample_period_s + timestamp_offset_s
                values = tuple(time_s * multiplier for multiplier in range(1, 8))
                adjusted_values = (values[0] + offset, *values[1:])
                writer.writerow(
                    [
                        format(time_s, ".17g"),
                        *(format(value, ".17g") for value in adjusted_values),
                    ]
                )
            ####
        ####
    ####
####


def _compare(
    baseline: CorpusAssessment,
    candidate: CorpusAssessment,
    tmp_path: Path,
    *,
    candidate_config: InferenceConfig | None = None,
    candidate_truth: ConventionHypothesis = TRUTH,
) -> CadenceAssessmentComparisonReport:
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    baseline_manifest = _manifest(baseline)
    candidate_manifest = _manifest(
        candidate,
        config=candidate_config,
        truth=candidate_truth,
    )
    _write_measurements(baseline_manifest, baseline_root)
    _write_measurements(candidate_manifest, candidate_root)
    return compare_cadence_assessments(
        baseline,
        candidate,
        baseline_manifest,
        candidate_manifest,
        baseline_root,
        candidate_root,
    )
####


def _cadence_assessments(
    *,
    candidate_evidence: tuple[str, ...] = ("truth",),
    candidate_period_s: float = 0.05,
    candidate_sample_count: int = 81,
    candidate_orientation_tau_s: float | None = 0.05,
    baseline_jitter_s: float = 0.0,
    candidate_jitter_s: float = 0.0,
    candidate_dropout: float | None = 0.0,
    candidate_noise_seed: int = 100,
    candidate_implementation_hash: str = IMPLEMENTATION_HASH,
) -> tuple[CorpusAssessment, CorpusAssessment]:
    baseline_case = _case(
        case_id="synthetic-n161-case",
        sample_count=161,
        sample_period_s=0.025,
        timestamp_jitter_amplitude_s=baseline_jitter_s,
    )
    candidate_case = _case(
        case_id="synthetic-n81-case",
        sample_count=candidate_sample_count,
        sample_period_s=candidate_period_s,
        evidence_contenders=candidate_evidence,
        nearest_alternative_log_risk_delta=(1.5 if len(candidate_evidence) > 1 else 2.5),
        orientation_noise_correlation_time_s=candidate_orientation_tau_s,
        timestamp_jitter_amplitude_s=candidate_jitter_s,
        dropout_probability=candidate_dropout,
        noise_seed=candidate_noise_seed,
    )
    return (
        _assessment(baseline_case),
        _assessment(candidate_case, implementation_hash=candidate_implementation_hash),
    )
####


def _with_rate_calibration(
    assessment: CorpusAssessment,
    *,
    calibration_id: str = "bias-p-150mrad-s",
    bias_rad_s: tuple[float, float, float] = (0.15, 0.0, 0.0),
) -> CorpusAssessment:
    case = assessment.cases[0]
    conditions = case.measurement_conditions.model_copy(
        update={
            "rate_sensor_calibration_id": calibration_id,
            "rate_sensor_misalignment_angle_deg": 0.0,
            "rate_sensor_misalignment_axis": (0.3, -0.5, 0.8),
            "rate_sensor_bias_rad_s": bias_rad_s,
            "rate_sensor_scale_by_axis": (1.0, 1.0, 1.0),
            "rate_sensor_polarity_by_axis": (1, 1, 1),
            "rate_sensor_clip_abs_rad_s": None,
        }
    )
    calibrated_case = case.model_copy(update={"measurement_conditions": conditions})
    return assessment.model_copy(update={"cases": (calibrated_case,)})
####


def test_pairs_same_seeded_case_across_nested_cadences(tmp_path: Path) -> None:
    baseline, candidate = _cadence_assessments(candidate_evidence=("truth", "other"))

    report = _compare(baseline, candidate, tmp_path)

    assert report.schema_version == 6
    assert report.shared_timestamp_stride_ratio == 2
    assert report.baseline_cadence.sample_count == 161
    assert report.candidate_cadence.sample_count == 81
    assert report.outcome_comparison.paired_case_count == 1
    assert report.outcome_comparison.seeded_noisy_case_count == 1
    assert report.shared_noisy_input_case_count == 1
    assert report.shared_common_timestamp_row_count == 81
    assert report.maximum_shared_input_absolute_difference == 0.0
    assert report.outcome_comparison.seeded_noisy_cases.changes.evidence_set_expansion_count == 1
    assert report.outcome_comparison.seeded_noisy_cases.changes.decision_status_change_count == 1
    risk_margin = report.risk_margin_comparison
    assert risk_margin.paired_case_count == 1
    assert risk_margin.baseline.median == 2.5
    assert risk_margin.candidate.median == 1.5
    assert risk_margin.candidate_minus_baseline.median == -1.0
    assert risk_margin.unique_to_multiple_count == 1
    assert risk_margin.baseline_outcomes.recommendation_count == 1
    assert risk_margin.baseline_outcomes.correct_recommendation_count == 1
    assert risk_margin.candidate_outcomes.recommendation_count == 0
    assert risk_margin.candidate_outcomes.abstention_count == 1
    assert report.risk_margin_by_motion_seed_and_truth_frame[0].truth_rate_frame == "body"
    replication = report.risk_margin_by_motion_noise_seed_and_truth_frame[0]
    assert replication.motion_profile_id == "piecewise-test"
    assert replication.motion_seed == 10
    assert replication.noise_seed == 100
    assert replication.truth_rate_frame == "body"
    assert replication.timestamp_jitter_amplitude_s == 0.0
    assert replication.dropout_probability == 0.0
    assert replication.orientation_latency_s == 0.0
    assert replication.rate_latency_s == 0.0
    assert replication.comparison.paired_case_count == 1
####


def test_cadence_comparison_reports_explicit_calibration_strata(tmp_path: Path) -> None:
    baseline, candidate = _cadence_assessments()
    baseline = _with_rate_calibration(baseline)
    candidate = _with_rate_calibration(candidate)

    report = _compare(baseline, candidate, tmp_path)

    assert len(report.risk_margin_by_motion_seed_and_truth_frame) == 1
    group = report.risk_margin_by_motion_seed_and_truth_frame[0]
    assert group.rate_sensor_calibration.calibration_id == "bias-p-150mrad-s"
    assert group.rate_sensor_calibration.bias_rad_s == (0.15, 0.0, 0.0)
    replication = report.risk_margin_by_motion_noise_seed_and_truth_frame[0]
    assert replication.rate_sensor_calibration == group.rate_sensor_calibration
####


def test_cadence_pairing_rejects_changed_rate_calibration_metadata(tmp_path: Path) -> None:
    baseline, candidate = _cadence_assessments()
    baseline = _with_rate_calibration(baseline)
    candidate = _with_rate_calibration(candidate, bias_rad_s=(0.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="same seeded motion/noise/truth cases"):
        _compare(baseline, candidate, tmp_path)
    ####
####


def test_pairs_seeded_jitter_and_dropout_across_nested_cadences(tmp_path: Path) -> None:
    def generate_paired_corpus(
        output_root: Path,
        *,
        sample_count: int,
        sample_period_s: float,
    ) -> Path:
        return generate_corpus(
            output_root,
            noise_profiles=(NOISE_PROFILES[1],),
            noise_replicates=1,
            noise_seed_start=990402,
            motion_seeds=(990401,),
            truth_convention_sweep=False,
            motion_profiles=(MotionProfile.SAMPLED_RATE_NONANALYTIC,),
            noise_sample_counts=(sample_count,),
            sample_period_s=sample_period_s,
            timestamp_jitter_amplitudes_s=(0.0, 0.002),
            dropout_probabilities=(0.0, 0.1),
            orientation_latencies_s=(0.0, 0.013),
            rate_latencies_s=(0.0, 0.027),
            orientation_noise_correlation_time_s=0.05,
            rate_noise_correlation_time_s=0.1,
            rate_noise_frame="body",
        )
    ####

    baseline_root = tmp_path / "jitter-dropout-40hz"
    candidate_root = tmp_path / "jitter-dropout-80hz"
    baseline_path = generate_paired_corpus(
        baseline_root,
        sample_count=161,
        sample_period_s=0.025,
    )
    candidate_path = generate_paired_corpus(
        candidate_root,
        sample_count=321,
        sample_period_s=0.0125,
    )
    baseline = validate_corpus(baseline_path)
    candidate = validate_corpus(candidate_path)
    baseline_manifest = ValidationManifest.model_validate_json(baseline_path.read_bytes())
    candidate_manifest = ValidationManifest.model_validate_json(candidate_path.read_bytes())

    report = compare_cadence_assessments(
        baseline,
        candidate,
        baseline_manifest,
        candidate_manifest,
        baseline_root,
        candidate_root,
    )

    assert report.shared_noisy_input_case_count == 16
    assert report.shared_common_timestamp_row_count > 0
    assert report.maximum_shared_input_absolute_difference == 0.0
    assert report.outcome_comparison.seeded_noisy_cases.case_count == 16
    acquisition_groups = report.risk_margin_by_motion_noise_seed_and_truth_frame
    assert len(acquisition_groups) == 16
    assert {
        (group.timestamp_jitter_amplitude_s, group.dropout_probability)
        for group in acquisition_groups
    } == {(0.0, 0.0), (0.0, 0.1), (0.002, 0.0), (0.002, 0.1)}
    assert {
        (group.orientation_latency_s, group.rate_latency_s) for group in acquisition_groups
    } == {(0.0, 0.0), (0.0, 0.027), (0.013, 0.0), (0.013, 0.027)}
    assert all(group.comparison.paired_case_count == 1 for group in acquisition_groups)
####


def test_risk_margin_replication_groups_keep_noise_seeds_separate(tmp_path: Path) -> None:
    baseline, candidate = _cadence_assessments()
    baseline_extra = _case(
        case_id="synthetic-n161-case-noise101",
        sample_count=161,
        sample_period_s=0.025,
        noise_seed=101,
    )
    candidate_extra = _case(
        case_id="synthetic-n81-case-noise101",
        sample_count=81,
        sample_period_s=0.05,
        noise_seed=101,
    )
    baseline = baseline.model_copy(
        update={
            "cases": (*baseline.cases, baseline_extra),
            "summary": baseline.summary.model_copy(
                update={
                    "case_count": 2,
                    "resolved_correct": 2,
                    "recommendation_count": 2,
                }
            ),
        }
    )
    candidate = candidate.model_copy(
        update={
            "cases": (*candidate.cases, candidate_extra),
            "summary": candidate.summary.model_copy(
                update={
                    "case_count": 2,
                    "resolved_correct": 2,
                    "recommendation_count": 2,
                }
            ),
        }
    )

    report = _compare(baseline, candidate, tmp_path)

    replications = report.risk_margin_by_motion_noise_seed_and_truth_frame
    assert [replication.noise_seed for replication in replications] == [100, 101]
    assert all(replication.comparison.paired_case_count == 1 for replication in replications)
####


def test_legacy_assessments_without_risk_margins_remain_comparable(tmp_path: Path) -> None:
    baseline, candidate = _cadence_assessments(candidate_evidence=("truth", "other"))
    baseline = baseline.model_copy(
        update={
            "cases": tuple(
                case.model_copy(
                    update={
                        "nearest_alternative_class_id": None,
                        "nearest_alternative_log_risk_delta": None,
                    }
                )
                for case in baseline.cases
            )
        }
    )
    candidate = candidate.model_copy(
        update={
            "cases": tuple(
                case.model_copy(
                    update={
                        "nearest_alternative_class_id": None,
                        "nearest_alternative_log_risk_delta": None,
                    }
                )
                for case in candidate.cases
            )
        }
    )

    report = _compare(baseline, candidate, tmp_path)

    assert report.risk_margin_comparison.paired_case_count == 1
    assert report.risk_margin_comparison.baseline.observed_case_count == 0
    assert report.risk_margin_comparison.candidate.observed_case_count == 0
    assert report.risk_margin_comparison.candidate_minus_baseline.median is None
####


def test_risk_margin_quantiles_use_linear_interpolation(tmp_path: Path) -> None:
    baseline, candidate = _cadence_assessments()
    baseline_cases = (
        baseline.cases[0].model_copy(update={"nearest_alternative_log_risk_delta": 0.0}),
        _case(
            case_id="synthetic-n161-case-2",
            sample_count=161,
            sample_period_s=0.025,
            nearest_alternative_log_risk_delta=10.0,
        ),
    )
    candidate_cases = (
        candidate.cases[0].model_copy(update={"nearest_alternative_log_risk_delta": 0.0}),
        _case(
            case_id="synthetic-n81-case-2",
            sample_count=81,
            sample_period_s=0.05,
            nearest_alternative_log_risk_delta=10.0,
        ),
    )
    report = _compare(
        baseline.model_copy(update={"cases": baseline_cases}),
        candidate.model_copy(update={"cases": candidate_cases}),
        tmp_path,
    )
    distribution = report.risk_margin_comparison.baseline

    assert distribution.tenth_percentile == pytest.approx(1.0)
    assert distribution.median == pytest.approx(5.0)
    assert distribution.ninetieth_percentile == pytest.approx(9.0)
####


def test_rejects_different_seeded_case_sets(tmp_path: Path) -> None:
    baseline, candidate = _cadence_assessments(candidate_noise_seed=101)

    with pytest.raises(ValueError, match="same seeded motion/noise/truth cases"):
        _compare(baseline, candidate, tmp_path)
    ####
####


@pytest.mark.parametrize(
    (
        "candidate_period_s",
        "candidate_sample_count",
        "candidate_orientation_tau_s",
        "candidate_jitter_s",
        "candidate_dropout",
        "message",
    ),
    (
        (0.04, 101, 0.05, 0.0, 0.0, "nested"),
        (0.025, 161, 0.05, 0.0, 0.0, "different cadences"),
        (0.05, 80, 0.05, 0.0, 0.0, "same duration"),
        (0.05, 81, None, 0.0, 0.0, "time-domain OU"),
        (0.05, 81, 0.05, 0.001, 0.0, "same seeded motion/noise/truth cases"),
        (0.05, 81, 0.05, 0.0, 0.1, "same seeded motion/noise/truth cases"),
    ),
)
def test_rejects_cadence_designs_that_do_not_support_pairing(
    candidate_period_s: float,
    candidate_sample_count: int,
    candidate_orientation_tau_s: float | None,
    candidate_jitter_s: float,
    candidate_dropout: float | None,
    message: str,
    tmp_path: Path,
) -> None:
    baseline, candidate = _cadence_assessments(
        candidate_period_s=candidate_period_s,
        candidate_sample_count=candidate_sample_count,
        candidate_orientation_tau_s=candidate_orientation_tau_s,
        candidate_jitter_s=candidate_jitter_s,
        candidate_dropout=candidate_dropout,
    )

    with pytest.raises(ValueError, match=message):
        _compare(baseline, candidate, tmp_path)
    ####
####


def test_rejects_different_inference_runtime_or_implementation(tmp_path: Path) -> None:
    baseline, candidate = _cadence_assessments(candidate_implementation_hash="b" * 64)

    with pytest.raises(ValueError, match="same runtime and implementation"):
        _compare(baseline, candidate, tmp_path)
    ####
####


def test_rejects_different_inference_configurations(tmp_path: Path) -> None:
    baseline, candidate = _cadence_assessments()

    with pytest.raises(ValueError, match="same inference configuration"):
        _compare(
            baseline,
            candidate,
            tmp_path,
            candidate_config=InferenceConfig(near_tie_delta_criterion=1.0),
        )
    ####
####


def test_rejects_different_manifest_truth_labels(tmp_path: Path) -> None:
    baseline, candidate = _cadence_assessments()
    alternate_truth = ConventionHypothesis(
        attitude_direction="world_to_body",
        rate_frame="world",
        rate_order="rqp",
        rate_unit="degrees_per_second",
        quaternion_order="scalar_first",
    )

    with pytest.raises(ValueError, match="same independently labeled truth"):
        _compare(baseline, candidate, tmp_path, candidate_truth=alternate_truth)
    ####
####


def test_rejects_different_measurements_at_shared_timestamps(tmp_path: Path) -> None:
    baseline, candidate = _cadence_assessments()
    baseline_manifest = _manifest(baseline)
    candidate_manifest = _manifest(candidate)
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    _write_measurements(baseline_manifest, baseline_root)
    _write_measurements(candidate_manifest, candidate_root, offset=1.0e-5)

    with pytest.raises(ValueError, match="common-time measurement values differ"):
        compare_cadence_assessments(
            baseline,
            candidate,
            baseline_manifest,
            candidate_manifest,
            baseline_root,
            candidate_root,
        )
    ####
####


def test_rejects_subnanosecond_differences_at_shared_timestamps(tmp_path: Path) -> None:
    baseline, candidate = _cadence_assessments(
        baseline_jitter_s=0.001,
        candidate_jitter_s=0.001,
    )
    baseline_manifest = _manifest(baseline)
    candidate_manifest = _manifest(candidate)
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    _write_measurements(baseline_manifest, baseline_root)
    _write_measurements(candidate_manifest, candidate_root, timestamp_offset_s=1.0e-10)

    with pytest.raises(ValueError, match="common-time timestamps differ"):
        compare_cadence_assessments(
            baseline,
            candidate,
            baseline_manifest,
            candidate_manifest,
            baseline_root,
            candidate_root,
        )
    ####
####
