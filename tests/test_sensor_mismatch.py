from __future__ import annotations

from math import pi

import pytest
from numpy import array, deg2rad, ones, rad2deg, sqrt

from attitude_convention_inference import (
    InferenceConfig,
    InferenceDecisionReason,
    InferenceDecisionStatus,
    InferenceReport,
    TrajectoryData,
)
from attitude_convention_inference.math_utils import (
    rotation_from_rotvec,
    rotation_product,
    rotation_rotvecs,
)
from scripts import validate_sensor_mismatch as mismatch_validation
from scripts.generate_validation_corpus import MotionProfile
from scripts.validate_sensor_mismatch import (
    DEFAULT_MIXED_TIMING_CALIBRATION_CONDITIONS,
    DEFAULT_SENSOR_MISMATCH_CONDITIONS,
    AcquisitionProfile,
    SensorMismatchCondition,
    run_sensor_mismatch_sweep,
)


def test_independent_sensor_mismatch_sweep_records_persistence_guard_policy() -> None:
    sweep = run_sensor_mismatch_sweep((703,))

    assert sweep.case_count == 22
    assert sweep.wrong_recommendation_count == 0
    assert sweep.persistence_guard_correlation_limit == 0.95
    assert sweep.persistence_guard_residual_floor_rad == 1.0e-5
    clean = next(case for case in sweep.cases if case.condition.condition_id == "clean")
    assert clean.recommendation_correct is True
    misaligned = next(
        case for case in sweep.cases if case.condition.condition_id == "axis-rotation-090deg"
    )
    assert misaligned.truth_supported_by_evidence
    assert misaligned.dependence_guard_triggered
    assert not misaligned.evidence_sufficient
    assert misaligned.decision_status is InferenceDecisionStatus.ABSTENTION
    assert misaligned.decision_reason is InferenceDecisionReason.EVIDENCE_INSUFFICIENT
####


def test_persistent_near_axis_mapping_suppresses_false_recommendation() -> None:
    permutation_axis = ones(3, dtype=float)
    permutation_axis /= sqrt(float((permutation_axis * permutation_axis).sum()))
    permutation = rotation_from_rotvec((2.0 * pi / 3.0) * permutation_axis)
    perturbation_axis = array((0.3, -0.5, 0.8), dtype=float)
    perturbation_axis /= sqrt(float((perturbation_axis * perturbation_axis).sum()))
    perturbation = rotation_from_rotvec(deg2rad(0.25) * perturbation_axis)
    rotation_vector = rotation_rotvecs(rotation_product(perturbation, permutation))
    angle_rad = sqrt(float((rotation_vector * rotation_vector).sum()))
    unit_axis = rotation_vector / angle_rad
    condition = SensorMismatchCondition(
        condition_id="near-discrete-axis-mapping",
        misalignment_angle_deg=float(rad2deg(angle_rad)),
        misalignment_axis=(float(unit_axis[0]), float(unit_axis[1]), float(unit_axis[2])),
    )

    unguarded = run_sensor_mismatch_sweep(
        (703,),
        conditions=(condition,),
        persistence_guard_correlation_limit=1.1,
    )
    assert unguarded.persistence_guard_correlation_limit == 1.1
    assert unguarded.wrong_recommendation_count == 1
    false_recommendation = unguarded.cases[0]
    assert false_recommendation.recommendation_correct is False
    assert not false_recommendation.dependence_guard_triggered
    assert abs(false_recommendation.residual_lag1_correlation) >= 0.95
    assert false_recommendation.residual_sigma_rad > 1.0e-5

    guarded = run_sensor_mismatch_sweep(
        (703,),
        conditions=(condition,),
        persistence_guard_correlation_limit=0.95,
    )

    assert guarded.persistence_guard_correlation_limit == 0.95
    assert guarded.wrong_recommendation_count == 0
    assert guarded.abstention_count == 1
    assert guarded.cases[0].dependence_guard_triggered
    assert guarded.cases[0].decision_status is InferenceDecisionStatus.ABSTENTION
    assert guarded.cases[0].decision_reason is InferenceDecisionReason.EVIDENCE_INSUFFICIENT
####


@pytest.mark.parametrize(
    "motion_profile",
    (
        MotionProfile.SAMPLED_RATE_NONANALYTIC,
        MotionProfile.SAMPLED_RATE_PIECEWISE_LINEAR,
        MotionProfile.SAMPLED_RATE_MANEUVER_BURSTS,
    ),
)
def test_mixed_timing_calibration_abstains_across_motion_families(
    motion_profile: MotionProfile,
) -> None:
    sweep = run_sensor_mismatch_sweep(
        (714,),
        conditions=DEFAULT_MIXED_TIMING_CALIBRATION_CONDITIONS,
        motion_profile=motion_profile,
    )

    assert sweep.motion_profile is motion_profile
    assert sweep.case_count == 3
    assert sweep.wrong_recommendation_count == 0
    assert sweep.abstention_count == 3
    assert all(case.truth_supported_by_evidence for case in sweep.cases)
    assert all(case.recommendation_correct is None for case in sweep.cases)
    assert all(case.dependence_guard_triggered for case in sweep.cases)
    assert all(
        case.condition.orientation_latency_s > 0.0 or case.condition.rate_latency_s > 0.0
        for case in sweep.cases
    )
####


@pytest.mark.parametrize(
    ("acquisition_profile", "orientation_sigma_rad", "rate_sigma_rad_s"),
    (
        (AcquisitionProfile.LOW_OU_JITTER_DROPOUT, 0.0015, 0.012),
        (AcquisitionProfile.MODERATE_OU_JITTER_DROPOUT, 0.003, 0.025),
        (AcquisitionProfile.HIGH_OU_JITTER_DROPOUT, 0.006, 0.05),
    ),
)
def test_noisy_jittered_dropout_screen_is_reproducible(
    acquisition_profile: AcquisitionProfile,
    orientation_sigma_rad: float,
    rate_sigma_rad_s: float,
) -> None:
    clean_control = SensorMismatchCondition(condition_id="clean")
    first = run_sensor_mismatch_sweep(
        (703,),
        conditions=(clean_control,),
        acquisition_profile=acquisition_profile,
        noise_seeds=(20314101,),
    )
    second = run_sensor_mismatch_sweep(
        (703,),
        conditions=(clean_control,),
        acquisition_profile=acquisition_profile,
        noise_seeds=(20314101,),
    )

    assert first == second
    assert first.acquisition_profile is acquisition_profile
    assert first.acquisition.orientation_noise_sigma_rad == orientation_sigma_rad
    assert first.acquisition.rate_noise_sigma_rad_s == rate_sigma_rad_s
    assert first.acquisition.timestamp_jitter_amplitude_s > 0.0
    assert first.acquisition.dropout_probability > 0.0
    assert first.acquisition.orientation_noise_correlation_time_s == 0.05
    assert first.acquisition.rate_noise_correlation_time_s == 0.1
    assert first.acquisition.timestamp_jitter_amplitude_s == 0.002
    assert first.acquisition.dropout_probability == 0.1
    assert 120 < first.cases[0].retained_sample_count < first.sample_count
    assert first.cases[0].noise_seed == 20314101
####


def test_large_model_mismatch_warns_and_abstains_without_convention_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_reports: list[InferenceReport] = []
    original_inference = mismatch_validation.infer_conventions

    def capture_report(data: TrajectoryData, config: InferenceConfig) -> InferenceReport:
        report = original_inference(data, config)
        captured_reports.append(report)
        return report
    ####

    condition = next(
        condition
        for condition in DEFAULT_SENSOR_MISMATCH_CONDITIONS
        if condition.condition_id == "axis-polarity-04"
    )
    monkeypatch.setattr(mismatch_validation, "infer_conventions", capture_report)
    sweep = mismatch_validation.run_sensor_mismatch_sweep(
        (744701,),
        conditions=(condition,),
        motion_profile=MotionProfile.SAMPLED_RATE_NONANALYTIC,
        acquisition_profile=AcquisitionProfile.HIGH_OU_JITTER_DROPOUT,
        noise_seeds=(20414101,),
    )
    case = sweep.cases[0]
    assert len(captured_reports) == 1
    report = captured_reports[0]
    assert not case.truth_supported_by_evidence
    assert case.recommendation_correct is None
    assert case.decision_reason is InferenceDecisionReason.MULTIPLE_EVIDENCE_CONTENDERS
    assert case.normalized_rmse > 0.5
    assert any(
        "Multiple convention hypotheses remain" in warning
        for warning in report.diagnostics.warnings
    )
    assert any(
        "best convention has a large residual" in warning for warning in report.diagnostics.warnings
    )
    assert report.diagnostics.evidence_sufficient
    assert report.decision.status is InferenceDecisionStatus.ABSTENTION
    assert report.decision.evidence_contender_class_ids == case.evidence_contender_class_ids
####
