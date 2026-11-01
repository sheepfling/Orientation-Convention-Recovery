from __future__ import annotations

import pytest
from pydantic import ValidationError

from attitude_convention_inference import InferenceDecisionStatus, InferenceReport
from scripts.generate_validation_corpus import MotionProfile
from scripts.validate_sparse_anchors import (
    SparseAnchorCase,
    SparseAnchorNoiseTier,
    SparseAnchorStudyConfig,
    SparseAnchorStudyReport,
    SparseAnchorTimingCondition,
    run_sparse_anchor_study,
    summarize_sparse_anchor_cases,
)


def test_sparse_anchor_configuration_requires_nested_cadences() -> None:
    with pytest.raises(ValidationError, match="nested integer multiples"):
        SparseAnchorStudyConfig(
            duration_s=3.0,
            rate_period_s=0.05,
            anchor_cadences_s=(0.5, 0.75),
        )
    ####
####


def test_sparse_anchor_configuration_rejects_jitter_that_can_reorder_rates() -> None:
    with pytest.raises(ValidationError, match="less than half the rate period"):
        SparseAnchorStudyConfig(
            duration_s=1.0,
            rate_period_s=0.025,
            anchor_cadences_s=(0.5,),
            timing_conditions=(
                SparseAnchorTimingCondition(
                    condition_id="invalid-rate-jitter",
                    rate_timestamp_jitter_s=0.0125,
                ),
            ),
        )
    ####
####


def test_sparse_anchor_timing_condition_rejects_nonpositive_clock_scale() -> None:
    with pytest.raises(ValidationError, match="greater than -1,000,000"):
        SparseAnchorTimingCondition(
            condition_id="invalid-clock-scale",
            clock_rate_error_ppm=-1_000_000.0,
        )
    ####
####


def test_sparse_anchor_summary_counts_winners_evidence_and_abstentions() -> None:
    cases = (
        SparseAnchorCase(
            motion_profile=MotionProfile.SAMPLED_RATE_NONANALYTIC,
            motion_seed=41,
            noise_tier=SparseAnchorNoiseTier.MODERATE,
            noise_seed=42,
            anchor_cadence_s=0.2,
            anchor_count=6,
            rate_sample_count=101,
            timing_condition_id="regular-exact",
            decision_status=InferenceDecisionStatus.RECOMMENDATION,
            truth_class_id="truth",
            data_winning_class_ids=("truth",),
            evidence_contender_class_ids=("truth",),
            recommended_class_id="truth",
            evidence_sufficient=True,
            normalized_rmse=0.1,
        ),
        SparseAnchorCase(
            motion_profile=MotionProfile.SAMPLED_RATE_NONANALYTIC,
            motion_seed=41,
            noise_tier=SparseAnchorNoiseTier.MODERATE,
            noise_seed=42,
            anchor_cadence_s=0.2,
            anchor_count=6,
            rate_sample_count=101,
            timing_condition_id="regular-exact",
            decision_status=InferenceDecisionStatus.ABSTENTION,
            truth_class_id="truth",
            data_winning_class_ids=("other",),
            evidence_contender_class_ids=("truth",),
            recommended_class_id=None,
            evidence_sufficient=False,
            normalized_rmse=0.2,
        ),
    )

    (summary,) = summarize_sparse_anchor_cases(cases)

    assert summary.trial_count == 2
    assert summary.data_winner_truth_count == 1
    assert summary.evidence_truth_count == 2
    assert summary.evidence_sufficient_count == 1
    assert summary.correct_recommendation_count == 1
    assert summary.wrong_recommendation_count == 0
    assert summary.abstention_count == 1
    assert summary.median_normalized_rmse == pytest.approx(0.15)
####


def test_sparse_anchor_study_runs_a_minimal_end_to_end_cell() -> None:
    report = run_sparse_anchor_study(
        SparseAnchorStudyConfig(
            duration_s=1.0,
            rate_period_s=0.025,
            anchor_cadences_s=(0.5,),
            motion_profiles=(MotionProfile.SAMPLED_RATE_NONANALYTIC,),
            motion_seeds=(43,),
            noise_tiers=(SparseAnchorNoiseTier.NONE,),
            timing_conditions=(SparseAnchorTimingCondition(condition_id="regular-exact"),),
        )
    )

    assert report.schema_version == 3
    assert len(report.cases) == len(report.summaries) == 1
    assert report.cases[0].anchor_count == 3
    assert report.cases[0].rate_sample_count == 41
    assert report.cases[0].decision_status is InferenceDecisionStatus.ABSTENTION
    assert report.cases[0].recommendation_correct is None
    assert SparseAnchorStudyReport.model_validate_json(report.model_dump_json()) == report
####


def test_sparse_anchor_study_uses_irregular_observed_timestamps_and_anchor_dropout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numpy import diff

    from attitude_convention_inference import SparseOrientationRateData
    from scripts import validate_sparse_anchors

    captured_data: list[SparseOrientationRateData] = []
    original_inference = validate_sparse_anchors.infer_sparse_orientation_conventions

    def capture_data(data: SparseOrientationRateData) -> InferenceReport:
        captured_data.append(data)
        return original_inference(data)
    ####

    monkeypatch.setattr(
        validate_sparse_anchors,
        "infer_sparse_orientation_conventions",
        capture_data,
    )
    report = run_sparse_anchor_study(
        SparseAnchorStudyConfig(
            duration_s=2.0,
            rate_period_s=0.025,
            anchor_cadences_s=(0.5,),
            motion_profiles=(MotionProfile.SAMPLED_RATE_NONANALYTIC,),
            motion_seeds=(43, 44, 45),
            noise_tiers=(SparseAnchorNoiseTier.NONE,),
            timing_conditions=(
                SparseAnchorTimingCondition(
                    condition_id="irregular-dropout-plus-jitter",
                    anchor_drop_probability=0.8,
                    anchor_timestamp_jitter_s=0.01,
                    rate_timestamp_jitter_s=0.002,
                ),
            ),
        )
    )

    assert len(report.cases) == len(captured_data) == 3
    assert all(3 <= case.anchor_count <= 5 for case in report.cases)
    assert any(case.anchor_count < 5 for case in report.cases)
    assert all(
        any(
            abs(float(interval_s) - 0.5) > 1.0e-6
            for interval_s in diff(data.orientation_anchor_times_s)
        )
        for data in captured_data
    )
    assert all(
        any(abs(float(interval_s) - 0.025) > 1.0e-7 for interval_s in diff(data.rate_times_s))
        for data in captured_data
    )
####


def test_sparse_anchor_study_applies_relative_value_latency_and_clock_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numpy import allclose

    from attitude_convention_inference import SparseOrientationRateData
    from scripts import validate_sparse_anchors

    captured_data: list[SparseOrientationRateData] = []
    original_inference = validate_sparse_anchors.infer_sparse_orientation_conventions

    def capture_data(data: SparseOrientationRateData) -> InferenceReport:
        captured_data.append(data)
        return original_inference(data)
    ####

    monkeypatch.setattr(
        validate_sparse_anchors,
        "infer_sparse_orientation_conventions",
        capture_data,
    )
    report = run_sparse_anchor_study(
        SparseAnchorStudyConfig(
            duration_s=2.0,
            rate_period_s=0.025,
            anchor_cadences_s=(0.5,),
            motion_profiles=(MotionProfile.SAMPLED_RATE_NONANALYTIC,),
            motion_seeds=(46,),
            noise_tiers=(SparseAnchorNoiseTier.NONE,),
            timing_conditions=(
                SparseAnchorTimingCondition(condition_id="regular-exact"),
                SparseAnchorTimingCondition(
                    condition_id="latency-and-clock-scale",
                    orientation_value_latency_s=0.01,
                    rate_value_latency_s=0.025,
                    clock_rate_error_ppm=1_000.0,
                ),
            ),
        )
    )

    assert len(report.cases) == len(captured_data) == 2
    baseline, perturbed = captured_data
    assert perturbed.orientation_anchor_times_s[-1] == pytest.approx(2.002)
    assert perturbed.rate_times_s[-1] == pytest.approx(2.002)
    assert not allclose(perturbed.orientation_anchors, baseline.orientation_anchors)
    assert not allclose(perturbed.rates, baseline.rates)
####
