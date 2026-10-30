from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from attitude_convention_inference import InferenceDecisionStatus
from scripts.generate_validation_corpus import MotionProfile, generate_corpus
from scripts.validate_corpus import MeasurementConditions, main, validate_corpus


def _manifest_case(quaternion_order: str, case_id: str) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    return {
        "case_id": case_id,
        "csv": str(repository / "examples" / "synthetic_quaternion.csv"),
        "label_source": "synthetic-generator-test-only",
        "representation": "quaternion",
        "orientation_columns": ["q0", "q1", "q2", "q3"],
        "rate_columns": ["w0", "w1", "w2"],
        "time_column": "time",
        "truth": {
            "attitude_direction": "world_to_body",
            "quaternion_order": quaternion_order,
            "rate_frame": "world",
            "rate_order": "rqp",
            "rate_unit": "radians_per_second",
        },
        "config": {
            "search_space": {
                "attitude_directions": ["world_to_body"],
                "quaternion_orders": ["scalar_first", "scalar_last"],
                "rate_frames": ["world"],
                "rate_orders": ["rqp"],
            }
        },
    }
####


def test_validation_corpus_counts_recovery_and_missed_label(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    _manifest_case("scalar_first", "known-correct"),
                    _manifest_case("scalar_last", "deliberately-wrong-label"),
                ],
            }
        ),
        encoding="utf-8",
    )

    result = validate_corpus(manifest)

    assert result.summary.case_count == 2
    assert result.summary.resolved_correct == 1
    assert result.summary.evidence_miss == 1
    assert result.summary.data_winner_miss == 1
    assert result.summary.operational_miss == 1
    assert result.summary.incorrect_evidence_sufficient == 1
    assert result.summary.recommendation_count == 2
    assert result.summary.abstention_count == 0
    assert result.summary.recommendation_miss == 1
    assert result.summary.recommendation_coverage == result.summary.recommendation_count / 2
    assert result.cases[0].truth_class_id in result.cases[0].evidence_contender_class_ids
    assert result.cases[1].truth_class_id not in result.cases[1].evidence_contender_class_ids
    assert result.cases[0].operational_correct
    assert not result.cases[1].operational_correct
    assert result.condition_summaries[0].case_count == 2
    assert result.condition_summaries[0].operational_miss == 1
    assert result.condition_summaries[0].evidence_miss_rate == 0.5
    assert result.condition_summaries[0].incorrect_evidence_sufficient_rate == 0.5
    assert result.condition_summaries[0].recommendation_accuracy == 0.5
    assert result.cases[1].recommendation_correct is False
####


def test_validation_corpus_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    case = _manifest_case("scalar_first", "duplicate")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "cases": [case, case]}), encoding="utf-8")

    with pytest.raises(ValueError, match="unique"):
        validate_corpus(manifest)
    ####
####


def test_validation_corpus_checks_declared_observation_count(tmp_path: Path) -> None:
    manifest = generate_corpus(
        tmp_path / "incorrect-observation-count-corpus",
        noise_sample_counts=(41,),
        dropout_probabilities=(0.1,),
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    noisy_case = next(
        case for case in payload["cases"] if "noise_seed" in case["measurement_conditions"]
    )
    noisy_case["measurement_conditions"]["observed_sample_count"] += 1
    noisy_case["measurement_conditions"]["dropped_sample_count"] -= 1
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"observed_sample_count.*does not match the CSV"):
        validate_corpus(manifest)
    ####
####


@pytest.mark.parametrize(
    "values",
    (
        {"timestamp_jitter_amplitude_s": -0.001},
        {"dropout_probability": 1.0},
        {"dropped_sample_count": -1},
        {"orientation_latency_s": -0.001},
        {"rate_latency_s": float("inf")},
        {"orientation_noise_sigma_rad": float("nan")},
        {"rate_noise_sigma_rad_s": float("inf")},
        {"noise_seed": -1},
        {"sample_count": 161},
        {"sample_period_s": 0.025},
        {"sample_count": 0, "sample_period_s": 0.025},
        {
            "sample_count": 161,
            "observed_sample_count": 140,
            "dropped_sample_count": 20,
            "sample_period_s": 0.025,
        },
        {
            "timestamp_jitter_amplitude_s": 0.013,
            "sample_count": 161,
            "sample_period_s": 0.025,
        },
        {"unrecognized_condition": True},
    ),
)
def test_measurement_conditions_reject_invalid_metadata(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        MeasurementConditions.model_validate(values)
    ####
####


def test_fail_on_miss_includes_wrong_operational_class(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = generate_corpus(
        tmp_path / "single-axis-corpus",
        motion_profiles=(MotionProfile.SINGLE_AXIS,),
    )
    assessment = validate_corpus(manifest)
    output = tmp_path / "assessment.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_corpus", str(manifest), "--output", str(output), "--fail-on-miss"],
    )

    assert assessment.summary.evidence_miss == 0
    assert assessment.summary.operational_miss > 0
    operational_misses = [case for case in assessment.cases if not case.operational_correct]
    assert all(
        case.decision.status is InferenceDecisionStatus.ABSTENTION for case in operational_misses
    )
    assert assessment.summary.recommendation_miss == 0
    single_axis_condition = next(
        summary
        for summary in assessment.condition_summaries
        if summary.measurement_conditions.motion_profile_id == MotionProfile.SINGLE_AXIS.value
    )
    assert single_axis_condition.recommendation_coverage == 0.0
    assert single_axis_condition.recommendation_accuracy is None
    with pytest.raises(SystemExit) as exit_info:
        main()
    ####
    assert exit_info.value.code == 1
    assert output.is_file()
####
