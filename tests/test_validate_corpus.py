from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from math import exp
from pathlib import Path
from typing import NoReturn, cast

import pytest
from pydantic import ValidationError

from attitude_convention_inference import InferenceDecisionStatus
from scripts.generate_validation_corpus import MotionProfile, generate_corpus
from scripts.validate_corpus import (
    CaseAssessment,
    MeasurementConditions,
    ValidationCheckpoint,
    ValidationManifest,
    main,
    validate_corpus,
)

validation_module = importlib.import_module("scripts.validate_corpus")


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
    parallel_result = validate_corpus(manifest, max_workers=2)

    assert parallel_result == result
    assert result.schema_version == 10
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
    assert result.cases[0].nearest_alternative_class_id is not None
    assert result.cases[0].nearest_alternative_log_risk_delta is not None
    assert result.cases[0].nearest_alternative_log_risk_delta >= 0.0
    assert result.cases[0].operational_correct
    assert not result.cases[1].operational_correct
    diagnostics = result.cases[0].inference_diagnostics
    assert 0.0 <= diagnostics.confidence <= 1.0
    assert diagnostics.estimated_noise_sigma_rad >= 0.0
    assert diagnostics.noise_residual_count > 0
    assert -1.0 <= diagnostics.noise_lag1_autocorrelation <= 1.0
    assert 0 <= diagnostics.evidence_intervals_used <= result.cases[0].sample_count - 1
    assert 0.0 <= diagnostics.evidence_fraction <= 1.0
    serialized_diagnostics = json.loads(result.model_dump_json())["cases"][0][
        "inference_diagnostics"
    ]
    assert (
        serialized_diagnostics["noise_lag1_autocorrelation"]
        == diagnostics.noise_lag1_autocorrelation
    )
    assert serialized_diagnostics["warnings"] == list(diagnostics.warnings)
    summaries_by_truth = {summary.truth_class_id: summary for summary in result.condition_summaries}
    assert set(summaries_by_truth) == {case.truth_class_id for case in result.cases}
    assert all(summary.case_count == 1 for summary in summaries_by_truth.values())
    correct_summary = summaries_by_truth[result.cases[0].truth_class_id]
    miss_summary = summaries_by_truth[result.cases[1].truth_class_id]
    assert correct_summary.mean_noise_lag1_autocorrelation == pytest.approx(
        result.cases[0].inference_diagnostics.noise_lag1_autocorrelation
    )
    assert correct_summary.operational_miss == 0
    assert correct_summary.evidence_miss_rate == 0.0
    assert correct_summary.recommendation_accuracy == 1.0
    assert miss_summary.operational_miss == 1
    assert miss_summary.evidence_miss_rate == 1.0
    assert miss_summary.incorrect_evidence_sufficient_rate == 1.0
    assert miss_summary.recommendation_accuracy == 0.0
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


@pytest.mark.parametrize("max_workers", (0, -1, True, 1.5))
def test_validation_corpus_requires_positive_worker_count(
    tmp_path: Path,
    max_workers: object,
) -> None:
    with pytest.raises(ValueError, match="max_workers must be a positive integer"):
        validate_corpus(tmp_path / "missing-manifest.json", max_workers=cast(int, max_workers))
    ####
####


def test_validation_corpus_retries_unfinished_cases_after_worker_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    _manifest_case("scalar_first", "case-1"),
                    _manifest_case("scalar_last", "case-2"),
                ],
            }
        ),
        encoding="utf-8",
    )
    expected = validate_corpus(manifest)

    class BrokenPool:
        def __init__(self, *, max_workers: int) -> None:
            assert max_workers == 2
        ####

        def __enter__(self) -> BrokenPool:
            return self
        ####

        def __exit__(self, *_args: object) -> None:
            return None
        ####

        def submit(self, *_args: object, **_kwargs: object) -> Future[object]:
            future: Future[object] = Future()
            future.set_exception(BrokenProcessPool("simulated worker crash"))
            return future
        ####
    ####

    monkeypatch.setattr(validation_module, "ProcessPoolExecutor", BrokenPool)
    recovered = validate_corpus(manifest, max_workers=2)

    assert recovered == expected
    assert "retrying 2 unfinished case(s) sequentially" in capsys.readouterr().err
####


def test_validation_checkpoint_resumes_and_rejects_changed_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.json"
    source_csv = tmp_path / "synthetic_quaternion.csv"
    source_csv.write_bytes(
        (Path(__file__).resolve().parents[1] / "examples" / "synthetic_quaternion.csv").read_bytes()
    )
    case = _manifest_case("scalar_first", "case-1")
    second_case = _manifest_case("scalar_last", "case-2")
    case["csv"] = str(source_csv)
    second_case["csv"] = str(source_csv)
    manifest.write_text(
        json.dumps({"schema_version": 1, "cases": [case, second_case]}),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "assessment.checkpoint.json"

    with pytest.raises(ValueError, match="must not overwrite"):
        validate_corpus(manifest, checkpoint_path=source_csv)
    ####

    first_result = validate_corpus(
        manifest,
        checkpoint_path=checkpoint,
        checkpoint_every=1,
        max_workers=2,
    )
    checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert checkpoint_payload["schema_version"] == 9
    assert [case["case_id"] for case in checkpoint_payload["completed_cases"]] == [
        "case-1",
        "case-2",
    ]
    legacy_checkpoint = tmp_path / "legacy-assessment.checkpoint.json"
    legacy_payload = dict(checkpoint_payload)
    legacy_payload["schema_version"] = 6
    legacy_checkpoint.write_text(json.dumps(legacy_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="predates nearest-alternative risk margins"):
        validate_corpus(manifest, checkpoint_path=legacy_checkpoint, resume=True)
    ####

    def unexpected_inference(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("resume re-evaluated a completed case")
    ####

    monkeypatch.setattr(validation_module, "infer_csv", unexpected_inference)
    resumed_result = validate_corpus(manifest, checkpoint_path=checkpoint, resume=True)
    assert resumed_result == first_result

    with monkeypatch.context() as patcher:
        patcher.setattr(validation_module, "_implementation_sha256", lambda: "0" * 64)
        with pytest.raises(ValueError, match="runtime environment"):
            validate_corpus(manifest, checkpoint_path=checkpoint, resume=True)
        ####
    ####

    original_csv = source_csv.read_bytes()
    source_csv.write_bytes(original_csv + b"\n")
    with pytest.raises(ValueError, match="source CSV hash"):
        validate_corpus(manifest, checkpoint_path=checkpoint, resume=True)
    ####
    source_csv.write_bytes(original_csv)

    changed_payload = json.loads(manifest.read_text(encoding="utf-8"))
    changed_payload["cases"][0]["label_source"] = "changed-label-source"
    manifest.write_text(json.dumps(changed_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash"):
        validate_corpus(manifest, checkpoint_path=checkpoint, resume=True)
    ####
####


def test_validation_checkpoint_recovers_a_partially_completed_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_csv = tmp_path / "synthetic_quaternion.csv"
    source_csv.write_bytes(
        (Path(__file__).resolve().parents[1] / "examples" / "synthetic_quaternion.csv").read_bytes()
    )
    first_case = _manifest_case("scalar_first", "case-1")
    second_case = _manifest_case("scalar_last", "case-2")
    first_case["csv"] = str(source_csv)
    second_case["csv"] = str(source_csv)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "cases": [first_case, second_case]}),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "assessment.checkpoint.json"
    write_checkpoint = cast(
        Callable[[Path, ValidationCheckpoint], None],
        validation_module._write_checkpoint,
    )

    def interrupt_after_checkpoint(path: Path, value: ValidationCheckpoint) -> None:
        write_checkpoint(path, value)
        raise RuntimeError("simulated interruption after durable checkpoint")
    ####

    with monkeypatch.context() as patcher:
        patcher.setattr(validation_module, "_write_checkpoint", interrupt_after_checkpoint)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            validate_corpus(manifest, checkpoint_path=checkpoint, checkpoint_every=1)
        ####
    ####

    saved_checkpoint = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert [case["case_id"] for case in saved_checkpoint["completed_cases"]] == ["case-1"]

    resumed_result = validate_corpus(manifest, checkpoint_path=checkpoint, resume=True)
    assert resumed_result.summary.case_count == 2
    assert [case.case_id for case in resumed_result.cases] == ["case-1", "case-2"]
    assert resumed_result.summary.evidence_miss == 1
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
        {"timestamp_clock_rate_error_ppm": -1_000_000.0},
        {"timestamp_clock_rate_error_ppm": float("nan")},
        {"dropout_probability": 1.0},
        {"dropped_sample_count": -1},
        {"orientation_latency_s": -0.001},
        {"rate_latency_s": float("inf")},
        {"orientation_noise_sigma_rad": float("nan")},
        {"rate_noise_sigma_rad_s": float("inf")},
        {"orientation_noise_ar1_coefficient": -1.0},
        {"orientation_noise_ar1_coefficient": 1.0},
        {"rate_noise_ar1_coefficient": float("nan")},
        {"rate_noise_ar1_coefficient": float("inf")},
        {"orientation_noise_correlation_time_s": 0.0},
        {"rate_noise_correlation_time_s": float("inf")},
        {"orientation_noise_correlation_time_s": 0.05},
        {"rate_noise_frame": "sensor"},
        {"rate_sensor_calibration_id": "named-without-values"},
        {"rate_sensor_misalignment_axis": (0.0, 0.0, 0.0)},
        {"rate_sensor_scale_by_axis": (1.0, 0.0, 1.0)},
        {"rate_sensor_polarity_by_axis": (1, 0, -1)},
        {"rate_sensor_clip_abs_rad_s": float("inf")},
        {
            "sample_count": 41,
            "sample_period_s": 0.025,
            "orientation_noise_correlation_time_s": 0.05,
            "orientation_noise_ar1_coefficient": 0.5,
        },
        {"noise_seed": -1},
        {"motion_seed": -1},
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


def test_measurement_conditions_validate_ou_time_constants_and_ar1_coefficients() -> None:
    conditions = MeasurementConditions.model_validate(
        {
            "sample_count": 41,
            "sample_period_s": 0.025,
            "orientation_noise_correlation_time_s": 0.05,
            "orientation_noise_ar1_coefficient": exp(-0.025 / 0.05),
            "rate_noise_correlation_time_s": 0.1,
            "rate_noise_ar1_coefficient": exp(-0.025 / 0.1),
        }
    )

    assert conditions.orientation_noise_correlation_time_s == 0.05
    assert conditions.rate_noise_correlation_time_s == 0.1
    assert conditions.rate_noise_frame == "truth-rate-frame"
####


def test_manifest_versions_and_rate_calibration_identifiers_are_consistent() -> None:
    def calibrated_case(case_id: str, bias_rad_s: tuple[float, float, float]) -> dict[str, object]:
        case = _manifest_case("scalar_last", case_id)
        case["measurement_conditions"] = {
            "rate_sensor_calibration_id": "known-bias",
            "rate_sensor_misalignment_angle_deg": 0.0,
            "rate_sensor_misalignment_axis": (0.3, -0.5, 0.8),
            "rate_sensor_bias_rad_s": bias_rad_s,
            "rate_sensor_scale_by_axis": (1.0, 1.0, 1.0),
            "rate_sensor_polarity_by_axis": (1, 1, 1),
        }
        return case
    ####

    first_case = calibrated_case("first", (0.15, 0.0, 0.0))
    matching_case = calibrated_case("matching", (0.15, 0.0, 0.0))
    conflicting_case = calibrated_case("conflicting", (0.0, 0.15, 0.0))

    with pytest.raises(ValidationError, match="schema_version 5"):
        ValidationManifest.model_validate({"schema_version": 4, "cases": [first_case]})
    ####

    manifest = ValidationManifest.model_validate(
        {"schema_version": 5, "cases": [first_case, matching_case]}
    )
    assert manifest.schema_version == 5
    with pytest.raises(ValidationError, match="one parameter set"):
        ValidationManifest.model_validate(
            {"schema_version": 5, "cases": [first_case, conflicting_case]}
        )
    ####
####


def test_condition_summaries_separate_time_domain_noise_models(tmp_path: Path) -> None:
    manifest = generate_corpus(tmp_path / "time-domain-noise", noise_sample_counts=(41,))
    assessment = validate_corpus(manifest)
    source_case = next(
        case for case in assessment.cases if case.measurement_conditions.sample_count == 41
    )
    variants: list[CaseAssessment] = []
    for correlation_time_s in (0.05, 0.1):
        conditions_payload = source_case.measurement_conditions.model_dump(exclude={"duration_s"})
        conditions_payload.update(
            orientation_noise_correlation_time_s=correlation_time_s,
            orientation_noise_ar1_coefficient=exp(-0.025 / correlation_time_s),
        )
        conditions = MeasurementConditions.model_validate(conditions_payload)
        variants.append(source_case.model_copy(update={"measurement_conditions": conditions}))
    ####
    summaries = validation_module._condition_summaries(variants)

    assert len(summaries) == 2
    assert {
        summary.measurement_conditions.orientation_noise_correlation_time_s for summary in summaries
    } == {0.05, 0.1}
    assert all(summary.case_count == 1 for summary in summaries)
####


def test_fail_on_miss_includes_wrong_operational_class(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
        [
            "validate_corpus",
            str(manifest),
            "--output",
            str(output),
            "--progress",
            "--workers",
            "2",
            "--fail-on-miss",
        ],
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
    captured = capsys.readouterr()
    assert "Validated 4/4 cases" in captured.err
    assert captured.out == ""
####
