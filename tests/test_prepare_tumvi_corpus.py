from __future__ import annotations

import csv
import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from numpy import asarray, float64
from numpy.typing import NDArray
from pydantic import ValidationError
from scipy.spatial.transform import Rotation

from scripts.prepare_tumvi_corpus import prepare_tumvi_corpus
from scripts.tumvi_data import sha256_file
from scripts.validate_corpus import ValidationManifest, validate_corpus
from scripts.validate_sparse_tumvi import run_sparse_tumvi_study

FloatArray = NDArray[float64]
tumvi_module = importlib.import_module("scripts.prepare_tumvi_corpus")


def _write_tumvi_fixture(
    sequence_directory: Path,
    *,
    duration_s: float = 0.2,
    imu_period_s: float = 0.005,
    pose_period_s: float = 0.01,
) -> None:
    imu_directory = sequence_directory / "mav0" / "imu0"
    pose_directory = sequence_directory / "mav0" / "mocap0"
    imu_directory.mkdir(parents=True)
    pose_directory.mkdir(parents=True)
    origin_ns = 1_700_000_000_000_000_000
    imu_count = round(duration_s / imu_period_s) + 1
    imu_times_s: FloatArray = asarray(
        [index * imu_period_s for index in range(imu_count)],
        dtype=float64,
    )
    pose_count = round(duration_s / pose_period_s) + 1
    pose_times_s: FloatArray = asarray(
        [index * pose_period_s for index in range(pose_count)],
        dtype=float64,
    )
    rates_rad_s: FloatArray = asarray((0.12, -0.08, 0.3), dtype=float64)
    rotation_class = cast(Any, Rotation)
    pose_rotations = rotation_class.from_rotvec(pose_times_s[:, None] * rates_rad_s)
    pose_quaternions_xyzw: FloatArray = asarray(
        pose_rotations.as_quat(),
        dtype=float64,
    )

    with (imu_directory / "data.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "#timestamp [ns]",
                "w_RS_S_x [rad s^-1]",
                "w_RS_S_y [rad s^-1]",
                "w_RS_S_z [rad s^-1]",
            )
        )
        for time_s in imu_times_s:
            timestamp_ns = origin_ns + round(float(time_s) * 1.0e9)
            writer.writerow((timestamp_ns, *rates_rad_s))
        ####
    ####
    with (pose_directory / "data.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "#timestamp [ns]",
                "p_RS_R_x [m]",
                "p_RS_R_y [m]",
                "p_RS_R_z [m]",
                "q_RS_x []",
                "q_RS_y []",
                "q_RS_z []",
                "q_RS_w []",
            )
        )
        for time_s, quaternion in zip(pose_times_s, pose_quaternions_xyzw, strict=True):
            timestamp_ns = origin_ns + round(float(time_s) * 1.0e9)
            writer.writerow((timestamp_ns, 0.0, 0.0, 0.0, *quaternion))
        ####
    ####
####


def test_tumvi_preparation_resamples_pose_and_preserves_typed_provenance(tmp_path: Path) -> None:
    sequence_directory = tmp_path / "dataset-room1_512_16"
    _write_tumvi_fixture(sequence_directory)
    output_directory = tmp_path / "prepared-room1"

    prepared = prepare_tumvi_corpus(
        sequence_directory,
        output_directory,
        start_s=0.0,
        duration_s=0.2,
    )

    assert prepared.sample_count == 41
    manifest = ValidationManifest.model_validate_json(prepared.manifest_json.read_text())
    assert manifest.schema_version == 3
    case = manifest.cases[0]
    assert case.truth.attitude_direction == "body_to_world"
    assert case.truth.quaternion_order == "scalar_last"
    assert case.truth.rate_frame == "body"
    assert case.truth.rate_order == "pqr"
    assert case.truth.rate_unit == "radians_per_second"
    assert case.case_id.startswith("tum-vi-dataset-room1_512_16-start-0ns-duration-200000000ns-")
    assert case.measurement_conditions.sample_count == 41
    assert case.measurement_conditions.observed_sample_count == 41
    assert case.measurement_conditions.timestamp_jitter_amplitude_s is None
    assert case.measurement_conditions.timestamp_clock_rate_error_ppm is None
    assert case.measurement_conditions.dropout_probability is None
    assert case.measurement_conditions.dropped_sample_count is None
    assert case.measurement_conditions.orientation_latency_s is None
    assert case.measurement_conditions.rate_latency_s is None
    assert case.measurement_conditions.orientation_noise_sigma_rad is None
    assert case.measurement_conditions.rate_noise_sigma_rad_s is None
    assert case.measurement_conditions.orientation_noise_ar1_coefficient is None
    assert case.measurement_conditions.rate_noise_ar1_coefficient is None
    assert case.source_provenance is not None
    assert case.source_provenance.sequence == "dataset-room1_512_16"
    assert "T_WI" in case.source_provenance.orientation_convention_basis
    assert "body-to-world" in case.source_provenance.orientation_convention_basis
    assert len(case.source_provenance.source_artifacts) == 2
    assert all(len(item.sha256) == 64 for item in case.source_provenance.source_artifacts)
    assert "not a blind test" in case.source_provenance.timing_contract
    assert any(
        "selected source rows span [0, 200000000] ns" in transformation
        for transformation in case.source_provenance.transformations
    )
    legacy_payload = json.loads(prepared.manifest_json.read_text(encoding="utf-8"))
    legacy_payload["schema_version"] = 1
    with pytest.raises(ValidationError, match="requires validation manifest schema_version 2"):
        ValidationManifest.model_validate(legacy_payload)
    ####
    with prepared.trajectory_csv.open("r", encoding="utf-8", newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    ####
    assert len(rows) == 41
    assert float(rows[-1]["time_s"]) == pytest.approx(0.2)
    rotation_class = cast(Any, Rotation)
    expected_midpoint_rotation = rotation_class.from_rotvec(
        asarray((0.12, -0.08, 0.3), dtype=float64) * 0.005
    )
    expected_midpoint_quaternion: FloatArray = asarray(
        expected_midpoint_rotation.as_quat(),
        dtype=float64,
    )
    assert float(rows[1]["time_s"]) == pytest.approx(0.005)
    assert tuple(float(rows[1][name]) for name in ("qx", "qy", "qz", "qw")) == pytest.approx(
        expected_midpoint_quaternion,
        abs=1.0e-8,
    )
    expected_last_rotation = rotation_class.from_rotvec(
        asarray((0.12, -0.08, 0.3), dtype=float64) * 0.2
    )
    expected_last_quaternion: FloatArray = asarray(
        expected_last_rotation.as_quat(),
        dtype=float64,
    )
    assert tuple(float(rows[-1][name]) for name in ("qx", "qy", "qz", "qw")) == pytest.approx(
        expected_last_quaternion,
        abs=1.0e-8,
    )
    assert float(rows[-1]["wz"]) == pytest.approx(0.3)
    assessment = validate_corpus(prepared.manifest_json)
    assert assessment.schema_version == 10
    assert assessment.cases[0].source_provenance == case.source_provenance
    assert assessment.condition_summaries[0].dropped_sample_count_min is None
    assert assessment.condition_summaries[0].dropped_sample_count_max is None
    assert assessment.condition_summaries[0].dropped_sample_count_mean is None
####


def test_tumvi_case_identity_includes_source_content(tmp_path: Path) -> None:
    first_sequence = tmp_path / "first" / "dataset-room1_512_16"
    second_sequence = tmp_path / "second" / "dataset-room1_512_16"
    _write_tumvi_fixture(first_sequence)
    _write_tumvi_fixture(second_sequence)
    second_imu = second_sequence / "mav0" / "imu0" / "data.csv"
    second_imu.write_text(
        second_imu.read_text(encoding="utf-8").replace("0.12", "0.13"),
        encoding="utf-8",
    )

    first = prepare_tumvi_corpus(
        first_sequence,
        tmp_path / "prepared-first",
        start_s=0.0,
        duration_s=0.2,
    )
    second = prepare_tumvi_corpus(
        second_sequence,
        tmp_path / "prepared-second",
        start_s=0.0,
        duration_s=0.2,
    )
    first_case = ValidationManifest.model_validate_json(first.manifest_json.read_text()).cases[0]
    second_case = ValidationManifest.model_validate_json(second.manifest_json.read_text()).cases[0]

    assert first_case.case_id != second_case.case_id
####


def test_tumvi_preparation_rejects_source_changes_during_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence_directory = tmp_path / "dataset-room1_512_16"
    _write_tumvi_fixture(sequence_directory)
    imu_path = sequence_directory / "mav0" / "imu0" / "data.csv"
    output_directory = tmp_path / "prepared-mutating-source"
    interpolate = cast(
        Callable[..., FloatArray],
        tumvi_module._interpolated_pose_quaternions,
    )

    def mutate_after_interpolation(
        pose_times_s: FloatArray,
        pose_quaternions_xyzw: FloatArray,
        query_times_s: FloatArray,
        *,
        max_pose_gap_s: float,
    ) -> FloatArray:
        result = interpolate(
            pose_times_s,
            pose_quaternions_xyzw,
            query_times_s,
            max_pose_gap_s=max_pose_gap_s,
        )
        imu_path.write_text(imu_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return result
    ####

    monkeypatch.setattr(
        tumvi_module,
        "_interpolated_pose_quaternions",
        mutate_after_interpolation,
    )
    with pytest.raises(RuntimeError, match="source CSV files changed"):
        prepare_tumvi_corpus(
            sequence_directory,
            output_directory,
            start_s=0.0,
            duration_s=0.2,
        )
    ####

    assert not output_directory.exists()
####


def test_tumvi_preparation_rejects_large_pose_gaps_without_overwriting(
    tmp_path: Path,
) -> None:
    sequence_directory = tmp_path / "dataset-room1_512_16"
    _write_tumvi_fixture(sequence_directory, pose_period_s=0.02)
    output_directory = tmp_path / "prepared-room1"

    with pytest.raises(ValueError, match="motion-capture gap"):
        prepare_tumvi_corpus(
            sequence_directory,
            output_directory,
            start_s=0.0,
            duration_s=0.2,
            max_pose_gap_s=0.01,
        )
    ####

    prepared = prepare_tumvi_corpus(
        sequence_directory,
        output_directory,
        start_s=0.0,
        duration_s=0.2,
        max_pose_gap_s=0.025,
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_tumvi_corpus(
            sequence_directory,
            output_directory,
            start_s=0.0,
            duration_s=0.2,
        )
    ####
    assert prepared.manifest_json.is_file()
####


def test_exact_pose_timestamps_do_not_count_as_interpolation_gaps(tmp_path: Path) -> None:
    sequence_directory = tmp_path / "dataset-room1_512_16"
    _write_tumvi_fixture(
        sequence_directory,
        duration_s=0.4,
        imu_period_s=0.02,
        pose_period_s=0.02,
    )
    prepared = prepare_tumvi_corpus(
        sequence_directory,
        tmp_path / "prepared-exact-poses",
        start_s=0.0,
        duration_s=0.4,
        max_pose_gap_s=0.01,
    )

    assert prepared.sample_count == 21
####


@pytest.mark.parametrize(
    ("relative_path", "original_header", "replacement_header", "expected_error"),
    (
        (
            Path("mav0/imu0/data.csv"),
            "#timestamp [ns]",
            "#timestamp [s]",
            "timestamp column must declare [ns]",
        ),
        (
            Path("mav0/imu0/data.csv"),
            "w_RS_S_x [rad s^-1]",
            "w_RS_S_x [deg s^-1]",
            "must declare radians per second units",
        ),
        (
            Path("mav0/mocap0/data.csv"),
            "q_RS_x []",
            "q_RS_x [deg]",
            "must declare unitless quaternion units",
        ),
        (
            Path("mav0/imu0/data.csv"),
            "w_RS_S_x [rad s^-1]",
            "gyro_x [rad s^-1]",
            "expected one TUM-VI 'w_rs_s' 'x' column",
        ),
    ),
)
def test_tumvi_preparation_rejects_ambiguous_schema_units_and_aliases(
    tmp_path: Path,
    relative_path: Path,
    original_header: str,
    replacement_header: str,
    expected_error: str,
) -> None:
    sequence_directory = tmp_path / "dataset-room1_512_16"
    _write_tumvi_fixture(sequence_directory)
    source_path = sequence_directory / relative_path
    source_text = source_path.read_text(encoding="utf-8")
    assert original_header in source_text
    source_path.write_text(
        source_text.replace(original_header, replacement_header, 1),
        encoding="utf-8",
    )
    output_directory = tmp_path / "prepared-invalid"

    with pytest.raises(ValueError) as error:
        prepare_tumvi_corpus(
            sequence_directory,
            output_directory,
            start_s=0.0,
            duration_s=0.2,
        )
    ####

    assert expected_error in str(error.value)
    assert not output_directory.exists()
####


def test_sparse_tumvi_study_keeps_native_pose_and_rate_sample_counts(tmp_path: Path) -> None:
    sequence_directory = tmp_path / "dataset-room1_512_16"
    _write_tumvi_fixture(sequence_directory)

    report = run_sparse_tumvi_study(
        sequence_directory,
        start_s=0.0,
        duration_s=0.2,
        pose_cadences_s=(0.1,),
    )

    assert report.observation_model == "sparse_orientation_anchors"
    assert report.lineage_manifest_sha256 is None
    assert report.upstream_lineage_provenance is None
    assert len(report.cadence_assessments) == 1
    assessment = report.cadence_assessments[0]
    assert assessment.orientation_anchor_count == 3
    assert assessment.dense_rate_sample_count == 41
    assert assessment.actual_anchor_span_s == pytest.approx(0.2)
    assert assessment.mean_actual_anchor_cadence_s == pytest.approx(0.1)
    assert assessment.maximum_grid_offset_s == pytest.approx(0.0)
    assert report.source_provenance.transformations
    assert "not independently verified" in report.source_provenance.license
    assert not any("SLERP" in item for item in report.source_provenance.transformations)
####


def test_sparse_tumvi_study_carries_only_verified_upstream_lineage(tmp_path: Path) -> None:
    sequence_directory = tmp_path / "dataset-room1_512_16"
    _write_tumvi_fixture(sequence_directory)
    prepared = prepare_tumvi_corpus(
        sequence_directory,
        tmp_path / "aligned-corpus",
        start_s=0.0,
        duration_s=0.2,
    )

    report = run_sparse_tumvi_study(
        sequence_directory,
        start_s=0.0,
        duration_s=0.2,
        pose_cadences_s=(0.1,),
        lineage_manifest_path=prepared.manifest_json,
    )

    assert report.lineage_manifest_sha256 == sha256_file(prepared.manifest_json)
    assert report.upstream_lineage_provenance is not None
    assert report.upstream_lineage_provenance.source_artifacts == (
        report.source_provenance.source_artifacts
    )
    assert not any("SLERP" in item for item in report.source_provenance.transformations)
    assert any("SLERP" in item for item in report.upstream_lineage_provenance.transformations)
####


def test_sparse_tumvi_study_rejects_lineage_for_different_input_bytes(tmp_path: Path) -> None:
    sequence_directory = tmp_path / "dataset-room1_512_16"
    _write_tumvi_fixture(sequence_directory)
    prepared = prepare_tumvi_corpus(
        sequence_directory,
        tmp_path / "aligned-corpus",
        start_s=0.0,
        duration_s=0.2,
    )
    imu_path = sequence_directory / "mav0" / "imu0" / "data.csv"
    imu_path.write_text(imu_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not fingerprint these exact input files"):
        run_sparse_tumvi_study(
            sequence_directory,
            start_s=0.0,
            duration_s=0.2,
            pose_cadences_s=(0.1,),
            lineage_manifest_path=prepared.manifest_json,
        )
    ####
####


def test_tumvi_preparation_rejects_timestamp_outside_signed_int64(
    tmp_path: Path,
) -> None:
    sequence_directory = tmp_path / "dataset-room1_512_16"
    _write_tumvi_fixture(sequence_directory)
    source_path = sequence_directory / "mav0" / "imu0" / "data.csv"
    with source_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    ####
    rows[1][0] = str(1 << 63)
    with source_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    ####

    with pytest.raises(ValueError, match="timestamp must fit signed 64-bit nanoseconds"):
        prepare_tumvi_corpus(
            sequence_directory,
            tmp_path / "prepared-out-of-range-timestamp",
            start_s=0.0,
            duration_s=0.2,
        )
    ####
####


def test_tumvi_preparation_rejects_timestamp_span_outside_signed_int64(
    tmp_path: Path,
) -> None:
    sequence_directory = tmp_path / "dataset-room1_512_16"
    _write_tumvi_fixture(sequence_directory)
    source_path = sequence_directory / "mav0" / "imu0" / "data.csv"
    with source_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    ####
    rows[1][0] = str(-(1 << 63))
    rows[-1][0] = str((1 << 63) - 1)
    with source_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    ####

    with pytest.raises(ValueError, match="timestamp span must fit signed 64-bit"):
        prepare_tumvi_corpus(
            sequence_directory,
            tmp_path / "prepared-overflowing-span",
            start_s=0.0,
            duration_s=0.2,
        )
    ####
####


@pytest.mark.parametrize(("start_s", "duration_s"), ((1.0e308, 0.2), (0.0, 1.0e308)))
def test_tumvi_preparation_rejects_extreme_finite_segment_bounds(
    tmp_path: Path,
    start_s: float,
    duration_s: float,
) -> None:
    sequence_directory = tmp_path / "dataset-room1_512_16"
    _write_tumvi_fixture(sequence_directory)

    with pytest.raises(ValueError, match="outside common IMU/mocap coverage"):
        prepare_tumvi_corpus(
            sequence_directory,
            tmp_path / "prepared-extreme-segment",
            start_s=start_s,
            duration_s=duration_s,
        )
    ####
####


def test_tumvi_preparation_rejects_subnanosecond_duration(tmp_path: Path) -> None:
    sequence_directory = tmp_path / "dataset-room1_512_16"
    _write_tumvi_fixture(sequence_directory)

    with pytest.raises(ValueError, match="at least one nanosecond"):
        prepare_tumvi_corpus(
            sequence_directory,
            tmp_path / "prepared-subnanosecond-segment",
            start_s=0.0,
            duration_s=1.0e-12,
        )
    ####
####
