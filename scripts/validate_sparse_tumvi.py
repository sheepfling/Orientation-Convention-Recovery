"""Run a provenance-recorded sparse-pose/dense-rate study on a TUM-VI room sequence."""

from __future__ import annotations

import argparse
import hashlib
import re
from math import isfinite
from pathlib import Path
from typing import Literal

from numpy import (
    abs as np_abs,
)
from numpy import (
    all as np_all,
)
from numpy import (
    any as np_any,
)
from numpy import (
    asarray,
    diff,
    float64,
    int64,
    searchsorted,
    sqrt,
)
from numpy import (
    isfinite as np_isfinite,
)
from numpy import (
    max as np_max,
)
from numpy import (
    sum as np_sum,
)
from numpy.typing import NDArray

from attitude_convention_inference import (
    AttitudeRepresentation,
    SparseOrientationRateData,
    __version__,
    infer_sparse_orientation_conventions,
)
from scripts.tumvi_data import NANOSECONDS_PER_SECOND, read_timed_vectors, sha256_file
from scripts.tumvi_sparse_models import (
    TUMVI_TRUTH,
    SparseCadenceAssessment,
    SparseTUMVIStudyReport,
)
from scripts.validate_corpus import (
    DatasetProvenance,
    SourceArtifactFingerprint,
    ValidationManifest,
)

__all__ = ["SparseCadenceAssessment", "SparseTUMVIStudyReport", "run_sparse_tumvi_study"]

FloatArray = NDArray[float64]
IntegerArray = NDArray[int64]
_TUMVI_SEQUENCE_PATTERN = re.compile(r"dataset-room\d+(?:_.+)?")


def _source_paths(sequence_directory: Path) -> tuple[Path, Path, Path]:
    sequence_root = sequence_directory.resolve()
    if not sequence_root.is_dir():
        raise FileNotFoundError(f"TUM-VI sequence directory does not exist: {sequence_root}")
    ####
    if _TUMVI_SEQUENCE_PATTERN.fullmatch(sequence_root.name) is None:
        raise ValueError(
            "Only TUM-VI room sequences with full-trajectory ground truth are supported."
        )
    ####
    imu_path = sequence_root / "mav0" / "imu0" / "data.csv"
    pose_path = sequence_root / "mav0" / "mocap0" / "data.csv"
    if not imu_path.is_file() or not pose_path.is_file():
        raise FileNotFoundError(
            "Expected calibrated TUM-VI files at mav0/imu0/data.csv and mav0/mocap0/data.csv."
        )
    ####
    return sequence_root, imu_path, pose_path
####


def _nearest_native_indices(
    source_timestamps_ns: IntegerArray,
    target_timestamps_ns: tuple[int, ...],
) -> IntegerArray:
    indices: list[int] = []
    for target_ns in target_timestamps_ns:
        right_index = int(searchsorted(source_timestamps_ns, target_ns, side="left"))
        candidates = tuple(
            index
            for index in (right_index - 1, right_index)
            if 0 <= index < source_timestamps_ns.size
        )
        if not candidates:
            raise ValueError("A requested pose-grid time has no native mocap sample nearby.")
        ####
        selected_index = min(
            candidates,
            key=lambda index: abs(int(source_timestamps_ns[index]) - target_ns),
        )
        if indices and selected_index <= indices[-1]:
            raise ValueError(
                "The requested pose cadence is finer than the native mocap stream; "
                "it selects duplicate or out-of-order anchors."
            )
        ####
        indices.append(selected_index)
    ####
    return asarray(indices, dtype=int64)
####


def _bracketing_rate_slice(
    rate_timestamps_ns: IntegerArray,
    first_anchor_ns: int,
    last_anchor_ns: int,
) -> slice:
    first_index = int(searchsorted(rate_timestamps_ns, first_anchor_ns, side="right")) - 1
    last_index = int(searchsorted(rate_timestamps_ns, last_anchor_ns, side="left"))
    if first_index < 0 or last_index >= rate_timestamps_ns.size:
        raise ValueError("The dense rate stream does not bracket all selected pose anchors.")
    ####
    return slice(first_index, last_index + 1)
####


def _validated_unit_quaternions(quaternions_xyzw: FloatArray) -> FloatArray:
    norms = sqrt(np_sum(quaternions_xyzw * quaternions_xyzw, axis=1))
    if not np_all(np_isfinite(norms)) or np_any(norms == 0.0):
        raise ValueError("TUM-VI pose quaternions must be finite and nonzero.")
    ####
    if np_max(np_abs(norms - 1.0)) > 1.0e-3:
        raise ValueError("TUM-VI pose quaternions must be unit length within 1e-3.")
    ####
    return quaternions_xyzw / norms[:, None]
####


def _timestamp_seconds(timestamps_ns: IntegerArray, origin_ns: int) -> FloatArray:
    return asarray(
        (timestamps_ns - origin_ns).astype(float64) / NANOSECONDS_PER_SECOND,
        dtype=float64,
    )
####


def _cadence_assessment(
    *,
    cadence_s: float,
    cadence_ns: int,
    window_start_ns: int,
    window_end_ns: int,
    common_origin_ns: int,
    pose_timestamps_ns: IntegerArray,
    pose_quaternions_xyzw: FloatArray,
    rate_timestamps_ns: IntegerArray,
    rates_rad_s: FloatArray,
) -> SparseCadenceAssessment:
    target_timestamps_ns = tuple(range(window_start_ns, window_end_ns + 1, cadence_ns))
    if len(target_timestamps_ns) < 3:
        raise ValueError(
            f"A {cadence_s:g} s pose cadence over the requested window yields fewer than "
            "three anchors."
        )
    ####
    anchor_indices = _nearest_native_indices(pose_timestamps_ns, target_timestamps_ns)
    anchor_timestamps_ns = pose_timestamps_ns[anchor_indices]
    anchor_quaternions_xyzw = _validated_unit_quaternions(pose_quaternions_xyzw[anchor_indices])
    rate_slice = _bracketing_rate_slice(
        rate_timestamps_ns,
        int(anchor_timestamps_ns[0]),
        int(anchor_timestamps_ns[-1]),
    )
    data = SparseOrientationRateData(
        representation=AttitudeRepresentation.QUATERNION,
        orientation_anchors=anchor_quaternions_xyzw,
        orientation_anchor_times_s=_timestamp_seconds(anchor_timestamps_ns, common_origin_ns),
        rates=rates_rad_s[rate_slice],
        rate_times_s=_timestamp_seconds(rate_timestamps_ns[rate_slice], common_origin_ns),
    )
    report = infer_sparse_orientation_conventions(data)
    labeled_candidate = next(
        (candidate for candidate in report.rankings if candidate.hypothesis == TUMVI_TRUTH),
        None,
    )
    labeled_class_id = None if labeled_candidate is None else labeled_candidate.equivalence_class_id
    label_in_evidence = (
        labeled_class_id is not None and labeled_class_id in report.evidence_contender_class_ids
    )
    label_is_data_winner = (
        labeled_class_id is not None and labeled_class_id in report.data_winning_class_ids
    )
    recommendation: Literal["matched_label", "different_class", "abstained"]
    if labeled_class_id is not None and report.operational_class_id == labeled_class_id:
        recommendation = "matched_label"
    elif report.operational_class_id:
        recommendation = "different_class"
    else:
        recommendation = "abstained"
    ####
    anchor_times_s = data.orientation_anchor_times_s
    actual_cadences_s = diff(anchor_times_s)
    maximum_grid_offset_ns = max(
        abs(int(anchor_timestamp_ns) - target_timestamp_ns)
        for anchor_timestamp_ns, target_timestamp_ns in zip(
            anchor_timestamps_ns, target_timestamps_ns, strict=True
        )
    )
    return SparseCadenceAssessment(
        target_pose_cadence_s=cadence_s,
        orientation_anchor_count=data.anchor_count,
        dense_rate_sample_count=data.rate_sample_count,
        actual_anchor_span_s=float(anchor_times_s[-1] - anchor_times_s[0]),
        mean_actual_anchor_cadence_s=float(np_sum(actual_cadences_s) / actual_cadences_s.size),
        maximum_grid_offset_s=maximum_grid_offset_ns / NANOSECONDS_PER_SECOND,
        labeled_hypothesis_rank=(None if labeled_candidate is None else labeled_candidate.rank),
        labeled_data_rank=(None if labeled_candidate is None else labeled_candidate.data_rank),
        label_in_evidence_contenders=label_in_evidence,
        label_is_data_winner=label_is_data_winner,
        recommendation=recommendation,
        operational_class_id=report.operational_class_id,
        operational_reason=report.operational_reason,
        evidence_contender_class_count=len(report.evidence_contender_class_ids),
    )
####


def _tumvi_provenance(
    *,
    sequence_name: str,
    imu_sha256: str,
    pose_sha256: str,
    start_offset_ns: int,
    duration_ns: int,
    cadences_ns: tuple[int, ...],
    license_statement: str,
) -> DatasetProvenance:
    return DatasetProvenance.model_validate(
        {
            "dataset_name": "TUM VI Visual-Inertial Dataset",
            "citation": (
                "D. Schubert et al., The TUM VI Benchmark for Evaluating Visual-Inertial "
                "Odometry, IROS 2018, arXiv:1804.06120."
            ),
            "source_url": "https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset",
            "license": license_statement,
            "sequence": sequence_name,
            "orientation_convention_basis": (
                "The calibrated room-sequence motion-capture pose is T_WI: it maps IMU/body "
                "coordinates into the motion-capture world/reference frame. Source "
                "q_RS_x/y/z/w columns are used as that body-to-world quaternion."
            ),
            "rate_convention_basis": (
                "The calibrated w_RS_S_x/y/z columns are angular velocity in the IMU "
                "sensor/body frame in rad/s; component order is x/y/z (p/q/r)."
            ),
            "timing_contract": (
                "The calibrated export provides aligned IMU and mocap timestamps. Upstream "
                "alignment uses gyro measurements and differentiated ground-truth orientation, "
                "so this is not a blind test of time-offset or raw sensor-axis recovery."
            ),
            "source_artifacts": (
                SourceArtifactFingerprint(file_name="mav0/imu0/data.csv", sha256=imu_sha256),
                SourceArtifactFingerprint(file_name="mav0/mocap0/data.csv", sha256=pose_sha256),
            ),
            "transformations": (
                "Read calibrated TUM-VI source files using explicit nanosecond, rad/s, and "
                "unitless-quaternion column declarations.",
                "Restrict target pose times to the requested window within common IMU/mocap "
                f"coverage; start offset is {start_offset_ns} ns and duration is {duration_ns} ns.",
                "For each requested cadence, select the nearest native mocap quaternion to "
                "each target time; do not interpolate or resample orientations.",
                "Retain native IMU rates and timestamps, adding only the samples needed to "
                "bracket the first and last selected pose anchors.",
                "Convert integer nanosecond timestamps to seconds relative to the common "
                "coverage origin before inference.",
                "Verify both source CSV hashes are unchanged after all cadence evaluations.",
                "Requested cadence offsets in nanoseconds: "
                + ", ".join(str(cadence_ns) for cadence_ns in cadences_ns),
            ),
        }
    )
####


def _load_lineage_provenance(
    lineage_manifest_path: Path | None,
    *,
    sequence_name: str,
    imu_sha256: str,
    pose_sha256: str,
) -> tuple[DatasetProvenance | None, str | None]:
    if lineage_manifest_path is None:
        return None, None
    ####
    if not lineage_manifest_path.is_file():
        raise FileNotFoundError(f"Lineage manifest does not exist: {lineage_manifest_path}")
    ####
    manifest_sha256 = sha256_file(lineage_manifest_path)
    manifest = ValidationManifest.model_validate_json(
        lineage_manifest_path.read_text(encoding="utf-8")
    )
    matching_cases = tuple(
        case
        for case in manifest.cases
        if case.source_provenance is not None and case.source_provenance.sequence == sequence_name
    )
    if not matching_cases:
        raise ValueError(
            f"The lineage manifest must contain a provenance record for sequence {sequence_name!r}."
        )
    ####
    provenance = matching_cases[0].source_provenance
    if provenance is None:
        raise RuntimeError("A matching lineage case unexpectedly has no source provenance.")
    ####
    known_hashes = {artifact.file_name: artifact.sha256 for artifact in provenance.source_artifacts}
    expected_hashes = {
        "mav0/imu0/data.csv": imu_sha256,
        "mav0/mocap0/data.csv": pose_sha256,
    }
    mismatches = tuple(
        name for name, digest in expected_hashes.items() if known_hashes.get(name) != digest
    )
    if mismatches:
        raise ValueError(
            "The lineage manifest does not fingerprint these exact input files: "
            + ", ".join(mismatches)
        )
    ####
    return provenance, manifest_sha256
####


def run_sparse_tumvi_study(
    sequence_directory: Path,
    *,
    start_s: float,
    duration_s: float,
    pose_cadences_s: tuple[float, ...],
    lineage_manifest_path: Path | None = None,
) -> SparseTUMVIStudyReport:
    """Evaluate sparse native poses against the separately timestamped dense IMU stream."""

    if not isfinite(start_s) or start_s < 0.0:
        raise ValueError("start_s must be finite and non-negative.")
    ####
    if not isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be positive and finite.")
    ####
    if not pose_cadences_s:
        raise ValueError("At least one pose cadence is required.")
    ####
    if any(not isfinite(cadence) or cadence <= 0.0 for cadence in pose_cadences_s):
        raise ValueError("Every pose cadence must be positive and finite.")
    ####
    cadence_values_ns = tuple(
        round(cadence_s * NANOSECONDS_PER_SECOND) for cadence_s in pose_cadences_s
    )
    if any(cadence_ns <= 0 for cadence_ns in cadence_values_ns):
        raise ValueError("Every pose cadence must resolve to at least one nanosecond.")
    ####
    if len(set(cadence_values_ns)) != len(cadence_values_ns):
        raise ValueError("Pose cadences must be unique at nanosecond precision.")
    ####
    sequence_root = sequence_directory.resolve()
    sequence_root, imu_path, pose_path = _source_paths(sequence_root)
    imu_sha256 = sha256_file(imu_path)
    pose_sha256 = sha256_file(pose_path)
    lineage_provenance, lineage_manifest_sha256 = _load_lineage_provenance(
        lineage_manifest_path,
        sequence_name=sequence_root.name,
        imu_sha256=imu_sha256,
        pose_sha256=pose_sha256,
    )
    imu_stream = read_timed_vectors(
        imu_path,
        axes=("x", "y", "z"),
        prefix="w_rs_s",
        expected_unit="rad/s",
    )
    pose_stream = read_timed_vectors(
        pose_path,
        axes=("x", "y", "z", "w"),
        prefix="q_rs",
        expected_unit="unitless",
    )
    imu_timestamps_ns = imu_stream.timestamps_ns
    pose_timestamps_ns = pose_stream.timestamps_ns
    common_origin_ns = max(int(imu_timestamps_ns[0]), int(pose_timestamps_ns[0]))
    common_end_ns = min(int(imu_timestamps_ns[-1]), int(pose_timestamps_ns[-1]))
    if common_end_ns <= common_origin_ns:
        raise ValueError("TUM-VI IMU and mocap files have no overlapping timestamp coverage.")
    ####
    available_duration_ns = common_end_ns - common_origin_ns
    start_offset_ns = round(start_s * NANOSECONDS_PER_SECOND)
    duration_ns = round(duration_s * NANOSECONDS_PER_SECOND)
    if duration_ns <= 0:
        raise ValueError("duration_s must resolve to at least one nanosecond.")
    ####
    if (
        start_offset_ns > available_duration_ns
        or duration_ns > available_duration_ns - start_offset_ns
    ):
        available_duration_s = available_duration_ns / NANOSECONDS_PER_SECOND
        raise ValueError(
            "Requested segment is outside common IMU/mocap coverage "
            f"({available_duration_s:.3f} seconds available)."
        )
    ####
    window_start_ns = common_origin_ns + start_offset_ns
    window_end_ns = window_start_ns + duration_ns
    first_pose_index = int(searchsorted(pose_timestamps_ns, common_origin_ns, side="left"))
    last_pose_index = int(searchsorted(pose_timestamps_ns, common_end_ns, side="right"))
    common_pose_timestamps_ns = pose_timestamps_ns[first_pose_index:last_pose_index]
    common_pose_quaternions_xyzw = pose_stream.values[first_pose_index:last_pose_index]
    common_pose_quaternions_xyzw = _validated_unit_quaternions(common_pose_quaternions_xyzw)
    if (
        common_pose_timestamps_ns.size < 3
        or int(common_pose_timestamps_ns[0]) > window_start_ns
        or int(common_pose_timestamps_ns[-1]) < window_end_ns
    ):
        raise ValueError("The requested interval lacks native mocap coverage at its bounds.")
    ####
    study_results = tuple(
        _cadence_assessment(
            cadence_s=cadence_s,
            cadence_ns=cadence_ns,
            window_start_ns=window_start_ns,
            window_end_ns=window_end_ns,
            common_origin_ns=common_origin_ns,
            pose_timestamps_ns=common_pose_timestamps_ns,
            pose_quaternions_xyzw=common_pose_quaternions_xyzw,
            rate_timestamps_ns=imu_timestamps_ns,
            rates_rad_s=imu_stream.values,
        )
        for cadence_s, cadence_ns in zip(
            pose_cadences_s,
            cadence_values_ns,
            strict=True,
        )
    )
    if sha256_file(imu_path) != imu_sha256 or sha256_file(pose_path) != pose_sha256:
        raise RuntimeError("TUM-VI source CSV files changed during sparse validation.")
    ####
    if (
        lineage_manifest_path is not None
        and sha256_file(lineage_manifest_path) != lineage_manifest_sha256
    ):
        raise RuntimeError("The lineage manifest changed during sparse validation.")
    ####
    identity_material = (
        f"{imu_sha256}\0{pose_sha256}\0{lineage_manifest_sha256}\0"
        f"{start_offset_ns}\0{duration_ns}\0"
        f"{','.join(str(value) for value in cadence_values_ns)}"
    )
    source_identity = hashlib.sha256(identity_material.encode("ascii")).hexdigest()[:16]
    provenance = _tumvi_provenance(
        sequence_name=sequence_root.name,
        imu_sha256=imu_sha256,
        pose_sha256=pose_sha256,
        start_offset_ns=start_offset_ns,
        duration_ns=duration_ns,
        cadences_ns=cadence_values_ns,
        license_statement=(
            "Upstream TUM-VI data is CC BY 4.0; the third-party mirror/re-export does not "
            "state a separate license."
            if lineage_provenance is not None
            and any(
                artifact.file_name.startswith("mirror/")
                for artifact in lineage_provenance.source_artifacts
            )
            else (
                "Upstream TUM-VI data is CC BY 4.0; provenance and license of these exact "
                "input files were not independently verified."
            )
        ),
    )
    return SparseTUMVIStudyReport(
        study_id=source_identity,
        package_version=__version__,
        observation_model="sparse_orientation_anchors",
        sequence=sequence_root.name,
        common_origin_timestamp_ns=common_origin_ns,
        requested_start_offset_ns=start_offset_ns,
        requested_duration_ns=duration_ns,
        truth=TUMVI_TRUTH,
        source_provenance=provenance,
        lineage_manifest_sha256=lineage_manifest_sha256,
        upstream_lineage_provenance=lineage_provenance,
        unmeasured_conditions=(
            "orientation_noise",
            "rate_noise",
            "orientation_latency",
            "rate_latency",
            "timestamp_jitter",
            "dropout",
            "timestamp_clock_rate_error",
        ),
        cadence_assessments=study_results,
    )
####


def _write_report(report: SparseTUMVIStudyReport, output_path: Path) -> None:
    payload = report.model_dump_json(indent=2) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(payload)
    ####
####


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence_directory", type=Path, help="extracted TUM-VI room sequence")
    parser.add_argument("--start-s", type=float, default=0.0, help="offset from common coverage")
    parser.add_argument("--duration-s", type=float, required=True, help="bounded window duration")
    parser.add_argument(
        "--pose-cadences-s",
        type=float,
        nargs="+",
        default=(0.2, 0.4, 0.8, 1.6, 3.2),
        help="nominal sparse pose cadences (default: 0.2 0.4 0.8 1.6 3.2)",
    )
    parser.add_argument(
        "--lineage-manifest",
        type=Path,
        help=(
            "optional validation manifest that fingerprints these exact TUM-VI inputs and "
            "records any upstream conversion"
        ),
    )
    parser.add_argument("--output", type=Path, help="new JSON report path; defaults to stdout")
    args = parser.parse_args()
    if args.output is not None:
        sequence_root = args.sequence_directory.resolve()
        output_path = args.output.resolve()
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing report: {output_path}")
        ####
        if output_path == sequence_root or sequence_root in output_path.parents:
            raise ValueError("The output report must be outside the source sequence directory.")
        ####
    ####
    report = run_sparse_tumvi_study(
        args.sequence_directory,
        start_s=args.start_s,
        duration_s=args.duration_s,
        pose_cadences_s=tuple(args.pose_cadences_s),
        lineage_manifest_path=args.lineage_manifest,
    )
    if args.output is None:
        print(report.model_dump_json(indent=2))
    else:
        _write_report(report, args.output)
        print(f"Sparse TUM-VI report written to {args.output}")
    ####
####


if __name__ == "__main__":
    main()
####
