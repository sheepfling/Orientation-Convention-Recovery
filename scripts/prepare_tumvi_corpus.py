"""Prepare a bounded, provenance-recorded TUM-VI room sequence for validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, cast

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
    median,
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
from scipy.spatial.transform import Rotation, Slerp

from scripts.tumvi_data import NANOSECONDS_PER_SECOND, read_timed_vectors, sha256_file
from scripts.validate_corpus import (
    DatasetProvenance,
    SourceArtifactFingerprint,
    ValidationCase,
    ValidationManifest,
)

FloatArray = NDArray[float64]


@dataclass(frozen=True, slots=True)
class PreparedTUMVICorpus:
    """Paths and sample count for a completed local TUM-VI validation corpus."""

    output_directory: Path
    trajectory_csv: Path
    manifest_json: Path
    sample_count: int
####


def _interpolated_pose_quaternions(
    pose_times_s: FloatArray,
    pose_quaternions_xyzw: FloatArray,
    query_times_s: FloatArray,
    *,
    max_pose_gap_s: float,
) -> FloatArray:
    quaternion_norms = sqrt(np_sum(pose_quaternions_xyzw * pose_quaternions_xyzw, axis=1))
    if not np_all(np_isfinite(quaternion_norms)) or np_any(quaternion_norms == 0.0):
        raise ValueError("TUM-VI pose quaternions must be finite and nonzero.")
    ####
    if np_max(np_abs(quaternion_norms - 1.0)) > 1.0e-3:
        raise ValueError("TUM-VI pose quaternions must be unit length within 1e-3.")
    ####
    normalized_quaternions = pose_quaternions_xyzw / quaternion_norms[:, None]
    right_indices = searchsorted(pose_times_s, query_times_s, side="right")
    left_indices = right_indices - 1
    right_indices = right_indices.clip(0, pose_times_s.size - 1)
    left_indices = left_indices.clip(0, pose_times_s.size - 1)
    pose_gaps_s = pose_times_s[right_indices] - pose_times_s[left_indices]
    exact_indices = searchsorted(pose_times_s, query_times_s, side="left")
    exact_indices = exact_indices.clip(0, pose_times_s.size - 1)
    exact_queries = pose_times_s[exact_indices] == query_times_s
    pose_gaps_s[exact_queries] = 0.0
    if np_max(pose_gaps_s) > max_pose_gap_s:
        raise ValueError(
            "TUM-VI pose interpolation would cross a motion-capture gap longer than "
            f"{max_pose_gap_s:g} seconds."
        )
    ####
    rotation_class = cast(Any, Rotation)
    rotations = rotation_class.from_quat(normalized_quaternions)
    interpolator = Slerp(pose_times_s, rotations)
    interpolated = interpolator(query_times_s)
    return asarray(cast(Any, interpolated).as_quat(), dtype=float64)
####


def prepare_tumvi_corpus(
    sequence_directory: Path,
    output_directory: Path,
    *,
    start_s: float,
    duration_s: float,
    max_pose_gap_s: float = 0.025,
) -> PreparedTUMVICorpus:
    """Resample one calibrated TUM-VI room sequence and record its provenance.

    The output contains a derived CSV and a manifest only; source dataset files are
    never copied. Time zero is the first common IMU/motion-capture timestamp.
    """

    if start_s < 0.0 or not isfinite(start_s):
        raise ValueError("start_s must be finite and non-negative.")
    ####
    if duration_s <= 0.0 or not isfinite(duration_s):
        raise ValueError("duration_s must be positive and finite.")
    ####
    if max_pose_gap_s <= 0.0 or not isfinite(max_pose_gap_s):
        raise ValueError("max_pose_gap_s must be positive and finite.")
    ####
    sequence_root = sequence_directory.resolve()
    output_root = output_directory.resolve()
    if not sequence_root.is_dir():
        raise FileNotFoundError(f"TUM-VI sequence directory does not exist: {sequence_root}")
    ####
    if re.fullmatch(r"dataset-room\d+(?:_.+)?", sequence_root.name) is None:
        raise ValueError(
            "Only TUM-VI room sequences with full-trajectory ground truth are supported."
        )
    ####
    if sequence_root == output_root or sequence_root in output_root.parents:
        raise ValueError("The output directory must be outside the source sequence directory.")
    ####
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_root}")
    ####
    imu_path = sequence_root / "mav0" / "imu0" / "data.csv"
    pose_path = sequence_root / "mav0" / "mocap0" / "data.csv"
    if not imu_path.is_file() or not pose_path.is_file():
        raise FileNotFoundError(
            "Expected calibrated TUM-VI files at mav0/imu0/data.csv and mav0/mocap0/data.csv."
        )
    ####
    imu_sha256 = sha256_file(imu_path)
    pose_sha256 = sha256_file(pose_path)
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
    rates_rad_s = imu_stream.values
    pose_timestamps_ns = pose_stream.timestamps_ns
    pose_quaternions_xyzw = pose_stream.values
    common_start_ns = max(int(imu_timestamps_ns[0]), int(pose_timestamps_ns[0]))
    common_end_ns = min(int(imu_timestamps_ns[-1]), int(pose_timestamps_ns[-1]))
    if common_end_ns <= common_start_ns:
        raise ValueError("TUM-VI IMU and mocap files have no overlapping timestamp coverage.")
    ####
    common_duration_ns = common_end_ns - common_start_ns
    common_duration_s = common_duration_ns / NANOSECONDS_PER_SECOND
    if start_s > common_duration_s or duration_s > common_duration_s:
        raise ValueError(
            "Requested segment is outside common IMU/mocap coverage "
            f"({common_duration_s:.3f} seconds available)."
        )
    ####
    requested_start_offset_ns = round(start_s * NANOSECONDS_PER_SECOND)
    requested_duration_ns = round(duration_s * NANOSECONDS_PER_SECOND)
    if requested_duration_ns == 0:
        raise ValueError("duration_s must resolve to at least one nanosecond.")
    ####
    if (
        requested_start_offset_ns > common_duration_ns
        or requested_duration_ns > common_duration_ns - requested_start_offset_ns
    ):
        raise ValueError(
            "Requested segment is outside common IMU/mocap coverage "
            f"({common_duration_s:.3f} seconds available)."
        )
    ####
    start_timestamp_ns = common_start_ns + requested_start_offset_ns
    end_timestamp_ns = start_timestamp_ns + requested_duration_ns
    selected = (imu_timestamps_ns >= start_timestamp_ns) & (imu_timestamps_ns <= end_timestamp_ns)
    selected_timestamps_ns = imu_timestamps_ns[selected]
    selected_rates_rad_s = rates_rad_s[selected]
    if selected_timestamps_ns.size < 21:
        raise ValueError("The selected TUM-VI segment must contain at least 21 IMU samples.")
    ####
    common_origin_ns = common_start_ns
    selected_start_offset_ns = int(selected_timestamps_ns[0]) - common_origin_ns
    selected_end_offset_ns = int(selected_timestamps_ns[-1]) - common_origin_ns
    pose_times_s = (pose_timestamps_ns - common_origin_ns).astype(float64) / NANOSECONDS_PER_SECOND
    query_times_s = (selected_timestamps_ns - common_origin_ns).astype(
        float64
    ) / NANOSECONDS_PER_SECOND
    interpolated_quaternions = _interpolated_pose_quaternions(
        pose_times_s,
        pose_quaternions_xyzw,
        query_times_s,
        max_pose_gap_s=max_pose_gap_s,
    )
    output_times_s = (selected_timestamps_ns - common_origin_ns).astype(
        float64
    ) / NANOSECONDS_PER_SECOND
    sample_period_s = float(
        median(diff(selected_timestamps_ns).astype(float64)) / NANOSECONDS_PER_SECOND
    )

    if sha256_file(imu_path) != imu_sha256 or sha256_file(pose_path) != pose_sha256:
        raise RuntimeError("TUM-VI source CSV files changed while the corpus was being prepared.")
    ####
    source_identity = hashlib.sha256(f"{imu_sha256}\0{pose_sha256}".encode("ascii")).hexdigest()[
        :12
    ]
    source_provenance = DatasetProvenance.model_validate(
        {
            "dataset_name": "TUM VI Visual-Inertial Dataset",
            "citation": (
                "D. Schubert et al., The TUM VI Benchmark for Evaluating Visual-Inertial "
                "Odometry, IROS 2018, arXiv:1804.06120."
            ),
            "source_url": "https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset",
            "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
            "sequence": sequence_root.name,
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
                SourceArtifactFingerprint(
                    file_name="mav0/imu0/data.csv",
                    sha256=imu_sha256,
                ),
                SourceArtifactFingerprint(
                    file_name="mav0/mocap0/data.csv",
                    sha256=pose_sha256,
                ),
            ),
            "transformations": (
                "Restrict to the common IMU/mocap timestamp range and requested start/duration.",
                "Select IMU rows in the inclusive requested interval "
                f"[{requested_start_offset_ns}, "
                f"{requested_start_offset_ns + requested_duration_ns}] "
                "ns relative to the common coverage start; the selected source rows span "
                f"[{selected_start_offset_ns}, {selected_end_offset_ns}] ns.",
                "Convert integer nanosecond timestamps to seconds relative to common coverage "
                "start.",
                "Verify both source CSV hashes are unchanged through conversion.",
                "SLERP labeled mocap orientations to gyro timestamps; reject pose gaps above the "
                "configured limit.",
                "Write canonical qx/qy/qz/qw and wx/wy/wz columns; no source files are copied.",
            ),
        }
    )
    case = ValidationCase.model_validate(
        {
            "case_id": (
                f"tum-vi-{sequence_root.name}-start-{requested_start_offset_ns}ns-"
                f"duration-{requested_duration_ns}ns-{source_identity}"
            ),
            "csv": "trajectory.csv",
            "label_source": (
                "Measured TUM VI motion-capture orientation and calibrated IMU gyroscope; "
                "see source_provenance for citation, convention basis, and input-file hashes."
            ),
            "representation": "quaternion",
            "truth": {
                "attitude_direction": "body_to_world",
                "quaternion_order": "scalar_last",
                "rate_frame": "body",
                "rate_order": "pqr",
                "rate_unit": "radians_per_second",
                "rate_sign": 1,
            },
            "orientation_columns": ("qx", "qy", "qz", "qw"),
            "rate_columns": ("wx", "wy", "wz"),
            "time_column": "time_s",
            "measurement_conditions": {
                "motion_profile_id": "tum-vi-mocap-resampled",
                "timestamp_jitter_amplitude_s": None,
                "dropout_probability": None,
                "dropped_sample_count": None,
                "orientation_latency_s": None,
                "rate_latency_s": None,
                "orientation_noise_sigma_rad": None,
                "rate_noise_sigma_rad_s": None,
                "orientation_noise_ar1_coefficient": None,
                "rate_noise_ar1_coefficient": None,
                "sample_count": int(selected_timestamps_ns.size),
                "observed_sample_count": int(selected_timestamps_ns.size),
                "sample_period_s": sample_period_s,
            },
            "source_provenance": source_provenance,
        }
    )
    manifest = ValidationManifest(schema_version=3, cases=(case,))

    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(exist_ok=False)
    trajectory_path = output_root / "trajectory.csv"
    with trajectory_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "qx", "qy", "qz", "qw", "wx", "wy", "wz"))
        for time_s, quaternion, rate in zip(
            output_times_s,
            interpolated_quaternions,
            selected_rates_rad_s,
            strict=True,
        ):
            writer.writerow(
                (
                    f"{float(time_s):.12g}",
                    *(f"{float(value):.17g}" for value in quaternion),
                    *(f"{float(value):.17g}" for value in rate),
                )
            )
        ####
    ####
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest.model_dump(
                mode="json",
                exclude={"cases": {"__all__": {"measurement_conditions": {"duration_s"}}}},
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return PreparedTUMVICorpus(
        output_directory=output_root,
        trajectory_csv=trajectory_path,
        manifest_json=manifest_path,
        sample_count=int(selected_timestamps_ns.size),
    )
####


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence_directory", type=Path, help="extracted TUM-VI room sequence")
    parser.add_argument("--outdir", type=Path, required=True, help="new output directory")
    parser.add_argument(
        "--start-s",
        type=float,
        default=0.0,
        help="offset from common coverage start",
    )
    parser.add_argument("--duration-s", type=float, required=True, help="bounded segment duration")
    parser.add_argument(
        "--max-pose-gap-s",
        type=float,
        default=0.025,
        help="reject orientation interpolation across larger mocap gaps",
    )
    args = parser.parse_args()
    result = prepare_tumvi_corpus(
        args.sequence_directory,
        args.outdir,
        start_s=args.start_s,
        duration_s=args.duration_s,
        max_pose_gap_s=args.max_pose_gap_s,
    )
    print(f"Prepared {result.sample_count} samples in {result.output_directory}")
    print(f"Manifest: {result.manifest_json}")
####


if __name__ == "__main__":
    main()
####
