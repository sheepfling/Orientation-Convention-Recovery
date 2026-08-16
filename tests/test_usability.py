from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from attitude_convention_inference import (
    AttitudeRepresentation,
    QuaternionOrder,
    RateFrame,
    __version__,
    detect_csv_layout,
    infer_attitude_conventions,
    infer_csv,
)
from attitude_convention_inference.cli import main


def _write_z_rotation_csv(path: Path, sample_count: int = 21) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("time_s", "qx", "qy", "qz", "qw", "p", "q", "r"),
        )
        writer.writeheader()
        for index in range(sample_count):
            angle = 0.01 * index
            writer.writerow(
                {
                    "time_s": 0.01 * index,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": np.sin(angle / 2.0),
                    "qw": np.cos(angle / 2.0),
                    "p": 0.0,
                    "q": 0.0,
                    "r": 1.0,
                }
            )
        ####
    ####
####


def test_high_level_api_accepts_lists_and_mapping_constraints() -> None:
    orientation: list[list[float]] = []
    rates: list[list[float]] = []
    time_s: list[float] = []
    for index in range(21):
        angle = 0.01 * index
        orientation.append([0.0, 0.0, float(np.sin(angle / 2.0)), float(np.cos(angle / 2.0))])
        rates.append([0.0, 0.0, 1.0])
        time_s.append(0.01 * index)
    ####

    report = infer_attitude_conventions(
        orientation,
        rates,
        representation="quaternion",
        time_s=time_s,
        constraints={
            "attitude_direction": "body_to_world",
            "quaternion_order": "scalar_last",
            "rate_frame": "body",
            "rate_order": "pqr",
        },
    )

    assert report.hypothesis_count == 1
    assert report.best.hypothesis.quaternion_order is QuaternionOrder.SCALAR_LAST
    assert report.best.hypothesis.rate_frame is RateFrame.BODY
    assert report.best.interval_rmse_rad < 1.0e-10
####


def test_detect_csv_layout_from_common_columns(tmp_path: Path) -> None:
    source = tmp_path / "trajectory.csv"
    _write_z_rotation_csv(source)

    layout = detect_csv_layout(source)

    assert layout.representation is AttitudeRepresentation.QUATERNION
    assert layout.orientation_columns == ("qx", "qy", "qz", "qw")
    assert layout.rate_columns == ("p", "q", "r")
    assert layout.time_column == "time_s"
####


def test_detect_csv_layout_is_case_and_spacing_tolerant(tmp_path: Path) -> None:
    source = tmp_path / "trajectory.csv"
    source.write_text(
        " Time S , QX , QY , QZ , QW , Omega X , Omega Y , Omega Z \n"
        "0,0,0,0,1,0,0,0\n"
        "1,0,0,0,1,0,0,0\n"
        "2,0,0,0,1,0,0,0\n",
        encoding="utf-8",
    )

    layout = detect_csv_layout(source)

    assert layout.orientation_columns == (" QX ", " QY ", " QZ ", " QW ")
    assert layout.rate_columns == (" Omega X ", " Omega Y ", " Omega Z ")
    assert layout.time_column == " Time S "
####


def test_detect_csv_layout_rejects_multiple_rate_groups(tmp_path: Path) -> None:
    source = tmp_path / "trajectory.csv"
    source.write_text(
        "time,qx,qy,qz,qw,p,q,r,wx,wy,wz\n0,0,0,0,1,0,0,0,0,0,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Multiple plausible angular-rate"):
        detect_csv_layout(source)
    ####
####


def test_infer_csv_uses_auto_detected_layout(tmp_path: Path) -> None:
    source = tmp_path / "trajectory.csv"
    _write_z_rotation_csv(source)

    report = infer_csv(
        source,
        constraints={
            "attitude_direction": "body_to_world",
            "quaternion_order": "scalar_last",
            "rate_frame": "body",
            "rate_order": "pqr",
        },
    )

    assert report.hypothesis_count == 1
    assert report.best.interval_rmse_rad < 1.0e-10
####


def test_cli_auto_detection_and_json_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "trajectory.csv"
    _write_z_rotation_csv(source)

    exit_code = main(
        [
            str(source),
            "--quaternion-order",
            "scalar_last",
            "--attitude-direction",
            "body_to_world",
            "--rate-frame",
            "body",
            "--rate-order",
            "pqr",
            "--format",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["representation"] == "quaternion"
    assert payload["hypothesis_count"] == 1
####


def test_cli_inspect_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "trajectory.csv"
    _write_z_rotation_csv(source)

    assert main([str(source), "--inspect", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["orientation_columns"] == ["qx", "qy", "qz", "qw"]
    assert payload["time_column"] == "time_s"
####


def test_python_module_entry_point(tmp_path: Path) -> None:
    source = tmp_path / "trajectory.csv"
    _write_z_rotation_csv(source)
    project_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root / "src")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "attitude_convention_inference",
            str(source),
            "--inspect",
            "--format",
            "json",
        ],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["representation"] == "quaternion"
####


def test_version_is_exported() -> None:
    from importlib.metadata import version

    assert __version__ == version("attitude-convention-inference")
    assert __version__
####


def test_detect_csv_layout_for_euler_columns(tmp_path: Path) -> None:
    source = tmp_path / "euler.csv"
    source.write_text(
        "time,roll,pitch,yaw,wx,wy,wz\n0,0,0,0,0,0,0\n1,0,0,0,0,0,0\n2,0,0,0,0,0,0\n",
        encoding="utf-8",
    )

    layout = detect_csv_layout(source)

    assert layout.representation is AttitudeRepresentation.EULER
    assert layout.orientation_columns == ("roll", "pitch", "yaw")
    assert layout.rate_columns == ("wx", "wy", "wz")
####


def test_negative_sample_period_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="sample_period_s"):
        infer_attitude_conventions(
            orientation=(
                (0.0, 0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            rates=((0.0, 0.0, 0.0),) * 3,
            representation="quaternion",
            sample_period_s=-0.1,
        )
    ####
####


def test_distribution_exposes_matching_primary_command() -> None:
    import tomllib

    project_root = Path(__file__).resolve().parents[1]
    payload = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))

    assert payload["project"]["name"] == "attitude-convention-inference"
    assert payload["project"]["scripts"]["attitude-conventions"] == (
        "attitude_convention_inference.cli:main"
    )
    assert payload["project"]["scripts"]["attitude-convention-inference"] == (
        "attitude_convention_inference.cli:main"
    )
    assert payload["project"]["dependencies"] == [
        "numpy>=1.26.4",
        "pydantic>=2.7",
        "scipy>=1.13",
    ]
####


def test_canonical_infer_conventions_accepts_arrays() -> None:
    from attitude_convention_inference import infer_conventions

    orientation: list[list[float]] = []
    rates: list[list[float]] = []
    time_s: list[float] = []
    for index in range(21):
        angle = 0.01 * index
        orientation.append([0.0, 0.0, float(np.sin(angle / 2.0)), float(np.cos(angle / 2.0))])
        rates.append([0.0, 0.0, 1.0])
        time_s.append(0.01 * index)
    ####

    report = infer_conventions(
        orientation,
        rates,
        representation="quaternion",
        time_s=time_s,
        constraints={
            "attitude_direction": "body_to_world",
            "quaternion_order": "scalar_last",
            "rate_frame": "body",
            "rate_order": "pqr",
        },
    )

    assert report.hypothesis_count == 1
    assert report.best.interval_rmse_rad < 1.0e-10
####


def test_runtime_models_use_pydantic() -> None:
    from attitude_convention_inference.fusion import EvidencePacket
    from attitude_convention_inference.models import InferenceConfig

    assert InferenceConfig.model_json_schema()
    assert EvidencePacket.model_json_schema()
####
