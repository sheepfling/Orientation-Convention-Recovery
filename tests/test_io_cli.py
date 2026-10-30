from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from attitude_convention_inference import (
    AttitudeRepresentation,
    InferenceConfig,
    QuaternionOrder,
    RateFrame,
    SearchSpace,
)
from attitude_convention_inference.cli import main
from attitude_convention_inference.io import (
    load_config,
    load_csv_trajectory,
    write_interval_diagnostics,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ####
####


def test_load_csv_quaternion(tmp_path: Path) -> None:
    source = tmp_path / "trajectory.csv"
    _write_csv(
        source,
        [
            {"t": 0.0, "q0": 0.0, "q1": 0.0, "q2": 0.0, "q3": 1.0, "a": 1, "b": 2, "c": 3},
            {"t": 0.1, "q0": 0.0, "q1": 0.0, "q2": 0.05, "q3": 0.9987, "a": 1, "b": 2, "c": 3},
            {"t": 0.2, "q0": 0.0, "q1": 0.0, "q2": 0.10, "q3": 0.9950, "a": 1, "b": 2, "c": 3},
        ],
    )
    data = load_csv_trajectory(
        source,
        AttitudeRepresentation.QUATERNION,
        ("q0", "q1", "q2", "q3"),
        ("a", "b", "c"),
        "t",
    )
    if data.orientation is None or data.rates is None:
        raise AssertionError("CSV loader should return attitudes and rates")
    ####
    assert data.orientation.shape == (3, 4)
    assert data.rates.shape == (3, 3)
    assert data.time_s is not None
    np.testing.assert_allclose(data.time_s, [0.0, 0.1, 0.2])
####


def test_load_csv_rejects_missing_columns(tmp_path: Path) -> None:
    source = tmp_path / "trajectory.csv"
    _write_csv(source, [{"q0": 0, "q1": 0, "q2": 0, "q3": 1, "a": 0, "b": 0}])
    with pytest.raises(ValueError, match="missing columns"):
        load_csv_trajectory(
            source,
            AttitudeRepresentation.QUATERNION,
            ("q0", "q1", "q2", "q3"),
            ("a", "b", "c"),
        )
    ####
####


def test_load_csv_rejects_wrong_orientation_width(tmp_path: Path) -> None:
    source = tmp_path / "trajectory.csv"
    source.write_text("x,y,z,a,b,c\n0,0,0,0,0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires 4 orientation columns"):
        load_csv_trajectory(
            source,
            AttitudeRepresentation.QUATERNION,
            ("x", "y", "z"),
            ("a", "b", "c"),
        )
    ####
####


def test_load_csv_reports_invalid_numeric_line(tmp_path: Path) -> None:
    source = tmp_path / "trajectory.csv"
    source.write_text(
        "q0,q1,q2,q3,a,b,c\n0,0,0,1,0,0,0\n0,broken,0,1,0,0,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="line 3"):
        load_csv_trajectory(
            source,
            AttitudeRepresentation.QUATERNION,
            ("q0", "q1", "q2", "q3"),
            ("a", "b", "c"),
        )
    ####
####


def test_load_config_round_trip(tmp_path: Path) -> None:
    expected = InferenceConfig(
        search_space=SearchSpace(
            quaternion_orders=(QuaternionOrder.SCALAR_FIRST,),
            rate_frames=(RateFrame.BODY,),
            rate_orders=("pqr",),
        ),
        sample_period_s=0.02,
    )
    path = tmp_path / "config.json"
    path.write_text(expected.model_dump_json(indent=2), encoding="utf-8")
    assert load_config(path) == expected
####


def test_write_interval_diagnostics(tmp_path: Path) -> None:
    target = tmp_path / "details.csv"
    write_interval_diagnostics(
        target,
        (
            {"interval": 0, "error_angle_rad": 0.01, "near_alias": False},
            {"interval": 1, "error_angle_rad": 0.02, "near_alias": True},
        ),
    )
    rows = list(csv.DictReader(target.open(encoding="utf-8")))
    assert len(rows) == 2
    assert rows[1]["near_alias"] == "True"
####


def test_cli_writes_report_and_details(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "trajectory.csv"
    rows: list[dict[str, object]] = []
    for index in range(21):
        angle = 0.01 * index
        rows.append(
            {
                "time": 0.01 * index,
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
    _write_csv(source, rows)
    config = InferenceConfig(
        search_space=SearchSpace(
            quaternion_orders=(QuaternionOrder.SCALAR_LAST,),
            rate_frames=(RateFrame.BODY,),
            rate_orders=("pqr",),
        )
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    report_path = tmp_path / "report.json"
    details_path = tmp_path / "details.csv"

    arguments = [
        str(source),
        "--representation",
        "quaternion",
        "--orientation-columns",
        "qx",
        "qy",
        "qz",
        "qw",
        "--rate-columns",
        "p",
        "q",
        "r",
        "--time-column",
        "time",
        "--config",
        str(config_path),
        "--report-json",
        str(report_path),
        "--details-csv",
        str(details_path),
        "--top",
        "2",
    ]
    exit_code = main(arguments)

    assert exit_code == 0
    assert "decision=" in capsys.readouterr().out
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["sample_count"] == 21
    assert payload["best"]["interval_rmse_rad"] < 1.0e-6
    assert len(list(csv.DictReader(details_path.open(encoding="utf-8")))) == 20

    decision_exit_code = main([*arguments, "--format", "decision-json"])
    decision_payload = json.loads(capsys.readouterr().out)
    assert decision_exit_code == 0
    assert decision_payload["status"] in ("recommendation", "abstention")
    assert "reason" in decision_payload
####


def test_cli_lists_and_inspects_packaged_profiles(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list-profiles"]) == 0
    listing = capsys.readouterr().out
    assert "neutral-v1" in listing
    assert "common-engineering-v1" in listing

    assert main(["--inspect-profile", "common-engineering-v1", "--format", "json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["profile_id"] == "common-engineering-v1"
    assert inspected["representative_preferences"]["quaternion_order"]["scalar_last"] == 2.0
####
