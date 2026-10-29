from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate_corpus import validate_corpus


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
    assert result.summary.miss == 1
    assert result.cases[0].truth_class_id in result.cases[0].evidence_contender_class_ids
    assert result.cases[1].truth_class_id not in result.cases[1].evidence_contender_class_ids
####


def test_validation_corpus_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    case = _manifest_case("scalar_first", "duplicate")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "cases": [case, case]}), encoding="utf-8")

    with pytest.raises(ValueError, match="unique"):
        validate_corpus(manifest)
    ####
####
