from __future__ import annotations

from pathlib import Path

from scripts.generate_schemas import check_generated_files, csv_matches_with_tolerance


def test_checked_in_example_data_matches_generator() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    check_generated_files(repository_root)
####


def test_synthetic_csv_comparison_tolerates_only_roundoff(tmp_path: Path) -> None:
    expected_path = tmp_path / "expected.csv"
    generated_path = tmp_path / "generated.csv"
    expected_path.write_text("time,value\n0,0.12345678912345\n", encoding="utf-8")
    generated_path.write_text("time,value\n0,0.12345678912346\n", encoding="utf-8")

    assert csv_matches_with_tolerance(expected_path, generated_path)

    generated_path.write_text("time,value\n0,0.1234567\n", encoding="utf-8")
    assert not csv_matches_with_tolerance(expected_path, generated_path)
####
