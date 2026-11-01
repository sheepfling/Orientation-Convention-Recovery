"""Generate optional local exchange schemas and committed example artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path

from attitude_convention_inference.fusion import CohortFusionResult, EvidencePacket

if __package__:
    from .generate_examples import generate_examples
else:
    from generate_examples import generate_examples
####


def generate_schemas(root: Path) -> None:
    schema_dir = root / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)
    for filename, model in (
        ("orientation_evidence_packet.schema.json", EvidencePacket),
        ("orientation_cohort_fusion_result.schema.json", CohortFusionResult),
    ):
        output = json.dumps(model.model_json_schema(), indent=2) + "\n"
        (schema_dir / filename).write_text(output, encoding="utf-8")
    ####
    generate_examples(root)
####


def _generated_example_paths() -> tuple[str, ...]:
    return (
        "examples/partially_known_config.json",
        "examples/preference_profile.json",
        "examples/synthetic_quaternion.csv",
        "examples/cohort_fusion/fusion_result.json",
        "examples/cohort_fusion/orientation-convention-domain-priors.json",
        "examples/cohort_fusion/packet_A.json",
        "examples/cohort_fusion/packet_B.json",
        "examples/cohort_fusion/packet_C.json",
    )
####


def csv_matches_with_tolerance(expected_path: Path, generated_path: Path) -> bool:
    """Compare generated numeric samples without requiring bit-identical integration."""

    with expected_path.open(encoding="utf-8", newline="") as expected_file:
        expected_rows = tuple(csv.reader(expected_file))
    ####
    with generated_path.open(encoding="utf-8", newline="") as generated_file:
        generated_rows = tuple(csv.reader(generated_file))
    ####
    if len(expected_rows) != len(generated_rows) or not expected_rows:
        return False
    ####
    if expected_rows[0] != generated_rows[0]:
        return False
    ####
    for expected_row, generated_row in zip(expected_rows[1:], generated_rows[1:], strict=True):
        if len(expected_row) != len(generated_row):
            return False
        ####
        for expected_value, generated_value in zip(expected_row, generated_row, strict=True):
            try:
                expected_number = float(expected_value)
                generated_number = float(generated_value)
            except ValueError:
                return False
            ####
            if not math.isclose(
                expected_number,
                generated_number,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                return False
            ####
        ####
    ####
    return True
####


def check_generated_files(root: Path) -> None:
    with tempfile.TemporaryDirectory(dir=root) as temporary_directory:
        generated_root = Path(temporary_directory)
        generate_schemas(generated_root)
        mismatches: list[str] = []
        for relative_path in _generated_example_paths():
            expected_path = root / relative_path
            generated_path = generated_root / relative_path
            if relative_path == "examples/synthetic_quaternion.csv":
                matches = csv_matches_with_tolerance(expected_path, generated_path)
            else:
                matches = expected_path.read_bytes() == generated_path.read_bytes()
            ####
            if not matches:
                mismatches.append(relative_path)
            ####
        ####
    ####
    if mismatches:
        details = "\n".join(f"  {path}" for path in mismatches)
        raise SystemExit(f"Generated files are out of date:\n{details}\nRun with --fix.")
    ####
####


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="verify generated schemas and examples without writing",
    )
    mode.add_argument(
        "--fix",
        action="store_true",
        help="regenerate schemas and examples in the repository",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.check:
        check_generated_files(root)
    else:
        generate_schemas(root)
    ####
####


if __name__ == "__main__":
    main()
####
