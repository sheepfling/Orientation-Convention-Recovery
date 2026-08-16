"""Generate optional local exchange schemas and committed example artifacts."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from generate_examples import generate_examples

from attitude_convention_inference.fusion import CohortFusionResult, EvidencePacket


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


def check_generated_files(root: Path) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        generated_root = Path(temporary_directory)
        generate_schemas(generated_root)
        mismatches = [
            relative_path
            for relative_path in _generated_example_paths()
            if (root / relative_path).read_bytes() != (generated_root / relative_path).read_bytes()
        ]
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
