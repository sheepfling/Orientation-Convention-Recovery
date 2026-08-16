from __future__ import annotations

from pathlib import Path

from scripts.generate_examples import generate_examples

EXAMPLE_ARTIFACTS = (
    "examples/partially_known_config.json",
    "examples/preference_profile.json",
    "examples/synthetic_quaternion.csv",
    "examples/cohort_fusion/fusion_result.json",
    "examples/cohort_fusion/orientation-convention-domain-priors.json",
    "examples/cohort_fusion/packet_A.json",
    "examples/cohort_fusion/packet_B.json",
    "examples/cohort_fusion/packet_C.json",
)


def test_checked_in_example_data_matches_generator(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    generate_examples(tmp_path)

    for relative_path in EXAMPLE_ARTIFACTS:
        expected = repository_root / relative_path
        generated = tmp_path / relative_path
        assert generated.read_bytes() == expected.read_bytes(), relative_path
    ####
####


