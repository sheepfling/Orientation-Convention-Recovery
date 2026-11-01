from __future__ import annotations

from pathlib import Path

from scripts.generate_euler_validation_corpus import (
    EulerCorpusConfig,
    generate_euler_corpus,
)
from scripts.generate_validation_corpus import (
    NO_NOISE_PROFILE,
    MotionProfile,
    generate_corpus,
)
from scripts.validate_corpus import validate_corpus


def test_independent_quaternion_convention_sweep_recovers_every_truth(
    tmp_path: Path,
) -> None:
    manifest_path = generate_corpus(
        tmp_path / "independent-quaternion-conventions",
        noise_profiles=(NO_NOISE_PROFILE,),
        noise_sample_counts=(41,),
        truth_convention_sweep=True,
        motion_profiles=(MotionProfile.RICH_THREE_AXIS,),
    )

    assessment = validate_corpus(manifest_path, max_workers=4)

    assert assessment.summary.case_count == 195
    assert assessment.summary.resolved_correct == 195
    assert assessment.summary.evidence_miss == 0
    assert assessment.summary.data_winner_miss == 0
    assert assessment.summary.operational_miss == 0
    assert assessment.summary.recommendation_miss == 0
    assert assessment.summary.recommendation_accuracy == 1.0
####


def test_independent_euler_convention_sweep_recovers_every_truth(
    tmp_path: Path,
) -> None:
    manifest_path = generate_euler_corpus(
        tmp_path / "independent-euler-conventions",
        EulerCorpusConfig(
            motion_seeds=(730301,),
            sample_count=101,
            orientation_noise_sigma_rad=0.0,
            rate_noise_sigma_rad_s=0.0,
        ),
    )

    assessment = validate_corpus(manifest_path, max_workers=4)

    assert assessment.summary.case_count == 192
    assert assessment.summary.resolved_correct == 192
    assert assessment.summary.evidence_miss == 0
    assert assessment.summary.data_winner_miss == 0
    assert assessment.summary.operational_miss == 0
    assert assessment.summary.recommendation_miss == 0
    assert assessment.summary.recommendation_accuracy == 1.0
####
