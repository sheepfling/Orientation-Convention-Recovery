from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from attitude_convention_inference.math_utils import rotation_from_quat, rotation_rotvecs
from scripts.generate_validation_corpus import (
    NOISE_PROFILES,
    MotionProfile,
    analytic_motion,
    generate_corpus,
)
from scripts.validate_corpus import validate_corpus


@pytest.mark.parametrize("profile", tuple(MotionProfile))
def test_analytic_rates_match_finite_difference_rotations(profile: MotionProfile) -> None:
    time_s = np.asarray([0.1, 0.7, 1.3, 2.1], dtype=np.float64)
    epsilon = 1.0e-6
    earlier = rotation_from_quat(analytic_motion(time_s - epsilon, profile).quaternion_xyzw)
    later = rotation_from_quat(analytic_motion(time_s + epsilon, profile).quaternion_xyzw)
    center = rotation_from_quat(analytic_motion(time_s, profile).quaternion_xyzw)
    body_rates = rotation_rotvecs(earlier.inv() * later) / (2.0 * epsilon)
    world_rates = np.asarray(cast(Any, center).apply(body_rates), dtype=np.float64)
    analytic = analytic_motion(time_s, profile)

    np.testing.assert_allclose(analytic.body_rates_rad_s, body_rates, atol=1.0e-9)
    np.testing.assert_allclose(analytic.world_rates_rad_s, world_rates, atol=1.0e-9)
####


def test_generated_analytic_corpus_reports_noisy_ambiguity_reproducibly(tmp_path: Path) -> None:
    manifest = generate_corpus(tmp_path / "analytic-corpus")
    repeated_manifest = generate_corpus(tmp_path / "repeated-corpus")

    for first_path in sorted(manifest.parent.iterdir()):
        filename = first_path.name
        assert (manifest.parent / filename).read_bytes() == (
            repeated_manifest.parent / filename
        ).read_bytes()
    ####

    assessment = validate_corpus(manifest)

    assert assessment.summary.case_count == 4
    assert assessment.summary.resolved_correct == 3
    assert assessment.summary.ambiguous_contains_truth == 1
    assert assessment.summary.evidence_miss == 0
    assert assessment.environment.python
    assert assessment.environment.numpy
    assert assessment.environment.scipy
    assert assessment.environment.package
    assert assessment.cases[1].measurement_conditions.dropped_sample_count > 0
    noisy_case = assessment.cases[3]
    assert noisy_case.measurement_conditions.noise_seed == 20261030
    assert noisy_case.measurement_conditions.orientation_noise_sigma_rad == 0.0015
    assert noisy_case.measurement_conditions.rate_noise_sigma_rad_s == 0.012
    assert noisy_case.outcome == "ambiguous_contains_truth"
    assert noisy_case.truth_class_id in noisy_case.evidence_contender_class_ids
    assert noisy_case.operational_correct
    low_noise_summary = next(
        summary
        for summary in assessment.condition_summaries
        if summary.measurement_conditions.orientation_noise_sigma_rad == 0.0015
    )
    assert low_noise_summary.case_count == 1
    assert low_noise_summary.measurement_conditions.motion_profile_id == (
        MotionProfile.RICH_THREE_AXIS.value
    )
    assert low_noise_summary.ambiguous_contains_truth == 1
    assert low_noise_summary.operational_miss == 0
    assert low_noise_summary.indeterminate_confidence == 1
    assert low_noise_summary.evidence_miss_rate == 0.0
    assert low_noise_summary.operational_miss_rate == 0.0
    assert low_noise_summary.indeterminate_confidence_rate == 1.0
####


def test_noise_replicates_are_aggregated_by_measurement_condition(tmp_path: Path) -> None:
    manifest = generate_corpus(
        tmp_path / "replicated-noise-corpus",
        noise_profiles=(NOISE_PROFILES[0],),
        noise_replicates=2,
        motion_profiles=(MotionProfile.RICH_THREE_AXIS, MotionProfile.SINGLE_AXIS),
    )

    assessment = validate_corpus(manifest)

    low_noise_summaries = tuple(
        summary
        for summary in assessment.condition_summaries
        if summary.measurement_conditions.orientation_noise_sigma_rad == 0.0015
    )
    assert len(low_noise_summaries) == 2
    assert {
        summary.measurement_conditions.motion_profile_id for summary in low_noise_summaries
    } == {MotionProfile.RICH_THREE_AXIS.value, MotionProfile.SINGLE_AXIS.value}
    assert all(summary.case_count == 2 for summary in low_noise_summaries)
    assert all(summary.measurement_conditions.noise_seed is None for summary in low_noise_summaries)
    assert all(
        summary.evidence_miss_rate == summary.evidence_miss / 2 for summary in low_noise_summaries
    )
####
