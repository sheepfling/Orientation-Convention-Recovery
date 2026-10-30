from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from attitude_convention_inference.math_utils import rotation_from_quat, rotation_rotvecs
from scripts.generate_validation_corpus import analytic_motion, generate_corpus
from scripts.validate_corpus import validate_corpus


def test_analytic_rates_match_finite_difference_rotations() -> None:
    time_s = np.asarray([0.1, 0.7, 1.3, 2.1], dtype=np.float64)
    epsilon = 1.0e-6
    earlier = rotation_from_quat(analytic_motion(time_s - epsilon).quaternion_xyzw)
    later = rotation_from_quat(analytic_motion(time_s + epsilon).quaternion_xyzw)
    center = rotation_from_quat(analytic_motion(time_s).quaternion_xyzw)
    body_rates = rotation_rotvecs(earlier.inv() * later) / (2.0 * epsilon)
    world_rates = np.asarray(cast(Any, center).apply(body_rates), dtype=np.float64)
    analytic = analytic_motion(time_s)

    np.testing.assert_allclose(analytic.body_rates_rad_s, body_rates, atol=1.0e-9)
    np.testing.assert_allclose(analytic.world_rates_rad_s, world_rates, atol=1.0e-9)
####


def test_generated_analytic_corpus_reports_noisy_ambiguity_reproducibly(tmp_path: Path) -> None:
    manifest = generate_corpus(tmp_path / "analytic-corpus")
    repeated_manifest = generate_corpus(tmp_path / "repeated-corpus")

    for filename in (
        "manifest.json",
        "quaternion_body.csv",
        "quaternion_world_jittered.csv",
        "euler_degrees.csv",
        "quaternion_body_noisy.csv",
    ):
        assert (manifest.parent / filename).read_bytes() == (
            repeated_manifest.parent / filename
        ).read_bytes()
    ####

    assessment = validate_corpus(manifest)

    assert assessment.summary.case_count == 4
    assert assessment.summary.resolved_correct == 3
    assert assessment.summary.ambiguous_contains_truth == 1
    assert assessment.summary.miss == 0
    assert assessment.cases[1].measurement_conditions.dropped_sample_count > 0
    noisy_case = assessment.cases[3]
    assert noisy_case.measurement_conditions.noise_seed == 20261030
    assert noisy_case.measurement_conditions.orientation_noise_sigma_rad == 0.0015
    assert noisy_case.measurement_conditions.rate_noise_sigma_rad_s == 0.012
    assert noisy_case.outcome == "ambiguous_contains_truth"
    assert noisy_case.truth_class_id in noisy_case.evidence_contender_class_ids
####
