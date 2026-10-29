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


def test_generated_analytic_corpus_recovers_all_three_cases(tmp_path: Path) -> None:
    manifest = generate_corpus(tmp_path / "analytic-corpus")

    assessment = validate_corpus(manifest)

    assert assessment.summary.case_count == 3
    assert assessment.summary.resolved_correct == 3
    assert assessment.summary.miss == 0
####
