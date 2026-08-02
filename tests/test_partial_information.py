from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from attitude_convention_inference import (
    AttitudeDirection,
    AttitudeRepresentation,
    ConventionConstraints,
    ConventionPriors,
    DimensionStatus,
    InferenceConfig,
    QuaternionOrder,
    RateFrame,
    SearchSpace,
    infer_conventions,
)
from attitude_convention_inference.cli import main
from attitude_convention_inference.core import TrajectoryData
from attitude_convention_inference.synthetic import (
    SyntheticConvention,
    encode_synthetic_trajectory,
    rich_three_axis_rates,
)


def _quaternion_data() -> tuple[np.ndarray, TrajectoryData]:
    time_s = np.arange(81, dtype=np.float64) * 0.025
    truth = SyntheticConvention(
        representation=AttitudeRepresentation.QUATERNION,
        attitude_direction=AttitudeDirection.WORLD_TO_BODY,
        quaternion_order=QuaternionOrder.SCALAR_FIRST,
        rate_frame=RateFrame.WORLD,
        rate_order="rqp",
    )
    data = encode_synthetic_trajectory(time_s, rich_three_axis_rates(time_s), truth)
    return time_s, data


####


def test_scalar_and_subset_constraints_are_accepted() -> None:
    config = InferenceConfig.model_validate(
        {
            "constraints": {
                "quaternion_order": "scalar_first",
                "rate_frame": "world",
                "rate_order": ["pqr", "rqp"],
            }
        }
    )

    assert config.constraints.quaternion_order == (QuaternionOrder.SCALAR_FIRST,)
    assert config.constraints.rate_frame == (RateFrame.WORLD,)
    assert config.constraints.rate_order == ("pqr", "rqp")


####


def test_constraints_intersect_legacy_search_space() -> None:
    config = InferenceConfig(
        search_space=SearchSpace(rate_orders=("pqr", "prq", "rqp")),
        constraints=ConventionConstraints(rate_order=("pqr", "rqp")),
    )

    assert config.resolved_search_space().rate_orders == ("pqr", "rqp")


####


def test_contradictory_constraint_is_rejected() -> None:
    with pytest.raises(ValueError, match="no overlap"):
        InferenceConfig(
            search_space=SearchSpace(rate_frames=(RateFrame.BODY,)),
            constraints=ConventionConstraints(rate_frame=RateFrame.WORLD),
        )


####


def test_partial_information_reduces_and_documents_hypothesis_space() -> None:
    _, data = _quaternion_data()
    config = InferenceConfig(
        constraints=ConventionConstraints(
            quaternion_order=QuaternionOrder.SCALAR_FIRST,
            rate_frame=RateFrame.WORLD,
            rate_order=("pqr", "rqp"),
        )
    )
    report = infer_conventions(data, config)

    assert report.hypothesis_count == 4
    assert report.hypothesis_space.base_hypothesis_count == 48
    assert report.hypothesis_space.constrained_hypothesis_count == 4
    assert report.hypothesis_space.reduction_fraction == pytest.approx(44.0 / 48.0)
    assert report.hypothesis_space.fixed_dimensions["quaternion_order"] == "scalar_first"
    assert report.hypothesis_space.fixed_dimensions["rate_frame"] == "world"
    assert report.hypothesis_space.searched_dimensions["rate_order"] == ("pqr", "rqp")
    assert report.best.hypothesis.rate_order == "rqp"

    assessments = report.diagnostics.dimension_assessments
    assert assessments["quaternion_order"].status is DimensionStatus.FIXED
    assert assessments["rate_order"].status is DimensionStatus.INFERRED
    assert assessments["euler_order"].status is DimensionStatus.NOT_APPLICABLE


####


def test_soft_prior_selects_tie_without_erasing_evidence_ambiguity() -> None:
    time_s = np.arange(121, dtype=np.float64) * 0.02
    rates = np.zeros((time_s.size, 3), dtype=np.float64)
    rates[:, 2] = 0.8
    truth = SyntheticConvention(
        representation=AttitudeRepresentation.QUATERNION,
        attitude_direction=AttitudeDirection.BODY_TO_WORLD,
        quaternion_order=QuaternionOrder.SCALAR_LAST,
        rate_frame=RateFrame.BODY,
        rate_order="pqr",
    )
    data = encode_synthetic_trajectory(
        time_s,
        rates,
        truth,
        initial_attitude=Rotation.identity(),
    )
    config = InferenceConfig(
        constraints=ConventionConstraints(
            attitude_direction=AttitudeDirection.BODY_TO_WORLD,
            quaternion_order=QuaternionOrder.SCALAR_LAST,
            rate_order="pqr",
        ),
        priors=ConventionPriors(
            rate_frame={RateFrame.BODY: 4.0, RateFrame.WORLD: 1.0}
        ),
    )
    report = infer_conventions(data, config)

    assert report.hypothesis_count == 2
    assert report.best.hypothesis.rate_frame is RateFrame.BODY
    assert report.best.prior_relative_weight == pytest.approx(1.0)
    assert report.rankings[1].prior_relative_weight == pytest.approx(0.25)
    assert report.diagnostics.near_tie_count == 2
    assert report.diagnostics.posterior_near_tie_count == 1
    assert report.diagnostics.unresolved_dimensions["rate_frame"] == ("body", "world")
    assert "rate_frame" not in report.diagnostics.posterior_unresolved_dimensions
    assert (
        report.diagnostics.dimension_assessments["rate_frame"].status
        is DimensionStatus.AMBIGUOUS
    )
    assert any("empirical ambiguity remains" in item for item in report.diagnostics.warnings)


####


def test_prior_json_uses_relative_weights() -> None:
    config = InferenceConfig.model_validate(
        {
            "priors": {
                "attitude_direction": {"body_to_world": 3.0},
                "rate_frame": {"body": 4.0, "world": 1.0},
                "rate_order": {"pqr": 2.0},
            }
        }
    )

    assert config.priors.attitude_direction[AttitudeDirection.BODY_TO_WORLD] == 3.0
    assert config.priors.rate_frame[RateFrame.BODY] == 4.0
    assert config.priors.rate_order["pqr"] == 2.0


####


def test_cli_convention_flags_lock_all_relevant_dimensions(tmp_path: Path) -> None:
    source = tmp_path / "trajectory.csv"
    report_path = tmp_path / "report.json"
    rows: list[dict[str, float]] = []
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
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    exit_code = main(
        [
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
            "--attitude-direction",
            "body_to_world",
            "--quaternion-order",
            "scalar_last",
            "--rate-frame",
            "body",
            "--rate-order",
            "pqr",
            "--rate-unit",
            "radians_per_second",
            "--rate-sign",
            "1",
            "--report-json",
            str(report_path),
            "--top",
            "1",
        ]
    )

    assert exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["hypothesis_count"] == 1
    assert payload["hypothesis_space"]["constrained_hypothesis_count"] == 1
    assert payload["diagnostics"]["dimension_assessments"]["rate_frame"]["status"] == "fixed"


####
