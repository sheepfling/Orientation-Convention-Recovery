from __future__ import annotations

import math

import numpy as np
import pytest

from attitude_convention_inference import (
    AttitudeRepresentation,
    ConventionConstraints,
    InferenceConfig,
    RateFrame,
    SearchSpace,
)
from attitude_convention_inference.core import TrajectoryData


def test_rejects_nonincreasing_time() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        TrajectoryData(
            representation=AttitudeRepresentation.EULER,
            orientation=np.zeros((3, 3)),
            rates=np.zeros((3, 3)),
            time_s=np.array([0.0, 1.0, 1.0]),
        )
    ####
####


def test_trajectory_data_owns_read_only_validated_arrays() -> None:
    orientation = np.zeros((3, 4), dtype=np.float64)
    orientation[:, 3] = 1.0
    rates = np.zeros((3, 3), dtype=np.float64)
    time_s = np.arange(3, dtype=np.float64)

    data = TrajectoryData(
        representation=AttitudeRepresentation.QUATERNION,
        orientation=orientation,
        rates=rates,
        time_s=time_s,
    )
    orientation[0, 0] = 0.5
    rates[0, 0] = 1.0
    time_s[0] = -1.0

    assert data.orientation is not None
    assert data.rates is not None
    assert data.time_s is not None
    assert data.orientation[0, 0] == 0.0
    assert data.rates[0, 0] == 0.0
    assert data.time_s[0] == 0.0
    with pytest.raises(ValueError, match="read-only"):
        data.orientation[0, 0] = 0.5
    ####
####


def test_rejects_invalid_rate_order() -> None:
    with pytest.raises(ValueError, match="p/q/r permutation"):
        SearchSpace(rate_orders=("xyz",))
    ####
####


@pytest.mark.parametrize("order", ("xxy", "xyy", "xxxx", "xyw"))
def test_rejects_invalid_euler_order(order: str) -> None:
    with pytest.raises(ValueError, match="valid three-axis Euler sequence"):
        SearchSpace(euler_orders=(order,))
    ####
####


@pytest.mark.parametrize(
    "field",
    [
        "sample_period_s",
        "trim_fraction",
        "residual_floor_rad",
        "near_alias_step_rad",
        "hard_alias_step_rad",
        "evidence_log_likelihood_ratio_threshold",
    ],
)
def test_rejects_nonfinite_inference_configuration(field: str) -> None:
    with pytest.raises(ValueError):
        InferenceConfig.model_validate({field: math.nan})
    ####
####


def test_rejects_nonfinite_sample_period_bounds() -> None:
    with pytest.raises(ValueError, match="finite"):
        InferenceConfig(inferred_sample_period_bounds_s=(1.0, math.inf))
    ####
####


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evidence_log_likelihood_ratio_threshold", 0.0),
        ("evidence_min_intervals", 0),
        ("evidence_stability_windows", 0),
    ],
)
def test_rejects_invalid_evidence_configuration(field: str, value: float | int) -> None:
    with pytest.raises(ValueError):
        InferenceConfig.model_validate({field: value})
    ####
####


def test_model_validate_normalizes_external_config_values() -> None:
    config = InferenceConfig.model_validate(
        {"constraints": {"rate_frame": "body", "rate_order": "pqr"}}
    )

    assert isinstance(config.constraints, ConventionConstraints)
    assert config.constraints.rate_frame == (RateFrame.BODY,)
    assert config.constraints.rate_order == ("pqr",)
####
