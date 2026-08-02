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


def test_rejects_invalid_rate_order() -> None:
    with pytest.raises(ValueError, match="p/q/r permutation"):
        SearchSpace(rate_orders=("xyz",))


@pytest.mark.parametrize(
    "field",
    [
        "sample_period_s",
        "trim_fraction",
        "residual_floor_rad",
        "near_alias_step_rad",
        "hard_alias_step_rad",
    ],
)
def test_rejects_nonfinite_inference_configuration(field: str) -> None:
    with pytest.raises(ValueError):
        InferenceConfig.model_validate({field: math.nan})


def test_rejects_nonfinite_sample_period_bounds() -> None:
    with pytest.raises(ValueError, match="finite"):
        InferenceConfig(inferred_sample_period_bounds_s=(1.0, math.inf))


def test_model_validate_normalizes_external_config_values() -> None:
    config = InferenceConfig.model_validate(
        {"constraints": {"rate_frame": "body", "rate_order": "pqr"}}
    )

    assert isinstance(config.constraints, ConventionConstraints)
    assert config.constraints.rate_frame == (RateFrame.BODY,)
    assert config.constraints.rate_order == ("pqr",)


####
