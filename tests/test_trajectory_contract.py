from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from attitude_convention_inference import AttitudeRepresentation
from attitude_convention_inference.core import TrajectoryData
from attitude_convention_inference.trajectory import (
    QuaternionNormStatistics,
    quaternion_norm_statistics,
)


def test_trajectory_accepts_rate_only_input_and_reports_sample_count() -> None:
    data = TrajectoryData(
        representation=AttitudeRepresentation.QUATERNION,
        orientation=None,
        rates=np.zeros((3, 3)),
    )

    assert data.sample_count == 3
    assert quaternion_norm_statistics(data) is None
####


def test_quaternion_norm_statistics_are_named_and_computed() -> None:
    data = TrajectoryData(
        representation=AttitudeRepresentation.QUATERNION,
        orientation=np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.5]]),
        rates=None,
    )

    statistics = quaternion_norm_statistics(data)

    assert statistics == QuaternionNormStatistics(mean_abs_error=0.5, max_abs_error=1.0)
####


def test_quaternion_norm_statistics_are_not_applicable_to_euler_data() -> None:
    data = TrajectoryData(
        representation=AttitudeRepresentation.EULER,
        orientation=np.zeros((3, 3)),
        rates=None,
    )

    assert quaternion_norm_statistics(data) is None
####


@pytest.mark.parametrize(
    ("orientation", "rates", "time_s", "message"),
    (
        (None, None, None, "At least one of orientation or rates"),
        (np.zeros((3, 4)), None, None, "shape"),
        (None, np.zeros((3, 4)), None, "rates must have shape"),
        (np.zeros((3, 3)), np.zeros((2, 3)), None, "same number of samples"),
        (np.zeros((3, 3)), None, np.zeros((3, 1)), "time_s must have shape"),
    ),
)
def test_trajectory_rejects_invalid_stream_contracts(
    orientation: NDArray[np.float64] | None,
    rates: NDArray[np.float64] | None,
    time_s: NDArray[np.float64] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TrajectoryData(
            representation=AttitudeRepresentation.EULER,
            orientation=orientation,
            rates=rates,
            time_s=time_s,
        )
    ####
####
