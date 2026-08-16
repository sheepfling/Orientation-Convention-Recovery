from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from attitude_convention_inference import (
    AttitudeDirection,
    AttitudeRepresentation,
    InferenceConfig,
    QuaternionOrder,
    RateFrame,
    RateUnit,
    SearchSpace,
    TrajectoryData,
    infer_conventions,
)
from attitude_convention_inference.core import FloatArray
from attitude_convention_inference.synthetic import (
    SyntheticConvention,
    encode_synthetic_trajectory,
    rich_three_axis_rates,
)


def _truth() -> SyntheticConvention:
    return SyntheticConvention(
        representation=AttitudeRepresentation.QUATERNION,
        attitude_direction=AttitudeDirection.BODY_TO_WORLD,
        quaternion_order=QuaternionOrder.SCALAR_LAST,
        rate_frame=RateFrame.BODY,
        rate_order="rqp",
    )
####


def _quaternions(rotation: Rotation) -> FloatArray:
    return np.asarray(cast(Any, rotation).as_quat(), dtype=np.float64)
####


def _from_quat(quaternions: FloatArray) -> Rotation:
    return cast(Rotation, cast(Any, Rotation).from_quat(quaternions))
####


def _from_rotvec(rotvecs: FloatArray) -> Rotation:
    return cast(Rotation, cast(Any, Rotation).from_rotvec(rotvecs))
####


def _config(*, trim_fraction: float = 0.02) -> InferenceConfig:
    return InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=(AttitudeDirection.BODY_TO_WORLD,),
            quaternion_orders=tuple(QuaternionOrder),
            rate_frames=(RateFrame.BODY,),
            rate_orders=("rqp",),
            rate_units=(RateUnit.RADIANS_PER_SECOND,),
            rate_signs=(1,),
        ),
        trim_fraction=trim_fraction,
        window_sizes=(1, 2, 4, 8),
    )
####


def _base_data(*, seed: int = 20260801, sample_count: int = 161) -> TrajectoryData:
    rng = np.random.default_rng(seed)
    time_s = np.arange(sample_count, dtype=np.float64) * 0.02
    rates = rich_three_axis_rates(time_s, scale=0.8)
    truth = _truth()
    data = encode_synthetic_trajectory(time_s, rates, truth, quaternion_sign_flips=True)
    if data.orientation is None or data.rates is None:
        raise AssertionError("synthetic trajectory must contain attitudes and rates")
    ####
    orientation = _from_quat(data.orientation)
    orientation = (
        _from_rotvec(np.asarray(rng.normal(0.0, 0.0015, (sample_count, 3)), dtype=np.float64))
        * orientation
    )
    noisy_rates = data.rates + rng.normal(0.0, 0.012, data.rates.shape)
    return replace(
        data,
        orientation=_quaternions(orientation),
        rates=noisy_rates,
    )
####


def test_stress_noise_and_sign_flips_recover_convention() -> None:
    report = infer_conventions(_base_data(), _config())

    assert report.best.hypothesis.quaternion_order is QuaternionOrder.SCALAR_LAST
    assert report.best.hypothesis.rate_order == "rqp"
    assert report.best.interval_rmse_rad < 0.03
####


def test_stress_timestamp_jitter_and_dropped_samples() -> None:
    rng = np.random.default_rng(7)
    time_s = np.arange(181, dtype=np.float64) * 0.018
    jitter = np.asarray(rng.uniform(-0.002, 0.002, time_s.size - 1), dtype=np.float64)
    cumulative_jitter = np.asarray(np.cumsum(jitter), dtype=np.float64)
    time_s += np.r_[0.0, cumulative_jitter]
    time_s = np.asarray(
        cast(Any, np.maximum).accumulate(time_s + np.arange(time_s.size) * 1.0e-5),
        dtype=np.float64,
    )
    data = encode_synthetic_trajectory(time_s, rich_three_axis_rates(time_s), _truth())
    if data.orientation is None or data.rates is None:
        raise AssertionError("synthetic trajectory must contain attitudes and rates")
    ####
    keep = np.ones(data.sample_count, dtype=bool)
    keep[[22, 23, 24, 87, 140]] = False
    dropped = TrajectoryData(
        representation=data.representation,
        orientation=data.orientation[keep],
        rates=data.rates[keep],
        time_s=time_s[keep],
    )

    report = infer_conventions(dropped, _config())

    assert report.best.hypothesis.quaternion_order is QuaternionOrder.SCALAR_LAST
    assert report.diagnostics.time_source == "timestamps"
####


def test_stress_burst_outliers_are_trimmed() -> None:
    data = _base_data(seed=19)
    if data.orientation is None:
        raise AssertionError("base trajectory must contain attitudes")
    ####
    raw = _from_quat(data.orientation)
    burst = np.zeros((data.sample_count, 3), dtype=np.float64)
    burst[60:68] = np.array([0.10, -0.06, 0.08])
    corrupted = _from_rotvec(burst) * raw
    contaminated = replace(data, orientation=_quaternions(corrupted))

    report = infer_conventions(contaminated, _config(trim_fraction=0.08))

    assert report.best.hypothesis.quaternion_order is QuaternionOrder.SCALAR_LAST
    assert report.best.interval_rmse_rad < 0.03
####


@pytest.mark.parametrize(
    ("scale", "expected_order"),
    ((1.0e-5, None), (5.0, QuaternionOrder.SCALAR_LAST)),
)
def test_stress_excitation_extremes_reduce_confidence(
    scale: float,
    expected_order: QuaternionOrder | None,
) -> None:
    time_s = np.arange(121, dtype=np.float64) * 0.02
    data = encode_synthetic_trajectory(
        time_s,
        rich_three_axis_rates(time_s, scale=scale),
        _truth(),
    )

    report = infer_conventions(data, _config())

    if expected_order is not None:
        assert report.best.hypothesis.quaternion_order is expected_order
    ####
    assert report.diagnostics.confidence < 0.95
####
