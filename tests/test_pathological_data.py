from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from attitude_convention_inference import (
    AngleUnit,
    AttitudeDirection,
    AttitudeRepresentation,
    EulerMode,
    InferenceConfig,
    InferenceDecisionReason,
    InferenceDecisionStatus,
    QuaternionOrder,
    RateFrame,
    RateUnit,
    SearchSpace,
    TrajectoryData,
    WrapUnitStatus,
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


def _config(*, rate_orders: tuple[str, ...] = ("rqp",)) -> InferenceConfig:
    return InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=(AttitudeDirection.BODY_TO_WORLD,),
            quaternion_orders=tuple(QuaternionOrder),
            rate_frames=(RateFrame.BODY,),
            rate_orders=rate_orders,
            rate_units=(RateUnit.RADIANS_PER_SECOND,),
            rate_signs=(1,),
        ),
        window_sizes=(1, 2, 4, 8),
    )
####


def _trajectory(*, scale: float = 0.8, count: int = 121) -> TrajectoryData:
    time_s = np.arange(count, dtype=np.float64) * 0.02
    return encode_synthetic_trajectory(time_s, rich_three_axis_rates(time_s, scale), _truth())
####


def _streams(data: TrajectoryData) -> tuple[FloatArray, FloatArray, FloatArray]:
    if data.orientation is None or data.rates is None or data.time_s is None:
        raise AssertionError("test trajectory must contain all streams")
    ####
    return data.orientation, data.rates, data.time_s
####


@pytest.mark.parametrize("bad_value", (np.nan, np.inf, -np.inf))
def test_nonfinite_streams_are_rejected(bad_value: float) -> None:
    data = _trajectory()
    orientation, rates, time_s = _streams(data)
    bad_rates = rates.copy()
    bad_rates[10, 1] = bad_value

    with pytest.raises(ValueError, match="finite"):
        TrajectoryData(
            representation=data.representation,
            orientation=orientation,
            rates=bad_rates,
            time_s=time_s,
        )
    ####
####


@pytest.mark.parametrize("bad_times", ((0.0, 0.1, 0.1), (0.0, 0.2, 0.1)))
def test_nonincreasing_timestamps_are_rejected(bad_times: tuple[float, float, float]) -> None:
    data = _trajectory(count=3)
    orientation, rates, _ = _streams(data)

    with pytest.raises(ValueError, match="strictly increasing"):
        TrajectoryData(
            representation=data.representation,
            orientation=orientation[:3],
            rates=rates[:3],
            time_s=np.asarray(bad_times),
        )
    ####
####


def test_zero_norm_quaternions_are_rejected() -> None:
    data = _trajectory()
    orientation, rates, time_s = _streams(data)
    orientation = orientation.copy()
    orientation[20] = 0.0

    with pytest.raises(ValueError, match="nonzero norm"):
        TrajectoryData(
            representation=data.representation,
            orientation=orientation,
            rates=rates,
            time_s=time_s,
        )
    ####
####


def test_zero_motion_reports_ambiguity_instead_of_confidence() -> None:
    time_s = np.arange(121, dtype=np.float64) * 0.02
    data = encode_synthetic_trajectory(time_s, np.zeros((time_s.size, 3)), _truth())

    report = infer_conventions(data, _config())

    assert len(report.evidence_contender_class_ids) > 1
    assert report.diagnostics.confidence < 0.5
    assert report.diagnostics.warnings
    assert report.decision.status is InferenceDecisionStatus.ABSTENTION
    assert report.decision.reason is InferenceDecisionReason.EVIDENCE_INSUFFICIENT
####


def test_single_axis_motion_reports_partial_excitation() -> None:
    time_s = np.arange(161, dtype=np.float64) * 0.02
    rates = np.zeros((time_s.size, 3), dtype=np.float64)
    rates[:, 0] = 0.8 * np.sin(0.9 * time_s) + 0.2
    data = encode_synthetic_trajectory(time_s, rates, _truth())

    report = infer_conventions(
        data,
        _config(rate_orders=("pqr", "prq", "qpr", "qrp", "rpq", "rqp")),
    )

    assert report.diagnostics.raw_excitation_rank < 3
    assert report.diagnostics.warnings
    assert report.diagnostics.confidence < 0.95
    assert report.decision.status is InferenceDecisionStatus.ABSTENTION
    assert report.decision.reason is InferenceDecisionReason.EVIDENCE_INSUFFICIENT
####


def test_hard_aliasing_is_reported() -> None:
    time_s = np.arange(121, dtype=np.float64) * 0.02
    rates = np.tile(np.asarray((200.0, 0.0, 0.0)), (time_s.size, 1))
    data = encode_synthetic_trajectory(time_s, rates, _truth())

    report = infer_conventions(data, _config())

    assert report.best.hard_alias_fraction > 0.0
    assert any("pi" in warning or "wrap" in warning for warning in report.diagnostics.warnings)
####


def test_inference_rejects_too_few_samples() -> None:
    data = _trajectory(count=3)
    orientation, rates, time_s = _streams(data)
    with pytest.raises(ValueError, match="At least three samples"):
        TrajectoryData(
            representation=data.representation,
            orientation=orientation[:2],
            rates=rates[:2],
            time_s=time_s[:2],
        )
    ####
####


def test_missing_timestamps_use_explicit_scale_gauge_label() -> None:
    data = _trajectory()
    without_time = replace(data, time_s=None)

    report = infer_conventions(without_time, _config())

    assert report.result_label.value == "scale-gauge"
####


@pytest.mark.parametrize("quaternion_order", tuple(QuaternionOrder))
def test_quaternion_scalar_position_variants_recover_under_stress(
    quaternion_order: QuaternionOrder,
) -> None:
    truth = _truth()
    truth = SyntheticConvention(
        representation=truth.representation,
        attitude_direction=truth.attitude_direction,
        quaternion_order=quaternion_order,
        rate_frame=truth.rate_frame,
        rate_order=truth.rate_order,
    )
    time_s = np.arange(161, dtype=np.float64) * 0.02
    data = encode_synthetic_trajectory(
        time_s,
        rich_three_axis_rates(time_s, scale=0.8),
        truth,
        quaternion_sign_flips=True,
    )

    report = infer_conventions(data, _config())

    assert report.best.hypothesis.quaternion_order is quaternion_order
    assert np.isfinite(report.best.interval_rmse_rad)
####


@pytest.mark.parametrize("storage_dtype", (np.float32, np.float64))
def test_float32_and_nonunit_quaternions_remain_numerically_stable(
    storage_dtype: type[np.floating],
) -> None:
    data = _trajectory(count=161)
    orientation, rates, time_s = _streams(data)
    candidate = TrajectoryData(
        representation=data.representation,
        orientation=cast(Any, (orientation * 1.0e6).astype(storage_dtype)),
        rates=cast(Any, rates.astype(storage_dtype)),
        time_s=cast(Any, time_s.astype(storage_dtype)),
    )

    report = infer_conventions(candidate, _config())

    assert report.best.hypothesis.quaternion_order is QuaternionOrder.SCALAR_LAST
    assert np.isfinite(report.best.interval_rmse_rad)
    assert all(np.isfinite(metric.residual_rmse_rad) for metric in report.best.lag_metrics)
####


def test_one_sample_rate_latency_is_measured_as_residual_degradation() -> None:
    data = _trajectory(count=161)
    _, rates, _ = _streams(data)
    clean = infer_conventions(data, _config())
    lagged = replace(data, rates=np.vstack((rates[0], rates[:-1])))

    report = infer_conventions(lagged, _config())

    assert np.isfinite(report.best.interval_rmse_rad)
    assert report.best.interval_rmse_rad > 1.0e-4
    assert report.best.interval_rmse_rad > clean.best.interval_rmse_rad * 100.0
####


def test_slow_timestamp_drift_stays_valid_but_reduces_fit_quality() -> None:
    data = _trajectory(count=161)
    nominal_time = np.arange(data.sample_count, dtype=np.float64) * 0.02
    drifted_time = np.r_[
        0.0,
        np.cumsum(
            np.diff(nominal_time) * (1.0 + 0.05 * np.linspace(-1.0, 1.0, data.sample_count - 1))
        ),
    ]

    report = infer_conventions(replace(data, time_s=drifted_time), _config())

    assert report.diagnostics.time_source == "timestamps"
    assert np.isfinite(report.best.interval_rmse_rad)
    assert report.best.interval_rmse_rad > 1.0e-4
####


def test_absolute_epoch_seconds_preserve_convention_recovery() -> None:
    data = _trajectory(count=161)
    if data.time_s is None:
        raise AssertionError("test trajectory must contain timestamps")
    ####
    baseline = infer_conventions(data, _config())
    epoch_data = replace(data, time_s=data.time_s + 1_700_000_000.0)

    epoch_report = infer_conventions(epoch_data, _config())

    assert epoch_report.best.hypothesis == baseline.best.hypothesis
    assert epoch_report.decision.status is baseline.decision.status
    assert epoch_report.best.interval_rmse_rad == pytest.approx(
        baseline.best.interval_rmse_rad,
        abs=1.0e-6,
    )
    assert epoch_report.diagnostics.time_source == "timestamps"
####


def test_euler_branch_crossing_is_diagnosed_without_invalid_rotation_math() -> None:
    time_s = np.arange(241, dtype=np.float64) * 0.05
    rates = np.zeros((time_s.size, 3), dtype=np.float64)
    rates[:, 2] = 1.8
    truth = SyntheticConvention(
        representation=AttitudeRepresentation.EULER,
        attitude_direction=AttitudeDirection.BODY_TO_WORLD,
        rate_frame=RateFrame.BODY,
        rate_order="pqr",
        euler_order="zyx",
        euler_mode=EulerMode.INTRINSIC,
        angle_unit=AngleUnit.RADIANS,
    )
    data = encode_synthetic_trajectory(time_s, rates, truth)
    config = InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=(AttitudeDirection.BODY_TO_WORLD,),
            euler_orders=("zyx",),
            euler_modes=(EulerMode.INTRINSIC,),
            angle_units=(AngleUnit.RADIANS,),
            rate_frames=(RateFrame.BODY,),
            rate_orders=("pqr",),
            rate_units=(RateUnit.RADIANS_PER_SECOND,),
            rate_signs=(1,),
        )
    )

    orientation, _, _ = _streams(data)
    raw_jump = np.max(np.linalg.norm(np.diff(orientation, axis=0), axis=1))
    report = infer_conventions(data, config)

    assert raw_jump > 3.0
    assert report.best.heuristic_features.wrap_unit_status is (
        WrapUnitStatus.UNIT_SUGGESTED_BY_WRAP
    )
    assert np.isfinite(report.best.interval_rmse_rad)
    assert report.best.euler_singularity_margin is not None
####
