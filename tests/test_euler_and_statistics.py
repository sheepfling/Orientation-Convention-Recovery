from __future__ import annotations

from itertools import product
from typing import Any, cast

import numpy as np
import pytest
from numpy import square
from numpy import sum as np_sum
from scipy.spatial.transform import Rotation

import attitude_convention_inference.integration as integration_module
from attitude_convention_inference import (
    AngleUnit,
    AttitudeDirection,
    AttitudeRepresentation,
    ConventionConstraints,
    ConventionHypothesis,
    EulerMode,
    InferenceConfig,
    QuaternionOrder,
    RateFrame,
    RateUnit,
    ResultLabel,
    SearchSpace,
    TrajectoryData,
    WrapUnitStatus,
    infer_conventions,
)
from attitude_convention_inference import heuristics as heuristic_module
from attitude_convention_inference.core import FloatArray
from attitude_convention_inference.heuristics import EulerHeuristicDiagnostics
from attitude_convention_inference.inference import estimate_noise_process
from attitude_convention_inference.statistics import (
    fixed_noise_log_likelihood,
    log_likelihood_advantage_by_prefix,
    log_likelihood_advantage_from_squared_errors_by_prefix,
)
from attitude_convention_inference.synthetic import (
    SyntheticConvention,
    encode_synthetic_trajectory,
    rich_three_axis_rates,
)

PROPER_EULER_ORDERS = ("xyx", "xzx", "yxy", "yzy", "zxz", "zyz")
PROPER_EULER_CASES = tuple(product(PROPER_EULER_ORDERS, tuple(EulerMode)))
REMAINING_TAIT_BRYAN_CASES = tuple(product(("xzy", "yxz", "yzx"), tuple(EulerMode)))


def _from_quat(quaternions: FloatArray) -> Rotation:
    return cast(Rotation, cast(Any, Rotation).from_quat(quaternions))
####


def _from_rotvec(rotvecs: FloatArray) -> Rotation:
    return cast(Rotation, cast(Any, Rotation).from_rotvec(rotvecs))
####


def _quaternions(rotation: Rotation) -> FloatArray:
    return np.asarray(cast(Any, rotation).as_quat(), dtype=np.float64)
####


def _euler_config(
    order: str,
    rate_orders: tuple[str, ...] = ("pqr",),
) -> InferenceConfig:
    return InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=(AttitudeDirection.BODY_TO_WORLD,),
            euler_orders=(order,),
            euler_modes=tuple(EulerMode),
            angle_units=(AngleUnit.RADIANS,),
            rate_frames=(RateFrame.BODY,),
            rate_orders=rate_orders,
            rate_units=(RateUnit.RADIANS_PER_SECOND,),
            rate_signs=(1,),
        )
    )
####


@pytest.mark.parametrize("rates_available", (True, False))
def test_inference_reuses_euler_heuristics_across_modes_and_rate_orders(
    monkeypatch: pytest.MonkeyPatch,
    rates_available: bool,
) -> None:
    time_s = np.arange(51, dtype=np.float64) * 0.025
    truth = SyntheticConvention(
        representation=AttitudeRepresentation.EULER,
        attitude_direction=AttitudeDirection.BODY_TO_WORLD,
        rate_frame=RateFrame.BODY,
        rate_order="pqr",
        euler_order="xyz",
        euler_mode=EulerMode.EXTRINSIC,
        angle_unit=AngleUnit.RADIANS,
    )
    data = encode_synthetic_trajectory(time_s, rich_three_axis_rates(time_s, 0.6), truth)
    if not rates_available:
        data = TrajectoryData(
            representation=AttitudeRepresentation.EULER,
            orientation=data.orientation,
            rates=None,
            time_s=data.time_s,
        )
    ####
    config = _euler_config("xyz", rate_orders=("pqr", "qpr"))
    original = heuristic_module.euler_heuristic_diagnostics
    calls = 0

    def counted_diagnostics(
        trajectory: TrajectoryData,
        hypothesis: ConventionHypothesis,
    ) -> EulerHeuristicDiagnostics:
        nonlocal calls
        calls += 1
        return original(trajectory, hypothesis)
    ####

    monkeypatch.setattr(
        heuristic_module,
        "euler_heuristic_diagnostics",
        counted_diagnostics,
    )

    report = infer_conventions(data, config)

    assert report.hypothesis_count == 4
    assert calls == 1
####


def test_euler_inference_reuses_rate_integrations_without_changing_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time_s = np.arange(41, dtype=np.float64) * 0.025
    truth = SyntheticConvention(
        representation=AttitudeRepresentation.EULER,
        attitude_direction=AttitudeDirection.BODY_TO_WORLD,
        rate_frame=RateFrame.BODY,
        rate_order="pqr",
        euler_order="xyz",
        euler_mode=EulerMode.EXTRINSIC,
        angle_unit=AngleUnit.RADIANS,
    )
    data = encode_synthetic_trajectory(time_s, rich_three_axis_rates(time_s, 0.6), truth)
    cache_type = integration_module.IntegratedIntervalsCache
    original_method = cache_type.integrate
    original_integrator = integration_module.integrate_interval_increments
    calls = 0

    def counted_integrator(
        rates_rad_s: FloatArray,
        durations_s: FloatArray,
        frame: RateFrame,
        substeps: int,
    ) -> integration_module.IntegratedIntervals:
        nonlocal calls
        calls += 1
        return original_integrator(rates_rad_s, durations_s, frame, substeps)
    ####

    def uncached_integration(
        cache: integration_module.IntegratedIntervalsCache,
        hypothesis: ConventionHypothesis,
        rates_rad_s: FloatArray,
        durations_s: FloatArray,
        frame: RateFrame,
        substeps: int,
    ) -> integration_module.IntegratedIntervals:
        del cache, hypothesis
        return integration_module.integrate_interval_increments(
            rates_rad_s,
            durations_s,
            frame,
            substeps,
        )
    ####

    monkeypatch.setattr(
        integration_module,
        "integrate_interval_increments",
        counted_integrator,
    )
    monkeypatch.setattr(cache_type, "integrate", uncached_integration)
    uncached_report = infer_conventions(data)
    uncached_calls = calls

    calls = 0
    monkeypatch.setattr(cache_type, "integrate", original_method)
    cached_report = infer_conventions(data)

    assert cached_report.model_dump_json() == uncached_report.model_dump_json()
    assert uncached_calls >= cached_report.hypothesis_count
    assert calls * 10 < uncached_calls
####


def test_integrated_interval_cache_invalidates_changed_rates_and_durations() -> None:
    time_s = np.arange(21, dtype=np.float64) * 0.05
    rates = rich_three_axis_rates(time_s)
    data = TrajectoryData(
        representation=AttitudeRepresentation.QUATERNION,
        orientation=np.tile(
            np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64),
            (time_s.size, 1),
        ),
        rates=rates,
        time_s=time_s,
    )
    hypothesis = ConventionHypothesis(
        attitude_direction=AttitudeDirection.BODY_TO_WORLD,
        quaternion_order=QuaternionOrder.SCALAR_LAST,
        rate_frame=RateFrame.BODY,
        rate_order="pqr",
        rate_unit=RateUnit.RADIANS_PER_SECOND,
    )
    rates_rad_s = rates.copy()
    durations_s = np.diff(time_s)
    cache = integration_module.IntegratedIntervalsCache(data)

    first = cache.integrate(hypothesis, rates_rad_s, durations_s, RateFrame.BODY, 4)
    rates_rad_s[1, 0] += 0.5
    changed_rates = cache.integrate(hypothesis, rates_rad_s, durations_s, RateFrame.BODY, 4)
    expected_after_rate_change = integration_module.integrate_interval_increments(
        rates_rad_s,
        durations_s,
        RateFrame.BODY,
        4,
    )
    np.testing.assert_allclose(
        _quaternions(changed_rates.predicted_increments),
        _quaternions(expected_after_rate_change.predicted_increments),
    )
    assert changed_rates is not first

    durations_s[2] *= 2.0
    changed_durations = cache.integrate(
        hypothesis,
        rates_rad_s,
        durations_s,
        RateFrame.BODY,
        4,
    )
    expected_after_duration_change = integration_module.integrate_interval_increments(
        rates_rad_s,
        durations_s,
        RateFrame.BODY,
        4,
    )
    np.testing.assert_allclose(
        _quaternions(changed_durations.predicted_increments),
        _quaternions(expected_after_duration_change.predicted_increments),
    )
    assert changed_durations is not changed_rates
####


@pytest.mark.parametrize(("order", "mode"), PROPER_EULER_CASES)
def test_all_proper_euler_orders_and_modes_are_inference_candidates(
    order: str,
    mode: EulerMode,
) -> None:
    time_s = np.arange(121, dtype=np.float64) * 0.02
    truth = SyntheticConvention(
        representation=AttitudeRepresentation.EULER,
        attitude_direction=AttitudeDirection.BODY_TO_WORLD,
        rate_frame=RateFrame.BODY,
        rate_order="pqr",
        euler_order=order,
        euler_mode=mode,
        angle_unit=AngleUnit.RADIANS,
    )
    data = encode_synthetic_trajectory(time_s, rich_three_axis_rates(time_s, 0.6), truth)

    report = infer_conventions(data, _euler_config(order))

    assert report.best.hypothesis.euler_order == order
    assert report.best.hypothesis.euler_mode is mode
    assert report.best.interval_rmse_rad < 1.0e-6
####


@pytest.mark.parametrize(("order", "mode"), REMAINING_TAIT_BRYAN_CASES)
def test_remaining_tait_bryan_orders_and_modes_are_inference_candidates(
    order: str,
    mode: EulerMode,
) -> None:
    time_s = np.arange(101, dtype=np.float64) * 0.02
    truth = SyntheticConvention(
        representation=AttitudeRepresentation.EULER,
        attitude_direction=AttitudeDirection.BODY_TO_WORLD,
        rate_frame=RateFrame.BODY,
        rate_order="pqr",
        euler_order=order,
        euler_mode=mode,
        angle_unit=AngleUnit.RADIANS,
    )
    data = encode_synthetic_trajectory(time_s, rich_three_axis_rates(time_s, 0.6), truth)

    report = infer_conventions(data, _euler_config(order))

    assert report.best.hypothesis.euler_order == order
    assert report.best.hypothesis.euler_mode is mode
    assert report.best.interval_rmse_rad < 1.0e-6
####


def test_proper_euler_gimbal_lock_uses_sine_singularity() -> None:
    angles = np.column_stack(
        (
            np.linspace(0.1, 0.4, 21),
            np.zeros(21),
            np.linspace(-0.2, 0.3, 21),
        )
    )
    data = TrajectoryData(
        representation=AttitudeRepresentation.EULER,
        orientation=angles,
        rates=np.zeros((21, 3)),
        time_s=np.arange(21, dtype=np.float64) * 0.1,
    )
    config = InferenceConfig(
        search_space=_euler_config("xyx").search_space,
        constraints=ConventionConstraints(
            euler_mode=EulerMode.INTRINSIC,
            angle_unit=AngleUnit.RADIANS,
        ),
    )

    report = infer_conventions(data, config)

    assert report.best.gimbal_lock_fraction == pytest.approx(1.0)
####


def test_euler_range_diagnostic_is_soft_and_preserves_rotation_validity() -> None:
    sample_count = 21
    data = TrajectoryData(
        representation=AttitudeRepresentation.EULER,
        orientation=np.tile((370.0, 10.0, -190.0), (sample_count, 1)),
        rates=np.zeros((sample_count, 3)),
        time_s=np.arange(sample_count, dtype=np.float64) * 0.1,
    )
    config = InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=(AttitudeDirection.BODY_TO_WORLD,),
            euler_orders=("xyz",),
            euler_modes=(EulerMode.EXTRINSIC,),
            angle_units=(AngleUnit.DEGREES,),
            rate_frames=(RateFrame.BODY,),
            rate_orders=("pqr",),
            rate_units=(RateUnit.RADIANS_PER_SECOND,),
            rate_signs=(1,),
        )
    )

    report = infer_conventions(data, config)

    assert report.best.interval_rmse_rad == pytest.approx(0.0)
    assert report.best.euler_range_penalty_rad2 > 0.0
    assert report.best.euler_singularity_margin is not None
    assert any("principal output ranges" in item for item in report.best.warnings)
####


@pytest.mark.parametrize("unit", tuple(AngleUnit))
def test_all_supported_raw_euler_angle_units_decode_consistently(unit: AngleUnit) -> None:
    time_s = np.arange(61, dtype=np.float64) * 0.02
    truth = SyntheticConvention(
        representation=AttitudeRepresentation.EULER,
        attitude_direction=AttitudeDirection.BODY_TO_WORLD,
        rate_frame=RateFrame.BODY,
        rate_order="pqr",
        euler_order="xyz",
        euler_mode=EulerMode.INTRINSIC,
        angle_unit=unit,
    )
    data = encode_synthetic_trajectory(time_s, rich_three_axis_rates(time_s, 0.4), truth)
    config = InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=(AttitudeDirection.BODY_TO_WORLD,),
            euler_orders=("xyz",),
            euler_modes=(EulerMode.INTRINSIC,),
            angle_units=(unit,),
            rate_frames=(RateFrame.BODY,),
            rate_orders=("pqr",),
            rate_units=(RateUnit.RADIANS_PER_SECOND,),
            rate_signs=(1,),
        )
    )

    report = infer_conventions(data, config)

    assert report.best.hypothesis.angle_unit is unit
    assert report.best.interval_rmse_rad < 1.0e-6
    assert report.result_label is ResultLabel.TRAJECTORY_CLASS
####


def test_wrap_classifier_abstains_without_a_visible_crossing() -> None:
    sample_count = 21
    data = TrajectoryData(
        representation=AttitudeRepresentation.EULER,
        orientation=np.tile((0.1, 0.2, 0.3), (sample_count, 1)),
        rates=np.zeros((sample_count, 3)),
        time_s=np.arange(sample_count, dtype=np.float64) * 0.1,
    )
    config = InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=(AttitudeDirection.BODY_TO_WORLD,),
            euler_orders=("xyz",),
            euler_modes=(EulerMode.EXTRINSIC,),
            angle_units=(AngleUnit.RADIANS,),
            rate_frames=(RateFrame.BODY,),
            rate_orders=("pqr",),
            rate_units=(RateUnit.RADIANS_PER_SECOND,),
            rate_signs=(1,),
        )
    )

    report = infer_conventions(data, config)

    assert report.best.heuristic_features.wrap_unit_status is (
        WrapUnitStatus.UNIT_UNDETERMINED_BY_WRAP
    )
####


def test_wrap_classifier_marks_a_visible_period_crossing_as_suggestive() -> None:
    angles = np.asarray((170.0, 175.0, -179.0, -174.0) * 6, dtype=np.float64)
    data = TrajectoryData(
        representation=AttitudeRepresentation.EULER,
        orientation=np.column_stack((angles, np.zeros_like(angles), np.zeros_like(angles))),
        rates=np.zeros((angles.size, 3)),
        time_s=np.arange(angles.size, dtype=np.float64) * 0.1,
    )
    config = InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=(AttitudeDirection.BODY_TO_WORLD,),
            euler_orders=("xyz",),
            euler_modes=(EulerMode.EXTRINSIC,),
            angle_units=(AngleUnit.DEGREES,),
            rate_frames=(RateFrame.BODY,),
            rate_orders=("pqr",),
            rate_units=(RateUnit.RADIANS_PER_SECOND,),
            rate_signs=(1,),
        )
    )

    report = infer_conventions(data, config)

    assert report.best.heuristic_features.wrap_unit_status is (
        WrapUnitStatus.UNIT_SUGGESTED_BY_WRAP
    )
    assert report.best.heuristic_features.inferred_wrap_count > 0
####


def test_noise_process_estimator_recovers_scale_after_trim() -> None:
    rng = np.random.default_rng(7)
    residuals = rng.normal(0.0, 0.002, (1000, 3))
    residuals[:20] += np.asarray([0.20, -0.15, 0.10])

    estimate = estimate_noise_process(residuals, 0.02)

    assert estimate.retained_count == 980
    assert estimate.sigma_rad == pytest.approx(0.002, rel=0.12)
    assert abs(estimate.lag1_autocorrelation) < 0.10
####


def test_prefix_evidence_matches_direct_fixed_noise_likelihood() -> None:
    rng = np.random.default_rng(41)
    winner = np.asarray(rng.normal(0.0, 0.002, (37, 3)), dtype=np.float64)
    competitors = (
        np.asarray(rng.normal(0.0, 0.003, (37, 3)), dtype=np.float64),
        np.asarray(rng.normal(0.0, 0.005, (37, 3)), dtype=np.float64),
    )
    sigma = 0.0025

    advantages = log_likelihood_advantage_by_prefix(winner, competitors, sigma)
    compact_advantages = log_likelihood_advantage_from_squared_errors_by_prefix(
        np_sum(square(winner), axis=1),
        (np_sum(square(item), axis=1) for item in competitors),
        sigma,
    )
    assert compact_advantages == pytest.approx(advantages)

    for prefix in range(1, len(winner) + 1):
        expected = fixed_noise_log_likelihood(winner[:prefix], sigma) - max(
            fixed_noise_log_likelihood(item[:prefix], sigma) for item in competitors
        )
        assert float(advantages[prefix - 1]) == pytest.approx(expected, abs=1.0e-9)
    ####
####


def _quaternion_config() -> InferenceConfig:
    return InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=(AttitudeDirection.BODY_TO_WORLD,),
            quaternion_orders=tuple(QuaternionOrder),
            rate_frames=(RateFrame.BODY,),
            rate_orders=("pqr",),
            rate_units=(RateUnit.RADIANS_PER_SECOND,),
            rate_signs=(1,),
        ),
        evidence_min_intervals=20,
        evidence_stability_windows=3,
    )
####


def test_estimates_rotation_noise_and_early_evidence() -> None:
    rng = np.random.default_rng(11)
    time_s = np.arange(161, dtype=np.float64) * 0.02
    truth = SyntheticConvention(
        representation=AttitudeRepresentation.QUATERNION,
        attitude_direction=AttitudeDirection.BODY_TO_WORLD,
        quaternion_order=QuaternionOrder.SCALAR_LAST,
        rate_frame=RateFrame.BODY,
        rate_order="pqr",
    )
    clean = encode_synthetic_trajectory(time_s, rich_three_axis_rates(time_s, 0.8), truth)
    if clean.orientation is None or clean.rates is None or clean.time_s is None:
        raise AssertionError("synthetic trajectory must contain all streams")
    ####
    attitudes = _from_quat(clean.orientation)
    noisy_attitudes = (
        _from_rotvec(
            np.asarray(
                rng.normal(0.0, 0.0015, (clean.sample_count, 3)),
                dtype=np.float64,
            )
        )
        * attitudes
    )
    data = TrajectoryData(
        representation=clean.representation,
        orientation=_quaternions(noisy_attitudes),
        rates=clean.rates + rng.normal(0.0, 0.01, clean.rates.shape),
        time_s=clean.time_s,
    )

    report = infer_conventions(data, _quaternion_config())

    assert report.best.hypothesis.quaternion_order is QuaternionOrder.SCALAR_LAST
    assert report.diagnostics.noise_model == "isotropic_rotation_vector_gaussian"
    assert report.diagnostics.estimated_noise_sigma_rad > 1.0e-6
    assert -1.0 <= report.diagnostics.noise_lag1_autocorrelation <= 1.0
    assert report.diagnostics.evidence_sufficient
    assert report.diagnostics.evidence_intervals_used < data.sample_count - 1
    assert report.diagnostics.evidence_log_likelihood_ratio > 5.0
####


def test_weak_motion_requires_the_full_file_without_separation() -> None:
    time_s = np.arange(161, dtype=np.float64) * 0.02
    truth = SyntheticConvention(
        representation=AttitudeRepresentation.QUATERNION,
        attitude_direction=AttitudeDirection.BODY_TO_WORLD,
        quaternion_order=QuaternionOrder.SCALAR_LAST,
        rate_frame=RateFrame.BODY,
        rate_order="pqr",
    )
    data = encode_synthetic_trajectory(time_s, rich_three_axis_rates(time_s, 1.0e-5), truth)

    report = infer_conventions(data, _quaternion_config())

    assert not report.diagnostics.evidence_sufficient
    assert report.diagnostics.evidence_intervals_used == data.sample_count - 1
    assert report.diagnostics.evidence_fraction == pytest.approx(1.0)
####
