from __future__ import annotations

from math import pi
from typing import Any, cast

import numpy as np
import pytest
from numpy import amax
from scipy.spatial.transform import Rotation

import attitude_convention_inference.integration as integration_module
from attitude_convention_inference import (
    AttitudeDirection,
    AttitudeRepresentation,
    InferenceConfig,
    QuaternionOrder,
    RateFrame,
    RateUnit,
    ResultLabel,
    SearchSpace,
    infer_conventions,
)
from attitude_convention_inference.core import (
    FloatArray,
    IncrementWindowComposer,
    ObservedIncrementCache,
    TrajectoryData,
    compose_increment_windows,
    observed_interval_increments,
    observed_window_increments,
)
from attitude_convention_inference.math_utils import rotation_product
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
        rate_order="pqr",
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


def _magnitudes(rotation: Rotation) -> FloatArray:
    return np.asarray(cast(Any, rotation).magnitude(), dtype=np.float64)
####


def test_direction_frame_sign_dual_has_one_exact_data_class() -> None:
    time_s = np.arange(101, dtype=np.float64) * 0.02
    data = encode_synthetic_trajectory(time_s, rich_three_axis_rates(time_s), _truth())
    config = InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=tuple(AttitudeDirection),
            quaternion_orders=(QuaternionOrder.SCALAR_LAST,),
            rate_frames=tuple(RateFrame),
            rate_orders=("pqr",),
            rate_units=(RateUnit.RADIANS_PER_SECOND,),
            rate_signs=(-1, 1),
        )
    )

    report = infer_conventions(data, config)
    truth = next(
        item
        for item in report.rankings
        if item.hypothesis.attitude_direction is AttitudeDirection.BODY_TO_WORLD
        and item.hypothesis.rate_frame is RateFrame.BODY
        and item.hypothesis.rate_sign == 1
    )
    dual = next(
        item
        for item in report.rankings
        if item.hypothesis.identifier() == truth.hypothesis.structural_dual().identifier()
    )

    assert truth.equivalence_class_id == dual.equivalence_class_id
    assert truth.raw_geometric_risk == pytest.approx(dual.raw_geometric_risk, abs=1.0e-15)
    assert truth.data_rank == dual.data_rank
    assert set(truth.exact_equivalent_hypothesis_ids) == {
        truth.hypothesis.identifier(),
        dual.hypothesis.identifier(),
    }
####


def test_rate_unit_and_period_share_an_exact_gauge_without_timestamps() -> None:
    time_s = np.arange(81, dtype=np.float64) * 0.025
    timed = encode_synthetic_trajectory(time_s, rich_three_axis_rates(time_s, 0.7), _truth())
    data = TrajectoryData(
        representation=timed.representation,
        orientation=timed.orientation,
        rates=timed.rates,
        time_s=None,
    )
    config = InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=(AttitudeDirection.BODY_TO_WORLD,),
            quaternion_orders=(QuaternionOrder.SCALAR_LAST,),
            rate_frames=(RateFrame.BODY,),
            rate_orders=("pqr",),
            rate_units=tuple(RateUnit),
            rate_signs=(1,),
        )
    )

    report = infer_conventions(data, config)
    radians, degrees = sorted(
        report.rankings,
        key=lambda item: str(
            getattr(cast(Any, item.hypothesis.rate_unit), "value", item.hypothesis.rate_unit)
        ),
    )

    assert radians.equivalence_class_id == degrees.equivalence_class_id
    assert radians.raw_geometric_risk == pytest.approx(degrees.raw_geometric_risk, abs=1.0e-12)
    assert radians.inferred_sample_period_s is not None
    assert degrees.inferred_sample_period_s is not None
    ratio = max(radians.inferred_sample_period_s, degrees.inferred_sample_period_s) / min(
        radians.inferred_sample_period_s, degrees.inferred_sample_period_s
    )
    assert ratio == pytest.approx(180.0 / pi, rel=5.0e-4)
    assert report.diagnostics.rate_scale_period_gauge_active
    assert report.result_label is ResultLabel.SCALE_GAUGE
    assert report.operational_hypothesis_id == report.best.hypothesis.identifier()
    assert report.evidence_contender_class_ids
####


def test_period_bound_breaks_rate_unit_scale_equivalence() -> None:
    time_s = np.arange(81, dtype=np.float64) * 0.025
    timed = encode_synthetic_trajectory(time_s, rich_three_axis_rates(time_s, 0.7), _truth())
    data = TrajectoryData(
        representation=timed.representation,
        orientation=timed.orientation,
        rates=timed.rates,
        time_s=None,
    )
    config = InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=(AttitudeDirection.BODY_TO_WORLD,),
            quaternion_orders=(QuaternionOrder.SCALAR_LAST,),
            rate_frames=(RateFrame.BODY,),
            rate_orders=("pqr",),
            rate_units=tuple(RateUnit),
            rate_signs=(1,),
        ),
        inferred_sample_period_bounds_s=(1.0e-6, 0.5),
    )

    report = infer_conventions(data, config)
    radians = next(
        item for item in report.rankings if item.hypothesis.rate_unit is RateUnit.RADIANS_PER_SECOND
    )
    degrees = next(
        item for item in report.rankings if item.hypothesis.rate_unit is RateUnit.DEGREES_PER_SECOND
    )

    assert degrees.inferred_sample_period_s == pytest.approx(0.5)
    assert radians.inferred_sample_period_s == pytest.approx(0.025, rel=5.0e-3)
    assert radians.equivalence_class_id != degrees.equivalence_class_id
####


def test_direction_dual_and_rate_scale_gauge_form_one_transitive_class() -> None:
    time_s = np.arange(81, dtype=np.float64) * 0.025
    timed = encode_synthetic_trajectory(time_s, rich_three_axis_rates(time_s, 0.7), _truth())
    data = TrajectoryData(
        representation=timed.representation,
        orientation=timed.orientation,
        rates=timed.rates,
        time_s=None,
    )
    config = InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=tuple(AttitudeDirection),
            quaternion_orders=(QuaternionOrder.SCALAR_LAST,),
            rate_frames=tuple(RateFrame),
            rate_orders=("pqr",),
            rate_units=tuple(RateUnit),
            rate_signs=(-1, 1),
        )
    )

    report = infer_conventions(data, config)
    truth = next(
        item
        for item in report.rankings
        if item.hypothesis.attitude_direction is AttitudeDirection.BODY_TO_WORLD
        and item.hypothesis.rate_frame is RateFrame.BODY
        and item.hypothesis.rate_unit is RateUnit.RADIANS_PER_SECOND
        and item.hypothesis.rate_sign == 1
    )

    assert len(truth.exact_equivalent_hypothesis_ids) == 4
    assert truth.hypothesis.structural_dual().identifier() in (
        truth.exact_equivalent_hypothesis_ids
    )
    assert {
        item.equivalence_class_id
        for item in report.rankings
        if item.hypothesis.identifier() in truth.exact_equivalent_hypothesis_ids
    } == {truth.equivalence_class_id}
####


def test_quaternion_orthogonal_coordinate_map_preserves_intrinsic_distances() -> None:
    random_rotation = cast(Rotation, cast(Any, Rotation).random(24, random_state=9))
    quaternions = _quaternions(random_rotation)
    orthogonal_map = np.array(
        [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    transformed = quaternions @ orthogonal_map.T
    original_distances = _magnitudes(
        _from_quat(quaternions[:-1]).inv() * _from_quat(quaternions[1:])
    )
    transformed_distances = _magnitudes(
        _from_quat(transformed[:-1]).inv() * _from_quat(transformed[1:])
    )

    assert transformed_distances == pytest.approx(original_distances, abs=1.0e-12)
####


def test_canonical_tie_order_is_invariant_to_search_enumeration() -> None:
    time_s = np.arange(101, dtype=np.float64) * 0.02
    data = encode_synthetic_trajectory(time_s, rich_three_axis_rates(time_s), _truth())
    forward = InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=tuple(AttitudeDirection),
            quaternion_orders=(QuaternionOrder.SCALAR_LAST,),
            rate_frames=tuple(RateFrame),
            rate_orders=("pqr",),
            rate_units=(RateUnit.RADIANS_PER_SECOND,),
            rate_signs=(-1, 1),
        )
    )
    reversed_search = SearchSpace(
        attitude_directions=tuple(reversed(tuple(AttitudeDirection))),
        quaternion_orders=(QuaternionOrder.SCALAR_LAST,),
        rate_frames=tuple(reversed(tuple(RateFrame))),
        rate_orders=("pqr",),
        rate_units=(RateUnit.RADIANS_PER_SECOND,),
        rate_signs=(1, -1),
    )

    first = infer_conventions(data, forward)
    second = infer_conventions(data, InferenceConfig(search_space=reversed_search))

    assert first.best.hypothesis.identifier() == second.best.hypothesis.identifier()
    assert tuple(item.hypothesis.identifier() for item in first.rankings) == tuple(
        item.hypothesis.identifier() for item in second.rankings
    )
####


@pytest.mark.parametrize("frame", tuple(RateFrame))
def test_prefix_window_products_match_direct_noncommutative_products(frame: RateFrame) -> None:
    increments = _from_rotvec(
        np.asarray(
            [
                (0.1, 0.2, -0.1),
                (-0.2, 0.05, 0.3),
                (0.15, -0.25, 0.12),
                (0.04, 0.11, -0.2),
                (-0.08, 0.17, 0.09),
                (0.2, -0.04, 0.13),
            ],
            dtype=np.float64,
        )
    )

    composer = IncrementWindowComposer.from_increments(increments, frame)
    for window_size in (1, 2, 3, 6):
        accelerated = composer.compose(window_size)
        expected: list[Rotation] = []
        for start in accelerated.starts:
            total = Rotation.identity()
            for offset in range(window_size):
                increment = increments[int(start + offset)]
                total = (
                    rotation_product(total, increment)
                    if frame is RateFrame.BODY
                    else rotation_product(increment, total)
                )
            ####
            expected.append(total)
        ####

        assert _quaternions(accelerated.window_increments) == pytest.approx(
            _quaternions(cast(Rotation, cast(Any, Rotation).concatenate(expected))),
            abs=1.0e-12,
        )
        one_shot = compose_increment_windows(increments, window_size, frame)
        assert _quaternions(one_shot.window_increments) == pytest.approx(
            _quaternions(accelerated.window_increments), abs=1.0e-12
        )
    ####
####


@pytest.mark.parametrize("frame", tuple(RateFrame))
def test_observed_increment_cache_matches_and_reuses_direct_rotations(frame: RateFrame) -> None:
    attitudes = _from_rotvec(
        np.asarray(
            [
                (0.12, -0.08, 0.03),
                (-0.04, 0.19, 0.11),
                (0.23, 0.02, -0.14),
                (-0.17, 0.09, 0.21),
                (0.06, -0.22, 0.07),
                (0.15, 0.13, -0.18),
                (-0.09, -0.05, 0.16),
            ],
            dtype=np.float64,
        )
    )
    cache = ObservedIncrementCache(attitudes)

    cached_intervals = cache.interval_increments(frame)
    assert cache.interval_increments(frame) is cached_intervals
    direct_intervals = observed_interval_increments(attitudes, frame)
    np.testing.assert_allclose(
        _magnitudes(cached_intervals.inv() * direct_intervals),
        0.0,
        atol=1e-12,
    )

    for window_size in (1, 2, 6):
        cached_windows = cache.window_composition(window_size, frame)
        assert cache.window_composition(window_size, frame) is cached_windows
        np.testing.assert_array_equal(
            cached_windows.starts,
            np.arange(0, len(attitudes) - window_size, dtype=np.int64),
        )
        direct_windows = observed_window_increments(
            attitudes,
            cached_windows.starts,
            window_size,
            frame,
        )
        np.testing.assert_allclose(
            _magnitudes(cached_windows.window_increments.inv() * direct_windows),
            0.0,
            atol=1e-12,
        )
    ####
####


def test_inference_computes_observed_windows_once_per_frame_and_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time_s = np.arange(31, dtype=np.float64) * 0.02
    data = encode_synthetic_trajectory(time_s, rich_three_axis_rates(time_s), _truth())
    window_sizes = (1, 2, 4)
    config = InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=(_truth().attitude_direction,),
            quaternion_orders=(QuaternionOrder.SCALAR_LAST,),
            rate_frames=tuple(RateFrame),
            rate_orders=("pqr",),
            rate_units=(RateUnit.RADIANS_PER_SECOND,),
            rate_signs=(-1, 1),
        ),
        window_sizes=window_sizes,
    )
    original = integration_module.observed_window_increments
    calls: dict[tuple[RateFrame, int], int] = {}

    def counted_observed_windows(
        attitudes: Rotation,
        starts: Any,
        window_size: int,
        frame: RateFrame,
    ) -> Rotation:
        key = (frame, window_size)
        calls[key] = calls.get(key, 0) + 1
        return original(attitudes, starts, window_size, frame)
    ####

    monkeypatch.setattr(
        integration_module,
        "observed_window_increments",
        counted_observed_windows,
    )

    report = infer_conventions(data, config)

    assert report.hypothesis_count == 4
    assert calls == {(frame, size): 1 for frame in RateFrame for size in window_sizes}
####


def test_missing_time_refinement_reuses_observed_rotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time_s = np.arange(31, dtype=np.float64) * 0.02
    timed_data = encode_synthetic_trajectory(time_s, rich_three_axis_rates(time_s), _truth())
    data = TrajectoryData(
        representation=timed_data.representation,
        orientation=timed_data.orientation,
        rates=timed_data.rates,
        time_s=None,
    )
    window_sizes = (1, 2)
    config = InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=(AttitudeDirection.BODY_TO_WORLD,),
            quaternion_orders=(QuaternionOrder.SCALAR_LAST,),
            rate_frames=tuple(RateFrame),
            rate_orders=("pqr",),
            rate_units=(RateUnit.RADIANS_PER_SECOND,),
            rate_signs=(-1, 1),
        ),
        window_sizes=window_sizes,
    )
    original_intervals = integration_module.observed_interval_increments
    original_windows = integration_module.observed_window_increments
    interval_calls: dict[RateFrame, int] = {}
    window_calls: dict[tuple[RateFrame, int], int] = {}

    def counted_intervals(attitudes: Rotation, frame: RateFrame) -> Rotation:
        interval_calls[frame] = interval_calls.get(frame, 0) + 1
        return original_intervals(attitudes, frame)
    ####

    def counted_windows(
        attitudes: Rotation,
        starts: Any,
        window_size: int,
        frame: RateFrame,
    ) -> Rotation:
        key = (frame, window_size)
        window_calls[key] = window_calls.get(key, 0) + 1
        return original_windows(attitudes, starts, window_size, frame)
    ####

    monkeypatch.setattr(integration_module, "observed_interval_increments", counted_intervals)
    monkeypatch.setattr(integration_module, "observed_window_increments", counted_windows)

    report = infer_conventions(data, config)

    assert report.hypothesis_count == 4
    assert report.diagnostics.time_source == "inferred_sample_period"
    detailed_frame = RateFrame(report.best.hypothesis.rate_frame)
    assert interval_calls == {frame: 1 + int(frame is detailed_frame) for frame in RateFrame}
    assert window_calls == {
        (frame, size): 1 + int(frame is detailed_frame)
        for frame in RateFrame
        for size in window_sizes
    }
####


@pytest.mark.parametrize("frame", tuple(RateFrame))
def test_long_prefix_products_match_direct_rotations(frame: RateFrame) -> None:
    rng = np.random.default_rng(804)
    vectors = rng.normal(size=(257, 3)) * 0.45
    vectors[::23] *= 6.0
    increments = _from_rotvec(np.asarray(vectors, dtype=np.float64))
    composer = IncrementWindowComposer.from_increments(increments, frame)

    for window_size in (1, 8, 31, 257):
        accelerated = composer.compose(window_size)
        expected: list[Rotation] = []
        for start in accelerated.starts:
            total = Rotation.identity()
            for offset in range(window_size):
                increment = increments[int(start + offset)]
                total = (
                    rotation_product(total, increment)
                    if frame is RateFrame.BODY
                    else rotation_product(increment, total)
                )
            ####
            expected.append(total)
        ####
        direct = cast(Rotation, cast(Any, Rotation).concatenate(expected))
        disagreement = accelerated.window_increments.inv() * direct
        angles = np.asarray(cast(Any, disagreement).magnitude(), dtype=np.float64)
        assert amax(angles) < 1.0e-11
    ####
####
