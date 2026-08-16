from __future__ import annotations

from math import pi
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
    ResultLabel,
    SearchSpace,
    infer_conventions,
)
from attitude_convention_inference.core import FloatArray, TrajectoryData, compose_increment_windows
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

    accelerated = compose_increment_windows(increments, 3, frame)
    expected: list[Rotation] = []
    for start in accelerated.starts:
        total = Rotation.identity()
        for offset in range(3):
            increment = increments[int(start + offset)]
            total = total * increment if frame is RateFrame.BODY else increment * total
        ####
        expected.append(total)
    ####

    assert _quaternions(accelerated.window_increments) == pytest.approx(
        _quaternions(cast(Rotation, cast(Any, Rotation).concatenate(expected))),
        abs=1.0e-12,
    )
####


