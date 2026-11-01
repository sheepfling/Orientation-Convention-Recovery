from __future__ import annotations

from itertools import pairwise
from typing import Any, cast

import pytest
from numpy import (
    abs as np_abs,
)
from numpy import (
    arange,
    arccos,
    array,
    asarray,
    clip,
    column_stack,
    concatenate,
    float64,
    interp,
    isclose,
    searchsorted,
    stack,
    unique,
)
from numpy import (
    sum as np_sum,
)
from scipy.spatial.transform import Rotation

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
    SparseOrientationRateData,
    infer_conventions,
    infer_sparse_orientation_conventions,
)
from attitude_convention_inference.integration import integrate_rates_between_anchors
from attitude_convention_inference.math_utils import (
    FloatArray,
    rotation_from_quat,
    rotation_from_rotvec,
    rotation_product,
    rotation_quaternions,
    rotation_rotvecs,
)
from attitude_convention_inference.models import ConventionHypothesis
from scripts.generate_validation_corpus import sampled_rate_motion


def test_rate_integration_uses_interpolated_anchor_boundaries_and_full_path() -> None:
    rate_times_s = array((0.0, 1.0, 2.0), dtype=float64)
    rates_rad_s = array(((0.0, 0.0, 1.0),) * 3, dtype=float64)
    anchor_times_s = array((0.0, 0.5, 2.0), dtype=float64)

    integrated = integrate_rates_between_anchors(
        rates_rad_s,
        rate_times_s,
        anchor_times_s,
        RateFrame.BODY,
        substeps=1,
    )

    rotation_vectors = rotation_rotvecs(integrated.predicted_increments)
    expected_angles = array((0.5, 1.5), dtype=float64)
    assert isclose(integrated.path_angles_rad[0], expected_angles[0], atol=1.0e-14)
    assert isclose(integrated.path_angles_rad[1], expected_angles[1], atol=1.0e-14)
    assert isclose(rotation_vectors[0, 2], expected_angles[0], atol=1.0e-14)
    assert isclose(rotation_vectors[1, 2], expected_angles[1], atol=1.0e-14)
####


@pytest.mark.parametrize("frame", (RateFrame.BODY, RateFrame.WORLD))
def test_sparse_rate_composition_matches_sequential_reference(frame: RateFrame) -> None:
    rate_times_s = array((0.0, 0.23, 0.51, 0.78, 1.0), dtype=float64)
    rates_rad_s = array(
        (
            (1.2, -0.1, 0.3),
            (0.2, 0.8, -0.4),
            (-0.5, 0.4, 1.1),
            (0.7, -0.6, 0.2),
            (0.1, 0.3, -0.9),
        ),
        dtype=float64,
    )
    anchor_times_s = array((0.0, 0.4, 1.0), dtype=float64)
    substeps = 7

    integrated = integrate_rates_between_anchors(
        rates_rad_s,
        rate_times_s,
        anchor_times_s,
        frame,
        substeps,
    )

    reference_times_s = unique(concatenate((rate_times_s, anchor_times_s)))
    reference_rates = column_stack(
        tuple(interp(reference_times_s, rate_times_s, rates_rad_s[:, axis]) for axis in range(3))
    )
    anchor_indices = searchsorted(reference_times_s, anchor_times_s)
    reference_rotations: list[Rotation] = []
    for start, end in pairwise(anchor_indices):
        reference = rotation_from_quat(array((0.0, 0.0, 0.0, 1.0), dtype=float64))
        for interval_index in range(int(start), int(end)):
            interval_s = reference_times_s[interval_index + 1] - reference_times_s[interval_index]
            for substep_index in range(substeps):
                fraction = (substep_index + 0.5) / substeps
                rate = (1.0 - fraction) * reference_rates[
                    interval_index
                ] + fraction * reference_rates[interval_index + 1]
                increment = rotation_from_rotvec(rate * interval_s / substeps)
                reference = (
                    rotation_product(reference, increment)
                    if frame is RateFrame.BODY
                    else rotation_product(increment, reference)
                )
            ####
        ####
        reference_rotations.append(reference)
    ####
    reference_quaternions = stack(tuple(rotation_quaternions(item) for item in reference_rotations))
    predicted_quaternions = rotation_quaternions(integrated.predicted_increments)
    absolute_quaternion_dot = np_abs(np_sum(reference_quaternions * predicted_quaternions, axis=1))
    error_angles = 2.0 * arccos(clip(absolute_quaternion_dot, 0.0, 1.0))
    assert (error_angles < 1.0e-12).all()
####


def test_sparse_orientation_and_dense_rates_recover_known_convention() -> None:
    rate_times_s = arange(0.0, 16.0001, 0.01, dtype=float64)
    anchor_times_s = arange(0.0, 16.0001, 0.2, dtype=float64)
    dense_motion = sampled_rate_motion(rate_times_s, seed=812_901)
    anchor_motion = sampled_rate_motion(anchor_times_s, seed=812_901)
    data = SparseOrientationRateData(
        representation=AttitudeRepresentation.QUATERNION,
        orientation_anchors=anchor_motion.quaternion_xyzw,
        orientation_anchor_times_s=anchor_times_s,
        rates=dense_motion.body_rates_rad_s,
        rate_times_s=rate_times_s,
    )

    report = infer_sparse_orientation_conventions(data)
    convenience_report = infer_conventions(data)

    truth = ConventionHypothesis(
        attitude_direction=AttitudeDirection.BODY_TO_WORLD,
        quaternion_order=QuaternionOrder.SCALAR_LAST,
        rate_frame=RateFrame.BODY,
        rate_order="pqr",
        rate_unit=RateUnit.RADIANS_PER_SECOND,
        rate_sign=1,
    )
    truth_result = next(item for item in report.rankings if item.hypothesis == truth)
    assert report.observation_model == "sparse_orientation_anchors"
    assert report.sample_count == data.anchor_count == anchor_times_s.size
    assert report.rate_sample_count == data.rate_sample_count == rate_times_s.size
    assert truth_result.equivalence_class_id in report.data_winning_class_ids
    assert report.decision.status is InferenceDecisionStatus.RECOMMENDATION
    assert "alternating anchor intervals" in " ".join(report.diagnostics.warnings)
    assert convenience_report == report
####


def test_sparse_euler_anchors_recover_known_convention() -> None:
    rate_times_s = arange(0.0, 16.0001, 0.01, dtype=float64)
    anchor_times_s = arange(0.0, 16.0001, 0.2, dtype=float64)
    dense_motion = sampled_rate_motion(rate_times_s, seed=812_903)
    anchor_motion = sampled_rate_motion(anchor_times_s, seed=812_903)
    euler_anchors: FloatArray = asarray(
        cast(Any, Rotation).from_quat(anchor_motion.quaternion_xyzw).as_euler("XYZ"),
        dtype=float64,
    )
    data = SparseOrientationRateData(
        representation=AttitudeRepresentation.EULER,
        orientation_anchors=euler_anchors,
        orientation_anchor_times_s=anchor_times_s,
        rates=dense_motion.body_rates_rad_s,
        rate_times_s=rate_times_s,
    )
    config = InferenceConfig(
        search_space=SearchSpace(
            euler_orders=("xyz",),
            euler_modes=(EulerMode.INTRINSIC, EulerMode.EXTRINSIC),
            angle_units=(AngleUnit.RADIANS,),
            rate_orders=("pqr",),
            rate_units=(RateUnit.RADIANS_PER_SECOND,),
            rate_signs=(-1, 1),
        )
    )

    report = infer_sparse_orientation_conventions(data, config)

    truth = ConventionHypothesis(
        attitude_direction=AttitudeDirection.BODY_TO_WORLD,
        euler_order="xyz",
        euler_mode=EulerMode.INTRINSIC,
        angle_unit=AngleUnit.RADIANS,
        rate_frame=RateFrame.BODY,
        rate_order="pqr",
        rate_unit=RateUnit.RADIANS_PER_SECOND,
        rate_sign=1,
    )
    truth_result = next(item for item in report.rankings if item.hypothesis == truth)
    assert truth_result.equivalence_class_id in report.data_winning_class_ids
    assert report.decision.status is InferenceDecisionStatus.RECOMMENDATION
####


def test_three_sparse_anchors_produce_fit_but_withhold_sequential_evidence() -> None:
    rate_times_s = arange(0.0, 1.0001, 0.01, dtype=float64)
    anchor_times_s = array((0.0, 0.5, 1.0), dtype=float64)
    dense_motion = sampled_rate_motion(rate_times_s, seed=812_904)
    anchor_motion = sampled_rate_motion(anchor_times_s, seed=812_904)
    data = SparseOrientationRateData(
        representation=AttitudeRepresentation.QUATERNION,
        orientation_anchors=anchor_motion.quaternion_xyzw,
        orientation_anchor_times_s=anchor_times_s,
        rates=dense_motion.body_rates_rad_s,
        rate_times_s=rate_times_s,
    )

    report = infer_sparse_orientation_conventions(data)

    assert report.sample_count == 3
    assert report.rate_sample_count == rate_times_s.size
    assert report.rankings
    assert report.diagnostics.evidence_sufficient is False
    assert report.decision.status is InferenceDecisionStatus.ABSTENTION
    assert report.decision.reason is InferenceDecisionReason.EVIDENCE_INSUFFICIENT
####


def test_sparse_input_rejects_missing_rate_coverage_and_owns_streams() -> None:
    rate_times_s = arange(0.0, 1.01, 0.1, dtype=float64)
    rates = array(((0.0, 0.0, 0.2),) * rate_times_s.size, dtype=float64)
    anchor_times_s = array((0.0, 0.5, 1.0), dtype=float64)
    anchor_motion = sampled_rate_motion(anchor_times_s, seed=812_902)
    orientation_anchors = anchor_motion.quaternion_xyzw.copy()
    data = SparseOrientationRateData(
        representation=AttitudeRepresentation.QUATERNION,
        orientation_anchors=orientation_anchors,
        orientation_anchor_times_s=anchor_times_s,
        rates=rates,
        rate_times_s=rate_times_s,
    )
    orientation_anchors[0, 0] = 0.5
    rates[0, 0] = 1.0

    assert data.orientation_anchors.flags.writeable is False
    assert data.rates.flags.writeable is False
    assert not isclose(data.orientation_anchors[0, 0], orientation_anchors[0, 0])
    assert not isclose(data.rates[0, 0], rates[0, 0])

    with pytest.raises(ValueError, match="must cover all orientation anchor times"):
        SparseOrientationRateData(
            representation=AttitudeRepresentation.QUATERNION,
            orientation_anchors=anchor_motion.quaternion_xyzw,
            orientation_anchor_times_s=array((0.0, 0.5, 1.1), dtype=float64),
            rates=rates,
            rate_times_s=rate_times_s,
        )
    ####
####
