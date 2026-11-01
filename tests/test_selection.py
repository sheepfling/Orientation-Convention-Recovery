from __future__ import annotations

from math import exp

from numpy import arange, float64, zeros
from scipy.spatial.transform import Rotation

from attitude_convention_inference import (
    AttitudeDirection,
    AttitudeRepresentation,
    ConventionConstraints,
    InferenceConfig,
    QuaternionOrder,
    RateFrame,
    RateUnit,
    infer_conventions,
)
from attitude_convention_inference.scoring import ScoredCandidate
from attitude_convention_inference.selection import _evidence_contender_classes
from attitude_convention_inference.synthetic import (
    SyntheticConvention,
    encode_synthetic_trajectory,
)


def test_contender_margin_is_log_geometric_risk_ratio() -> None:
    time_s = arange(81, dtype=float64) * 0.025
    rates = zeros((time_s.size, 3), dtype=float64)
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
    report = infer_conventions(
        data,
        InferenceConfig(
            constraints=ConventionConstraints(
                attitude_direction=AttitudeDirection.BODY_TO_WORLD,
                quaternion_order=QuaternionOrder.SCALAR_LAST,
                rate_order="pqr",
                rate_unit=RateUnit.RADIANS_PER_SECOND,
                rate_sign=1,
            )
        ),
    )
    class_results = {result.equivalence_class_id: result for result in report.rankings}
    assert len(class_results) == 2
    results = tuple(class_results.values())
    controlled = (
        results[0].model_copy(update={"class_data_risk": 1.0}),
        results[1].model_copy(update={"class_data_risk": exp(2.0)}),
    )
    ranked = tuple(
        ScoredCandidate(result=result, interval_squared_errors_rad2=zeros(1, dtype=float64))
        for result in controlled
    )

    included_at_default_margin = _evidence_contender_classes(
        ranked,
        InferenceConfig(near_tie_delta_criterion=2.0),
    )
    excluded_below_margin = _evidence_contender_classes(
        ranked,
        InferenceConfig(near_tie_delta_criterion=2.0 - 1.0e-6),
    )

    assert len(included_at_default_margin) == 2
    assert excluded_below_margin == (controlled[0].equivalence_class_id,)
####
