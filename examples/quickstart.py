"""Minimal library usage example.

Run from a checkout with ``PYTHONPATH=src python examples/quickstart.py`` or run
it after installing the package.
"""

from __future__ import annotations

from numpy import arange

from attitude_convention_inference import (
    AttitudeDirection,
    AttitudeRepresentation,
    QuaternionOrder,
    RateFrame,
    infer_conventions,
)
from attitude_convention_inference.synthetic import (
    SyntheticConvention,
    encode_synthetic_trajectory,
    rich_three_axis_rates,
)


def main() -> None:
    # In an application, replace these generated arrays with the measured data.
    time_s = arange(0.0, 4.0, 0.025)
    angular_rates = rich_three_axis_rates(time_s)
    data = encode_synthetic_trajectory(
        time_s,
        angular_rates,
        SyntheticConvention(
            representation=AttitudeRepresentation.QUATERNION,
            attitude_direction=AttitudeDirection.WORLD_TO_BODY,
            quaternion_order=QuaternionOrder.SCALAR_FIRST,
            rate_frame=RateFrame.WORLD,
            rate_order="rqp",
        ),
    )

    report = infer_conventions(
        data.orientation,
        data.rates,
        representation="quaternion",
        time_s=data.time_s,
    )

    print("Best convention:")
    print(report.best.hypothesis.model_dump_json(indent=2))
    print(f"Confidence: {report.diagnostics.confidence:.3f}")
    print(
        "Noise sigma (rad/axis): "
        f"{report.diagnostics.estimated_noise_sigma_rad:.6g}; "
        "evidence intervals used: "
        f"{report.diagnostics.evidence_intervals_used}"
    )
####




if __name__ == "__main__":
    main()
####
