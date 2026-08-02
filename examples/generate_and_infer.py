from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

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
    time_s = np.arange(0.0, 4.0, 0.025)
    rates = rich_three_axis_rates(time_s)
    convention = SyntheticConvention(
        representation=AttitudeRepresentation.QUATERNION,
        attitude_direction=AttitudeDirection.WORLD_TO_BODY,
        quaternion_order=QuaternionOrder.SCALAR_FIRST,
        rate_frame=RateFrame.WORLD,
        rate_order="rqp",
    )
    data = encode_synthetic_trajectory(time_s, rates, convention)
    report = infer_conventions(data)

    print(report.best.hypothesis.model_dump_json(indent=2))
    print(report.diagnostics.model_dump_json(indent=2))

    output = Path(__file__).with_name("synthetic_quaternion.csv")
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time", "q0", "q1", "q2", "q3", "w0", "w1", "w2"))
        for index in range(data.sample_count):
            writer.writerow(
                (
                    data.time_s[index],
                    *data.orientation[index],
                    *data.rates[index],
                )
            )
    print(f"Wrote {output}")


####


if __name__ == "__main__":
    main()
####
