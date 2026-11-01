from __future__ import annotations

from pathlib import Path

from attitude_convention_inference import infer_csv, inspect_csv

SOURCE = Path(__file__).with_name("synthetic_quaternion.csv")

layout = inspect_csv(
    SOURCE,
    representation="quaternion",
    orientation_columns=("q0", "q1", "q2", "q3"),
    rate_columns=("w0", "w1", "w2"),
    time_column="time",
)
print(layout)

report = infer_csv(
    SOURCE,
    representation="quaternion",
    orientation_columns=("q0", "q1", "q2", "q3"),
    rate_columns=("w0", "w1", "w2"),
    time_column="time",
    constraints={
        "attitude_direction": "world_to_body",
        "quaternion_order": "scalar_first",
        "rate_frame": "world",
        "rate_order": "rqp",
    },
)
print(report.best.hypothesis.model_dump_json(indent=2))
print(report.diagnostics.model_dump_json(indent=2))
