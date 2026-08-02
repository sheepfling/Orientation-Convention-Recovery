"""Run reproducible adversarial trajectory scenarios.

This is intentionally a manual harness: it prints one compact JSON record per
scenario and exits nonzero if a scenario selects the wrong known convention.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np
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
from attitude_convention_inference.synthetic import (
    SyntheticConvention,
    encode_synthetic_trajectory,
    rich_three_axis_rates,
)


TRUTH = SyntheticConvention(
    representation=AttitudeRepresentation.QUATERNION,
    attitude_direction=AttitudeDirection.BODY_TO_WORLD,
    quaternion_order=QuaternionOrder.SCALAR_LAST,
    rate_frame=RateFrame.BODY,
    rate_order="rqp",
)


def _config(trim_fraction: float = 0.02) -> InferenceConfig:
    return InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=(TRUTH.attitude_direction,),
            quaternion_orders=tuple(QuaternionOrder),
            rate_frames=(TRUTH.rate_frame,),
            rate_orders=(TRUTH.rate_order,),
            rate_units=(RateUnit.RADIANS_PER_SECOND,),
            rate_signs=(1,),
        ),
        trim_fraction=trim_fraction,
    )


def _clean(seed: int, count: int) -> TrajectoryData:
    rng = np.random.default_rng(seed)
    time_s = np.arange(count, dtype=np.float64) * 0.02
    rates = rich_three_axis_rates(time_s, scale=0.8)
    data = encode_synthetic_trajectory(time_s, rates, TRUTH, quaternion_sign_flips=True)
    attitudes = Rotation.from_quat(data.orientation)
    attitudes = Rotation.from_rotvec(rng.normal(0.0, 0.0015, (count, 3))) * attitudes
    return replace(
        data,
        orientation=attitudes.as_quat(),
        rates=data.rates + rng.normal(0.0, 0.012, data.rates.shape),
    )


def _scenarios(seed: int) -> dict[str, tuple[TrajectoryData, InferenceConfig]]:
    noisy = _clean(seed, 241)
    raw = Rotation.from_quat(noisy.orientation)
    burst = np.zeros((noisy.sample_count, 3), dtype=np.float64)
    burst[90:102] = [0.10, -0.06, 0.08]
    burst_data = replace(noisy, orientation=(Rotation.from_rotvec(burst) * raw).as_quat())

    rng = np.random.default_rng(seed + 1)
    time_s = np.arange(241, dtype=np.float64) * 0.018
    time_s += np.r_[0.0, np.cumsum(rng.uniform(-0.002, 0.002, 240))]
    time_s = np.maximum.accumulate(time_s + np.arange(241) * 1.0e-5)
    irregular = encode_synthetic_trajectory(time_s, rich_three_axis_rates(time_s), TRUTH)
    keep = np.ones(irregular.sample_count, dtype=bool)
    keep[[20, 21, 22, 85, 170]] = False
    dropped = TrajectoryData(
        representation=irregular.representation,
        orientation=irregular.orientation[keep],
        rates=irregular.rates[keep],
        time_s=irregular.time_s[keep],
    )
    return {
        "noise_sign_flips": (noisy, _config()),
        "burst_outliers": (burst_data, _config(0.08)),
        "jitter_and_drops": (dropped, _config()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()

    failures = 0
    for name, (data, config) in _scenarios(args.seed).items():
        report = infer_conventions(data, config)
        selected = report.best.hypothesis.quaternion_order
        ok = selected is TRUTH.quaternion_order
        failures += not ok
        print(
            json.dumps(
                {
                    "scenario": name,
                    "ok": ok,
                    "samples": data.sample_count,
                    "selected_quaternion_order": selected.value,
                    "interval_rmse_rad": report.best.interval_rmse_rad,
                    "confidence": report.diagnostics.confidence,
                    "warnings": list(report.diagnostics.warnings),
                }
            )
        )
    return int(failures)


if __name__ == "__main__":
    raise SystemExit(main())
