"""Run reproducible adversarial trajectory scenarios.

This is intentionally a manual harness: it prints one compact JSON record per
scenario and exits nonzero if a scenario selects the wrong known convention.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from numpy import (
    arange,
    asarray,
    cumsum,
    diff,
    float32,
    float64,
    linspace,
    maximum,
    ones,
    r_,
    random,
    tile,
    vstack,
    zeros,
)
from numpy import (
    isfinite as np_isfinite,
)
from numpy.typing import NDArray
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


@dataclass(frozen=True, slots=True)
class StressScenario:
    data: TrajectoryData
    config: InferenceConfig
    expectation: Literal["recover", "abstain", "warn", "degrade"] = "recover"
####


def _quaternions(rotation: Rotation) -> NDArray[float64]:
    return asarray(cast(Any, rotation).as_quat(), dtype=float64)
####


def _rotation_from_quat(quaternions: NDArray[float64]) -> Rotation:
    return cast(Rotation, cast(Any, Rotation).from_quat(quaternions))
####


def _rotation_from_rotvec(rotvecs: NDArray[float64]) -> Rotation:
    return cast(Rotation, cast(Any, Rotation).from_rotvec(rotvecs))
####


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
####


def _clean(seed: int, count: int) -> TrajectoryData:
    rng = random.default_rng(seed)
    time_s = arange(count, dtype=float64) * 0.02
    rates = rich_three_axis_rates(time_s, scale=0.8)
    data = encode_synthetic_trajectory(time_s, rates, TRUTH, quaternion_sign_flips=True)
    if data.orientation is None or data.rates is None:
        raise AssertionError("synthetic trajectory must contain attitudes and rates")
    ####
    attitudes = _rotation_from_quat(data.orientation)
    attitudes = (
        _rotation_from_rotvec(asarray(rng.normal(0.0, 0.0015, (count, 3)), dtype=float64))
        * attitudes
    )
    return replace(
        data,
        orientation=_quaternions(attitudes),
        rates=data.rates + rng.normal(0.0, 0.012, data.rates.shape),
    )
####


def _scenarios(seed: int) -> dict[str, StressScenario]:
    noisy = _clean(seed, 241)
    if noisy.orientation is None or noisy.rates is None or noisy.time_s is None:
        raise AssertionError("noisy trajectory must contain all streams")
    ####
    raw = _rotation_from_quat(noisy.orientation)
    burst = zeros((noisy.sample_count, 3), dtype=float64)
    burst[90:102] = [0.10, -0.06, 0.08]
    burst_data = replace(
        noisy,
        orientation=_quaternions(_rotation_from_rotvec(burst) * raw),
    )

    rng = random.default_rng(seed + 1)
    time_s = arange(241, dtype=float64) * 0.018
    time_s += r_[0.0, cumsum(rng.uniform(-0.002, 0.002, 240))]
    time_s = maximum.accumulate(time_s + arange(241) * 1.0e-5)
    irregular = encode_synthetic_trajectory(time_s, rich_three_axis_rates(time_s), TRUTH)
    if irregular.orientation is None or irregular.rates is None or irregular.time_s is None:
        raise AssertionError("irregular trajectory must contain all streams")
    ####
    keep = ones(irregular.sample_count, dtype=bool)
    keep[[20, 21, 22, 85, 170]] = False
    dropped = TrajectoryData(
        representation=irregular.representation,
        orientation=irregular.orientation[keep],
        rates=irregular.rates[keep],
        time_s=irregular.time_s[keep],
    )
    time_s = arange(121, dtype=float64) * 0.02
    zero_motion = encode_synthetic_trajectory(
        time_s,
        zeros((time_s.size, 3), dtype=float64),
        TRUTH,
    )
    alias_rates = tile(asarray((200.0, 0.0, 0.0)), (time_s.size, 1))
    hard_alias = encode_synthetic_trajectory(time_s, alias_rates, TRUTH)
    float32_data = TrajectoryData(
        representation=noisy.representation,
        orientation=cast(Any, noisy.orientation.astype(float32)),
        rates=cast(Any, noisy.rates.astype(float32)),
        time_s=cast(Any, noisy.time_s.astype(float32)),
    )
    lagged = replace(noisy, rates=vstack((noisy.rates[0], noisy.rates[:-1])))
    nominal_time = arange(noisy.sample_count, dtype=float64) * 0.02
    drifted_time = r_[
        0.0,
        cumsum(diff(nominal_time) * (1.0 + 0.05 * linspace(-1.0, 1.0, noisy.sample_count - 1))),
    ]
    drifted = replace(noisy, time_s=drifted_time)
    return {
        "noise_sign_flips": StressScenario(noisy, _config()),
        "burst_outliers": StressScenario(burst_data, _config(0.08)),
        "jitter_and_drops": StressScenario(dropped, _config()),
        "zero_motion_abstention": StressScenario(zero_motion, _config(), "abstain"),
        "hard_alias_warning": StressScenario(hard_alias, _config(), "warn"),
        "float32_stability": StressScenario(float32_data, _config()),
        "one_sample_latency": StressScenario(lagged, _config(), "degrade"),
        "timestamp_drift": StressScenario(drifted, _config(), "degrade"),
    }
####


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()

    failures = 0
    for name, scenario in _scenarios(args.seed).items():
        report = infer_conventions(scenario.data, scenario.config)
        selected = report.best.hypothesis.quaternion_order
        if selected is None:
            raise AssertionError("stress scenario should select a quaternion order")
        ####
        selected_value = cast(str, getattr(cast(Any, selected), "value", selected))
        if scenario.expectation == "recover":
            ok = selected is TRUTH.quaternion_order
        elif scenario.expectation == "abstain":
            ok = (
                len(report.evidence_contender_class_ids) > 1 and report.diagnostics.confidence < 0.5
            )
        elif scenario.expectation == "warn":
            ok = report.best.hard_alias_fraction > 0.0 and any(
                "pi" in warning or "wrap" in warning for warning in report.diagnostics.warnings
            )
        else:
            ok = (
                np_isfinite(report.best.interval_rmse_rad)
                and report.best.interval_rmse_rad > 1.0e-4
            )
        ####
        failures += not ok
        print(
            json.dumps(
                {
                    "scenario": name,
                    "ok": ok,
                    "expectation": scenario.expectation,
                    "samples": scenario.data.sample_count,
                    "selected_quaternion_order": selected_value,
                    "evidence_contenders": list(report.evidence_contender_class_ids),
                    "interval_rmse_rad": report.best.interval_rmse_rad,
                    "hard_alias_fraction": report.best.hard_alias_fraction,
                    "confidence": report.diagnostics.confidence,
                    "warnings": list(report.diagnostics.warnings),
                }
            )
        )
    ####
    return int(failures)
####


if __name__ == "__main__":
    raise SystemExit(main())
####
