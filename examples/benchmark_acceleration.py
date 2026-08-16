from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from numpy import amax, arange, asarray, empty, float64, int64, median, random, stack
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from attitude_convention_inference.core import FloatArray, compose_increment_windows
from attitude_convention_inference.models import RateFrame


@dataclass(frozen=True, slots=True)
class _DirectWindowComposition:
    window_increments: Rotation
    starts: NDArray[int64]
####




def _rotation_from_quat(quaternions: FloatArray) -> Rotation:
    return cast(Rotation, cast(Any, Rotation).from_quat(quaternions))
####




def _rotation_from_rotvec(rotvecs: FloatArray) -> Rotation:
    return cast(Rotation, cast(Any, Rotation).from_rotvec(rotvecs))
####




def _rotation_magnitudes(rotation: Rotation) -> FloatArray:
    return asarray(cast(Any, rotation).magnitude(), dtype=float64)
####




def _direct_windows(
    increments: Rotation,
    window_size: int,
    frame: RateFrame,
) -> _DirectWindowComposition:
    interval_count = len(increments)
    starts = arange(0, interval_count - window_size + 1, dtype=int64)
    outputs: list[FloatArray] = []
    for start in starts:
        total = Rotation.identity()
        for offset in range(window_size):
            increment = increments[int(start + offset)]
            total = total * increment if frame is RateFrame.BODY else increment * total
        ####
        outputs.append(asarray(cast(Any, total).as_quat(), dtype=float64))
    ####
    if not outputs:
        return _DirectWindowComposition(_rotation_from_quat(empty((0, 4))), starts)
    ####
    return _DirectWindowComposition(_rotation_from_quat(stack(outputs)), starts)
####




def _elapsed(function: Callable[[], object], repeats: int) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    ####
    return float(median(samples))
####




def _benchmark_case(
    sample_count: int,
    window_size: int,
    repeats: int,
    rng: random.Generator,
) -> dict[str, float | int | str]:
    increments = _rotation_from_rotvec(
        asarray(rng.normal(0.0, 0.02, (sample_count - 1, 3)), dtype=float64)
    )
    frame = RateFrame.BODY
    expected = _direct_windows(increments, window_size, frame)
    accelerated = compose_increment_windows(increments, window_size, frame)
    mismatch = _rotation_magnitudes(
        expected.window_increments.inv() * accelerated.window_increments
    )
    if mismatch.size and float(amax(mismatch)) > 1.0e-12:
        raise AssertionError("Prefix-product windows disagree with the direct reference.")
    ####

    direct_seconds = _elapsed(lambda: _direct_windows(increments, window_size, frame), repeats)
    accelerated_seconds = _elapsed(
        lambda: compose_increment_windows(increments, window_size, frame), repeats
    )
    return {
        "samples": sample_count,
        "window_size": window_size,
        "direct_seconds": direct_seconds,
        "prefix_seconds": accelerated_seconds,
        "speedup": direct_seconds / accelerated_seconds
        if accelerated_seconds > 0.0
        else float("inf"),
    }
####




def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark direct versus prefix-product rotation windows."
    )
    parser.add_argument("--samples", nargs="+", type=int, default=(1000, 5000, 20000))
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260814)
    arguments = parser.parse_args(argv)
    if arguments.window_size < 1 or arguments.repeats < 1:
        parser.error("--window-size and --repeats must be positive.")
    ####
    if any(sample_count < 3 for sample_count in arguments.samples):
        parser.error("Every --samples value must be at least 3.")
    ####

    rng = random.default_rng(arguments.seed)
    results = [
        _benchmark_case(sample_count, arguments.window_size, arguments.repeats, rng)
        for sample_count in arguments.samples
    ]
    print(json.dumps({"seed": arguments.seed, "results": results}, indent=2))
    return 0
####




if __name__ == "__main__":
    raise SystemExit(main())
####
