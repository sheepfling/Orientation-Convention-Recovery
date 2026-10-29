"""Measure full inference latency and traced peak allocation on analytic trajectories.

The measurements are diagnostic, not CI pass/fail thresholds. Run the same cases on
the same interpreter and machine when comparing revisions.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import tracemalloc
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median
from time import perf_counter

from numpy import __version__ as numpy_version
from numpy import arange, float64

from attitude_convention_inference import (
    AttitudeDirection,
    AttitudeRepresentation,
    InferenceConfig,
    QuaternionOrder,
    RateFrame,
    SearchSpace,
    TrajectoryData,
    __version__,
    infer_conventions,
)

if __package__:
    from scripts.generate_validation_corpus import analytic_motion
else:
    from generate_validation_corpus import analytic_motion
####


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    samples: int
    search: str
    hypotheses: int
    median_seconds: float
    traced_peak_bytes: int
####


def _config(search: str) -> InferenceConfig:
    if search == "default":
        return InferenceConfig()
    ####
    if search != "compact":
        raise ValueError(f"Unknown search space: {search}")
    ####
    return InferenceConfig(
        search_space=replace(
            SearchSpace(),
            attitude_directions=(AttitudeDirection.BODY_TO_WORLD,),
            quaternion_orders=(QuaternionOrder.SCALAR_LAST,),
            rate_frames=(RateFrame.BODY,),
            rate_orders=("pqr", "rqp"),
        )
    )
####


def benchmark_case(samples: int, search: str, repeats: int) -> BenchmarkResult:
    if samples < 3 or repeats < 1:
        raise ValueError("samples must be at least 3 and repeats must be positive")
    ####
    motion = analytic_motion(arange(samples, dtype=float64) * 0.025)
    data = TrajectoryData(
        representation=AttitudeRepresentation.QUATERNION,
        orientation=motion.quaternion_xyzw,
        rates=motion.body_rates_rad_s,
        time_s=motion.time_s,
    )
    config = _config(search)

    timings: list[float] = []
    for _ in range(repeats):
        gc.collect()
        start = perf_counter()
        report = infer_conventions(data, config)
        timings.append(perf_counter() - start)
    ####

    gc.collect()
    tracemalloc.start()
    try:
        report = infer_conventions(data, config)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    ####
    return BenchmarkResult(
        samples=samples,
        search=search,
        hypotheses=report.hypothesis_count,
        median_seconds=median(timings),
        traced_peak_bytes=peak,
    )
####


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", type=int, default=[101, 501])
    parser.add_argument(
        "--searches", nargs="+", choices=("compact", "default"), default=["compact", "default"]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, help="write machine-readable results as JSON")
    args = parser.parse_args()

    results = [
        benchmark_case(samples, search, args.repeats)
        for samples in args.samples
        for search in args.searches
    ]
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": numpy_version,
        "package_version": __version__,
        "method": "perf_counter median; tracemalloc peak in a separate inference run",
        "results": [asdict(result) for result in results],
    }
    output = json.dumps(payload, indent=2) + "\n"
    if args.output is None:
        print(output, end="")
    else:
        args.output.write_text(output, encoding="utf-8")
        print(args.output)
    ####
####


if __name__ == "__main__":
    main()
####
