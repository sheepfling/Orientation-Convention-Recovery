"""Measure full inference latency and memory on analytic trajectories.

The measurements are diagnostic, not CI pass/fail thresholds. Run the same cases on
the same interpreter and machine when comparing revisions.
Use --process to include first-call latency and peak RSS from a fresh subprocess.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import tracemalloc
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median
from time import perf_counter

from numpy import __version__ as numpy_version
from numpy import arange, float64
from scipy import __version__ as scipy_version

from attitude_convention_inference import (
    AngleUnit,
    AttitudeDirection,
    AttitudeRepresentation,
    EulerMode,
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
class BenchmarkCase:
    """Validated trajectory and search configuration for one benchmark case."""

    data: TrajectoryData
    config: InferenceConfig
####


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """In-process measurements with optional fresh-process measurements."""

    samples: int
    search: str
    representation: str
    hypotheses: int
    median_seconds: float
    traced_peak_bytes: int
    cold_inference_seconds: float | None = None
    cold_process_seconds: float | None = None
    peak_process_rss_bytes: int | None = None
####


@dataclass(frozen=True, slots=True)
class ColdProcessResult:
    """First-call measurements; process time is added by the parent runner."""

    hypotheses: int
    inference_seconds: float
    peak_rss_bytes: int | None
    process_seconds: float | None = None
####


def _config(search: str) -> InferenceConfig:
    if search in ("default", "euler-default"):
        return InferenceConfig()
    ####
    if search == "euler-compact":
        return InferenceConfig(
            search_space=replace(
                SearchSpace(),
                attitude_directions=(AttitudeDirection.BODY_TO_WORLD,),
                euler_orders=("xyz",),
                euler_modes=(EulerMode.EXTRINSIC,),
                angle_units=(AngleUnit.RADIANS,),
                rate_frames=(RateFrame.BODY,),
                rate_orders=("pqr", "rqp"),
            )
        )
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


def _case_data(samples: int, search: str) -> BenchmarkCase:
    if samples < 3:
        raise ValueError("samples must be at least 3")
    ####
    motion = analytic_motion(arange(samples, dtype=float64) * 0.025)
    config = _config(search)
    representation = (
        AttitudeRepresentation.EULER
        if search.startswith("euler-")
        else AttitudeRepresentation.QUATERNION
    )
    data = TrajectoryData(
        representation=representation,
        orientation=(
            motion.euler_xyz_rad
            if representation is AttitudeRepresentation.EULER
            else motion.quaternion_xyzw
        ),
        rates=motion.body_rates_rad_s,
        time_s=motion.time_s,
    )
    return BenchmarkCase(data=data, config=config)
####


def benchmark_case(samples: int, search: str, repeats: int) -> BenchmarkResult:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    ####
    case = _case_data(samples, search)

    timings: list[float] = []
    for _ in range(repeats):
        gc.collect()
        start = perf_counter()
        report = infer_conventions(case.data, case.config)
        timings.append(perf_counter() - start)
    ####

    gc.collect()
    tracemalloc.start()
    try:
        report = infer_conventions(case.data, case.config)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    ####
    return BenchmarkResult(
        samples=samples,
        search=search,
        representation=case.data.representation.value,
        hypotheses=report.hypothesis_count,
        median_seconds=median(timings),
        traced_peak_bytes=peak,
    )
####


def _peak_process_rss_bytes() -> int | None:
    """Read the process lifetime peak working set/RSS on supported platforms."""

    if sys.platform == "win32":
        from psutil import Process

        return int(Process().memory_info().peak_wset)
    ####
    if sys.platform == "linux":
        from resource import RUSAGE_SELF, getrusage

        return int(getrusage(RUSAGE_SELF).ru_maxrss) * 1024
    ####
    return None
####


def _cold_process_case(samples: int, search: str) -> ColdProcessResult:
    case = _case_data(samples, search)
    start = perf_counter()
    report = infer_conventions(case.data, case.config)
    inference_seconds = perf_counter() - start
    return ColdProcessResult(
        hypotheses=report.hypothesis_count,
        inference_seconds=inference_seconds,
        peak_rss_bytes=_peak_process_rss_bytes(),
    )
####


def benchmark_cold_process(samples: int, search: str) -> ColdProcessResult:
    """Measure a fresh process, including interpreter startup and data creation."""

    start = perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.benchmark_inference",
            "--cold-worker",
            str(samples),
            search,
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    process_seconds = perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(f"Cold benchmark worker failed: {completed.stderr}")
    ####
    payload = json.loads(completed.stdout)
    cold = ColdProcessResult(
        hypotheses=int(payload["hypotheses"]),
        inference_seconds=float(payload["inference_seconds"]),
        peak_rss_bytes=(
            None if payload["peak_rss_bytes"] is None else int(payload["peak_rss_bytes"])
        ),
    )
    return replace(cold, process_seconds=process_seconds)
####


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", type=int, default=[101, 501])
    parser.add_argument(
        "--searches",
        nargs="+",
        choices=("compact", "default", "euler-compact", "euler-default"),
        default=["compact", "default"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--process",
        action="store_true",
        help="also measure cold inference and peak process RSS in a fresh subprocess per case",
    )
    parser.add_argument("--cold-worker", nargs=2, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help="write machine-readable results as JSON")
    args = parser.parse_args()

    if args.cold_worker is not None:
        samples_text, search = args.cold_worker
        print(json.dumps(asdict(_cold_process_case(int(samples_text), search))))
        return
    ####

    results: list[BenchmarkResult] = []
    for samples in args.samples:
        for search in args.searches:
            result = benchmark_case(samples, search, args.repeats)
            if args.process:
                cold = benchmark_cold_process(samples, search)
                if cold.hypotheses != result.hypotheses:
                    raise RuntimeError("Cold process and in-process hypothesis counts differ")
                ####
                result = replace(
                    result,
                    cold_inference_seconds=cold.inference_seconds,
                    cold_process_seconds=cold.process_seconds,
                    peak_process_rss_bytes=cold.peak_rss_bytes,
                )
            ####
            results.append(result)
        ####
    ####
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": numpy_version,
        "scipy": scipy_version,
        "package_version": __version__,
        "method": (
            "perf_counter median; tracemalloc peak in a separate inference run; "
            "optional fresh-process cold inference, end-to-end wall time, and peak RSS"
        ),
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
