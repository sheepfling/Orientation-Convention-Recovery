"""Measure full inference latency and memory on synthetic trajectories.

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
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from math import isfinite
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch

from numpy import __version__ as numpy_version
from numpy import arange, float64, int64
from scipy import __version__ as scipy_version
from scipy.spatial.transform import Rotation

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
from attitude_convention_inference.core import (
    ObservedIncrementCache,
    WindowComposition,
    observed_interval_increments,
    observed_window_increments,
)
from attitude_convention_inference.heuristics import (
    EulerHeuristicDiagnostics,
    EulerHeuristicDiagnosticsCache,
    euler_heuristic_diagnostics,
)
from attitude_convention_inference.models import ConventionHypothesis, InferenceReport

if __package__:
    from scripts.generate_validation_corpus import (
        DEFAULT_MOTION_SEED,
        MotionProfile,
        analytic_motion,
        maneuver_burst_rate_motion,
        piecewise_linear_rate_motion,
        sampled_rate_motion,
    )
else:
    from generate_validation_corpus import (
        DEFAULT_MOTION_SEED,
        MotionProfile,
        analytic_motion,
        maneuver_burst_rate_motion,
        piecewise_linear_rate_motion,
        sampled_rate_motion,
    )
####


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """Validated trajectory and search configuration for one benchmark case."""

    data: TrajectoryData
    config: InferenceConfig
    sample_period_s: float
####


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """In-process measurements with optional fresh-process measurements."""

    samples: int
    sample_period_s: float
    search: str
    representation: str
    motion_profile: str
    motion_seed: int
    timestamps_available: bool
    euler_heuristic_cache_enabled: bool
    observed_increment_cache_enabled: bool
    hypotheses: int
    timing_sample_count: int
    minimum_seconds: float
    median_seconds: float
    maximum_seconds: float
    traced_peak_bytes: int
    cold_inference_seconds: float | None = None
    cold_process_seconds: float | None = None
    peak_process_rss_bytes: int | None = None
####


@dataclass(frozen=True, slots=True)
class EulerCacheComparison:
    """Paired measurements for the uncached and cached Euler heuristic paths."""

    uncached: BenchmarkResult
    cached: BenchmarkResult
    median_speedup_factor: float
####


@dataclass(frozen=True, slots=True)
class ObservedIncrementCacheComparison:
    """Paired measurements for cached/uncached observed rotations."""

    uncached: BenchmarkResult
    cached: BenchmarkResult
    median_speedup_factor: float
####


@dataclass(frozen=True, slots=True)
class ColdProcessResult:
    """First-call measurements; process time is added by the parent runner."""

    motion_profile: str
    motion_seed: int
    sample_period_s: float
    euler_heuristic_cache_enabled: bool
    observed_increment_cache_enabled: bool
    timestamps_available: bool
    hypotheses: int
    report_fingerprint: str
    inference_seconds: float
    peak_rss_bytes: int | None
    process_seconds: float | None = None
####


@dataclass(frozen=True, slots=True)
class ObservedIncrementCacheProcessComparison:
    """Paired fresh-process timings, RSS, and report fingerprints."""

    samples: int
    search: str
    representation: str
    motion_profile: str
    motion_seed: int
    sample_period_s: float
    timestamps_available: bool
    hypotheses: int
    repeats: int
    uncached_runs: tuple[ColdProcessResult, ...]
    cached_runs: tuple[ColdProcessResult, ...]
    median_speedup_factor: float
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


def _positive_finite_float(value: str) -> float:
    """Parse a finite positive float for command-line benchmark inputs."""

    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    ####
    if not isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    ####
    return parsed
####


def _case_data(
    samples: int,
    search: str,
    motion_profile: MotionProfile,
    motion_seed: int,
    include_timestamps: bool = True,
    sample_period_s: float = 0.025,
) -> BenchmarkCase:
    if samples < 3:
        raise ValueError("samples must be at least 3")
    ####
    if not isfinite(sample_period_s) or sample_period_s <= 0.0:
        raise ValueError("sample_period_s must be finite and positive")
    ####
    time_s = arange(samples, dtype=float64) * sample_period_s
    if motion_profile is MotionProfile.SAMPLED_RATE_NONANALYTIC:
        motion = sampled_rate_motion(time_s, seed=motion_seed)
    elif motion_profile is MotionProfile.SAMPLED_RATE_PIECEWISE_LINEAR:
        motion = piecewise_linear_rate_motion(time_s, seed=motion_seed)
    elif motion_profile is MotionProfile.SAMPLED_RATE_MANEUVER_BURSTS:
        motion = maneuver_burst_rate_motion(time_s, seed=motion_seed)
    else:
        motion = analytic_motion(time_s, motion_profile)
    ####
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
        time_s=motion.time_s if include_timestamps else None,
    )
    return BenchmarkCase(data=data, config=config, sample_period_s=sample_period_s)
####


def _uncached_euler_diagnostics(
    cache: EulerHeuristicDiagnosticsCache,
    hypothesis: ConventionHypothesis,
) -> EulerHeuristicDiagnostics:
    """Reference calculation for comparing the Euler heuristic cache."""

    return euler_heuristic_diagnostics(cache.data, hypothesis)
####


@contextmanager
def _euler_cache_mode(enabled: bool) -> Generator[None, None, None]:
    if enabled:
        yield
        return
    ####
    with patch.object(
        EulerHeuristicDiagnosticsCache,
        "for_hypothesis",
        new=_uncached_euler_diagnostics,
    ):
        yield
    ####
####


def _uncached_observed_intervals(
    cache: ObservedIncrementCache,
    frame: RateFrame,
) -> Rotation:
    """Reference calculation for comparing observed-increment caching."""

    return observed_interval_increments(cache.attitudes, frame)
####


def _uncached_observed_windows(
    cache: ObservedIncrementCache,
    window_size: int,
    frame: RateFrame,
) -> WindowComposition:
    """Recompute observed windows for the benchmark's uncached reference path."""

    if window_size < 1:
        raise ValueError("window_size must be positive")
    ####
    interval_count = len(cache.attitudes) - 1
    starts = arange(0, interval_count - window_size + 1, dtype=int64)
    return WindowComposition(
        window_increments=observed_window_increments(
            cache.attitudes,
            starts,
            window_size,
            frame,
        ),
        starts=starts,
    )
####


@contextmanager
def _observed_increment_cache_mode(enabled: bool) -> Generator[None, None, None]:
    if enabled:
        yield
        return
    ####
    with (
        patch.object(
            ObservedIncrementCache,
            "interval_increments",
            new=_uncached_observed_intervals,
        ),
        patch.object(
            ObservedIncrementCache,
            "window_composition",
            new=_uncached_observed_windows,
        ),
    ):
        yield
    ####
####


def _traced_peak_bytes(
    case: BenchmarkCase,
    euler_cache_enabled: bool,
    observed_cache_enabled: bool = True,
) -> int:
    gc.collect()
    tracemalloc.start()
    try:
        with (
            _euler_cache_mode(euler_cache_enabled),
            _observed_increment_cache_mode(observed_cache_enabled),
        ):
            infer_conventions(case.data, case.config)
            _, peak = tracemalloc.get_traced_memory()
        ####
    finally:
        tracemalloc.stop()
    ####
    return peak
####


def _benchmark_result(
    case: BenchmarkCase,
    *,
    samples: int,
    search: str,
    motion_profile: MotionProfile,
    motion_seed: int,
    euler_cache_enabled: bool,
    observed_cache_enabled: bool,
    report: InferenceReport,
    timings: list[float],
    traced_peak_bytes: int,
) -> BenchmarkResult:
    return BenchmarkResult(
        samples=samples,
        sample_period_s=case.sample_period_s,
        search=search,
        representation=case.data.representation.value,
        motion_profile=motion_profile.value,
        motion_seed=motion_seed,
        timestamps_available=case.data.time_s is not None,
        euler_heuristic_cache_enabled=euler_cache_enabled,
        observed_increment_cache_enabled=observed_cache_enabled,
        hypotheses=report.hypothesis_count,
        timing_sample_count=len(timings),
        minimum_seconds=min(timings),
        median_seconds=median(timings),
        maximum_seconds=max(timings),
        traced_peak_bytes=traced_peak_bytes,
    )
####


def benchmark_case(
    samples: int,
    search: str,
    repeats: int,
    motion_profile: MotionProfile = MotionProfile.RICH_THREE_AXIS,
    motion_seed: int = DEFAULT_MOTION_SEED,
    use_euler_heuristic_cache: bool = True,
    use_observed_increment_cache: bool = True,
    sample_period_s: float = 0.025,
) -> BenchmarkResult:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    ####
    case = _case_data(
        samples,
        search,
        motion_profile,
        motion_seed,
        sample_period_s=sample_period_s,
    )

    timings: list[float] = []
    report: InferenceReport | None = None
    for _ in range(repeats):
        gc.collect()
        with (
            _euler_cache_mode(use_euler_heuristic_cache),
            _observed_increment_cache_mode(use_observed_increment_cache),
        ):
            start = perf_counter()
            report = infer_conventions(case.data, case.config)
            timings.append(perf_counter() - start)
        ####
    ####

    if report is None:
        raise RuntimeError("Benchmark did not produce an inference report")
    ####
    return _benchmark_result(
        case,
        samples=samples,
        search=search,
        motion_profile=motion_profile,
        motion_seed=motion_seed,
        euler_cache_enabled=use_euler_heuristic_cache,
        observed_cache_enabled=use_observed_increment_cache,
        report=report,
        timings=timings,
        traced_peak_bytes=_traced_peak_bytes(
            case,
            use_euler_heuristic_cache,
            use_observed_increment_cache,
        ),
    )
####


def benchmark_euler_cache_comparison(
    samples: int,
    search: str,
    repeats: int,
    motion_profile: MotionProfile = MotionProfile.RICH_THREE_AXIS,
    motion_seed: int = DEFAULT_MOTION_SEED,
    sample_period_s: float = 0.025,
) -> EulerCacheComparison:
    """Compare cached/uncached scoring on one trajectory with alternating order."""

    if not search.startswith("euler-"):
        raise ValueError("Euler cache comparisons require an Euler search")
    ####
    if repeats < 1:
        raise ValueError("repeats must be positive")
    ####
    case = _case_data(
        samples,
        search,
        motion_profile,
        motion_seed,
        sample_period_s=sample_period_s,
    )
    uncached_timings: list[float] = []
    cached_timings: list[float] = []
    uncached_report: InferenceReport | None = None
    cached_report: InferenceReport | None = None

    for repeat_index in range(repeats):
        modes = (False, True) if repeat_index % 2 == 0 else (True, False)
        for cache_enabled in modes:
            gc.collect()
            with _euler_cache_mode(cache_enabled), _observed_increment_cache_mode(True):
                start = perf_counter()
                report = infer_conventions(case.data, case.config)
                elapsed = perf_counter() - start
            ####
            if cache_enabled:
                cached_timings.append(elapsed)
                cached_report = report
            else:
                uncached_timings.append(elapsed)
                uncached_report = report
            ####
        ####
    ####

    if cached_report is None or uncached_report is None:
        raise RuntimeError("Cache comparison did not produce both inference results")
    ####
    if cached_report != uncached_report:
        raise RuntimeError("Euler heuristic caching changed the inference report")
    ####

    peak_order = (True, False) if repeats % 2 else (False, True)
    traced_peaks: dict[bool, int] = {}
    for cache_enabled in peak_order:
        traced_peaks[cache_enabled] = _traced_peak_bytes(case, cache_enabled, True)
    ####
    uncached_result = _benchmark_result(
        case,
        samples=samples,
        search=search,
        motion_profile=motion_profile,
        motion_seed=motion_seed,
        euler_cache_enabled=False,
        observed_cache_enabled=True,
        report=uncached_report,
        timings=uncached_timings,
        traced_peak_bytes=traced_peaks[False],
    )
    cached_result = _benchmark_result(
        case,
        samples=samples,
        search=search,
        motion_profile=motion_profile,
        motion_seed=motion_seed,
        euler_cache_enabled=True,
        observed_cache_enabled=True,
        report=cached_report,
        timings=cached_timings,
        traced_peak_bytes=traced_peaks[True],
    )
    return EulerCacheComparison(
        uncached=uncached_result,
        cached=cached_result,
        median_speedup_factor=(uncached_result.median_seconds / cached_result.median_seconds),
    )
####


def benchmark_observed_increment_cache_comparison(
    samples: int,
    search: str,
    repeats: int,
    motion_profile: MotionProfile = MotionProfile.RICH_THREE_AXIS,
    motion_seed: int = DEFAULT_MOTION_SEED,
    include_timestamps: bool = True,
    sample_period_s: float = 0.025,
) -> ObservedIncrementCacheComparison:
    """Compare observed-rotation caching modes on one trajectory, alternating order."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    ####
    case = _case_data(
        samples,
        search,
        motion_profile,
        motion_seed,
        include_timestamps,
        sample_period_s,
    )
    uncached_timings: list[float] = []
    cached_timings: list[float] = []
    uncached_report: InferenceReport | None = None
    cached_report: InferenceReport | None = None

    for repeat_index in range(repeats):
        modes = (False, True) if repeat_index % 2 == 0 else (True, False)
        for cache_enabled in modes:
            gc.collect()
            with _euler_cache_mode(True), _observed_increment_cache_mode(cache_enabled):
                start = perf_counter()
                report = infer_conventions(case.data, case.config)
                elapsed = perf_counter() - start
            ####
            if cache_enabled:
                cached_timings.append(elapsed)
                cached_report = report
            else:
                uncached_timings.append(elapsed)
                uncached_report = report
            ####
        ####
    ####

    if cached_report is None or uncached_report is None:
        raise RuntimeError("Observed-increment cache comparison did not run both modes")
    ####
    if cached_report != uncached_report:
        raise RuntimeError("Observed-increment caching changed the inference report")
    ####

    peak_order = (True, False) if repeats % 2 else (False, True)
    traced_peaks: dict[bool, int] = {}
    for cache_enabled in peak_order:
        traced_peaks[cache_enabled] = _traced_peak_bytes(case, True, cache_enabled)
    ####
    uncached_result = _benchmark_result(
        case,
        samples=samples,
        search=search,
        motion_profile=motion_profile,
        motion_seed=motion_seed,
        euler_cache_enabled=True,
        observed_cache_enabled=False,
        report=uncached_report,
        timings=uncached_timings,
        traced_peak_bytes=traced_peaks[False],
    )
    cached_result = _benchmark_result(
        case,
        samples=samples,
        search=search,
        motion_profile=motion_profile,
        motion_seed=motion_seed,
        euler_cache_enabled=True,
        observed_cache_enabled=True,
        report=cached_report,
        timings=cached_timings,
        traced_peak_bytes=traced_peaks[True],
    )
    return ObservedIncrementCacheComparison(
        uncached=uncached_result,
        cached=cached_result,
        median_speedup_factor=(uncached_result.median_seconds / cached_result.median_seconds),
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


def _cold_process_case(
    samples: int,
    search: str,
    motion_profile: MotionProfile,
    motion_seed: int,
    observed_cache_enabled: bool = True,
    include_timestamps: bool = True,
    sample_period_s: float = 0.025,
) -> ColdProcessResult:
    case = _case_data(
        samples,
        search,
        motion_profile,
        motion_seed,
        include_timestamps,
        sample_period_s,
    )
    with _euler_cache_mode(True), _observed_increment_cache_mode(observed_cache_enabled):
        start = perf_counter()
        report = infer_conventions(case.data, case.config)
        inference_seconds = perf_counter() - start
    ####
    peak_rss_bytes = _peak_process_rss_bytes()
    report_fingerprint = sha256(report.model_dump_json().encode("utf-8")).hexdigest()
    return ColdProcessResult(
        motion_profile=motion_profile.value,
        motion_seed=motion_seed,
        sample_period_s=case.sample_period_s,
        euler_heuristic_cache_enabled=True,
        observed_increment_cache_enabled=observed_cache_enabled,
        timestamps_available=case.data.time_s is not None,
        hypotheses=report.hypothesis_count,
        report_fingerprint=report_fingerprint,
        inference_seconds=inference_seconds,
        peak_rss_bytes=peak_rss_bytes,
    )
####


def benchmark_cold_process(
    samples: int,
    search: str,
    motion_profile: MotionProfile = MotionProfile.RICH_THREE_AXIS,
    motion_seed: int = DEFAULT_MOTION_SEED,
    observed_cache_enabled: bool = True,
    include_timestamps: bool = True,
    sample_period_s: float = 0.025,
) -> ColdProcessResult:
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
            motion_profile.value,
            str(motion_seed),
            str(observed_cache_enabled).lower(),
            str(include_timestamps).lower(),
            str(sample_period_s),
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
        motion_profile=str(payload["motion_profile"]),
        motion_seed=int(payload["motion_seed"]),
        sample_period_s=float(payload["sample_period_s"]),
        euler_heuristic_cache_enabled=bool(payload["euler_heuristic_cache_enabled"]),
        observed_increment_cache_enabled=bool(payload["observed_increment_cache_enabled"]),
        timestamps_available=bool(payload["timestamps_available"]),
        hypotheses=int(payload["hypotheses"]),
        report_fingerprint=str(payload["report_fingerprint"]),
        inference_seconds=float(payload["inference_seconds"]),
        peak_rss_bytes=(
            None if payload["peak_rss_bytes"] is None else int(payload["peak_rss_bytes"])
        ),
    )
    return replace(cold, process_seconds=process_seconds)
####


def benchmark_observed_increment_cache_process_comparison(
    samples: int,
    search: str,
    repeats: int,
    motion_profile: MotionProfile = MotionProfile.RICH_THREE_AXIS,
    motion_seed: int = DEFAULT_MOTION_SEED,
    include_timestamps: bool = True,
    sample_period_s: float = 0.025,
) -> ObservedIncrementCacheProcessComparison:
    """Compare cache modes in interleaved fresh processes, including native RSS."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    ####
    runs: dict[bool, list[ColdProcessResult]] = {False: [], True: []}
    for repeat_index in range(repeats):
        modes = (False, True) if repeat_index % 2 == 0 else (True, False)
        for cache_enabled in modes:
            result = benchmark_cold_process(
                samples,
                search,
                motion_profile,
                motion_seed,
                observed_cache_enabled=cache_enabled,
                include_timestamps=include_timestamps,
                sample_period_s=sample_period_s,
            )
            if (
                result.motion_profile != motion_profile.value
                or result.motion_seed != motion_seed
                or result.sample_period_s != sample_period_s
                or result.euler_heuristic_cache_enabled is not True
                or result.observed_increment_cache_enabled is not cache_enabled
                or result.timestamps_available is not include_timestamps
            ):
                raise RuntimeError("Fresh-process cache benchmark metadata differs")
            ####
            runs[cache_enabled].append(result)
        ####
    ####

    uncached_runs = tuple(runs[False])
    cached_runs = tuple(runs[True])
    all_runs = (*uncached_runs, *cached_runs)
    if not all_runs:
        raise RuntimeError("Fresh-process cache benchmark produced no runs")
    ####
    if len({item.hypotheses for item in all_runs}) != 1:
        raise RuntimeError("Fresh-process cache modes evaluated different hypothesis counts")
    ####
    if len({item.report_fingerprint for item in all_runs}) != 1:
        raise RuntimeError("Observed-increment caching changed a fresh-process report")
    ####
    uncached_median = median(item.inference_seconds for item in uncached_runs)
    cached_median = median(item.inference_seconds for item in cached_runs)

    return ObservedIncrementCacheProcessComparison(
        samples=samples,
        search=search,
        representation="euler" if search.startswith("euler-") else "quaternion",
        motion_profile=motion_profile.value,
        motion_seed=motion_seed,
        sample_period_s=sample_period_s,
        timestamps_available=include_timestamps,
        hypotheses=uncached_runs[0].hypotheses,
        repeats=repeats,
        uncached_runs=uncached_runs,
        cached_runs=cached_runs,
        median_speedup_factor=uncached_median / cached_median,
    )
####


def _median_peak_rss(runs: tuple[ColdProcessResult, ...]) -> int | None:
    values = [item.peak_rss_bytes for item in runs]
    if any(value is None for value in values):
        return None
    ####
    return int(median(value for value in values if value is not None))
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
        "--sample-period-s",
        type=_positive_finite_float,
        default=0.025,
        help="time between generated samples in seconds (default: 0.025)",
    )
    parser.add_argument(
        "--motion-profile",
        choices=(
            MotionProfile.RICH_THREE_AXIS.value,
            MotionProfile.SAMPLED_RATE_NONANALYTIC.value,
            MotionProfile.SAMPLED_RATE_PIECEWISE_LINEAR.value,
            MotionProfile.SAMPLED_RATE_MANEUVER_BURSTS.value,
        ),
        default=MotionProfile.RICH_THREE_AXIS.value,
        help="synthetic family; sampled-rate options use independently integrated seeded rates",
    )
    parser.add_argument("--motion-seed", type=int, default=DEFAULT_MOTION_SEED)
    parser.add_argument(
        "--disable-euler-cache",
        action="store_true",
        help="measure the uncached reference path for performance comparisons",
    )
    parser.add_argument(
        "--compare-euler-cache",
        action="store_true",
        help="interleave cached and uncached Euler runs and verify equal reports",
    )
    parser.add_argument(
        "--disable-observed-cache",
        action="store_true",
        help="recompute observed interval/window rotations for each hypothesis",
    )
    parser.add_argument(
        "--compare-observed-cache",
        action="store_true",
        help="interleave cached and uncached observed rotations and verify equal reports",
    )
    parser.add_argument(
        "--missing-timestamps",
        action="store_true",
        help="benchmark constant-period inference (requires --compare-observed-cache)",
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help=(
            "measure fresh-process inference and RSS; with observed-cache comparison, "
            "run both modes"
        ),
    )
    parser.add_argument("--cold-worker", nargs=7, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help="write machine-readable results as JSON")
    args = parser.parse_args()

    if args.process and (
        args.disable_euler_cache or args.compare_euler_cache or args.disable_observed_cache
    ):
        parser.error("--process cannot be combined with Euler-cache comparisons or disable options")
    ####
    if args.compare_euler_cache and args.disable_euler_cache:
        parser.error("--compare-euler-cache cannot be combined with --disable-euler-cache")
    ####
    if args.compare_observed_cache and args.disable_observed_cache:
        parser.error("--compare-observed-cache cannot be combined with --disable-observed-cache")
    ####
    if args.compare_euler_cache and args.compare_observed_cache:
        parser.error("Choose one cache comparison per benchmark invocation")
    ####
    if args.compare_euler_cache and args.disable_observed_cache:
        parser.error("--compare-euler-cache requires the observed-increment cache enabled")
    ####
    if args.compare_observed_cache and args.disable_euler_cache:
        parser.error("--compare-observed-cache requires the Euler heuristic cache enabled")
    ####
    if args.missing_timestamps and not args.compare_observed_cache:
        parser.error("--missing-timestamps requires --compare-observed-cache")
    ####
    if args.compare_euler_cache and any(
        not search.startswith("euler-") for search in args.searches
    ):
        parser.error("--compare-euler-cache requires Euler search spaces")
    ####
    if args.cold_worker is not None:
        (
            samples_text,
            search,
            motion_profile_text,
            motion_seed_text,
            observed_cache_text,
            timestamps_text,
            sample_period_text,
        ) = args.cold_worker
        if observed_cache_text not in ("true", "false") or timestamps_text not in (
            "true",
            "false",
        ):
            parser.error("Cold-worker cache and timestamp flags must be true or false")
        ####
        cold_case = _cold_process_case(
            int(samples_text),
            search,
            MotionProfile(motion_profile_text),
            int(motion_seed_text),
            observed_cache_enabled=observed_cache_text == "true",
            include_timestamps=timestamps_text == "true",
            sample_period_s=float(sample_period_text),
        )
        print(json.dumps(asdict(cold_case)))
        return
    ####

    results: list[BenchmarkResult] = []
    cache_comparisons: list[EulerCacheComparison] = []
    observed_cache_comparisons: list[ObservedIncrementCacheComparison] = []
    observed_cache_process_comparisons: list[ObservedIncrementCacheProcessComparison] = []
    for samples in args.samples:
        for search in args.searches:
            motion_profile = MotionProfile(args.motion_profile)
            if args.compare_euler_cache:
                print(
                    f"Comparing Euler cache modes for {motion_profile.value}, "
                    f"seed={args.motion_seed}, samples={samples}, "
                    f"dt={args.sample_period_s:g}s, search={search}",
                    file=sys.stderr,
                    flush=True,
                )
                comparison = benchmark_euler_cache_comparison(
                    samples,
                    search,
                    args.repeats,
                    motion_profile,
                    args.motion_seed,
                    sample_period_s=args.sample_period_s,
                )
                cache_comparisons.append(comparison)
                print(
                    f"Completed paired benchmark: "
                    f"uncached={comparison.uncached.median_seconds:.3f}s, "
                    f"cached={comparison.cached.median_seconds:.3f}s, "
                    f"speedup={comparison.median_speedup_factor:.2f}x",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            ####
            if args.compare_observed_cache:
                print(
                    f"Comparing observed-increment cache modes for {motion_profile.value}, "
                    f"seed={args.motion_seed}, samples={samples}, "
                    f"dt={args.sample_period_s:g}s, search={search}",
                    file=sys.stderr,
                    flush=True,
                )
                if args.process:
                    process_comparison = benchmark_observed_increment_cache_process_comparison(
                        samples,
                        search,
                        args.repeats,
                        motion_profile,
                        args.motion_seed,
                        include_timestamps=not args.missing_timestamps,
                        sample_period_s=args.sample_period_s,
                    )
                    observed_cache_process_comparisons.append(process_comparison)
                    median_rss = {
                        "uncached": _median_peak_rss(process_comparison.uncached_runs),
                        "cached": _median_peak_rss(process_comparison.cached_runs),
                    }
                    print(
                        f"Fresh-process median peak RSS: uncached={median_rss['uncached']}, "
                        f"cached={median_rss['cached']} B",
                        file=sys.stderr,
                        flush=True,
                    )
                    speedup = process_comparison.median_speedup_factor
                else:
                    comparison = benchmark_observed_increment_cache_comparison(
                        samples,
                        search,
                        args.repeats,
                        motion_profile,
                        args.motion_seed,
                        include_timestamps=not args.missing_timestamps,
                        sample_period_s=args.sample_period_s,
                    )
                    observed_cache_comparisons.append(comparison)
                    speedup = comparison.median_speedup_factor
                ####
                print(
                    f"Completed paired benchmark: inference speedup={speedup:.2f}x",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            ####
            print(
                f"Benchmarking {motion_profile.value}, seed={args.motion_seed}, "
                f"samples={samples}, dt={args.sample_period_s:g}s, search={search}",
                file=sys.stderr,
                flush=True,
            )
            result = benchmark_case(
                samples,
                search,
                args.repeats,
                motion_profile,
                args.motion_seed,
                use_euler_heuristic_cache=not args.disable_euler_cache,
                use_observed_increment_cache=not args.disable_observed_cache,
                sample_period_s=args.sample_period_s,
            )
            if args.process:
                cold = benchmark_cold_process(
                    samples,
                    search,
                    motion_profile,
                    args.motion_seed,
                    sample_period_s=args.sample_period_s,
                )
                if (
                    cold.hypotheses != result.hypotheses
                    or cold.motion_profile != result.motion_profile
                    or cold.motion_seed != result.motion_seed
                    or cold.sample_period_s != result.sample_period_s
                    or cold.timestamps_available != result.timestamps_available
                    or cold.euler_heuristic_cache_enabled != result.euler_heuristic_cache_enabled
                    or cold.observed_increment_cache_enabled
                    != result.observed_increment_cache_enabled
                ):
                    raise RuntimeError("Cold and in-process benchmark case metadata differs")
                ####
                result = replace(
                    result,
                    cold_inference_seconds=cold.inference_seconds,
                    cold_process_seconds=cold.process_seconds,
                    peak_process_rss_bytes=cold.peak_rss_bytes,
                )
            ####
            results.append(result)
            print(
                f"Completed {samples} samples / {search}: "
                f"median={result.median_seconds:.3f}s, "
                f"range={result.minimum_seconds:.3f}-{result.maximum_seconds:.3f}s "
                f"(n={result.timing_sample_count}), "
                f"traced_peak={result.traced_peak_bytes} B",
                file=sys.stderr,
                flush=True,
            )
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
        "euler_cache_comparisons": [asdict(comparison) for comparison in cache_comparisons],
        "observed_increment_cache_comparisons": [
            asdict(comparison) for comparison in observed_cache_comparisons
        ],
        "observed_increment_cache_process_comparisons": [
            asdict(comparison) for comparison in observed_cache_process_comparisons
        ],
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
