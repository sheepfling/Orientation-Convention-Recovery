"""Smoke-test the reproducible full-inference benchmark harness."""

from __future__ import annotations

import sys

import pytest

from scripts.benchmark_inference import (
    benchmark_case,
    benchmark_cold_process,
    benchmark_euler_cache_comparison,
    benchmark_observed_increment_cache_comparison,
    benchmark_observed_increment_cache_process_comparison,
)
from scripts.generate_validation_corpus import MotionProfile


def test_compact_benchmark_measures_two_hypotheses() -> None:
    result = benchmark_case(21, "compact", 1)

    assert result.samples == 21
    assert result.sample_period_s == 0.025
    assert result.representation == "quaternion"
    assert result.timestamps_available
    assert result.euler_heuristic_cache_enabled
    assert result.observed_increment_cache_enabled
    assert result.hypotheses == 2
    assert result.median_seconds > 0.0
    assert result.timing_sample_count == 1
    assert result.minimum_seconds == result.median_seconds == result.maximum_seconds
    assert result.traced_peak_bytes > 0
####


def test_benchmark_records_requested_sample_period() -> None:
    result = benchmark_case(21, "compact", 1, sample_period_s=0.05)

    assert result.sample_period_s == 0.05
####


@pytest.mark.parametrize("sample_period_s", (0.0, -0.025, float("inf"), float("nan")))
def test_benchmark_rejects_nonpositive_or_nonfinite_sample_period(
    sample_period_s: float,
) -> None:
    with pytest.raises(ValueError, match="sample_period_s"):
        benchmark_case(21, "compact", 1, sample_period_s=sample_period_s)
    ####
####


def test_euler_compact_benchmark_measures_two_hypotheses() -> None:
    result = benchmark_case(21, "euler-compact", 1)

    assert result.representation == "euler"
    assert result.hypotheses == 2
    assert result.euler_heuristic_cache_enabled
    assert result.observed_increment_cache_enabled
    assert result.median_seconds > 0.0
    assert result.timing_sample_count == 1
    assert result.minimum_seconds == result.median_seconds == result.maximum_seconds
    assert result.traced_peak_bytes > 0
####


@pytest.mark.parametrize(
    ("motion_profile", "expected_profile"),
    (
        (MotionProfile.SAMPLED_RATE_NONANALYTIC, "sampled-rate-nonanalytic"),
        (MotionProfile.SAMPLED_RATE_PIECEWISE_LINEAR, "sampled-rate-piecewise-linear"),
        (MotionProfile.SAMPLED_RATE_MANEUVER_BURSTS, "sampled-rate-maneuver-bursts"),
    ),
)
def test_sampled_rate_benchmark_records_synthetic_motion_metadata(
    motion_profile: MotionProfile,
    expected_profile: str,
) -> None:
    result = benchmark_case(
        21,
        "euler-compact",
        3,
        motion_profile,
        motion_seed=913,
    )

    assert result.motion_profile == expected_profile
    assert result.motion_seed == 913
    assert result.representation == "euler"
    assert result.hypotheses == 2
    assert result.timing_sample_count == 3
    assert 0.0 < result.minimum_seconds <= result.median_seconds <= result.maximum_seconds
####


def test_benchmark_can_measure_uncached_euler_heuristics() -> None:
    result = benchmark_case(
        21,
        "euler-compact",
        1,
        MotionProfile.SAMPLED_RATE_NONANALYTIC,
        motion_seed=913,
        use_euler_heuristic_cache=False,
    )

    assert not result.euler_heuristic_cache_enabled
    assert result.hypotheses == 2
####


def test_euler_cache_comparison_pairs_modes_and_checks_report_equivalence() -> None:
    comparison = benchmark_euler_cache_comparison(
        21,
        "euler-compact",
        1,
        MotionProfile.SAMPLED_RATE_NONANALYTIC,
        motion_seed=913,
    )

    assert not comparison.uncached.euler_heuristic_cache_enabled
    assert comparison.cached.euler_heuristic_cache_enabled
    assert comparison.uncached.hypotheses == comparison.cached.hypotheses == 2
    assert comparison.uncached.observed_increment_cache_enabled
    assert comparison.cached.observed_increment_cache_enabled
    assert comparison.median_speedup_factor > 0.0
####


def test_observed_increment_cache_comparison_checks_report_equivalence() -> None:
    comparison = benchmark_observed_increment_cache_comparison(
        21,
        "compact",
        1,
        MotionProfile.SAMPLED_RATE_NONANALYTIC,
        motion_seed=913,
    )

    assert comparison.uncached.euler_heuristic_cache_enabled
    assert comparison.cached.euler_heuristic_cache_enabled
    assert not comparison.uncached.observed_increment_cache_enabled
    assert comparison.cached.observed_increment_cache_enabled
    assert comparison.uncached.hypotheses == comparison.cached.hypotheses == 2
    assert comparison.median_speedup_factor > 0.0
####


def test_observed_increment_cache_comparison_covers_missing_timestamps() -> None:
    comparison = benchmark_observed_increment_cache_comparison(
        31,
        "euler-compact",
        1,
        MotionProfile.SAMPLED_RATE_NONANALYTIC,
        motion_seed=913,
        include_timestamps=False,
    )

    assert not comparison.uncached.timestamps_available
    assert not comparison.cached.timestamps_available
    assert comparison.uncached.observed_increment_cache_enabled is False
    assert comparison.cached.observed_increment_cache_enabled is True
    assert comparison.median_speedup_factor > 0.0
####


def test_observed_increment_process_comparison_pairs_rss_and_report_fingerprints() -> None:
    comparison = benchmark_observed_increment_cache_process_comparison(
        21,
        "compact",
        1,
        MotionProfile.SAMPLED_RATE_NONANALYTIC,
        motion_seed=913,
        sample_period_s=0.05,
    )

    assert comparison.timestamps_available
    assert comparison.sample_period_s == 0.05
    assert comparison.hypotheses == 2
    assert len(comparison.uncached_runs) == len(comparison.cached_runs) == 1
    uncached = comparison.uncached_runs[0]
    cached = comparison.cached_runs[0]
    assert not uncached.observed_increment_cache_enabled
    assert cached.observed_increment_cache_enabled
    assert uncached.sample_period_s == cached.sample_period_s == 0.05
    assert uncached.report_fingerprint == cached.report_fingerprint
    assert uncached.inference_seconds > 0.0
    assert cached.inference_seconds > 0.0
    assert uncached.process_seconds is not None
    assert cached.process_seconds is not None
    if sys.platform in ("win32", "linux"):
        assert uncached.peak_rss_bytes is not None and uncached.peak_rss_bytes > 0
        assert cached.peak_rss_bytes is not None and cached.peak_rss_bytes > 0
    else:
        assert uncached.peak_rss_bytes is None
        assert cached.peak_rss_bytes is None
    ####
####


def test_benchmark_can_measure_uncached_observed_increments() -> None:
    result = benchmark_case(
        21,
        "compact",
        1,
        MotionProfile.SAMPLED_RATE_NONANALYTIC,
        motion_seed=913,
        use_observed_increment_cache=False,
    )

    assert result.euler_heuristic_cache_enabled
    assert not result.observed_increment_cache_enabled
    assert result.hypotheses == 2
####


def test_cold_benchmark_measures_a_fresh_process() -> None:
    cold = benchmark_cold_process(21, "euler-compact")

    assert cold.hypotheses == 2
    assert cold.motion_profile == "rich-three-axis"
    assert cold.motion_seed == 730201
    assert cold.euler_heuristic_cache_enabled
    assert cold.observed_increment_cache_enabled
    assert cold.timestamps_available
    assert cold.inference_seconds > 0.0
    assert len(cold.report_fingerprint) == 64
    assert cold.process_seconds is not None
    assert cold.process_seconds >= cold.inference_seconds
    if sys.platform in ("win32", "linux"):
        assert cold.peak_rss_bytes is not None and cold.peak_rss_bytes > 0
    else:
        assert cold.peak_rss_bytes is None
    ####
####


def test_sampled_rate_cold_benchmark_uses_requested_seed() -> None:
    cold = benchmark_cold_process(
        21,
        "euler-compact",
        MotionProfile.SAMPLED_RATE_NONANALYTIC,
        motion_seed=913,
        sample_period_s=0.05,
    )

    assert cold.hypotheses == 2
    assert cold.motion_profile == "sampled-rate-nonanalytic"
    assert cold.motion_seed == 913
    assert cold.sample_period_s == 0.05
    assert cold.euler_heuristic_cache_enabled
    assert cold.observed_increment_cache_enabled
    assert cold.inference_seconds > 0.0
####


def test_benchmark_rejects_invalid_workloads() -> None:
    with pytest.raises(ValueError, match="samples"):
        benchmark_case(2, "compact", 1)
    ####
    with pytest.raises(ValueError, match="Unknown search"):
        benchmark_case(21, "tiny", 1)
    ####
####
