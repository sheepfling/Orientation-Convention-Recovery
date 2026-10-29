"""Smoke-test the reproducible full-inference benchmark harness."""

from __future__ import annotations

import sys

import pytest

from scripts.benchmark_inference import benchmark_case, benchmark_cold_process


def test_compact_benchmark_measures_two_hypotheses() -> None:
    result = benchmark_case(21, "compact", 1)

    assert result.samples == 21
    assert result.representation == "quaternion"
    assert result.hypotheses == 2
    assert result.median_seconds > 0.0
    assert result.traced_peak_bytes > 0
####


def test_euler_compact_benchmark_measures_two_hypotheses() -> None:
    result = benchmark_case(21, "euler-compact", 1)

    assert result.representation == "euler"
    assert result.hypotheses == 2
    assert result.median_seconds > 0.0
    assert result.traced_peak_bytes > 0
####


def test_cold_benchmark_measures_a_fresh_process() -> None:
    cold = benchmark_cold_process(21, "euler-compact")

    assert cold.hypotheses == 2
    assert cold.inference_seconds > 0.0
    assert cold.process_seconds is not None
    assert cold.process_seconds >= cold.inference_seconds
    if sys.platform in ("win32", "linux"):
        assert cold.peak_rss_bytes is not None and cold.peak_rss_bytes > 0
    else:
        assert cold.peak_rss_bytes is None
    ####
####


def test_benchmark_rejects_invalid_workloads() -> None:
    with pytest.raises(ValueError, match="samples"):
        benchmark_case(2, "compact", 1)
    ####
    with pytest.raises(ValueError, match="Unknown search"):
        benchmark_case(21, "tiny", 1)
    ####
####
