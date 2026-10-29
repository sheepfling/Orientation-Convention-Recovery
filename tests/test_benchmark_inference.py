"""Smoke-test the reproducible full-inference benchmark harness."""

from __future__ import annotations

import pytest

from scripts.benchmark_inference import benchmark_case


def test_compact_benchmark_measures_two_hypotheses() -> None:
    result = benchmark_case(21, "compact", 1)

    assert result.samples == 21
    assert result.hypotheses == 2
    assert result.median_seconds > 0.0
    assert result.traced_peak_bytes > 0
####


def test_benchmark_rejects_invalid_workloads() -> None:
    with pytest.raises(ValueError, match="samples"):
        benchmark_case(2, "compact", 1)
    ####
    with pytest.raises(ValueError, match="Unknown search"):
        benchmark_case(21, "tiny", 1)
    ####
####
