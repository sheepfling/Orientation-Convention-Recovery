"""Shared seeded measurement-noise processes for synthetic validation data."""

from __future__ import annotations

from math import exp, expm1, isclose, isfinite, sqrt

from numpy import arange, array, bool_, float64, int64, ones, zeros
from numpy.random import Generator
from numpy.typing import NDArray

__all__ = [
    "OU_PROCESS_BASE_PERIOD_S",
    "OU_PROCESS_OUTPUT_PERIOD_S",
    "sample_time_domain_dropout_mask",
    "sample_time_domain_ou_noise",
    "sample_time_domain_uniform_values",
    "validate_ou_sample_period",
]

FloatArray = NDArray[float64]
OU_PROCESS_BASE_PERIOD_S = 0.001
OU_PROCESS_OUTPUT_PERIOD_S = OU_PROCESS_BASE_PERIOD_S / 2.0


def validate_ou_sample_period(sample_period_s: float) -> int:
    """Return the shared half-millisecond-grid stride for a positive period."""

    if not isfinite(sample_period_s) or sample_period_s <= 0.0:
        raise ValueError("sample_period_s must be positive and finite.")
    ####
    stride_value = sample_period_s / OU_PROCESS_OUTPUT_PERIOD_S
    rounded_stride = round(stride_value)
    if rounded_stride < 1 or not isclose(stride_value, rounded_stride, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("Sample period must be a multiple of 0.5 ms when OU noise is enabled.")
    ####
    return int(rounded_stride)
####


def _shared_grid_indices(
    sample_count: int,
    sample_period_s: float,
) -> NDArray[int64] | None:
    """Map cadence samples onto the shared half-millisecond acquisition grid."""

    if sample_count < 1:
        raise ValueError("sample_count must be positive.")
    ####
    if not isfinite(sample_period_s) or sample_period_s <= 0.0:
        raise ValueError("sample_period_s must be positive and finite.")
    ####
    try:
        stride = validate_ou_sample_period(sample_period_s)
    except ValueError:
        return None
    ####
    return arange(sample_count, dtype=int64) * stride
####


def sample_time_domain_uniform_values(
    random: Generator,
    sample_count: int,
    sample_period_s: float,
    lower_bound: float,
    upper_bound: float,
) -> FloatArray:
    """Draw uniform values whose nested cadences share common-time realizations.

    For cadences supported by the half-millisecond grid, values are drawn on that
    fine grid and decimated. Unsupported periods retain per-sample independent
    draws so synthetic corpus generation continues to support arbitrary cadences.
    """

    if not isfinite(lower_bound) or not isfinite(upper_bound) or lower_bound > upper_bound:
        raise ValueError("uniform bounds must be finite and ordered.")
    ####
    indices = _shared_grid_indices(sample_count, sample_period_s)
    if lower_bound == upper_bound:
        return zeros(sample_count, dtype=float64)
    ####
    if indices is None:
        return array(random.uniform(lower_bound, upper_bound, sample_count), dtype=float64)
    ####
    fine_grid = random.uniform(lower_bound, upper_bound, int(indices[-1]) + 1)
    return array(fine_grid[indices], dtype=float64)
####


def sample_time_domain_dropout_mask(
    random: Generator,
    sample_count: int,
    sample_period_s: float,
    dropout_probability: float,
) -> NDArray[bool_]:
    """Draw Bernoulli retention flags with cadence-paired common-time outcomes."""

    if not isfinite(dropout_probability) or not 0.0 <= dropout_probability < 1.0:
        raise ValueError("dropout_probability must be finite and in [0, 1).")
    ####
    indices = _shared_grid_indices(sample_count, sample_period_s)
    if dropout_probability == 0.0:
        return ones(sample_count, dtype=bool_)
    ####
    if indices is None:
        draws = random.random(sample_count)
    else:
        draws = random.random(int(indices[-1]) + 1)[indices]
    ####
    return array(draws >= dropout_probability, dtype=bool_)
####


def sample_time_domain_ou_noise(
    random: Generator,
    sample_count: int,
    sample_period_s: float,
    sigma: float,
    correlation_time_s: float,
) -> FloatArray:
    """Sample stationary OU noise on a shared 0.5 ms grid, then decimate it.

    The marginal standard deviation and real-time correlation time stay fixed as
    the observation cadence changes. Equal seeds, durations, and time constants
    therefore share exactly the same noise values at common timestamps. Integer-
    millisecond values follow the legacy recurrence; intermediate half-ms values
    are sampled from the OU bridge conditional on their neighboring endpoints.
    """

    if sample_count < 1:
        raise ValueError("sample_count must be positive.")
    ####
    if not isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigma must be finite and non-negative.")
    ####
    if not isfinite(correlation_time_s) or correlation_time_s <= 0.0:
        raise ValueError("correlation_time_s must be positive and finite.")
    ####
    output_stride = validate_ou_sample_period(sample_period_s)
    last_output_index = (sample_count - 1) * output_stride
    base_sample_count = (last_output_index + 1) // 2 + 1
    base_step_correlation = exp(-OU_PROCESS_BASE_PERIOD_S / correlation_time_s)
    innovation_sigma = sigma * sqrt(-expm1(-2.0 * OU_PROCESS_BASE_PERIOD_S / correlation_time_s))
    process = zeros((base_sample_count, 3), dtype=float64)
    process[0] = random.normal(0.0, sigma, 3)
    innovations = random.normal(0.0, innovation_sigma, (base_sample_count - 1, 3))
    for sample_index in range(1, base_sample_count):
        process[sample_index] = (
            base_step_correlation * process[sample_index - 1] + innovations[sample_index - 1]
        )
    ####
    output_indices = arange(sample_count, dtype=int64) * output_stride
    integer_indices = output_indices // 2
    sampled_process = zeros((sample_count, 3), dtype=float64)
    even_indices = output_indices % 2 == 0
    sampled_process[even_indices] = process[integer_indices[even_indices]]
    if output_stride % 2 == 1:
        bridge_correlation = exp(-OU_PROCESS_OUTPUT_PERIOD_S / correlation_time_s)
        bridge_step_variance = sigma**2 * -expm1(
            -2.0 * OU_PROCESS_OUTPUT_PERIOD_S / correlation_time_s
        )
        bridge_variance = bridge_step_variance / (1.0 + bridge_correlation**2)
        bridge_sigma = sqrt(bridge_variance)
        bridge_innovations = random.normal(
            0.0,
            bridge_sigma,
            (base_sample_count - 1, 3),
        )
        left_endpoints = process[:-1]
        right_endpoints = process[1:]
        bridge_mean = bridge_correlation * left_endpoints + (
            bridge_correlation / (1.0 + bridge_correlation**2)
        ) * (right_endpoints - bridge_correlation**2 * left_endpoints)
        bridge_values = bridge_mean + bridge_innovations
        odd_indices = ~even_indices
        sampled_process[odd_indices] = bridge_values[integer_indices[odd_indices]]
    ####
    return array(sampled_process, dtype=float64)
####
