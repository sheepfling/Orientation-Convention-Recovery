"""Numerical summaries used to diagnose excitation and residual evidence."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from math import exp, log

from numpy import (
    argsort,
    ceil,
    cumsum,
    diff,
    float64,
    mean,
    median,
    minimum,
    newaxis,
    ones,
    pi,
    sqrt,
    square,
    zeros,
)
from numpy import (
    log as np_log,
)
from numpy import (
    sum as np_sum,
)
from numpy.linalg import eigvalsh, norm, svd

from .core import TrajectoryData
from .math_utils import FloatArray
from .models import InferenceConfig

__all__ = [
    "ExcitationSummary",
    "NoiseProcessEstimate",
    "RawExcitationSummary",
    "effective_excitation_rank",
    "estimate_noise_process",
    "fixed_noise_log_likelihood",
    "log_likelihood_advantage_by_prefix",
    "log_likelihood_advantage_from_squared_errors_by_prefix",
    "raw_rate_excitation",
]


@dataclass(frozen=True, slots=True)
class ExcitationSummary:
    """Singular-value summary of integrated candidate-rate excitation."""

    effective_rank: float
    singular_values: tuple[float, float, float]
####


@dataclass(frozen=True, slots=True)
class RawExcitationSummary:
    """Eigenvalue summary of raw rate excitation before candidate decoding."""

    rank: int
    eigenvalues: tuple[float, float, float]
    weakest_axis_ratio: float
####


@dataclass(frozen=True, slots=True)
class NoiseProcessEstimate:
    """Robust residual scale and temporal-correlation estimates."""

    sigma_rad: float
    retained_count: int
    lag1_autocorrelation: float
####


def effective_excitation_rank(
    rates_rad_s: FloatArray,
    durations_s: FloatArray,
) -> ExcitationSummary:
    """Summarize the effective dimensionality of integrated rate excitation."""

    integrated = 0.5 * (rates_rad_s[:-1] + rates_rad_s[1:]) * durations_s[:, newaxis]
    singular_values = svd(integrated, compute_uv=False)
    padded = zeros(3, dtype=float64)
    padded[: singular_values.size] = singular_values
    energy = square(padded)
    total = float(np_sum(energy))
    if total <= 1.0e-30:
        return ExcitationSummary(0.0, (0.0, 0.0, 0.0))
    ####
    probabilities = energy / total
    nonzero = probabilities > 1.0e-15
    entropy = -float(np_sum(probabilities[nonzero] * np_log(probabilities[nonzero])))
    return ExcitationSummary(
        effective_rank=float(exp(entropy)),
        singular_values=(float(padded[0]), float(padded[1]), float(padded[2])),
    )
####


def raw_rate_excitation(
    data: TrajectoryData,
    config: InferenceConfig,
) -> RawExcitationSummary:
    """Return decoder-invariant rank and weakest-axis ratio for raw rate channels."""

    if data.rates is None:
        return RawExcitationSummary(0, (0.0, 0.0, 0.0), 0.0)
    ####
    if data.time_s is None:
        weights = ones(data.rates.shape[0], dtype=float64)
    else:
        weights = diff(data.time_s, prepend=data.time_s[0])
        weights[0] = weights[1] if weights.size > 1 else 1.0
    ####
    gramian = (data.rates.T * weights) @ data.rates
    eigenvalues = eigvalsh(gramian)[::-1]
    largest = float(eigenvalues[0]) if eigenvalues.size else 0.0
    if largest <= 1.0e-30:
        return RawExcitationSummary(0, (0.0, 0.0, 0.0), 0.0)
    ####
    rank = int(np_sum(eigenvalues >= config.excitation_rank_relative_threshold * largest))
    ratio = float(eigenvalues[-1] / largest)
    padded = zeros(3, dtype=float64)
    padded[: eigenvalues.size] = eigenvalues
    return RawExcitationSummary(
        rank=rank,
        eigenvalues=(float(padded[0]), float(padded[1]), float(padded[2])),
        weakest_axis_ratio=ratio,
    )
####


def estimate_noise_process(
    interval_error_rotvecs: FloatArray,
    trim_fraction: float,
) -> NoiseProcessEstimate:
    """Estimate isotropic rotation-vector noise and lag-one correlation."""

    if interval_error_rotvecs.ndim != 2 or interval_error_rotvecs.shape[0] == 0:
        return NoiseProcessEstimate(0.0, 0, 0.0)
    ####
    robust_center = median(interval_error_rotvecs, axis=0)
    centered = interval_error_rotvecs - robust_center
    squared_norms = np_sum(centered * centered, axis=1)
    keep_count = max(1, int(ceil(squared_norms.size * (1.0 - trim_fraction))))
    keep_indices = argsort(squared_norms)[:keep_count]
    robust_center = mean(interval_error_rotvecs[keep_indices], axis=0)
    centered = interval_error_rotvecs - robust_center
    squared_norms = np_sum(centered * centered, axis=1)
    keep_indices = argsort(squared_norms)[:keep_count]
    kept = squared_norms[keep_indices]
    sigma = float(sqrt(max(float(mean(kept)) / 3.0, 1.0e-24)))

    keep_mask = zeros(centered.shape[0], dtype=bool)
    keep_mask[keep_indices] = True
    valid_pairs = keep_mask[:-1] & keep_mask[1:]
    if int(np_sum(valid_pairs)) < 2:
        autocorrelation = 0.0
    else:
        previous = centered[:-1][valid_pairs]
        following = centered[1:][valid_pairs]
        denominator = float(norm(previous) * norm(following))
        autocorrelation = (
            float(np_sum(previous * following) / denominator) if denominator > 1.0e-24 else 0.0
        )
    ####
    return NoiseProcessEstimate(sigma, keep_count, autocorrelation)
####


def fixed_noise_log_likelihood(
    interval_error_rotvecs: FloatArray,
    noise_sigma_rad: float,
) -> float:
    """Evaluate an isotropic Gaussian log-likelihood at a fixed noise scale."""

    if interval_error_rotvecs.ndim != 2 or interval_error_rotvecs.shape[0] == 0:
        return float("-inf")
    ####
    variance = max(float(noise_sigma_rad) ** 2, 1.0e-24)
    degrees_of_freedom = 3 * interval_error_rotvecs.shape[0]
    sum_squared_error = float(np_sum(square(interval_error_rotvecs)))
    return -0.5 * (sum_squared_error / variance + degrees_of_freedom * log(2.0 * pi * variance))
####


def log_likelihood_advantage_by_prefix(
    winner_errors: FloatArray,
    competitor_errors: Sequence[FloatArray],
    noise_sigma_rad: float,
) -> FloatArray:
    """Winner advantage at each prefix under the shared fixed Gaussian noise scale."""

    if winner_errors.ndim != 2 or winner_errors.shape[1] != 3 or len(winner_errors) == 0:
        raise ValueError("winner_errors must have shape (N, 3) with N > 0")
    ####
    if not competitor_errors:
        raise ValueError("At least one competitor is required")
    ####

    def squared_energies() -> Iterator[FloatArray]:
        for errors in competitor_errors:
            if errors.shape != winner_errors.shape:
                raise ValueError("Competitor residuals must match winner residual shape")
            ####
            yield np_sum(square(errors), axis=1)
        ####
    ####

    return log_likelihood_advantage_from_squared_errors_by_prefix(
        np_sum(square(winner_errors), axis=1), squared_energies(), noise_sigma_rad
    )
####


def log_likelihood_advantage_from_squared_errors_by_prefix(
    winner_squared_errors: FloatArray,
    competitor_squared_errors: Iterable[FloatArray],
    noise_sigma_rad: float,
) -> FloatArray:
    """Fixed-noise winner advantage from compact per-interval squared errors."""

    if winner_squared_errors.ndim != 1 or winner_squared_errors.size == 0:
        raise ValueError("winner_squared_errors must have shape (N,) with N > 0")
    ####

    def cumulative_energy(squared_errors: FloatArray) -> FloatArray:
        if squared_errors.shape != winner_squared_errors.shape:
            raise ValueError("Competitor squared errors must match winner shape")
        ####
        return cumsum(squared_errors)
    ####

    competitors = iter(competitor_squared_errors)
    first_competitor = next(competitors, None)
    if first_competitor is None:
        raise ValueError("At least one competitor is required")
    ####
    best_competitor = cumulative_energy(first_competitor)
    for squared_errors in competitors:
        best_competitor = minimum(best_competitor, cumulative_energy(squared_errors))
    ####
    winner = cumulative_energy(winner_squared_errors)
    variance = max(float(noise_sigma_rad) ** 2, 1.0e-24)
    return (best_competitor - winner) / (2.0 * variance)
####
