"""Residual-noise estimates and fixed-noise evidence sufficiency."""

from __future__ import annotations

from dataclasses import dataclass

from .core import TrajectoryData
from .models import InferenceConfig
from .scoring import ScoredCandidate as _ScoredCandidate
from .scoring import score_hypothesis as _score_hypothesis
from .scoring import score_partial_hypothesis as _score_partial_hypothesis
from .statistics import (
    estimate_noise_process,
    log_likelihood_advantage_from_squared_errors_by_prefix,
)

__all__ = ["_EvidenceSummary", "_evidence_summary"]


@dataclass(frozen=True, slots=True)
class _EvidenceSummary:
    """Statistical evidence and sufficiency metrics for a candidate hypothesis."""

    noise_sigma_rad: float
    noise_residual_count: int
    noise_lag1_autocorrelation: float
    sufficient: bool
    intervals_used: int
    fraction: float
    log_likelihood_ratio: float
####


def _evidence_summary(
    data: TrajectoryData,
    ranked: tuple[_ScoredCandidate, ...],
    detailed: _ScoredCandidate,
    config: InferenceConfig,
) -> _EvidenceSummary:
    """Estimate noise and find the earliest stable evidence prefix."""

    representatives: dict[str, _ScoredCandidate] = {}
    for item in sorted(ranked, key=lambda candidate: candidate.result.data_rank):
        representatives.setdefault(item.result.equivalence_class_id, item)
    ####
    data_ranked = tuple(representatives.values())
    data_winner = data_ranked[0]
    winner_details = detailed
    if detailed.result.hypothesis != data_winner.result.hypothesis:
        winner_details = (
            _score_partial_hypothesis(
                data, data_winner.result.hypothesis, config, retain_interval_details=True
            )
            if data.orientation is None or data.rates is None
            else _score_hypothesis(
                data, data_winner.result.hypothesis, config, retain_interval_details=True
            )
        )
    ####
    winner_errors = winner_details.interval_error_rotvecs
    if winner_errors is None:
        raise RuntimeError("Data-winning residual vectors were not retained")
    ####
    noise = estimate_noise_process(
        winner_errors,
        config.trim_fraction,
    )
    interval_count = data_winner.interval_squared_errors_rad2.shape[0]
    if data_winner.result.observed_step_rms_rad < config.slow_step_threshold_rad:
        return _EvidenceSummary(
            noise.sigma_rad,
            noise.retained_count,
            noise.lag1_autocorrelation,
            False,
            interval_count,
            1.0,
            0.0,
        )
    ####
    if len(data_ranked) < 2:
        return _EvidenceSummary(
            noise.sigma_rad,
            noise.retained_count,
            noise.lag1_autocorrelation,
            False,
            interval_count,
            1.0,
            0.0,
        )
    ####

    prefix_advantages = log_likelihood_advantage_from_squared_errors_by_prefix(
        data_winner.interval_squared_errors_rad2,
        (item.interval_squared_errors_rad2 for item in data_ranked[1:]),
        noise.sigma_rad,
    )
    final_delta = float(prefix_advantages[-1])
    minimum = min(config.evidence_min_intervals, interval_count)
    stable_count = 0
    earliest = interval_count
    sufficient = False
    for prefix in range(minimum, interval_count + 1):
        if prefix_advantages[prefix - 1] >= config.evidence_log_likelihood_ratio_threshold:
            stable_count += 1
        else:
            stable_count = 0
        ####
        if stable_count >= config.evidence_stability_windows:
            sufficient = True
            earliest = prefix
            break
        ####
    ####
    fraction = float(earliest / interval_count) if interval_count else 1.0
    return _EvidenceSummary(
        noise.sigma_rad,
        noise.retained_count,
        noise.lag1_autocorrelation,
        sufficient,
        earliest,
        fraction,
        final_delta,
    )
####
