"""Resolve sample intervals from timestamps or trajectory-consistency estimates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

import scipy.optimize as scipy_optimize
from numpy import (
    abs,
    argmin,
    asarray,
    clip,
    diff,
    float64,
    full,
    geomspace,
    mean,
    quantile,
    sqrt,
)
from numpy import any as np_any
from numpy import isfinite as np_isfinite
from numpy import sum as np_sum
from scipy.spatial.transform import Rotation

from .integration import (
    IncrementWindowComposer,
    ObservedIncrementCache,
    integrate_interval_increments,
    observed_interval_increments,
    observed_window_increments,
)
from .math_utils import FloatArray, rotation_magnitudes, trimmed_mean_square
from .models import InferenceConfig, RateFrame
from .trajectory import TrajectoryData

__all__ = [
    "IntervalDurationResolution",
    "SamplePeriodEstimate",
    "infer_constant_sample_period",
    "refine_constant_sample_period",
    "resolve_interval_durations",
]


@dataclass(frozen=True, slots=True)
class SamplePeriodEstimate:
    """Estimated constant sample period and diagnostic warnings."""

    period_s: float
    warnings: tuple[str, ...]
####


@dataclass(frozen=True, slots=True)
class IntervalDurationResolution:
    """Resolved interval durations together with their provenance and warnings."""

    durations_s: FloatArray
    inferred_period_s: float | None
    source: str
    warnings: tuple[str, ...]
####


class _ScalarMinimizeResult(Protocol):
    success: bool
    x: float
####


_typed_minimize_scalar = cast(
    Callable[..., _ScalarMinimizeResult],
    getattr(scipy_optimize, "minimize_scalar"),  # noqa: B009
)


def infer_constant_sample_period(
    attitudes: Rotation,
    rates_rad_s: FloatArray,
    frame: RateFrame,
    config: InferenceConfig,
    *,
    observed_increment_cache: ObservedIncrementCache | None = None,
) -> SamplePeriodEstimate:
    """Estimate one constant sample period from observed rotations and rates."""

    if observed_increment_cache is not None and observed_increment_cache.attitudes is not attitudes:
        raise ValueError("Observed-increment cache must belong to the estimated attitudes")
    ####
    observed_rotations = (
        observed_interval_increments(attitudes, frame)
        if observed_increment_cache is None
        else observed_increment_cache.interval_increments(frame)
    )
    observed = asarray(cast(Any, observed_rotations).as_rotvec(), dtype=float64)
    midpoint_rates = 0.5 * (rates_rad_s[:-1] + rates_rad_s[1:])
    rate_energy = np_sum(midpoint_rates * midpoint_rates, axis=1)
    valid = rate_energy > 1.0e-20
    warnings: list[str] = []

    lower, configured_upper = config.inferred_sample_period_bounds_s
    if not np_any(valid):
        warnings.append(
            "Sample period is unidentifiable because all candidate rates are near zero."
        )
        return SamplePeriodEstimate(lower, tuple(warnings))
    ####

    numerator = float(np_sum(midpoint_rates[valid] * observed[valid]))
    denominator = float(np_sum(rate_energy[valid]))
    estimate = numerator / denominator

    rate_norm = sqrt(rate_energy[valid])
    representative_rate = float(quantile(rate_norm, 0.99))
    no_alias_upper = (
        config.near_alias_step_rad / representative_rate
        if representative_rate > 1.0e-15
        else configured_upper
    )
    upper = max(lower * 1.01, min(configured_upper, no_alias_upper))

    if not np_isfinite(estimate) or estimate <= 0.0:
        warnings.append(
            "The unconstrained sample-period fit was nonpositive; this hypothesis is "
            "directionally inconsistent."
        )
        estimate = lower
    ####
    if estimate > upper:
        warnings.append(
            "The fitted sample period crossed the no-alias bound and was clipped; "
            "timing is ambiguous."
        )
    ####
    estimate = float(clip(estimate, lower, upper))
    return SamplePeriodEstimate(estimate, tuple(warnings))
####


def _period_geometric_risk(
    period_s: float,
    attitudes: Rotation,
    rates_rad_s: FloatArray,
    frame: RateFrame,
    config: InferenceConfig,
    observed_increment_cache: ObservedIncrementCache | None = None,
) -> float:
    """Evaluate the same raw multiscale geometric risk used for ranking."""

    durations = full(rates_rad_s.shape[0] - 1, period_s, dtype=float64)
    integrated = integrate_interval_increments(
        rates_rad_s,
        durations,
        frame,
        config.integration_substeps,
    )
    window_composer = IncrementWindowComposer.from_increments(
        integrated.predicted_increments,
        frame,
    )
    scale_risks: list[float] = []
    for window_size in config.window_sizes:
        if window_size > durations.size:
            continue
        ####
        composition = window_composer.compose(window_size)
        observed_windows = (
            observed_window_increments(attitudes, composition.starts, window_size, frame)
            if observed_increment_cache is None
            else observed_increment_cache.window_composition(window_size, frame).window_increments
        )
        scaled_errors = rotation_magnitudes(
            composition.window_increments.inv() * observed_windows
        ) / sqrt(float(window_size))
        scale_risks.append(trimmed_mean_square(scaled_errors, config.trim_fraction))
    ####
    return float(mean(scale_risks)) if scale_risks else float("inf")
####


def refine_constant_sample_period(
    attitudes: Rotation,
    rates_rad_s: FloatArray,
    frame: RateFrame,
    config: InferenceConfig,
    *,
    observed_increment_cache: ObservedIncrementCache | None = None,
) -> SamplePeriodEstimate:
    """Seed from the linear estimate and refine every promising group-risk basin."""

    if observed_increment_cache is not None and observed_increment_cache.attitudes is not attitudes:
        raise ValueError("Observed-increment cache must belong to the refined attitudes")
    ####
    initial_estimate = infer_constant_sample_period(
        attitudes,
        rates_rad_s,
        frame,
        config,
        observed_increment_cache=observed_increment_cache,
    )
    lower, upper = config.inferred_sample_period_bounds_s
    # A logarithmic grid covers broad timing ranges without spending dozens of
    # integrations on every discrete convention. The linear initializer receives
    # its own local refinement because it is usually the physically relevant basin.
    grid = geomspace(lower, upper, 9, dtype=float64)
    values = asarray(
        [
            _period_geometric_risk(
                float(period),
                attitudes,
                rates_rad_s,
                frame,
                config,
                observed_increment_cache,
            )
            for period in grid
        ],
        dtype=float64,
    )
    candidate_periods = [lower, upper, initial_estimate.period_s]

    def refine(lower_bound: float, upper_bound: float) -> None:
        def objective(period: float) -> float:
            return _period_geometric_risk(
                float(period),
                attitudes,
                rates_rad_s,
                frame,
                config,
                observed_increment_cache,
            )
        ####

        refined = _typed_minimize_scalar(
            objective,
            bounds=(lower_bound, upper_bound),
            method="bounded",
            options={"xatol": max(1.0e-12, (upper - lower) * 1.0e-10)},
        )
        if refined.success:
            candidate_periods.append(float(refined.x))
        ####
    ####

    refine(
        max(lower, initial_estimate.period_s / 1.5),
        min(upper, initial_estimate.period_s * 1.5),
    )
    coarse_index = int(argmin(values))
    if 0 < coarse_index < grid.size - 1:
        refine(float(grid[coarse_index - 1]), float(grid[coarse_index + 1]))
    ####

    best = min(
        candidate_periods,
        key=lambda period: _period_geometric_risk(
            period,
            attitudes,
            rates_rad_s,
            frame,
            config,
            observed_increment_cache,
        ),
    )
    warnings = list(initial_estimate.warnings)
    bound_tolerance = max(1.0e-12, (upper - lower) * 1.0e-6)
    if abs(best - lower) <= bound_tolerance or abs(best - upper) <= bound_tolerance:
        warnings.append(
            "The full geometric sample-period fit selected an active search bound; "
            "timing remains ambiguous."
        )
    ####
    return SamplePeriodEstimate(float(best), tuple(warnings))
####


def resolve_interval_durations(
    data: TrajectoryData,
    attitudes: Rotation,
    rates_rad_s: FloatArray,
    frame: RateFrame,
    config: InferenceConfig,
    *,
    observed_increment_cache: ObservedIncrementCache | None = None,
) -> IntervalDurationResolution:
    """Resolve interval timing in order: timestamps, configured, or inferred."""

    if observed_increment_cache is not None and observed_increment_cache.attitudes is not attitudes:
        raise ValueError("Observed-increment cache must belong to the resolved attitudes")
    ####
    if data.time_s is not None:
        return IntervalDurationResolution(diff(data.time_s), None, "timestamps", ())
    ####
    if config.sample_period_s is not None:
        intervals = full(data.sample_count - 1, config.sample_period_s, dtype=float64)
        return IntervalDurationResolution(intervals, None, "provided_sample_period", ())
    ####
    if not config.infer_sample_period_when_missing:
        raise ValueError(
            "No timestamps or sample_period_s were supplied and sample-period "
            "inference is disabled."
        )
    ####

    estimate = refine_constant_sample_period(
        attitudes,
        rates_rad_s,
        frame,
        config,
        observed_increment_cache=observed_increment_cache,
    )
    intervals = full(data.sample_count - 1, estimate.period_s, dtype=float64)
    return IntervalDurationResolution(
        intervals,
        estimate.period_s,
        "inferred_sample_period",
        estimate.warnings,
    )
####
