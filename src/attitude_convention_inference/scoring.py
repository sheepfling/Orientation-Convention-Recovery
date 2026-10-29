"""Candidate kinematic scoring, without ranking or operational policy."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt

from numpy import amax, asarray, diff, float64, mean, median, ones, quantile, square, zeros
from numpy import sqrt as np_sqrt
from numpy import sum as np_sum
from numpy.linalg import norm
from scipy.spatial.transform import Rotation

from .core import (
    FloatArray,
    IncrementWindowComposer,
    TrajectoryData,
    decode_attitudes,
    decode_rates,
    integrate_interval_increments,
    observed_interval_increments,
    observed_window_increments,
    resolve_interval_durations,
)
from .heuristics import euler_heuristic_diagnostics as _euler_heuristic_diagnostics
from .heuristics import gimbal_lock_fraction as _gimbal_lock_fraction
from .heuristics import wrap_unit_diagnostics as _wrap_unit_diagnostics
from .math_utils import rotation_magnitudes as _rotation_magnitudes
from .math_utils import rotation_rotvecs as _rotation_rotvecs
from .math_utils import trimmed_mean_square as _trimmed_mean_square
from .models import (
    CandidateResult,
    ConventionHypothesis,
    HeuristicFeatures,
    InferenceConfig,
    LagMetric,
    RateFrame,
    ScaleRisk,
)

__all__ = [
    "IntervalDetails",
    "ScoredCandidate",
    "replace_result",
    "score_hypothesis",
    "score_partial_hypothesis",
]


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """A result with compact interval energy and optional detailed residuals."""

    result: CandidateResult
    interval_squared_errors_rad2: FloatArray
    interval_error_rotvecs: FloatArray | None = None
    interval_details: IntervalDetails | None = None
####


@dataclass(frozen=True, slots=True)
class IntervalDetails:
    """Large per-interval arrays retained only for an explicit diagnostics request."""

    rates_rad_s: FloatArray
    interval_durations_s: FloatArray
    predicted_intervals: Rotation
    observed_intervals: Rotation
    path_angles: FloatArray
####


def score_hypothesis(
    data: TrajectoryData,
    hypothesis: ConventionHypothesis,
    config: InferenceConfig,
    *,
    attitudes: Rotation | None = None,
    rates_rad_s: FloatArray | None = None,
    retain_interval_details: bool = False,
) -> ScoredCandidate:
    """Score one complete-stream hypothesis against geometric trajectory evidence."""

    attitudes = decode_attitudes(data, hypothesis) if attitudes is None else attitudes
    rates_rad_s = decode_rates(data, hypothesis) if rates_rad_s is None else rates_rad_s
    frame = RateFrame(hypothesis.rate_frame)
    duration_resolution = resolve_interval_durations(
        data,
        attitudes,
        rates_rad_s,
        frame,
        config,
    )
    durations = duration_resolution.durations_s
    inferred_period = duration_resolution.inferred_period_s
    time_warnings = duration_resolution.warnings
    integrated = integrate_interval_increments(
        rates_rad_s,
        duration_resolution.durations_s,
        frame,
        config.integration_substeps,
    )
    predicted = integrated.predicted_increments
    path_angles = integrated.path_angles_rad
    rate_change_angles = integrated.rate_change_angles_rad
    observed = observed_interval_increments(attitudes, frame)
    interval_errors = _rotation_rotvecs(predicted.inv() * observed)
    interval_error_angles = norm(interval_errors, axis=1)
    observed_interval_angles = _rotation_magnitudes(observed)

    valid_windows = tuple(
        window for window in config.window_sizes if window <= data.sample_count - 1
    )
    window_composer = IncrementWindowComposer.from_increments(predicted, frame)
    lag_metrics: list[LagMetric] = []
    scale_risks: list[ScaleRisk] = []

    for window_size in valid_windows:
        composition = window_composer.compose(window_size)
        observed_windows = observed_window_increments(
            attitudes,
            composition.starts,
            window_size,
            frame,
        )
        errors = _rotation_magnitudes(composition.window_increments.inv() * observed_windows)
        observed_angles = _rotation_magnitudes(observed_windows)
        scale = sqrt(float(window_size))
        scaled_errors = asarray(errors, dtype=float64) / scale
        scale_risks.append(
            ScaleRisk(
                window_size=window_size,
                trimmed_mse_rad2=_trimmed_mean_square(scaled_errors, config.trim_fraction),
                untrimmed_mse_rad2=float(mean(square(scaled_errors))),
                weight=1.0 / len(valid_windows),
            )
        )
        lag_metrics.append(
            LagMetric(
                window_size=window_size,
                window_count=int(errors.size),
                residual_rmse_rad=float(np_sqrt(mean(square(errors)))),
                residual_median_rad=float(median(errors)),
                residual_p95_rad=float(quantile(errors, 0.95)),
                observed_rotation_rms_rad=float(np_sqrt(mean(square(observed_angles)))),
            )
        )
    ####

    raw_geometric_risk = float(sum(item.weight * item.trimmed_mse_rad2 for item in scale_risks))
    untrimmed_geometric_risk = float(
        sum(item.weight * item.untrimmed_mse_rad2 for item in scale_risks)
    )
    multiscale_rmse = sqrt(raw_geometric_risk)
    # The criterion is only a monotone contrast transform for tie thresholds and
    # support display. Candidate ordering is the raw geometric risk above.
    data_criterion = log(max(raw_geometric_risk, config.residual_floor_rad**2))

    near_alias_fraction = float(mean(path_angles >= config.near_alias_step_rad))
    hard_alias_fraction = float(mean(path_angles >= config.hard_alias_step_rad))
    rate_change_rms = float(np_sqrt(mean(square(rate_change_angles))))
    warnings = list(time_warnings)
    if hard_alias_fraction > 0.0:
        warnings.append(
            "At least one sample interval contains an unwrapped rate integral of pi "
            "radians or more."
        )
    elif near_alias_fraction > 0.0:
        warnings.append("At least one sample interval is close to rotational aliasing.")
    ####
    if rate_change_rms >= config.high_rate_change_angle_rad:
        warnings.append(
            "Angular rate changes substantially within a sample interval; interpolation "
            "error may matter."
        )
    ####

    gimbal_fraction = _gimbal_lock_fraction(data, hypothesis, config)
    euler_diagnostics = _euler_heuristic_diagnostics(data, hypothesis)
    euler_range_penalty = euler_diagnostics.range_penalty_rad2
    euler_branch_switches = euler_diagnostics.branch_switch_count
    euler_singularity_margin = euler_diagnostics.singularity_margin
    euler_branch_cost = euler_diagnostics.branch_cost_rad2
    wrap_diagnostics = _wrap_unit_diagnostics(data, hypothesis)
    wrap_gain = wrap_diagnostics.gain_raw
    wrap_count = wrap_diagnostics.wrap_count
    wrap_status = wrap_diagnostics.status
    wrap_score = wrap_diagnostics.score
    if gimbal_fraction is not None and gimbal_fraction > 0.0:
        warnings.append("Some Euler samples are near this hypothesis's gimbal-lock surface.")
    ####
    if euler_range_penalty > 0.0:
        warnings.append(
            "Some Euler samples fall outside this hypothesis's principal output ranges; "
            "the represented rotations may still be valid under an unwrapped branch."
        )
    ####
    heuristic_features = HeuristicFeatures(
        branch_penalty=euler_range_penalty,
        singularity_penalty=(
            0.0 if euler_singularity_margin is None else max(0.0, 1.0 - euler_singularity_margin)
        ),
        coarse_risk=0.0,
        wrap_score=wrap_score,
        wrap_gain_raw=wrap_gain,
        inferred_wrap_count=wrap_count,
        wrap_unit_status=wrap_status,
        representation_score=float(wrap_score + euler_range_penalty + euler_branch_cost),
        representation_feature_vector={
            "wrap_score": float(wrap_score),
            "wrap_gain_raw": float(wrap_gain),
            "range_penalty_rad2": float(euler_range_penalty),
            "branch_cost_rad2": float(euler_branch_cost),
            "singularity_margin": (
                0.0 if euler_singularity_margin is None else float(euler_singularity_margin)
            ),
        },
        classifier_probability=None,
    )

    result = CandidateResult(
        hypothesis=hypothesis,
        criterion=float(data_criterion),
        data_criterion=float(data_criterion),
        fit_score=0.0,
        normalized_rmse=float(multiscale_rmse),
        multiscale_rmse_rad=float(multiscale_rmse),
        raw_geometric_risk=raw_geometric_risk,
        untrimmed_geometric_risk=untrimmed_geometric_risk,
        scale_risks=tuple(scale_risks),
        interval_rmse_rad=float(np_sqrt(mean(square(interval_error_angles)))),
        interval_median_rad=float(median(interval_error_angles)),
        interval_p95_rad=float(quantile(interval_error_angles, 0.95)),
        interval_max_rad=float(amax(interval_error_angles)),
        observed_step_rms_rad=float(np_sqrt(mean(square(observed_interval_angles)))),
        inferred_sample_period_s=inferred_period,
        valid_interval_count=data.sample_count - 1,
        near_alias_fraction=near_alias_fraction,
        hard_alias_fraction=hard_alias_fraction,
        rate_change_rms_angle_rad=rate_change_rms,
        gimbal_lock_fraction=gimbal_fraction,
        euler_range_penalty_rad2=euler_range_penalty,
        euler_branch_switch_count=euler_branch_switches,
        euler_singularity_margin=euler_singularity_margin,
        euler_branch_cost_rad2=euler_branch_cost,
        heuristic_features=heuristic_features,
        lag_metrics=tuple(lag_metrics),
        warnings=tuple(warnings),
    )
    return ScoredCandidate(
        result=result,
        interval_squared_errors_rad2=asarray(
            np_sum(square(interval_errors), axis=1), dtype=float64
        ),
        interval_error_rotvecs=(
            asarray(interval_errors, dtype=float64) if retain_interval_details else None
        ),
        interval_details=(
            IntervalDetails(
                rates_rad_s=rates_rad_s,
                interval_durations_s=durations,
                predicted_intervals=predicted,
                observed_intervals=observed,
                path_angles=path_angles,
            )
            if retain_interval_details
            else None
        ),
    )
####


def replace_result(
    item: ScoredCandidate,
    result: CandidateResult,
) -> ScoredCandidate:
    """Update ranking metadata while retaining a candidate's numerical state."""

    return ScoredCandidate(
        result=result,
        interval_squared_errors_rad2=item.interval_squared_errors_rad2,
        interval_error_rotvecs=item.interval_error_rotvecs,
        interval_details=item.interval_details,
    )
####


def score_partial_hypothesis(
    data: TrajectoryData,
    hypothesis: ConventionHypothesis,
    config: InferenceConfig,
    *,
    retain_interval_details: bool = False,
) -> ScoredCandidate:
    """Create an explicit heuristic/non-identifying result for partial streams."""

    sample_count = data.sample_count
    interval_count = sample_count - 1
    if data.orientation is not None:
        euler_diagnostics = _euler_heuristic_diagnostics(data, hypothesis)
        euler_range_penalty = euler_diagnostics.range_penalty_rad2
        euler_branch_switches = euler_diagnostics.branch_switch_count
        euler_singularity_margin = euler_diagnostics.singularity_margin
        euler_branch_cost = euler_diagnostics.branch_cost_rad2
        wrap_diagnostics = _wrap_unit_diagnostics(data, hypothesis)
        wrap_gain = wrap_diagnostics.gain_raw
        wrap_count = wrap_diagnostics.wrap_count
        wrap_status = wrap_diagnostics.status
        wrap_score = wrap_diagnostics.score
        representation_score = wrap_score + euler_range_penalty + euler_branch_cost
        gimbal_fraction = _gimbal_lock_fraction(data, hypothesis, config)
        heuristic_features = HeuristicFeatures(
            wrap_score=wrap_score,
            wrap_gain_raw=wrap_gain,
            inferred_wrap_count=wrap_count,
            wrap_unit_status=wrap_status,
            branch_penalty=euler_range_penalty,
            singularity_penalty=(
                0.0
                if euler_singularity_margin is None
                else max(0.0, 1.0 - euler_singularity_margin)
            ),
            representation_score=float(representation_score),
            representation_feature_vector={
                "wrap_score": float(wrap_score),
                "range_penalty_rad2": float(euler_range_penalty),
                "branch_cost_rad2": float(euler_branch_cost),
            },
        )
    else:
        euler_range_penalty = 0.0
        euler_branch_switches = 0
        euler_singularity_margin = None
        euler_branch_cost = 0.0
        gimbal_fraction = None
        representation_score = 0.0
        heuristic_features = HeuristicFeatures()
    ####

    squared_errors = zeros(interval_count, dtype=float64)
    score = max(float(representation_score), 0.0)
    result = CandidateResult(
        hypothesis=hypothesis,
        criterion=log(max(score, config.residual_floor_rad**2)),
        data_criterion=log(max(score, config.residual_floor_rad**2)),
        fit_score=0.0,
        normalized_rmse=0.0,
        multiscale_rmse_rad=sqrt(score),
        raw_geometric_risk=score,
        untrimmed_geometric_risk=score,
        scale_risks=(),
        interval_rmse_rad=0.0,
        interval_median_rad=0.0,
        interval_p95_rad=0.0,
        interval_max_rad=0.0,
        observed_step_rms_rad=0.0,
        valid_interval_count=interval_count,
        near_alias_fraction=0.0,
        hard_alias_fraction=0.0,
        rate_change_rms_angle_rad=0.0,
        lag_metrics=(),
        inferred_sample_period_s=None,
        gimbal_lock_fraction=gimbal_fraction,
        euler_range_penalty_rad2=euler_range_penalty,
        euler_branch_switch_count=euler_branch_switches,
        euler_singularity_margin=euler_singularity_margin,
        euler_branch_cost_rad2=euler_branch_cost,
        heuristic_features=heuristic_features,
    )
    details = None
    if retain_interval_details:
        rates = (
            zeros((sample_count, 3), dtype=float64)
            if data.orientation is not None
            else decode_rates(data, hypothesis)
        )
        durations = (
            diff(data.time_s) if data.time_s is not None else ones(interval_count, dtype=float64)
        )
        details = IntervalDetails(
            rates_rad_s=rates,
            interval_durations_s=durations,
            predicted_intervals=Rotation.identity(interval_count),
            observed_intervals=Rotation.identity(interval_count),
            path_angles=zeros(interval_count, dtype=float64),
        )
    ####
    return ScoredCandidate(
        result=result,
        interval_squared_errors_rad2=squared_errors,
        interval_error_rotvecs=(
            zeros((interval_count, 3), dtype=float64) if retain_interval_details else None
        ),
        interval_details=details,
    )
####
