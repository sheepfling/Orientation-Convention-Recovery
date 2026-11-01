"""Convention inference from sparse attitude anchors and densely sampled rates."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

from numpy import arange, clip, diff, int64, mean, square
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from .assessments import _dimension_assessments, _unresolved_dimensions
from .decoding import decode_attitudes, decode_rates
from .evidence import PERSISTENT_RESIDUAL_CORRELATION_LIMIT
from .heuristics import EulerHeuristicDiagnosticsCache
from .hypotheses import generate_hypotheses
from .integration import (
    IntegratedAnchorIntervals as _IntegratedAnchorIntervals,
)
from .integration import (
    ObservedIncrementCache,
    integrate_rates_between_anchors,
)
from .math_utils import FloatArray
from .model_common import (
    AngleUnit,
    AttitudeDirection,
    EulerMode,
    QuaternionOrder,
    RateFrame,
    RateUnit,
    confidence_label_for_score,
)
from .models import (
    ConventionHypothesis,
    InferenceConfig,
    InferenceReport,
    OverallDiagnostics,
)
from .ranking import _assign_ranks_and_support
from .reporting import _build_inference_report
from .scoring import ScoredCandidate, score_increment_sequences
from .statistics import (
    effective_excitation_rank,
    estimate_noise_process,
    log_likelihood_advantage_from_squared_errors_by_prefix,
    raw_rate_excitation,
)
from .trajectory import SparseOrientationRateData, TrajectoryData

__all__ = ["infer_sparse_orientation_trajectory"]


@dataclass(frozen=True, slots=True)
class _AttitudeCacheKey:
    """Fields that determine a decoded sparse attitude-anchor series."""

    direction: AttitudeDirection | str
    quaternion_order: QuaternionOrder | str | None
    euler_order: str | None
    euler_mode: EulerMode | str | None
    angle_unit: AngleUnit | str | None
####


@dataclass(frozen=True, slots=True)
class _RateCacheKey:
    """Fields that determine decoded rate samples, before frame composition."""

    order: str
    unit: RateUnit | str
    sign: int
####


@dataclass(frozen=True, slots=True)
class _SparseEvidenceSummary:
    """Evidence computed only from non-overlapping anchor-to-anchor intervals."""

    noise_sigma_rad: float
    noise_residual_count: int
    noise_lag1_autocorrelation: float
    sufficient: bool
    intervals_used: int
    fraction: float
    log_likelihood_ratio: float
    dependence_limited: bool
####


def infer_sparse_orientation_trajectory(
    data: SparseOrientationRateData,
    config: InferenceConfig | None = None,
) -> InferenceReport:
    """Infer conventions by integrating dense rates between sparse orientation anchors.

    Rate samples are treated as a piecewise-linear signal on their own timestamps.
    The returned ``sample_count`` is the number of orientation anchors; the report
    separately records ``rate_sample_count``. Sequential evidence uses only every
    other adjacent anchor interval so a shared endpoint anchor is not counted twice.
    """

    resolved_config = config or InferenceConfig()
    anchor_data = TrajectoryData(
        representation=data.representation,
        orientation=data.orientation_anchors,
        rates=None,
        time_s=data.orientation_anchor_times_s,
    )
    rate_data = TrajectoryData(
        representation=data.representation,
        orientation=None,
        rates=data.rates,
        time_s=data.rate_times_s,
    )
    hypotheses = generate_hypotheses(data.representation, resolved_config)
    euler_diagnostics_cache = EulerHeuristicDiagnosticsCache(anchor_data)
    attitude_cache: dict[_AttitudeCacheKey, Rotation] = {}
    observed_cache: dict[_AttitudeCacheKey, ObservedIncrementCache] = {}
    rate_cache: dict[_RateCacheKey, FloatArray] = {}
    integration_cache: dict[tuple[_RateCacheKey, RateFrame], _IntegratedAnchorIntervals] = {}
    scored: list[ScoredCandidate] = []
    failures: list[str] = []

    for hypothesis in hypotheses:
        try:
            attitude_key = _attitude_cache_key(hypothesis)
            attitudes = attitude_cache.get(attitude_key)
            if attitudes is None:
                attitudes = decode_attitudes(anchor_data, hypothesis)
                attitude_cache[attitude_key] = attitudes
                observed_cache[attitude_key] = ObservedIncrementCache(attitudes)
            ####
            rate_key = _rate_cache_key(hypothesis)
            rates_rad_s = rate_cache.get(rate_key)
            if rates_rad_s is None:
                rates_rad_s = decode_rates(rate_data, hypothesis)
                rate_cache[rate_key] = rates_rad_s
            ####
            frame = RateFrame(hypothesis.rate_frame)
            integration_key = (rate_key, frame)
            integrated = integration_cache.get(integration_key)
            if integrated is None:
                integrated = integrate_rates_between_anchors(
                    rates_rad_s,
                    data.rate_times_s,
                    data.orientation_anchor_times_s,
                    frame,
                    resolved_config.integration_substeps,
                )
                integration_cache[integration_key] = integrated
            ####
            attitude_observed_cache = observed_cache[attitude_key]
            scored.append(
                score_increment_sequences(
                    anchor_data,
                    hypothesis,
                    resolved_config,
                    attitudes=attitudes,
                    predicted=integrated.predicted_increments,
                    observed=attitude_observed_cache.interval_increments(frame),
                    path_angles=integrated.path_angles_rad,
                    rate_change_rms_angle_rad=integrated.rate_change_rms_angle_rad,
                    inferred_period=None,
                    time_warnings=(),
                    euler_diagnostics_cache=euler_diagnostics_cache,
                    observed_increment_cache=attitude_observed_cache,
                    retain_interval_error_rotvecs=True,
                )
            )
        except (ValueError, FloatingPointError) as error:
            failures.append(f"{hypothesis.identifier()}: {error}")
        ####
    ####

    if not scored:
        details = "\n".join(failures[:10])
        raise ValueError(f"All sparse-anchor convention hypotheses failed to score.\n{details}")
    ####

    ranked = _assign_ranks_and_support(scored, anchor_data, resolved_config)
    diagnostics = _build_sparse_overall_diagnostics(
        data,
        anchor_data,
        rate_data,
        ranked,
        resolved_config,
    )
    return _build_inference_report(
        anchor_data,
        hypotheses,
        ranked,
        resolved_config,
        diagnostics,
        observation_model="sparse_orientation_anchors",
        rate_sample_count=data.rate_sample_count,
    )
####


def _attitude_cache_key(hypothesis: ConventionHypothesis) -> _AttitudeCacheKey:
    return _AttitudeCacheKey(
        direction=hypothesis.attitude_direction,
        quaternion_order=hypothesis.quaternion_order,
        euler_order=hypothesis.euler_order,
        euler_mode=hypothesis.euler_mode,
        angle_unit=hypothesis.angle_unit,
    )
####


def _rate_cache_key(hypothesis: ConventionHypothesis) -> _RateCacheKey:
    return _RateCacheKey(
        order=hypothesis.rate_order,
        unit=hypothesis.rate_unit,
        sign=int(hypothesis.rate_sign),
    )
####


def _independent_anchor_interval_indices(interval_count: int) -> NDArray[int64]:
    """Select alternating adjacent-anchor intervals with no shared endpoints."""

    return arange(0, interval_count, 2, dtype=int64)
####


def _sparse_anchor_evidence(
    ranked: tuple[ScoredCandidate, ...],
    config: InferenceConfig,
) -> _SparseEvidenceSummary:
    """Evaluate sequential evidence without reusing adjacent orientation anchors."""

    representatives: dict[str, ScoredCandidate] = {}
    for item in sorted(ranked, key=lambda candidate: candidate.result.data_rank):
        representatives.setdefault(item.result.equivalence_class_id, item)
    ####
    data_ranked = tuple(representatives.values())
    data_winner = data_ranked[0]
    independent_indices = _independent_anchor_interval_indices(
        data_winner.result.valid_interval_count
    )
    winner_errors = data_winner.interval_error_rotvecs
    if winner_errors is None:
        raise RuntimeError("Sparse-anchor residual vectors were not retained for evidence")
    ####
    independent_errors = winner_errors[independent_indices]
    noise = estimate_noise_process(independent_errors, config.trim_fraction)
    interval_count = int(independent_indices.size)
    no_evidence = _SparseEvidenceSummary(
        noise_sigma_rad=noise.sigma_rad,
        noise_residual_count=noise.retained_count,
        noise_lag1_autocorrelation=noise.lag1_autocorrelation,
        sufficient=False,
        intervals_used=interval_count,
        fraction=1.0,
        log_likelihood_ratio=0.0,
        dependence_limited=False,
    )
    if data_winner.result.observed_step_rms_rad < config.slow_step_threshold_rad:
        return no_evidence
    ####
    if len(data_ranked) < 2:
        return no_evidence
    ####
    competitor_energies = (
        item.interval_squared_errors_rad2[independent_indices] for item in data_ranked[1:]
    )
    prefix_advantages = log_likelihood_advantage_from_squared_errors_by_prefix(
        data_winner.interval_squared_errors_rad2[independent_indices],
        competitor_energies,
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
    dependence_limited = (
        abs(noise.lag1_autocorrelation) >= PERSISTENT_RESIDUAL_CORRELATION_LIMIT
        and noise.sigma_rad > config.residual_floor_rad
    )
    if dependence_limited:
        sufficient = False
        earliest = interval_count
    ####
    return _SparseEvidenceSummary(
        noise_sigma_rad=noise.sigma_rad,
        noise_residual_count=noise.retained_count,
        noise_lag1_autocorrelation=noise.lag1_autocorrelation,
        sufficient=sufficient,
        intervals_used=earliest,
        fraction=float(earliest / interval_count) if interval_count else 1.0,
        log_likelihood_ratio=final_delta,
        dependence_limited=dependence_limited,
    )
####


def _build_sparse_overall_diagnostics(
    sparse_data: SparseOrientationRateData,
    anchor_data: TrajectoryData,
    rate_data: TrajectoryData,
    ranked: tuple[ScoredCandidate, ...],
    config: InferenceConfig,
) -> OverallDiagnostics:
    """Build conservative diagnostics with sample counts tied to their streams."""

    best = ranked[0]
    equivalence_classes = {
        item.result.equivalence_class_id: item.result.exact_equivalent_hypothesis_ids
        for item in ranked
    }
    evidence_near_ties = tuple(
        sorted(
            (
                item
                for item in ranked
                if item.result.data_delta_criterion <= config.near_tie_delta_criterion
            ),
            key=lambda item: item.result.data_rank,
        )
    )
    posterior_near_ties = tuple(
        item for item in ranked if item.result.delta_criterion <= config.near_tie_delta_criterion
    )
    unresolved = _unresolved_dimensions(evidence_near_ties)
    posterior_unresolved = _unresolved_dimensions(posterior_near_ties)
    evidence = _sparse_anchor_evidence(ranked, config)

    best_rates = decode_rates(rate_data, best.result.hypothesis)
    rate_durations = diff(sparse_data.rate_times_s)
    excitation = effective_excitation_rank(best_rates, rate_durations)
    raw_excitation = raw_rate_excitation(rate_data, config)
    best_integration = integrate_rates_between_anchors(
        best_rates,
        sparse_data.rate_times_s,
        sparse_data.orientation_anchor_times_s,
        RateFrame(best.result.hypothesis.rate_frame),
        config.integration_substeps,
    )
    path_angles = best_integration.path_angles_rad
    total_motion = float(sum(path_angles))
    step_rms = float(mean(square(path_angles)) ** 0.5)

    fit_factor = exp(-0.5 * (best.result.normalized_rmse / 0.25) ** 2)
    uniqueness_factor = min(1.0, best.result.data_relative_support / 0.90)
    step_factor = min(1.0, step_rms / config.slow_step_threshold_rad)
    total_factor = min(1.0, total_motion / config.slow_total_motion_threshold_rad)
    motion_factor = sqrt(step_factor * total_factor)
    alias_factor = max(
        0.02,
        1.0
        - best.result.hard_alias_fraction
        - 0.5 * (best.result.near_alias_fraction - best.result.hard_alias_fraction),
    )
    integration_factor = min(
        1.0,
        config.high_rate_change_angle_rad
        / max(config.high_rate_change_angle_rad, best.result.rate_change_rms_angle_rad),
    )
    excitation_factor = 0.30 + 0.70 * min(1.0, max(0.0, (excitation.effective_rank - 1.0) / 1.5))
    independent_interval_count = int(
        _independent_anchor_interval_indices(sparse_data.anchor_count - 1).size
    )
    sample_factor = min(1.0, independent_interval_count / 20.0) ** 0.25
    confidence = float(
        clip(
            fit_factor
            * uniqueness_factor
            * motion_factor
            * alias_factor
            * integration_factor
            * excitation_factor
            * sample_factor,
            0.0,
            1.0,
        )
    )
    if unresolved or (len(ranked) > 1 and not evidence.sufficient):
        confidence = min(confidence, 0.49)
    ####
    if best.result.hard_alias_fraction > 0.0:
        confidence = min(confidence, 0.35)
    ####
    if step_rms < config.slow_step_threshold_rad and total_motion < (
        config.slow_total_motion_threshold_rad
    ):
        confidence = min(confidence, 0.35)
    ####

    warnings: list[str] = [
        f"Inference used {sparse_data.anchor_count} orientation anchors and "
        f"{sparse_data.rate_sample_count} rate samples.",
        "Sequential evidence uses alternating anchor intervals to avoid reusing a shared "
        "orientation endpoint.",
        "Rate values are linearly interpolated between their timestamps; no extrapolation is used.",
    ]
    if len(best.result.exact_equivalent_hypothesis_ids) > 1:
        warnings.append(
            "The selected result is an exact observational equivalence class; attitude "
            "direction, rate frame, and global rate sign cannot be separated by this "
            "trajectory alone."
        )
    ####
    if len(evidence_near_ties) > 1:
        warnings.append(
            "Multiple convention hypotheses remain within the configured empirical evidence margin."
        )
    ####
    if evidence.dependence_limited:
        warnings.append(
            "Residuals from non-overlapping anchor intervals remain strongly temporally "
            "dependent at a measurable scale, so sequential evidence is withheld."
        )
    elif len(ranked) > 1 and not evidence.sufficient:
        warnings.append(
            "Non-overlapping anchor intervals did not meet the evidence stability threshold; "
            "collect more anchors or more informative motion."
        )
    ####
    if len(evidence_near_ties) > len(posterior_near_ties):
        warnings.append(
            "Soft priors select among evidence-near-tied hypotheses; the underlying "
            "empirical ambiguity remains."
        )
    ####
    if best.result.data_rank != 1:
        warnings.append("Soft priors changed the top-ranked convention from the data-only winner.")
    ####
    if step_rms < config.slow_step_threshold_rad:
        warnings.append("Integrated rotational path per anchor interval is small.")
    ####
    if total_motion < config.slow_total_motion_threshold_rad:
        warnings.append("The integrated rate stream contains little total rotational motion.")
    ####
    if excitation.effective_rank < 1.35:
        warnings.append(
            "Motion is nearly coaxial; frame and unused-axis permutations may be "
            "fundamentally unidentifiable."
        )
    elif excitation.effective_rank < 2.0:
        warnings.append("Rotational excitation does not strongly cover all three axes.")
    ####
    if best.result.hard_alias_fraction > 0.0:
        warnings.append(
            "One or more anchor intervals contain at least pi radians of integrated rate path; "
            "their endpoint orientations cannot uniquely reveal wrap count."
        )
    elif best.result.near_alias_fraction > 0.0:
        warnings.append("One or more anchor intervals approach the rotational alias boundary.")
    ####
    if best.result.rate_change_rms_angle_rad >= config.high_rate_change_angle_rad:
        warnings.append("Angular rate changes substantially between rate samples.")
    ####
    if best.result.normalized_rmse > 0.50:
        warnings.append(
            "Even the best convention has a large residual relative to observed motion."
        )
    ####
    if best.result.gimbal_lock_fraction is not None and best.result.gimbal_lock_fraction > 0.0:
        warnings.append("The winning Euler hypothesis passes near gimbal lock.")
    ####
    if abs(evidence.noise_lag1_autocorrelation) >= 0.25:
        warnings.append(
            "Residuals across non-overlapping anchor intervals show noticeable lag-one "
            "correlation; the iid Gaussian evidence scale should be treated as approximate."
        )
    ####

    return OverallDiagnostics(
        confidence=confidence,
        confidence_label=confidence_label_for_score(confidence),
        best_relative_support=best.result.relative_support,
        best_data_relative_support=best.result.data_relative_support,
        near_tie_count=len(evidence_near_ties),
        near_tie_hypothesis_ids=tuple(
            item.result.hypothesis.identifier() for item in evidence_near_ties[: config.top_k]
        ),
        unresolved_dimensions=unresolved,
        posterior_near_tie_count=len(posterior_near_ties),
        posterior_near_tie_hypothesis_ids=tuple(
            item.result.hypothesis.identifier() for item in posterior_near_ties[: config.top_k]
        ),
        posterior_unresolved_dimensions=posterior_unresolved,
        dimension_assessments=_dimension_assessments(
            sparse_data.representation,
            best,
            evidence_near_ties,
            posterior_near_ties,
            config,
        ),
        effective_excitation_rank=excitation.effective_rank,
        excitation_singular_values=excitation.singular_values,
        total_observed_motion_rad=total_motion,
        observed_step_rms_rad=step_rms,
        near_alias_fraction=best.result.near_alias_fraction,
        hard_alias_fraction=best.result.hard_alias_fraction,
        time_source="timestamps",
        warnings=tuple(warnings),
        estimated_noise_sigma_rad=evidence.noise_sigma_rad,
        noise_residual_count=evidence.noise_residual_count,
        noise_lag1_autocorrelation=evidence.noise_lag1_autocorrelation,
        evidence_sufficient=evidence.sufficient,
        evidence_intervals_used=evidence.intervals_used,
        evidence_fraction=evidence.fraction,
        evidence_log_likelihood_ratio=evidence.log_likelihood_ratio,
        exact_equivalence_class_count=len(equivalence_classes),
        best_equivalence_class_hypothesis_ids=best.result.exact_equivalent_hypothesis_ids,
        rate_scale_period_gauge_active=False,
        raw_excitation_rank=raw_excitation.rank,
        raw_excitation_eigenvalues=raw_excitation.eigenvalues,
        raw_excitation_condition_ratio=raw_excitation.weakest_axis_ratio,
    )
####
