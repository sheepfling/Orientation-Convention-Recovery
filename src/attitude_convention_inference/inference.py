"""Inference orchestration, policy selection, and report assembly."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isclose, log, sqrt

from numpy import clip, mean, square
from numpy import (
    sqrt as np_sqrt,
)
from numpy import (
    sum as np_sum,
)
from numpy.linalg import norm
from scipy.spatial.transform import Rotation

from .core import (
    FloatArray,
    TrajectoryData,
    decode_attitudes,
    decode_rates,
)
from .hypotheses import DIMENSIONS as _DIMENSIONS
from .hypotheses import generate_hypotheses
from .hypotheses import hypothesis_count as _hypothesis_count
from .hypotheses import relevant_dimensions as _relevant_dimensions
from .hypotheses import search_values as _search_values
from .math_utils import rotation_magnitudes as _rotation_magnitudes
from .math_utils import rotation_quaternions as _rotation_quaternions
from .models import (
    AttitudeRepresentation,
    ConfidenceLabel,
    ConventionHypothesis,
    DimensionAssessment,
    DimensionStatus,
    HypothesisSpaceSummary,
    InferenceConfig,
    InferenceReport,
    OverallDiagnostics,
    ResultLabel,
)
from .ranking import _assign_ranks_and_support, _canonical_key, _hypothesis_value
from .scoring import ScoredCandidate as _ScoredCandidate
from .scoring import score_hypothesis as _score_hypothesis
from .scoring import score_partial_hypothesis as _score_partial_hypothesis
from .statistics import effective_excitation_rank as _effective_excitation_rank
from .statistics import (
    estimate_noise_process,
    log_likelihood_advantage_from_squared_errors_by_prefix,
)
from .statistics import raw_rate_excitation as _raw_rate_excitation

__all__ = [
    "best_interval_diagnostics",
    "estimate_noise_process",
    "generate_hypotheses",
    "infer_trajectory_conventions",
]


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


@dataclass(frozen=True, slots=True)
class _OperationalSelection:
    """Operational class and hypothesis selection with policy explanations."""

    class_id: str
    hypothesis_id: str
    reason: str
    domain_prior_abstained: bool
    domain_prior_supported_class_ids: tuple[str, ...]
    preference_profile_abstained: bool
    preference_profile_supported_class_ids: tuple[str, ...]
    representative_reason: str
####


def _result_label(data: TrajectoryData, config: InferenceConfig) -> ResultLabel:
    if data.orientation is None and data.rates is None:
        return ResultLabel.NO_DATA
    ####
    if data.rates is None:
        return ResultLabel.ATTITUDE_HEURISTIC
    ####
    if data.orientation is None:
        return ResultLabel.RATE_NON_ID
    ####
    if data.time_s is None and config.sample_period_s is None:
        return ResultLabel.SCALE_GAUGE
    ####
    return ResultLabel.TRAJECTORY_QUOTIENT
####


def _evidence_contender_classes(
    ranked: tuple[_ScoredCandidate, ...],
    config: InferenceConfig,
) -> tuple[str, ...]:
    """Return distinct quotient classes within the configured data-risk margin."""

    representatives: dict[str, _ScoredCandidate] = {}
    for item in ranked:
        representatives.setdefault(item.result.equivalence_class_id, item)
    ####
    ordered = sorted(
        representatives.values(),
        key=lambda item: (
            item.result.class_data_risk,
            item.result.equivalence_class_id,
            _canonical_key(item.result.hypothesis),
        ),
    )
    if not ordered:
        return ()
    ####
    minimum = max(ordered[0].result.class_data_risk, config.residual_floor_rad**2)
    minimum_criterion = log(minimum)
    return tuple(
        item.result.equivalence_class_id
        for item in ordered
        if log(max(item.result.class_data_risk, config.residual_floor_rad**2)) - minimum_criterion
        <= config.near_tie_delta_criterion
    )
####


def _select_operational_result(
    ranked: tuple[_ScoredCandidate, ...],
    config: InferenceConfig,
) -> _OperationalSelection:
    """Select an operational class only after preserving data contenders."""

    contender_ids = _evidence_contender_classes(ranked, config)
    by_class: dict[str, list[_ScoredCandidate]] = {}
    for item in ranked:
        by_class.setdefault(item.result.equivalence_class_id, []).append(item)
    ####
    domain_prior = config.domain_prior
    profile = config.preference_profile
    domain_supported = tuple(
        class_id
        for class_id in contender_ids
        if float(domain_prior.class_weights.get(class_id, 0.0)) > 0.0
    )
    profile_supported = tuple(
        class_id
        for class_id in contender_ids
        if float(profile.class_weights.get(class_id, 0.0)) > 0.0
    )
    effective_class_scores: dict[str, float] = {}
    for class_id in contender_ids:
        if class_id in profile.class_weights:
            effective_weight = float(profile.class_weights[class_id])
        elif class_id in domain_prior.class_weights:
            effective_weight = float(domain_prior.class_weights[class_id])
        else:
            effective_weight = 0.0
        ####
        if effective_weight <= 0.0:
            continue
        ####
        score = 0.0
        if class_id in domain_prior.class_weights:
            domain_weight = float(domain_prior.class_weights[class_id])
            if domain_weight > 0.0:
                score += float(domain_prior.strength) * log(domain_weight)
            ####
        ####
        if class_id in profile.class_weights:
            profile_weight = float(profile.class_weights[class_id])
            if profile_weight > 0.0:
                score += float(profile.strength) * log(profile_weight)
            ####
        ####
        effective_class_scores[class_id] = score
    ####
    supported = tuple(class_id for class_id in contender_ids if class_id in effective_class_scores)
    abstained = False
    profile_abstained = False
    if len(contender_ids) == 1:
        selected_class = contender_ids[0]
        selected_members = by_class[selected_class]
        reason = (
            "exact-gauge-class"
            if len(selected_members[0].result.exact_equivalent_hypothesis_ids) > 1
            else "trajectory-unique"
        )
    elif supported:
        selected_class = max(
            supported,
            key=lambda class_id: (
                effective_class_scores[class_id],
                class_id,
            ),
        )
        reason = "preference-profile" if profile.class_configured() else "domain-prior"
        selected_members = by_class[selected_class]
    else:
        selected_class = (
            contender_ids[0] if contender_ids else ranked[0].result.equivalence_class_id
        )
        reason = "canonical-policy"
        abstained = domain_prior.configured()
        profile_abstained = profile.class_configured()
        selected_members = by_class[selected_class]
    ####
    canonical = min(selected_members, key=lambda item: _canonical_key(item.result.hypothesis))
    operational = canonical
    representative_reason = "canonical-policy"
    if profile.representative_configured():

        def preference_log_weight(item: _ScoredCandidate) -> float:
            total = 0.0
            hypothesis = item.result.hypothesis
            for dimension in _DIMENSIONS:
                value = _hypothesis_value(hypothesis, dimension)
                if value is None:
                    continue
                ####
                weights = getattr(profile.representative_preferences, dimension)
                total += float(profile.strength) * log(float(weights.get(value, 1.0)))
            ####
            return total
        ####

        preference_scores = {
            item.result.hypothesis.identifier(): preference_log_weight(item)
            for item in selected_members
        }
        maximum = max(preference_scores.values())
        preferred_members = [
            item
            for item in selected_members
            if isclose(
                preference_scores[item.result.hypothesis.identifier()],
                maximum,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ]
        operational = min(
            preferred_members,
            key=lambda item: _canonical_key(item.result.hypothesis),
        )
        if operational.result.hypothesis.identifier() != canonical.result.hypothesis.identifier():
            representative_reason = "preference-profile"
        ####
    ####
    return _OperationalSelection(
        class_id=selected_class,
        hypothesis_id=operational.result.hypothesis.identifier(),
        reason=reason,
        domain_prior_abstained=abstained,
        domain_prior_supported_class_ids=domain_supported,
        preference_profile_abstained=profile_abstained,
        preference_profile_supported_class_ids=profile_supported,
        representative_reason=representative_reason,
    )
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


def _hypothesis_dimensions(hypothesis: ConventionHypothesis) -> dict[str, str]:
    values = hypothesis.model_dump(mode="json")
    return {key: str(value) for key, value in values.items() if value is not None}
####


def _unresolved_dimensions(
    near_ties: tuple[_ScoredCandidate, ...],
) -> dict[str, tuple[str, ...]]:
    gathered: dict[str, set[str]] = {}
    for item in near_ties:
        for key, value in _hypothesis_dimensions(item.result.hypothesis).items():
            gathered.setdefault(key, set()).add(value)
        ####
    ####
    return {key: tuple(sorted(values)) for key, values in gathered.items() if len(values) > 1}
####


def _relative_prior_weights(
    dimension: str,
    allowed_values: tuple[object, ...],
    config: InferenceConfig,
) -> dict[str, float]:
    configured = getattr(config.priors, dimension)
    if not configured:
        return {}
    ####
    weights = {str(value): float(configured.get(value, 1.0)) for value in allowed_values}
    maximum = max(weights.values())
    return {key: weight / maximum for key, weight in weights.items()}
####


def _hypothesis_space_summary(
    representation: AttitudeRepresentation,
    config: InferenceConfig,
) -> HypothesisSpaceSummary:
    base = config.search_space
    constrained = config.resolved_search_space()
    relevant = set(_relevant_dimensions(representation))
    fixed: dict[str, str] = {}
    searched: dict[str, tuple[str, ...]] = {}
    prior_weights: dict[str, dict[str, float]] = {}

    for dimension in _DIMENSIONS:
        if dimension not in relevant:
            continue
        ####
        values = _search_values(constrained, dimension)
        string_values = tuple(str(value) for value in values)
        if len(values) == 1:
            fixed[dimension] = string_values[0]
        else:
            searched[dimension] = string_values
        ####
        relative = _relative_prior_weights(dimension, values, config)
        if relative:
            prior_weights[dimension] = relative
        ####
    ####

    base_count = _hypothesis_count(representation, base)
    constrained_count = _hypothesis_count(representation, constrained)
    return HypothesisSpaceSummary(
        base_hypothesis_count=base_count,
        constrained_hypothesis_count=constrained_count,
        reduction_fraction=1.0 - constrained_count / base_count,
        fixed_dimensions=fixed,
        searched_dimensions=searched,
        not_applicable_dimensions=tuple(
            dimension for dimension in _DIMENSIONS if dimension not in relevant
        ),
        relative_prior_weights=prior_weights,
    )
####


def _dimension_values(
    candidates: tuple[_ScoredCandidate, ...],
    dimension: str,
) -> tuple[str, ...]:
    values = {
        str(value)
        for item in candidates
        if (value := _hypothesis_value(item.result.hypothesis, dimension)) is not None
    }
    return tuple(sorted(values))
####


def _dimension_assessments(
    representation: AttitudeRepresentation,
    best: _ScoredCandidate,
    evidence_near_ties: tuple[_ScoredCandidate, ...],
    posterior_near_ties: tuple[_ScoredCandidate, ...],
    config: InferenceConfig,
) -> dict[str, DimensionAssessment]:
    search = config.resolved_search_space()
    relevant = set(_relevant_dimensions(representation))
    assessments: dict[str, DimensionAssessment] = {}

    for dimension in _DIMENSIONS:
        if dimension not in relevant:
            assessments[dimension] = DimensionAssessment(
                status=DimensionStatus.NOT_APPLICABLE,
                selected_value=None,
                allowed_values=(),
                evidence_near_tie_values=(),
                posterior_near_tie_values=(),
            )
            continue
        ####

        allowed_raw = _search_values(search, dimension)
        allowed = tuple(str(value) for value in allowed_raw)
        selected_raw = _hypothesis_value(best.result.hypothesis, dimension)
        selected = None if selected_raw is None else str(selected_raw)
        evidence_values = _dimension_values(evidence_near_ties, dimension)
        posterior_values = _dimension_values(posterior_near_ties, dimension)

        if len(allowed) == 1:
            status = DimensionStatus.FIXED
        elif len(evidence_values) > 1:
            status = DimensionStatus.AMBIGUOUS
        else:
            status = DimensionStatus.INFERRED
        ####

        assessments[dimension] = DimensionAssessment(
            status=status,
            selected_value=selected,
            allowed_values=allowed,
            evidence_near_tie_values=evidence_values,
            posterior_near_tie_values=posterior_values,
            relative_prior_weights=_relative_prior_weights(
                dimension,
                allowed_raw,
                config,
            ),
        )
    ####
    return assessments
####


def _confidence_label(value: float) -> ConfidenceLabel:
    if value >= 0.75:
        return ConfidenceLabel.HIGH
    ####
    if value >= 0.50:
        return ConfidenceLabel.MODERATE
    ####
    if value >= 0.20:
        return ConfidenceLabel.LOW
    ####
    return ConfidenceLabel.INDETERMINATE
####


def _build_overall_diagnostics(
    data: TrajectoryData,
    ranked: tuple[_ScoredCandidate, ...],
    config: InferenceConfig,
) -> OverallDiagnostics:
    best = ranked[0]
    equivalence_classes = {
        item.result.equivalence_class_id: item.result.exact_equivalent_hypothesis_ids
        for item in ranked
    }
    rate_scale_period_gauge_active = (
        data.time_s is None
        and config.sample_period_s is None
        and len(config.resolved_search_space().rate_units) > 1
    )
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
    detailed = (
        _score_partial_hypothesis(
            data, best.result.hypothesis, config, retain_interval_details=True
        )
        if data.orientation is None or data.rates is None
        else _score_hypothesis(data, best.result.hypothesis, config, retain_interval_details=True)
    )
    details = detailed.interval_details
    if details is None:
        raise RuntimeError("Interval details were not retained for diagnostics")
    ####
    observed_angles = _rotation_magnitudes(details.observed_intervals)
    total_motion = float(np_sum(observed_angles))
    step_rms = float(np_sqrt(mean(square(observed_angles))))
    excitation = _effective_excitation_rank(
        details.rates_rad_s,
        details.interval_durations_s,
    )
    raw_excitation = _raw_rate_excitation(
        data,
        config,
    )
    evidence = _evidence_summary(data, ranked, detailed, config)
    effective_rank = excitation.effective_rank
    singular_values = excitation.singular_values
    raw_excitation_rank = raw_excitation.rank
    raw_excitation_eigenvalues = raw_excitation.eigenvalues
    raw_excitation_ratio = raw_excitation.weakest_axis_ratio
    noise_sigma = evidence.noise_sigma_rad
    noise_residual_count = evidence.noise_residual_count
    noise_autocorrelation = evidence.noise_lag1_autocorrelation
    evidence_sufficient = evidence.sufficient
    evidence_intervals_used = evidence.intervals_used
    evidence_fraction = evidence.fraction
    evidence_log_likelihood_ratio = evidence.log_likelihood_ratio

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
    excitation_factor = 0.30 + 0.70 * min(1.0, max(0.0, (effective_rank - 1.0) / 1.5))
    sample_factor = min(1.0, (data.sample_count - 1) / 20.0) ** 0.25

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

    if unresolved:
        confidence = min(confidence, 0.49)
    ####
    if len(ranked) > 1 and not evidence_sufficient:
        confidence = min(confidence, 0.49)
    ####
    if best.result.hard_alias_fraction > 0.0:
        confidence = min(confidence, 0.35)
    ####
    if (
        step_rms < config.slow_step_threshold_rad
        and total_motion < config.slow_total_motion_threshold_rad
    ):
        confidence = min(confidence, 0.35)
    ####

    warnings: list[str] = []
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
    if len(ranked) > 1 and not evidence_sufficient:
        warnings.append(
            "The leading convention did not achieve a stable evidence threshold before "
            "the end of the trajectory; use the full file or collect more informative motion."
        )
    elif len(ranked) > 1 and evidence_intervals_used < data.sample_count - 1:
        warnings.append(
            f"The leading convention separated after {evidence_intervals_used} of "
            f"{data.sample_count - 1} intervals; later samples remain useful for validation."
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
        warnings.append(
            "Per-sample attitude motion is small, so convention evidence is noise-sensitive."
        )
    ####
    if total_motion < config.slow_total_motion_threshold_rad:
        warnings.append("The trajectory contains little total rotational excitation.")
    ####
    if effective_rank < 1.35:
        warnings.append(
            "Motion is nearly coaxial; frame and unused-axis permutations may be "
            "fundamentally unidentifiable."
        )
    elif effective_rank < 2.0:
        warnings.append("Rotational excitation does not strongly cover all three axes.")
    ####
    if best.result.hard_alias_fraction > 0.0:
        warnings.append(
            "One or more rate integrals exceed pi radians per sample; orientation "
            "samples cannot uniquely reveal wrap count."
        )
    elif best.result.near_alias_fraction > 0.0:
        warnings.append("One or more intervals approach the pi-radian alias boundary.")
    ####
    if best.result.rate_change_rms_angle_rad >= config.high_rate_change_angle_rad:
        warnings.append("Within-step rate variation is large relative to the sampling interval.")
    ####
    if best.result.normalized_rmse > 0.50:
        warnings.append(
            "Even the best convention has a large residual relative to the observed motion."
        )
    ####
    if rate_scale_period_gauge_active:
        warnings.append(
            "Without timestamps, rate-unit scale and sample period form an exact gauge "
            "unless external bounds or metadata anchor one of them."
        )
    ####
    if best.result.gimbal_lock_fraction is not None and best.result.gimbal_lock_fraction > 0.0:
        warnings.append("The winning Euler hypothesis passes near gimbal lock.")
    ####
    if abs(noise_autocorrelation) >= 0.25:
        warnings.append(
            "Residuals show noticeable lag-one correlation; the iid Gaussian evidence "
            "scale should be treated as approximate."
        )
    ####

    time_source = (
        "timestamps"
        if data.time_s is not None
        else "provided_sample_period"
        if config.sample_period_s is not None
        else "inferred_sample_period"
    )
    return OverallDiagnostics(
        confidence=confidence,
        confidence_label=_confidence_label(confidence),
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
            data.representation,
            best,
            evidence_near_ties,
            posterior_near_ties,
            config,
        ),
        effective_excitation_rank=effective_rank,
        excitation_singular_values=singular_values,
        total_observed_motion_rad=total_motion,
        observed_step_rms_rad=step_rms,
        near_alias_fraction=best.result.near_alias_fraction,
        hard_alias_fraction=best.result.hard_alias_fraction,
        time_source=time_source,
        warnings=tuple(warnings),
        estimated_noise_sigma_rad=noise_sigma,
        noise_residual_count=noise_residual_count,
        noise_lag1_autocorrelation=noise_autocorrelation,
        evidence_sufficient=evidence_sufficient,
        evidence_intervals_used=evidence_intervals_used,
        evidence_fraction=evidence_fraction,
        evidence_log_likelihood_ratio=evidence_log_likelihood_ratio,
        exact_equivalence_class_count=len(equivalence_classes),
        best_equivalence_class_hypothesis_ids=best.result.exact_equivalent_hypothesis_ids,
        rate_scale_period_gauge_active=rate_scale_period_gauge_active,
        raw_excitation_rank=raw_excitation_rank,
        raw_excitation_eigenvalues=raw_excitation_eigenvalues,
        raw_excitation_condition_ratio=raw_excitation_ratio,
    )
####


def _build_inference_report(
    data: TrajectoryData,
    hypotheses: tuple[ConventionHypothesis, ...],
    ranked: tuple[_ScoredCandidate, ...],
    config: InferenceConfig,
    diagnostics: OverallDiagnostics,
) -> InferenceReport:
    rankings = tuple(item.result for item in ranked)
    hypothesis_space = _hypothesis_space_summary(data.representation, config)
    contender_class_ids = _evidence_contender_classes(ranked, config)
    operational = _select_operational_result(ranked, config)
    data_minimum = min(item.result.class_data_risk for item in ranked)
    data_winning_class_ids = tuple(
        dict.fromkeys(
            item.result.equivalence_class_id
            for item in ranked
            if item.result.class_data_risk == data_minimum
        )
    )
    operational_item = next(
        item for item in ranked if item.result.hypothesis.identifier() == operational.hypothesis_id
    )
    return InferenceReport(
        representation=data.representation,
        sample_count=data.sample_count,
        hypothesis_count=len(hypotheses),
        hypothesis_space=hypothesis_space,
        best=operational_item.result,
        rankings=rankings,
        diagnostics=diagnostics,
        result_label=_result_label(data, config),
        evidence_contender_class_ids=contender_class_ids,
        data_winning_class_ids=data_winning_class_ids,
        empirical_gap_to_next_class=operational_item.result.empirical_gap_to_next_class,
        operational_class_id=operational.class_id,
        operational_hypothesis_id=operational.hypothesis_id,
        operational_reason=operational.reason,
        domain_prior_configured=config.domain_prior.configured(),
        domain_prior_abstained=operational.domain_prior_abstained,
        domain_prior_supported_class_ids=operational.domain_prior_supported_class_ids,
        preference_profile_id=config.preference_profile.profile_id,
        preference_profile_source=config.preference_profile.source,
        preference_profile_configured=config.preference_profile.configured(),
        preference_profile_abstained=operational.preference_profile_abstained,
        preference_profile_supported_class_ids=operational.preference_profile_supported_class_ids,
        representative_selection_reason=operational.representative_reason,
    )
####


def infer_trajectory_conventions(
    data: TrajectoryData,
    config: InferenceConfig | None = None,
) -> InferenceReport:
    resolved_config = config or InferenceConfig()
    hypotheses = generate_hypotheses(data.representation, resolved_config)
    if data.orientation is None or data.rates is None:
        partial_scored = tuple(
            _score_partial_hypothesis(data, hypothesis, resolved_config)
            for hypothesis in hypotheses
        )
        ranked = _assign_ranks_and_support(partial_scored, data, resolved_config)
        diagnostics = _build_overall_diagnostics(data, ranked, resolved_config)
        return _build_inference_report(
            data,
            hypotheses,
            ranked,
            resolved_config,
            diagnostics,
        )
    ####

    scored: list[_ScoredCandidate] = []
    failures: list[str] = []
    # Hypotheses are grouped by attitude convention. Retain only the current
    # decoded rotation series instead of every Euler variant on a long log.
    current_attitude_key: tuple[object, ...] | None = None
    current_attitudes: Rotation | None = None
    rate_cache: dict[tuple[object, ...], FloatArray] = {}

    for hypothesis in hypotheses:
        try:
            attitude_key = (
                hypothesis.attitude_direction,
                hypothesis.quaternion_order,
                hypothesis.euler_order,
                hypothesis.euler_mode,
                hypothesis.angle_unit,
            )
            if current_attitudes is None or attitude_key != current_attitude_key:
                current_attitudes = decode_attitudes(data, hypothesis)
                current_attitude_key = attitude_key
            ####

            rate_key = (hypothesis.rate_order, hypothesis.rate_unit, hypothesis.rate_sign)
            if rate_key not in rate_cache:
                rate_cache[rate_key] = decode_rates(data, hypothesis)
            ####

            scored.append(
                _score_hypothesis(
                    data,
                    hypothesis,
                    resolved_config,
                    attitudes=current_attitudes,
                    rates_rad_s=rate_cache[rate_key],
                )
            )
        except (ValueError, FloatingPointError) as error:
            failures.append(f"{hypothesis.identifier()}: {error}")
        ####
    ####

    if not scored:
        details = "\n".join(failures[:10])
        raise ValueError(f"All hypotheses failed to score.\n{details}")
    ####

    ranked = _assign_ranks_and_support(scored, data, resolved_config)
    diagnostics = _build_overall_diagnostics(data, ranked, resolved_config)
    return _build_inference_report(
        data,
        hypotheses,
        ranked,
        resolved_config,
        diagnostics,
    )
####


def best_interval_diagnostics(
    data: TrajectoryData,
    report: InferenceReport,
    config: InferenceConfig | None = None,
) -> tuple[dict[str, float | int | bool], ...]:
    if data.orientation is None or data.rates is None:
        raise ValueError("Interval diagnostics require both orientation and angular rates")
    ####
    resolved_config = config or InferenceConfig()
    scored = _score_hypothesis(
        data, report.best.hypothesis, resolved_config, retain_interval_details=True
    )
    details = scored.interval_details
    if details is None:
        raise RuntimeError("Interval details were not retained for diagnostics")
    ####
    predicted_quaternions = _rotation_quaternions(details.predicted_intervals)
    observed_quaternions = _rotation_quaternions(details.observed_intervals)
    errors = scored.interval_error_rotvecs
    if errors is None:
        raise RuntimeError("Interval residual vectors were not retained for diagnostics")
    ####
    rows: list[dict[str, float | int | bool]] = []

    for index in range(data.sample_count - 1):
        row: dict[str, float | int | bool] = {
            "interval": index,
            "dt_s": float(details.interval_durations_s[index]),
            "omega_x_rad_s": float(
                0.5 * (details.rates_rad_s[index, 0] + details.rates_rad_s[index + 1, 0])
            ),
            "omega_y_rad_s": float(
                0.5 * (details.rates_rad_s[index, 1] + details.rates_rad_s[index + 1, 1])
            ),
            "omega_z_rad_s": float(
                0.5 * (details.rates_rad_s[index, 2] + details.rates_rad_s[index + 1, 2])
            ),
            "observed_delta_qx": float(observed_quaternions[index, 0]),
            "observed_delta_qy": float(observed_quaternions[index, 1]),
            "observed_delta_qz": float(observed_quaternions[index, 2]),
            "observed_delta_qw": float(observed_quaternions[index, 3]),
            "predicted_delta_qx": float(predicted_quaternions[index, 0]),
            "predicted_delta_qy": float(predicted_quaternions[index, 1]),
            "predicted_delta_qz": float(predicted_quaternions[index, 2]),
            "predicted_delta_qw": float(predicted_quaternions[index, 3]),
            "error_rotvec_x_rad": float(errors[index, 0]),
            "error_rotvec_y_rad": float(errors[index, 1]),
            "error_rotvec_z_rad": float(errors[index, 2]),
            "error_angle_rad": float(norm(errors[index, :])),
            "unwrapped_rate_angle_rad": float(details.path_angles[index]),
            "near_alias": bool(details.path_angles[index] >= resolved_config.near_alias_step_rad),
            "hard_alias": bool(details.path_angles[index] >= resolved_config.hard_alias_step_rad),
        }
        if data.time_s is not None:
            row["time_start_s"] = float(data.time_s[index])
            row["time_end_s"] = float(data.time_s[index + 1])
        ####
        rows.append(row)
    ####
    return tuple(rows)
####
