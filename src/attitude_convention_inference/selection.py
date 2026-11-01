"""Operational class and representative selection after trajectory ranking."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, log

from .hypotheses import DIMENSIONS as _DIMENSIONS
from .models import InferenceConfig, ResultLabel
from .ranking import _canonical_key, _hypothesis_value
from .scoring import ScoredCandidate as _ScoredCandidate
from .trajectory import TrajectoryData

__all__ = ["_evidence_contender_classes", "_result_label", "_select_operational_result"]


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
    """Return classes within the configured log-ratio of minimum floored geometric risk.

    This descriptive fit margin is independent of the sequential likelihood
    sufficiency check; it is not itself a likelihood-ratio or confidence threshold.
    """

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
