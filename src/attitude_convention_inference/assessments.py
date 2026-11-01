"""Hypothesis-space and per-dimension report assessments."""

from __future__ import annotations

from .hypotheses import DIMENSIONS as _DIMENSIONS
from .hypotheses import DimensionName as _DimensionName
from .hypotheses import SearchDimensionValue as _SearchDimensionValue
from .hypotheses import hypothesis_count as _hypothesis_count
from .hypotheses import relevant_dimensions as _relevant_dimensions
from .hypotheses import search_values as _search_values
from .models import (
    AttitudeRepresentation,
    ConventionHypothesis,
    DimensionAssessment,
    DimensionStatus,
    HypothesisSpaceSummary,
    InferenceConfig,
)
from .ranking import _hypothesis_value
from .scoring import ScoredCandidate as _ScoredCandidate

__all__ = ["_dimension_assessments", "_hypothesis_space_summary", "_unresolved_dimensions"]


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
    dimension: _DimensionName,
    allowed_values: tuple[_SearchDimensionValue, ...],
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
    dimension: _DimensionName,
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
