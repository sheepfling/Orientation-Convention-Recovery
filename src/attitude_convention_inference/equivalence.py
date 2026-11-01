"""Exact observational equivalence classes for trajectory candidates."""

from __future__ import annotations

from collections.abc import Iterable
from math import log, sqrt

from numpy import mean

from .models import CandidateResult, ConventionHypothesis, InferenceConfig, RateUnit
from .scoring import ScoredCandidate as _ScoredCandidate
from .scoring import replace_result as _replace_result
from .trajectory import TrajectoryData

__all__ = ["_annotate_exact_equivalences"]


def _rate_unit_partner(hypothesis: ConventionHypothesis) -> ConventionHypothesis:
    rate_unit = (
        RateUnit.DEGREES_PER_SECOND
        if hypothesis.rate_unit is RateUnit.RADIANS_PER_SECOND
        else RateUnit.RADIANS_PER_SECOND
    )
    return hypothesis.model_copy(update={"rate_unit": rate_unit})
####


def _period_is_active_bound(result: CandidateResult, config: InferenceConfig) -> bool:
    if result.inferred_sample_period_s is None:
        return False
    ####
    lower, upper = config.inferred_sample_period_bounds_s
    tolerance = max(1.0e-12, (upper - lower) * 1.0e-6)
    return (
        abs(result.inferred_sample_period_s - lower) <= tolerance
        or abs(result.inferred_sample_period_s - upper) <= tolerance
    )
####


def _annotate_exact_equivalences(
    scored: Iterable[_ScoredCandidate],
    data: TrajectoryData,
    config: InferenceConfig,
) -> tuple[_ScoredCandidate, ...]:
    """Collapse exact observational symmetries before numeric ranking.

    The direction--frame--global-sign dual is exact for every trajectory.  Missing
    timestamps add the rate-unit/sample-period scale gauge unless a configured period
    bound is active.  Members receive one shared raw risk so floating-point noise cannot
    manufacture a winner inside either class.
    """

    candidates = tuple(scored)
    identifiers = {
        item.result.hypothesis.identifier(): index for index, item in enumerate(candidates)
    }
    parents = list(range(len(candidates)))
    scale_gauge = data.time_s is None and config.sample_period_s is None

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        ####
        return index
    ####

    def join(left: int, right: int) -> None:
        parents[find(left)] = find(right)
    ####

    for index, item in enumerate(candidates):
        partner = item.result.hypothesis.structural_dual().identifier()
        if partner in identifiers:
            join(index, identifiers[partner])
        ####
        if scale_gauge and not _period_is_active_bound(item.result, config):
            unit_partner = _rate_unit_partner(item.result.hypothesis).identifier()
            partner_index = identifiers.get(unit_partner)
            if partner_index is not None and not _period_is_active_bound(
                candidates[partner_index].result, config
            ):
                join(index, partner_index)
            ####
        ####
    ####

    groups: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        groups.setdefault(find(index), []).append(index)
    ####

    updated = list(candidates)
    for group_indices in groups.values():
        members = [candidates[index] for index in group_indices]
        identifiers_in_group = tuple(
            sorted(item.result.hypothesis.identifier() for item in members)
        )
        class_id = identifiers_in_group[0]
        shared_risk = float(mean([item.result.raw_geometric_risk for item in members]))
        shared_untrimmed_risk = float(
            mean([item.result.untrimmed_geometric_risk for item in members])
        )
        shared_rmse = sqrt(shared_risk)
        shared_criterion = log(max(shared_risk, config.residual_floor_rad**2))
        for index in group_indices:
            result = candidates[index].result.model_copy(
                update={
                    "raw_geometric_risk": shared_risk,
                    "untrimmed_geometric_risk": shared_untrimmed_risk,
                    "multiscale_rmse_rad": shared_rmse,
                    "data_criterion": shared_criterion,
                    "equivalence_class_id": class_id,
                    "exact_equivalent_hypothesis_ids": identifiers_in_group,
                }
            )
            updated[index] = _replace_result(candidates[index], result)
        ####
    ####
    return tuple(updated)
####
