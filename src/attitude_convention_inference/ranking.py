"""Data ranking and soft-prior support for scored convention hypotheses."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable
from math import exp, log

from numpy import all as np_all
from numpy import amax, amin, asarray, float64, intp, median
from numpy import isfinite as np_isfinite
from numpy.typing import NDArray

from .equivalence import _annotate_exact_equivalences
from .hypotheses import DIMENSIONS as _DIMENSIONS
from .hypotheses import DimensionName as _DimensionName
from .hypotheses import SearchDimensionValue as _SearchDimensionValue
from .math_utils import FloatArray
from .math_utils import support_weights as _support_weights
from .models import (
    AttitudeDirection,
    ConventionHypothesis,
    InferenceConfig,
    QuaternionOrder,
    RateFrame,
    RateUnit,
)
from .scoring import ScoredCandidate as _ScoredCandidate
from .scoring import replace_result as _replace_result
from .trajectory import TrajectoryData

__all__ = ["_assign_ranks_and_support", "_canonical_key", "_hypothesis_value"]


def _hypothesis_value(
    hypothesis: ConventionHypothesis,
    dimension: _DimensionName,
) -> _SearchDimensionValue | None:
    return getattr(hypothesis, dimension)
####


def _prior_log_weight(
    hypothesis: ConventionHypothesis,
    config: InferenceConfig,
) -> float:
    total = 0.0
    for dimension in _DIMENSIONS:
        value = _hypothesis_value(hypothesis, dimension)
        if value is None:
            continue
        ####
        weights = getattr(config.priors, dimension)
        total += log(float(weights.get(value, 1.0)))
    ####
    return total
####


def _canonical_key(hypothesis: ConventionHypothesis) -> tuple[str, ...]:
    """Return the stable minimum-intervention key used only after evidence ties."""

    return (
        "0" if hypothesis.attitude_direction is AttitudeDirection.BODY_TO_WORLD else "1",
        "0" if hypothesis.rate_frame is RateFrame.BODY else "1",
        "0" if hypothesis.rate_sign == 1 else "1",
        "0" if hypothesis.rate_unit is RateUnit.RADIANS_PER_SECOND else "1",
        "0" if hypothesis.rate_order == "pqr" else "1",
        "0"
        if hypothesis.quaternion_order is QuaternionOrder.SCALAR_LAST
        else "1"
        if hypothesis.quaternion_order is QuaternionOrder.SCALAR_FIRST
        else "0",
        hypothesis.identifier(),
    )
####


def _ranks_with_exact_ties(order: NDArray[intp], values: FloatArray) -> dict[int, int]:
    ranks: dict[int, int] = {}
    previous: float | None = None
    current_rank = 0
    for position, candidate_index_raw in enumerate(order, start=1):
        candidate_index = int(candidate_index_raw)
        value = float(values[candidate_index])
        if previous is None or value != previous:
            current_rank = position
            previous = value
        ####
        ranks[candidate_index] = current_rank
    ####
    return ranks
####


def _assign_ranks_and_support(
    scored: Iterable[_ScoredCandidate],
    data: TrajectoryData,
    config: InferenceConfig,
) -> tuple[_ScoredCandidate, ...]:
    candidates = _annotate_exact_equivalences(scored, data, config)
    if not candidates:
        raise ValueError("No convention hypothesis could be scored.")
    ####

    class_risks: dict[str, float] = {}
    class_keys: dict[str, tuple[str, ...]] = {}
    for item in candidates:
        class_id = item.result.equivalence_class_id
        class_risks[class_id] = min(
            class_risks.get(class_id, float("inf")),
            float(item.result.raw_geometric_risk),
        )
        class_keys.setdefault(class_id, _canonical_key(item.result.hypothesis))
    ####
    ordered_classes = sorted(
        class_risks,
        key=lambda class_id: (class_risks[class_id], class_id, class_keys[class_id]),
    )
    distinct_risks = sorted(set(class_risks.values()))
    next_class_risk: dict[str, float | None] = {}
    for class_id in ordered_classes:
        next_index = bisect_right(distinct_risks, class_risks[class_id])
        next_class_risk[class_id] = (
            distinct_risks[next_index] if next_index < len(distinct_risks) else None
        )
    ####

    def _class_gap(item: _ScoredCandidate) -> float | None:
        next_risk = next_class_risk[item.result.equivalence_class_id]
        if next_risk is None:
            return None
        ####
        return max(0.0, next_risk - class_risks[item.result.equivalence_class_id])
    ####

    candidates = tuple(
        _replace_result(
            item,
            item.result.model_copy(
                update={
                    "class_data_risk": class_risks[item.result.equivalence_class_id],
                    "empirical_gap_to_next_class": _class_gap(item),
                }
            ),
        )
        for item in candidates
    )

    display_scale = max(
        float(median([item.result.observed_step_rms_rad for item in candidates])),
        config.residual_floor_rad,
    )
    candidates = tuple(
        _replace_result(
            item,
            item.result.model_copy(
                update={
                    "normalized_rmse": item.result.multiscale_rmse_rad / display_scale,
                    "fit_score": 100.0
                    * exp(-0.5 * (item.result.multiscale_rmse_rad / display_scale) ** 2),
                }
            ),
        )
        for item in candidates
    )

    data_criteria: FloatArray = asarray(
        [item.result.data_criterion for item in candidates],
        dtype=float64,
    )
    if not np_all(np_isfinite(data_criteria)):
        raise ValueError("No finite convention hypothesis could be scored.")
    ####

    data_supports = _support_weights(data_criteria)
    data_minimum = float(amin(data_criteria))
    raw_risks: FloatArray = asarray(
        [item.result.raw_geometric_risk for item in candidates], dtype=float64
    )

    def _data_sort_key(index: int) -> tuple[float, str, tuple[str, ...]]:
        return (
            float(raw_risks[index]),
            candidates[index].result.equivalence_class_id,
            _canonical_key(candidates[index].result.hypothesis),
        )
    ####

    data_order: NDArray[intp] = asarray(
        sorted(range(len(candidates)), key=_data_sort_key),
        dtype=intp,
    )
    data_rank_by_index = _ranks_with_exact_ties(data_order, raw_risks)

    prior_logs = asarray(
        [_prior_log_weight(item.result.hypothesis, config) for item in candidates],
        dtype=float64,
    )
    maximum_prior_log = float(amax(prior_logs))
    prior_penalties = 2.0 * (maximum_prior_log - prior_logs)
    posterior_criteria: FloatArray = data_criteria + prior_penalties
    posterior_supports = _support_weights(posterior_criteria)
    posterior_minimum = float(amin(posterior_criteria))

    def _posterior_sort_key(index: int) -> tuple[float, str, tuple[str, ...]]:
        return (
            float(posterior_criteria[index]),
            candidates[index].result.equivalence_class_id,
            _canonical_key(candidates[index].result.hypothesis),
        )
    ####

    posterior_order: NDArray[intp] = asarray(
        sorted(range(len(candidates)), key=_posterior_sort_key),
        dtype=intp,
    )

    ranked: list[_ScoredCandidate] = []
    for rank, candidate_index_raw in enumerate(posterior_order, start=1):
        candidate_index = int(candidate_index_raw)
        item = candidates[candidate_index]
        updated_result = item.result.model_copy(
            update={
                "rank": rank,
                "data_rank": data_rank_by_index[candidate_index],
                "relative_support": float(posterior_supports[candidate_index]),
                "data_relative_support": float(data_supports[candidate_index]),
                "delta_criterion": float(posterior_criteria[candidate_index] - posterior_minimum),
                "data_delta_criterion": float(data_criteria[candidate_index] - data_minimum),
                "criterion": float(posterior_criteria[candidate_index]),
                "prior_relative_weight": float(
                    exp(prior_logs[candidate_index] - maximum_prior_log)
                ),
                "prior_penalty": float(prior_penalties[candidate_index]),
            }
        )
        ranked.append(_replace_result(item, updated_result))
    ####
    return tuple(ranked)
####
