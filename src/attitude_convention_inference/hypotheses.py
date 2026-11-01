"""Hypothesis-space construction and dimension normalization."""

from __future__ import annotations

from itertools import product
from typing import Literal, TypeAlias

from .models import (
    AttitudeRepresentation,
    ConventionHypothesis,
    InferenceConfig,
    SearchSpace,
)

__all__ = [
    "DIMENSIONS",
    "SEARCH_FIELD_BY_DIMENSION",
    "generate_hypotheses",
    "hypothesis_count",
    "relevant_dimensions",
    "search_values",
]

DimensionName: TypeAlias = Literal[
    "attitude_direction",
    "quaternion_order",
    "euler_order",
    "euler_mode",
    "angle_unit",
    "rate_frame",
    "rate_order",
    "rate_unit",
    "rate_sign",
]
SearchDimensionValue: TypeAlias = str | int

DIMENSIONS: tuple[DimensionName, ...] = (
    "attitude_direction",
    "quaternion_order",
    "euler_order",
    "euler_mode",
    "angle_unit",
    "rate_frame",
    "rate_order",
    "rate_unit",
    "rate_sign",
)
SEARCH_FIELD_BY_DIMENSION: dict[DimensionName, str] = {
    "attitude_direction": "attitude_directions",
    "quaternion_order": "quaternion_orders",
    "euler_order": "euler_orders",
    "euler_mode": "euler_modes",
    "angle_unit": "angle_units",
    "rate_frame": "rate_frames",
    "rate_order": "rate_orders",
    "rate_unit": "rate_units",
    "rate_sign": "rate_signs",
}


def relevant_dimensions(representation: AttitudeRepresentation) -> tuple[DimensionName, ...]:
    """Return the search dimensions applicable to an attitude representation."""

    attitude_dimensions = (
        ("quaternion_order",)
        if representation is AttitudeRepresentation.QUATERNION
        else ("euler_order", "euler_mode", "angle_unit")
    )
    return (
        "attitude_direction",
        *attitude_dimensions,
        "rate_frame",
        "rate_order",
        "rate_unit",
        "rate_sign",
    )
####


def search_values(
    search: SearchSpace,
    dimension: DimensionName,
) -> tuple[SearchDimensionValue, ...]:
    """Return the configured values for one named search dimension."""

    return tuple(getattr(search, SEARCH_FIELD_BY_DIMENSION[dimension]))
####


def hypothesis_count(
    representation: AttitudeRepresentation,
    search: SearchSpace,
) -> int:
    """Return the number of syntactically valid hypotheses in a search space."""

    count = 1
    for dimension in relevant_dimensions(representation):
        count *= len(search_values(search, dimension))
    ####
    return count
####


def generate_hypotheses(
    representation: AttitudeRepresentation,
    config: InferenceConfig,
) -> tuple[ConventionHypothesis, ...]:
    """Generate normalized hypotheses for the configured representation and search."""

    search = config.resolved_search_space()
    rate_options = tuple(
        product(search.rate_frames, search.rate_orders, search.rate_units, search.rate_signs)
    )
    hypotheses: list[ConventionHypothesis] = []

    if representation is AttitudeRepresentation.QUATERNION:
        for direction, quaternion_order in product(
            search.attitude_directions,
            search.quaternion_orders,
        ):
            for rate_frame, rate_order, rate_unit, rate_sign in rate_options:
                hypotheses.append(
                    ConventionHypothesis(
                        attitude_direction=direction,
                        quaternion_order=quaternion_order,
                        rate_frame=rate_frame,
                        rate_order=rate_order,
                        rate_unit=rate_unit,
                        rate_sign=rate_sign,
                    )
                )
            ####
        ####
    else:
        for direction, euler_order, euler_mode, angle_unit in product(
            search.attitude_directions,
            search.euler_orders,
            search.euler_modes,
            search.angle_units,
        ):
            for rate_frame, rate_order, rate_unit, rate_sign in rate_options:
                hypotheses.append(
                    ConventionHypothesis(
                        attitude_direction=direction,
                        euler_order=euler_order,
                        euler_mode=euler_mode,
                        angle_unit=angle_unit,
                        rate_frame=rate_frame,
                        rate_order=rate_order,
                        rate_unit=rate_unit,
                        rate_sign=rate_sign,
                    )
                )
            ####
        ####
    ####
    return tuple(hypotheses)
####
