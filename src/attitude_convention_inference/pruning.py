"""Exact lower-bound pruning, separate from statistical evidence sufficiency."""

from __future__ import annotations

from math import isfinite

__all__ = ["should_prune_by_exact_lower_bound"]


def should_prune_by_exact_lower_bound(
    partial_lower_bound: float,
    incumbent_score: float,
) -> bool:
    """Return whether a verified exact lower bound can abandon a candidate safely.

    This predicate is deliberately separate from statistical evidence elimination:
    it is valid only when the caller supplies a mathematically verified lower bound
    on the same final risk and a verified incumbent score.
    """

    lower_bound = float(partial_lower_bound)
    incumbent = float(incumbent_score)
    if not isfinite(lower_bound) or not isfinite(incumbent):
        raise ValueError("Exact pruning scores must be finite.")
    ####
    return lower_bound > incumbent
####
