"""Coercion boundaries for the convenience-facing inference API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

from .models import (
    ConventionConstraints,
    ConventionPriors,
    DomainPrior,
    InferenceConfig,
    PreferenceProfile,
)

__all__ = [
    "ConfigInput",
    "ConstraintInput",
    "DomainPriorInput",
    "PreferenceProfileInput",
    "PriorInput",
    "coerce_config",
    "coerce_constraints",
    "coerce_domain_prior",
    "coerce_preference_profile",
    "coerce_priors",
    "merge_priors",
]

ConfigInput: TypeAlias = InferenceConfig | Mapping[str, Any] | None
ConstraintInput: TypeAlias = ConventionConstraints | Mapping[str, Any] | None
PriorInput: TypeAlias = ConventionPriors | Mapping[str, Any] | None
DomainPriorInput: TypeAlias = DomainPrior | Mapping[str, Any] | None
PreferenceProfileInput: TypeAlias = PreferenceProfile | Mapping[str, Any] | None


def coerce_config(value: ConfigInput) -> InferenceConfig:
    """Coerce a config model or mapping into an immutable config model."""

    if value is None:
        return InferenceConfig()
    ####
    if isinstance(value, InferenceConfig):
        return value
    ####
    return InferenceConfig.model_validate(value)
####




def coerce_constraints(value: ConstraintInput) -> ConventionConstraints | None:
    """Coerce optional constraint input while preserving an omitted value."""

    if value is None:
        return None
    ####
    if isinstance(value, ConventionConstraints):
        return value
    ####
    return ConventionConstraints.model_validate(value)
####




def coerce_priors(value: PriorInput) -> ConventionPriors | None:
    """Coerce optional dimension-prior input."""

    if value is None:
        return None
    ####
    if isinstance(value, ConventionPriors):
        return value
    ####
    return ConventionPriors.model_validate(value)
####




def coerce_domain_prior(value: DomainPriorInput) -> DomainPrior | None:
    """Coerce optional domain-prior input."""

    if value is None:
        return None
    ####
    if isinstance(value, DomainPrior):
        return value
    ####
    return DomainPrior.model_validate(value)
####




def coerce_preference_profile(
    value: PreferenceProfileInput,
) -> PreferenceProfile | None:
    """Coerce optional preference-profile input."""

    if value is None:
        return None
    ####
    if isinstance(value, PreferenceProfile):
        return value
    ####
    return PreferenceProfile.model_validate(value)
####




def merge_priors(base: ConventionPriors, override: ConventionPriors) -> ConventionPriors:
    """Merge dimension weights, with override values taking precedence."""

    payload = base.model_dump(mode="python")
    override_payload = override.model_dump(mode="python")
    for dimension, weights in override_payload.items():
        payload[dimension] = {**payload[dimension], **weights}
    ####
    return ConventionPriors.model_validate(payload)
####

