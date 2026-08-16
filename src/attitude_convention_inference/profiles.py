"""Packaged, versioned preference profiles for deployed inference."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from importlib.resources import files
from importlib.resources.abc import Traversable
from math import exp, isfinite, log
from typing import Any, cast

from .models import ConventionPriors, FrozenModel, PreferenceProfile

__all__ = [
    "DEFAULT_PROFILE_ID",
    "WeightedPreferenceProfile",
    "available_preference_profiles",
    "compose_preference_profiles",
    "load_preference_profile",
]

DEFAULT_PROFILE_ID = "neutral-v1"
_PROFILE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PROFILE_DIRECTORY = ("data", "profiles")


@dataclass(frozen=True, slots=True)
class WeightedPreferenceProfile(FrozenModel):
    """A profile and its positive contribution to a weighted composition."""

    profile: PreferenceProfile
    weight: float

    def __post_init__(self) -> None:
        if not isfinite(float(self.weight)) or self.weight <= 0.0:
            raise ValueError("Profile composition weights must be positive and finite.")
        ####
    ####
####





def _profile_directory() -> Traversable:
    resource = files("attitude_convention_inference")
    for component in _PROFILE_DIRECTORY:
        resource = resource.joinpath(component)
    ####
    return resource
####




def available_preference_profiles() -> tuple[str, ...]:
    """Return the IDs of preference profiles bundled in this installation."""

    return tuple(
        sorted(
            resource.name.removesuffix(".json")
            for resource in _profile_directory().iterdir()
            if resource.is_file() and resource.name.endswith(".json")
        )
    )
####




def load_preference_profile(profile_id: str = DEFAULT_PROFILE_ID) -> PreferenceProfile:
    """Load and validate a packaged preference profile by its stable ID.

    Profiles are package data so the selected policy remains available in wheels,
    containers, and offline deployments. External policies can still be supplied
    through :class:`PreferenceProfile` or the CLI configuration file.
    """

    normalized_id = str(profile_id).strip()
    if _PROFILE_ID_PATTERN.fullmatch(normalized_id) is None:
        raise ValueError(f"Invalid preference profile ID: {profile_id!r}.")
    ####
    resource = _profile_directory().joinpath(f"{normalized_id}.json")
    if not resource.is_file():
        available = ", ".join(available_preference_profiles()) or "<none>"
        raise ValueError(
            f"Unknown preference profile {normalized_id!r}; available profiles: {available}."
        )
    ####
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Preference profile {normalized_id!r} must contain a JSON object.")
    ####
    profile = PreferenceProfile.model_validate(cast(Mapping[str, Any], payload))
    if profile.profile_id != normalized_id:
        raise ValueError(
            f"Preference profile resource {normalized_id!r} declares ID {profile.profile_id!r}."
        )
    ####
    return profile
####




def _combined_mapping(
    profiles: tuple[WeightedPreferenceProfile, ...],
    mappings: tuple[Mapping[Any, float], ...],
    dimension: str,
    total_weight: float,
) -> dict[Any, float]:
    keys: set[Any] = set()
    for mapping in mappings:
        keys.update(mapping)
    ####

    result: dict[Any, float] = {}
    for key in keys:
        combined_log_weight = 0.0
        zero_weight = False
        for item, mapping in zip(profiles, mappings, strict=True):
            value = float(mapping.get(key, 1.0))
            if value == 0.0:
                zero_weight = True
                break
            ####
            combined_log_weight += (
                float(item.weight) / total_weight * float(item.profile.strength) * log(value)
            )
        ####
        if zero_weight:
            result[key] = 0.0
            continue
        ####
        try:
            combined = exp(combined_log_weight)
        except OverflowError as error:
            raise ValueError(
                f"Profile composition overflowed while combining {dimension!r} weights."
            ) from error
        ####
        if not isfinite(combined) or combined <= 0.0:
            raise ValueError(f"Profile composition produced an invalid {dimension!r} weight.")
        ####
        result[key] = combined
    ####
    return result
####




def compose_preference_profiles(
    profiles: Sequence[WeightedPreferenceProfile],
    *,
    profile_id: str,
    source: str = "weighted-geometric-composition",
    context: Mapping[str, str] | None = None,
    strength: float = 1.0,
) -> PreferenceProfile:
    """Compose profiles with a normalized weighted geometric combination.

    Relative weights are combined in log space. Missing dimension values have
    neutral weight ``1.0``; an explicit zero class weight remains an exclusion.
    Each member profile's ``strength`` is included in its contribution, and the
    resulting profile has the requested overall ``strength``.
    """

    members = tuple(profiles)
    if not members:
        raise ValueError("At least one profile is required for composition.")
    ####
    total_weight = sum(float(item.weight) for item in members)
    if not isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("Profile composition weights must have a finite positive sum.")
    ####

    representative_preferences: dict[str, dict[Any, float]] = {}
    for field in fields(ConventionPriors):
        mappings = tuple(
            cast(Mapping[Any, float], getattr(item.profile.representative_preferences, field.name))
            for item in members
        )
        representative_preferences[field.name] = _combined_mapping(
            members,
            mappings,
            field.name,
            total_weight,
        )
    ####

    class_mappings = tuple(
        cast(Mapping[Any, float], item.profile.class_weights) for item in members
    )
    class_weights = {
        str(key): value
        for key, value in _combined_mapping(
            members,
            class_mappings,
            "class_weights",
            total_weight,
        ).items()
    }

    member_summary = ",".join(
        f"{item.profile.profile_id}={float(item.weight):g}" for item in members
    )
    composition_context = {str(key): str(value) for key, value in (context or {}).items()}
    composition_context["composition_method"] = "weighted-geometric"
    composition_context["composition_members"] = member_summary
    return PreferenceProfile(
        profile_id=profile_id,
        source=source,
        context=composition_context,
        class_weights=class_weights,
        representative_preferences=ConventionPriors.model_validate(representative_preferences),
        strength=strength,
    )
####


