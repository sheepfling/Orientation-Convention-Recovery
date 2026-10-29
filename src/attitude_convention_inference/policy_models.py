"""Validated prior, preference, and inference-configuration contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import field
from math import isfinite, pi
from typing import Any, Literal, Self, cast

from pydantic.dataclasses import dataclass

from .immutable import ReadOnlyMapping, freeze_mapping
from .model_common import (
    PRIOR_FIELDS,
    AngleUnit,
    AttitudeDirection,
    EulerMode,
    FrozenModel,
    QuaternionOrder,
    RateFrame,
    RateUnit,
    enum_value,
    normalize_euler_orders,
    normalize_rate_orders,
    normalize_signs,
    validated_payload,
)
from .search_models import ConventionConstraints, SearchSpace

__all__ = ["ConventionPriors", "DomainPrior", "InferenceConfig", "PreferenceProfile"]


def _prior_mapping(
    value: Mapping[Any, Any] | None,
    *,
    key_converter: Callable[[Any], Any],
    label: str,
) -> dict[Any, float]:
    if value is None:
        return {}
    ####
    result: dict[Any, float] = {}
    for raw_key, raw_weight in value.items():
        key = key_converter(raw_key)
        weight = float(raw_weight)
        if not isfinite(weight) or weight <= 0.0:
            raise ValueError(f"Prior weight for {key!r} must be positive and finite.")
        ####
        result[key] = weight
    ####
    return result
####


@dataclass(frozen=True, slots=True)
class ConventionPriors(FrozenModel):
    """Relative soft-prior weights; missing values have neutral weight ``1.0``."""

    attitude_direction: ReadOnlyMapping[AttitudeDirection | str, float] = field(
        default_factory=dict[AttitudeDirection | str, float]
    )
    quaternion_order: ReadOnlyMapping[QuaternionOrder | str, float] = field(
        default_factory=dict[QuaternionOrder | str, float]
    )
    euler_order: ReadOnlyMapping[str, float] = field(default_factory=dict[str, float])
    euler_mode: ReadOnlyMapping[EulerMode | str, float] = field(
        default_factory=dict[EulerMode | str, float]
    )
    angle_unit: ReadOnlyMapping[AngleUnit | str, float] = field(
        default_factory=dict[AngleUnit | str, float]
    )
    rate_frame: ReadOnlyMapping[RateFrame | str, float] = field(
        default_factory=dict[RateFrame | str, float]
    )
    rate_order: ReadOnlyMapping[str, float] = field(default_factory=dict[str, float])
    rate_unit: ReadOnlyMapping[RateUnit | str, float] = field(
        default_factory=dict[RateUnit | str, float]
    )
    rate_sign: ReadOnlyMapping[Literal[-1, 1] | str, float] = field(
        default_factory=dict[Literal[-1, 1] | str, float]
    )

    @classmethod
    def model_validate(cls, value: Self | Mapping[str, Any]) -> Self:
        return cls(**validated_payload(cls, value))
    ####

    def __post_init__(self) -> None:
        converters: tuple[tuple[str, Callable[[Any], Any], str], ...] = (
            (
                "attitude_direction",
                lambda item: enum_value(AttitudeDirection, item),
                "attitude_direction",
            ),
            (
                "quaternion_order",
                lambda item: enum_value(QuaternionOrder, item),
                "quaternion_order",
            ),
            ("euler_order", lambda item: normalize_euler_orders((str(item),))[0], "euler_order"),
            ("euler_mode", lambda item: enum_value(EulerMode, item), "euler_mode"),
            ("angle_unit", lambda item: enum_value(AngleUnit, item), "angle_unit"),
            ("rate_frame", lambda item: enum_value(RateFrame, item), "rate_frame"),
            ("rate_order", lambda item: normalize_rate_orders((str(item),))[0], "rate_order"),
            ("rate_unit", lambda item: enum_value(RateUnit, item), "rate_unit"),
            ("rate_sign", lambda item: normalize_signs((cast(int | str, item),))[0], "rate_sign"),
        )
        for name, converter, label in converters:
            weights = cast(Mapping[Any, Any], getattr(self, name))
            normalized = _prior_mapping(weights, key_converter=converter, label=label)
            object.__setattr__(self, name, freeze_mapping(normalized))
        ####
    ####

    def configured_dimensions(self) -> tuple[str, ...]:
        return tuple(field_name for field_name in PRIOR_FIELDS if bool(getattr(self, field_name)))
    ####
####


@dataclass(frozen=True, slots=True)
class DomainPrior(FrozenModel):
    """Class-level provenance prior applied only after trajectory contenders exist."""

    class_weights: ReadOnlyMapping[str, float] = field(default_factory=dict[str, float])
    context: ReadOnlyMapping[str, str] = field(default_factory=dict[str, str])
    strength: float = 1.0

    @classmethod
    def model_validate(cls, value: Self | Mapping[str, Any]) -> Self:
        payload = validated_payload(cls, value)
        raw_weights: Any = payload.get("class_weights", {})
        if not isinstance(raw_weights, Mapping):
            raise ValueError("Domain prior class_weights must be a mapping.")
        ####
        raw_weights = cast(Mapping[Any, Any], raw_weights)
        weights: dict[str, float] = {}
        for raw_key, raw_value in raw_weights.items():
            key = str(raw_key)
            weight = float(raw_value)
            if not isfinite(weight) or weight < 0.0:
                raise ValueError("Domain prior weights must be finite and nonnegative.")
            ####
            weights[key] = weight
        ####
        raw_context: Any = payload.get("context", {})
        if not isinstance(raw_context, Mapping):
            raise ValueError("Domain prior context must be a mapping.")
        ####
        raw_context = cast(Mapping[Any, Any], raw_context)
        payload["class_weights"] = weights
        payload["context"] = {str(key): str(value) for key, value in raw_context.items()}
        payload["strength"] = float(payload.get("strength", 1.0))
        return cls(**payload)
    ####

    def __post_init__(self) -> None:
        if not isfinite(float(self.strength)) or self.strength <= 0.0:
            raise ValueError("Domain prior strength must be positive and finite.")
        ####
        for weight in self.class_weights.values():
            if not isfinite(float(weight)) or float(weight) < 0.0:
                raise ValueError("Domain prior weights must be finite and nonnegative.")
            ####
        ####
        object.__setattr__(self, "class_weights", freeze_mapping(self.class_weights))
        object.__setattr__(self, "context", freeze_mapping(self.context))
    ####

    def configured(self) -> bool:
        return bool(self.class_weights) or bool(self.context)
    ####
####


@dataclass(frozen=True, slots=True)
class PreferenceProfile(FrozenModel):
    """Persisted operational preferences for classes and representatives.

    Class weights may choose among trajectory-supported quotient classes.  The
    representative preferences are applied only after that class has been
    selected, so they cannot manufacture trajectory evidence or break an exact
    observational equivalence.
    """

    profile_id: str = "neutral-v1"
    source: str = "package-default"
    context: ReadOnlyMapping[str, str] = field(default_factory=dict[str, str])
    class_weights: ReadOnlyMapping[str, float] = field(default_factory=dict[str, float])
    representative_preferences: ConventionPriors = field(default_factory=ConventionPriors)
    strength: float = 1.0

    @classmethod
    def model_validate(cls, value: Self | Mapping[str, Any]) -> Self:
        payload = validated_payload(cls, value)
        payload["profile_id"] = str(payload.get("profile_id", "neutral-v1"))
        payload["source"] = str(payload.get("source", "package-default"))

        raw_context: Any = payload.get("context", {})
        if not isinstance(raw_context, Mapping):
            raise ValueError("Preference profile context must be a mapping.")
        ####
        raw_context = cast(Mapping[Any, Any], raw_context)
        payload["context"] = {str(key): str(item) for key, item in raw_context.items()}

        raw_weights: Any = payload.get("class_weights", {})
        if not isinstance(raw_weights, Mapping):
            raise ValueError("Preference profile class_weights must be a mapping.")
        ####
        raw_weights = cast(Mapping[Any, Any], raw_weights)
        weights: dict[str, float] = {}
        for raw_key, raw_value in raw_weights.items():
            key = str(raw_key)
            weight = float(raw_value)
            if not isfinite(weight) or weight < 0.0:
                raise ValueError("Preference profile class weights must be finite and nonnegative.")
            ####
            weights[key] = weight
        ####
        payload["class_weights"] = weights
        payload["representative_preferences"] = ConventionPriors.model_validate(
            payload.get("representative_preferences", {})
        )
        payload["strength"] = float(payload.get("strength", 1.0))
        return cls(**payload)
    ####

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("Preference profile profile_id must not be empty.")
        ####
        if not isfinite(float(self.strength)) or self.strength <= 0.0:
            raise ValueError("Preference profile strength must be positive and finite.")
        ####
        for weight in self.class_weights.values():
            if not isfinite(float(weight)) or float(weight) < 0.0:
                raise ValueError("Preference profile class weights must be finite and nonnegative.")
            ####
        ####
        if not isinstance(cast(Any, self.representative_preferences), ConventionPriors):
            object.__setattr__(
                self,
                "representative_preferences",
                ConventionPriors.model_validate(self.representative_preferences),
            )
        ####
        object.__setattr__(self, "context", freeze_mapping(self.context))
        object.__setattr__(self, "class_weights", freeze_mapping(self.class_weights))
    ####

    def class_configured(self) -> bool:
        return bool(self.class_weights)
    ####

    def representative_configured(self) -> bool:
        return bool(self.representative_preferences.configured_dimensions())
    ####

    def configured(self) -> bool:
        return bool(self.context) or self.class_configured() or self.representative_configured()
    ####
####


@dataclass(frozen=True, slots=True)
class InferenceConfig(FrozenModel):
    search_space: SearchSpace = field(default_factory=SearchSpace)
    constraints: ConventionConstraints = field(default_factory=ConventionConstraints)
    priors: ConventionPriors = field(default_factory=ConventionPriors)
    domain_prior: DomainPrior = field(default_factory=DomainPrior)
    preference_profile: PreferenceProfile = field(default_factory=PreferenceProfile)
    sample_period_s: float | None = None
    infer_sample_period_when_missing: bool = True
    inferred_sample_period_bounds_s: tuple[float, float] = (1.0e-6, 10.0)
    integration_substeps: int = 4
    window_sizes: tuple[int, ...] = (1, 2, 4, 8)
    trim_fraction: float = 0.02
    residual_floor_rad: float = 1.0e-5
    near_tie_delta_criterion: float = 2.0
    slow_step_threshold_rad: float = 0.002
    slow_total_motion_threshold_rad: float = 0.10
    near_alias_step_rad: float = 0.80 * pi
    hard_alias_step_rad: float = pi
    high_rate_change_angle_rad: float = 0.50
    gimbal_cosine_threshold: float = 0.05
    excitation_rank_relative_threshold: float = 1.0e-3
    top_k: int = 20
    evidence_log_likelihood_ratio_threshold: float = 5.0
    evidence_min_intervals: int = 20
    evidence_stability_windows: int = 3

    @classmethod
    def model_validate(cls, value: Self | Mapping[str, Any]) -> Self:
        payload = validated_payload(cls, value)
        if "search_space" in payload:
            payload["search_space"] = SearchSpace.model_validate(payload["search_space"])
        ####
        if "constraints" in payload:
            payload["constraints"] = ConventionConstraints.model_validate(payload["constraints"])
        ####
        if "priors" in payload:
            payload["priors"] = ConventionPriors.model_validate(payload["priors"])
        ####
        if "domain_prior" in payload:
            payload["domain_prior"] = DomainPrior.model_validate(payload["domain_prior"])
        ####
        if "preference_profile" in payload:
            payload["preference_profile"] = PreferenceProfile.model_validate(
                payload["preference_profile"]
            )
        ####
        if "inferred_sample_period_bounds_s" in payload:
            payload["inferred_sample_period_bounds_s"] = tuple(
                float(value) for value in payload["inferred_sample_period_bounds_s"]
            )
        ####
        if "window_sizes" in payload:
            payload["window_sizes"] = tuple(int(value) for value in payload["window_sizes"])
        ####
        if "evidence_log_likelihood_ratio_threshold" in payload:
            payload["evidence_log_likelihood_ratio_threshold"] = float(
                payload["evidence_log_likelihood_ratio_threshold"]
            )
        ####
        for name in ("evidence_min_intervals", "evidence_stability_windows"):
            if name in payload:
                payload[name] = int(payload[name])
            ####
        ####
        return cls(**payload)
    ####

    def __post_init__(self) -> None:
        if self.sample_period_s is not None and (
            not isfinite(float(self.sample_period_s)) or self.sample_period_s <= 0.0
        ):
            raise ValueError("sample_period_s must be positive and finite.")
        ####
        lower, upper = (float(value) for value in self.inferred_sample_period_bounds_s)
        if not isfinite(lower) or not isfinite(upper) or lower <= 0.0 or upper <= lower:
            raise ValueError(
                "inferred_sample_period_bounds_s must be finite, positive, and strictly increasing."
            )
        ####
        if self.inferred_sample_period_bounds_s != (lower, upper):
            raise ValueError("inferred_sample_period_bounds_s must contain floats.")
        ####

        if not 1 <= int(self.integration_substeps) <= 64:
            raise ValueError("integration_substeps must be between 1 and 64.")
        ####
        windows = tuple(sorted(set(int(window) for window in self.window_sizes)))
        if not windows or any(window < 1 for window in windows):
            raise ValueError("Window sizes must be positive integers.")
        ####
        if tuple(self.window_sizes) != windows:
            raise ValueError("window_sizes must contain unique positive integers.")
        ####

        bounded_checks = (
            (
                "trim_fraction",
                self.trim_fraction,
                isfinite(float(self.trim_fraction)) and 0.0 <= self.trim_fraction < 0.25,
            ),
            (
                "gimbal_cosine_threshold",
                self.gimbal_cosine_threshold,
                isfinite(float(self.gimbal_cosine_threshold))
                and 0.0 < self.gimbal_cosine_threshold < 1.0,
            ),
            (
                "excitation_rank_relative_threshold",
                self.excitation_rank_relative_threshold,
                isfinite(float(self.excitation_rank_relative_threshold))
                and 0.0 < self.excitation_rank_relative_threshold < 1.0,
            ),
        )
        for name, _, valid in bounded_checks:
            if not valid:
                raise ValueError(f"{name} is outside its valid range.")
            ####
        ####

        positive_fields = (
            "residual_floor_rad",
            "near_tie_delta_criterion",
            "slow_step_threshold_rad",
            "slow_total_motion_threshold_rad",
            "high_rate_change_angle_rad",
        )
        for name in positive_fields:
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
            ####
        ####
        if not isfinite(float(self.near_alias_step_rad)) or not 0.0 < self.near_alias_step_rad < pi:
            raise ValueError("near_alias_step_rad must be between 0 and pi.")
        ####
        if not isfinite(float(self.hard_alias_step_rad)) or self.hard_alias_step_rad < pi:
            raise ValueError("hard_alias_step_rad must be at least pi.")
        ####
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1.")
        ####
        if not isfinite(float(self.evidence_log_likelihood_ratio_threshold)) or (
            self.evidence_log_likelihood_ratio_threshold <= 0.0
        ):
            raise ValueError("evidence_log_likelihood_ratio_threshold must be positive and finite.")
        ####
        if self.evidence_min_intervals < 1:
            raise ValueError("evidence_min_intervals must be at least 1.")
        ####
        if self.evidence_stability_windows < 1:
            raise ValueError("evidence_stability_windows must be at least 1.")
        ####

        if not isinstance(cast(Any, self.domain_prior), DomainPrior):
            object.__setattr__(self, "domain_prior", DomainPrior.model_validate(self.domain_prior))
        ####
        if not isinstance(cast(Any, self.preference_profile), PreferenceProfile):
            object.__setattr__(
                self,
                "preference_profile",
                PreferenceProfile.model_validate(self.preference_profile),
            )
        ####
        self.constraints.apply(self.search_space)
    ####

    def resolved_search_space(self) -> SearchSpace:
        return self.constraints.apply(self.search_space)
    ####
####
