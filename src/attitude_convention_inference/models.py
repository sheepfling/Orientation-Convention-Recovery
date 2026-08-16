"""Validated domain models and serialized inference report contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import field, fields, replace
from enum import Enum, StrEnum
from math import isfinite, pi
from typing import Any, Literal, Self, cast

from pydantic import TypeAdapter
from pydantic.dataclasses import dataclass

__all__ = [
    "AngleUnit",
    "AttitudeDirection",
    "AttitudeRepresentation",
    "CandidateResult",
    "ConfidenceLabel",
    "ConventionConstraints",
    "ConventionHypothesis",
    "ConventionPriors",
    "DimensionAssessment",
    "DimensionStatus",
    "DomainPrior",
    "EulerMode",
    "HeuristicFeatures",
    "HypothesisSpaceSummary",
    "InferenceConfig",
    "InferenceReport",
    "LagMetric",
    "OverallDiagnostics",
    "PreferenceProfile",
    "QuaternionOrder",
    "RateFrame",
    "RateUnit",
    "ResultLabel",
    "ScaleRisk",
    "SearchSpace",
    "WrapUnitStatus",
    "angle_unit_scale",
]

EULER_ORDERS: tuple[str, ...] = (
    # Tait--Bryan sequences (all three axes distinct).
    "xyz",
    "xzy",
    "yxz",
    "yzx",
    "zxy",
    "zyx",
    # Proper-Euler sequences (first and third axes repeated).
    "xyx",
    "xzx",
    "yxy",
    "yzy",
    "zxz",
    "zyz",
)
RATE_ORDERS: tuple[str, ...] = ("pqr", "prq", "qpr", "qrp", "rpq", "rqp")
_CONSTRAINT_FIELDS: tuple[str, ...] = (
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
_PRIOR_FIELDS: tuple[str, ...] = _CONSTRAINT_FIELDS


class AttitudeRepresentation(StrEnum):
    EULER = "euler"
    QUATERNION = "quaternion"
####




class AttitudeDirection(StrEnum):
    BODY_TO_WORLD = "body_to_world"
    WORLD_TO_BODY = "world_to_body"
####




class QuaternionOrder(StrEnum):
    SCALAR_FIRST = "scalar_first"
    SCALAR_LAST = "scalar_last"
####




class EulerMode(StrEnum):
    INTRINSIC = "intrinsic"
    EXTRINSIC = "extrinsic"
####




class AngleUnit(StrEnum):
    RADIANS = "radians"
    DEGREES = "degrees"
    TURNS = "turns"
    GRADS = "grads"
####




def angle_unit_scale(unit: AngleUnit) -> float:
    """Convert one raw angle unit to radians."""

    return {
        AngleUnit.RADIANS: 1.0,
        AngleUnit.DEGREES: pi / 180.0,
        AngleUnit.TURNS: 2.0 * pi,
        AngleUnit.GRADS: pi / 200.0,
    }[unit]
####




class RateFrame(StrEnum):
    BODY = "body"
    WORLD = "world"
####




class RateUnit(StrEnum):
    RADIANS_PER_SECOND = "radians_per_second"
    DEGREES_PER_SECOND = "degrees_per_second"
####




class ConfidenceLabel(StrEnum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INDETERMINATE = "indeterminate"
####




class ResultLabel(StrEnum):
    TRAJECTORY_QUOTIENT = "trajectory-quotient"
    TRAJECTORY_CLASS = "trajectory-quotient"
    SCALE_GAUGE = "scale-gauge"
    ORDER_PARTIAL = "order-partial"
    OVERFIT_NON_ID = "overfit/non-ID"
    ATTITUDE_HEURISTIC = "attitude-heuristic"
    RATE_EXTERNAL = "rate-external"
    RATE_NON_ID = "rate-non-ID"
    NO_DATA = "no-data"
####




class WrapUnitStatus(StrEnum):
    NOT_APPLICABLE = "not-applicable"
    UNIT_UNDETERMINED_BY_WRAP = "unit-undetermined-by-wrap"
    UNIT_SUGGESTED_BY_WRAP = "unit-suggested-by-wrap"
####




class DimensionStatus(StrEnum):
    FIXED = "fixed"
    INFERRED = "inferred"
    AMBIGUOUS = "ambiguous"
    NOT_APPLICABLE = "not_applicable"
####




class FrozenModel:
    """Shared Pydantic-backed behavior for immutable domain dataclasses."""

    @classmethod
    def model_validate(cls: type[Self], value: Any) -> Self:
        return TypeAdapter(cls).validate_python(value)
    ####


    def model_dump(self, *, mode: Literal["python", "json"] = "python") -> dict[str, Any]:
        if mode not in ("python", "json"):
            raise ValueError("mode must be 'python' or 'json'.")
        ####
        converted = TypeAdapter(type(self)).dump_python(self, mode=mode)
        return cast(dict[str, Any], converted)
    ####


    def model_dump_json(self, *, indent: int | None = None) -> str:
        return TypeAdapter(type(self)).dump_json(self, indent=indent).decode("utf-8")
    ####


    @classmethod
    def model_json_schema(cls) -> dict[str, Any]:
        """Return the JSON Schema generated from the Pydantic dataclass."""

        return TypeAdapter(cls).json_schema()
    ####


    def model_copy(self: Self, *, update: Mapping[str, object] | None = None) -> Self:
        return cast(Self, replace(cast(Any, self), **dict(update or {})))
    ####
####





def _as_tuple[T](value: T | Sequence[T] | set[T] | None) -> tuple[T, ...] | None:
    if value is None:
        return None
    ####
    if isinstance(value, (str, bytes, Enum)) or not isinstance(value, (Sequence, set)):
        return (cast(T, value),)
    ####
    if isinstance(value, set):
        return tuple(cast(set[T], value))
    ####
    return tuple(cast(Sequence[T], value))
####




def _enum_value[EnumT: Enum](enum_type: type[EnumT], value: EnumT | str) -> EnumT:
    return value if isinstance(value, enum_type) else enum_type(value)
####




def _enum_tuple[EnumT: Enum](
    enum_type: type[EnumT],
    value: EnumT | str | Sequence[EnumT | str],
) -> tuple[EnumT, ...]:
    raw = _as_tuple(value)
    if raw is None:
        raise ValueError("A non-empty value is required.")
    ####
    return tuple(_enum_value(enum_type, item) for item in raw)
####




def _validate_nonempty_unique(value: tuple[object, ...], label: str) -> None:
    if not value:
        raise ValueError(f"{label} may not be empty.")
    ####
    if len(set(value)) != len(value):
        raise ValueError(f"{label} may not contain duplicates.")
    ####
####




def _normalize_euler_orders(value: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(order).lower() for order in value)
    for order in normalized:
        if (
            len(order) != 3
            or any(axis not in "xyz" for axis in order)
            or order[0] == order[1]
            or order[1] == order[2]
        ):
            raise ValueError(f"Euler order {order!r} is not a valid three-axis Euler sequence.")
        ####
    ####
    return normalized
####




def _normalize_rate_orders(value: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(order).lower() for order in value)
    for order in normalized:
        if len(order) != 3 or set(order) != {"p", "q", "r"}:
            raise ValueError(f"Rate order {order!r} is not a p/q/r permutation.")
        ####
    ####
    return normalized
####




def _normalize_signs(value: Sequence[int | str]) -> tuple[Literal[-1, 1], ...]:
    signs = tuple(int(item) for item in value)
    for sign in signs:
        if sign not in (-1, 1):
            raise ValueError("Rate signs must be either -1 or 1.")
        ####
    ####
    return cast(tuple[Literal[-1, 1], ...], signs)
####




@dataclass(frozen=True, slots=True)
class SearchSpace(FrozenModel):
    attitude_directions: tuple[AttitudeDirection, ...] = (
        AttitudeDirection.BODY_TO_WORLD,
        AttitudeDirection.WORLD_TO_BODY,
    )
    quaternion_orders: tuple[QuaternionOrder, ...] = (
        QuaternionOrder.SCALAR_FIRST,
        QuaternionOrder.SCALAR_LAST,
    )
    euler_orders: tuple[str, ...] = EULER_ORDERS
    euler_modes: tuple[EulerMode, ...] = (EulerMode.INTRINSIC, EulerMode.EXTRINSIC)
    angle_units: tuple[AngleUnit, ...] = (AngleUnit.RADIANS,)
    rate_frames: tuple[RateFrame, ...] = (RateFrame.BODY, RateFrame.WORLD)
    rate_orders: tuple[str, ...] = RATE_ORDERS
    rate_units: tuple[RateUnit, ...] = (RateUnit.RADIANS_PER_SECOND,)
    rate_signs: tuple[Literal[-1, 1], ...] = (1,)

    @classmethod
    def model_validate(cls, value: Self | Mapping[str, Any]) -> Self:
        payload = _validated_payload(cls, value)
        enum_fields: tuple[tuple[str, type[Enum]], ...] = (
            ("attitude_directions", AttitudeDirection),
            ("quaternion_orders", QuaternionOrder),
            ("euler_modes", EulerMode),
            ("angle_units", AngleUnit),
            ("rate_frames", RateFrame),
            ("rate_units", RateUnit),
        )
        for name, enum_type in enum_fields:
            if name in payload:
                payload[name] = _enum_tuple(enum_type, payload[name])
            ####
        ####
        if "euler_orders" in payload:
            payload["euler_orders"] = _normalize_euler_orders(tuple(payload["euler_orders"]))
        ####
        if "rate_orders" in payload:
            payload["rate_orders"] = _normalize_rate_orders(tuple(payload["rate_orders"]))
        ####
        if "rate_signs" in payload:
            payload["rate_signs"] = _normalize_signs(tuple(payload["rate_signs"]))
        ####
        return cls(**payload)
    ####


    def __post_init__(self) -> None:
        for item in fields(self):
            _validate_nonempty_unique(tuple(getattr(self, item.name)), "Search-space fields")
        ####
        _normalize_euler_orders(tuple(str(item) for item in self.euler_orders))
        _normalize_rate_orders(tuple(str(item) for item in self.rate_orders))
        for name, enum_type in (
            ("attitude_directions", AttitudeDirection),
            ("quaternion_orders", QuaternionOrder),
            ("euler_modes", EulerMode),
            ("angle_units", AngleUnit),
            ("rate_frames", RateFrame),
            ("rate_units", RateUnit),
        ):
            for value in getattr(self, name):
                enum_type(value)
            ####
        ####
        _normalize_signs(tuple(self.rate_signs))
    ####
####





@dataclass(frozen=True, slots=True)
class ConventionConstraints(FrozenModel):
    """Hard constraints layered on top of :class:`SearchSpace`.

    A scalar locks a dimension, a sequence retains an allowed subset, and ``None``
    leaves the dimension unknown.
    """

    attitude_direction: tuple[AttitudeDirection, ...] | AttitudeDirection | str | None = None
    quaternion_order: tuple[QuaternionOrder, ...] | QuaternionOrder | str | None = None
    euler_order: tuple[str, ...] | str | None = None
    euler_mode: tuple[EulerMode, ...] | EulerMode | str | None = None
    angle_unit: tuple[AngleUnit, ...] | AngleUnit | str | None = None
    rate_frame: tuple[RateFrame, ...] | RateFrame | str | None = None
    rate_order: tuple[str, ...] | str | None = None
    rate_unit: tuple[RateUnit, ...] | RateUnit | str | None = None
    rate_sign: tuple[Literal[-1, 1], ...] | Literal[-1, 1] | str | None = None

    @classmethod
    def model_validate(cls, value: Self | Mapping[str, Any]) -> Self:
        payload = _validated_payload(cls, value)
        enum_fields: tuple[tuple[str, type[Enum]], ...] = (
            ("attitude_direction", AttitudeDirection),
            ("quaternion_order", QuaternionOrder),
            ("euler_mode", EulerMode),
            ("angle_unit", AngleUnit),
            ("rate_frame", RateFrame),
            ("rate_unit", RateUnit),
        )
        for name, enum_type in enum_fields:
            if payload.get(name) is not None:
                payload[name] = _enum_tuple(enum_type, payload[name])
            ####
        ####
        if payload.get("euler_order") is not None:
            payload["euler_order"] = _normalize_euler_orders(
                tuple(str(item) for item in _as_tuple(payload["euler_order"]) or ())
            )
        ####
        if payload.get("rate_order") is not None:
            payload["rate_order"] = _normalize_rate_orders(
                tuple(str(item) for item in _as_tuple(payload["rate_order"]) or ())
            )
        ####
        if payload.get("rate_sign") is not None:
            payload["rate_sign"] = _normalize_signs(
                tuple(cast(int | str, item) for item in _as_tuple(payload["rate_sign"]) or ())
            )
        ####
        return cls(**payload)
    ####


    def __post_init__(self) -> None:
        for field_name in _CONSTRAINT_FIELDS:
            value = getattr(self, field_name)
            if value is not None:
                _validate_nonempty_unique(tuple(_as_tuple(value) or ()), "Constraint fields")
            ####
        ####
    ####


    @staticmethod
    def _intersect_values[T](
        base: tuple[T, ...],
        constrained: tuple[T, ...] | None,
        dimension: str,
    ) -> tuple[T, ...]:
        if constrained is None:
            return base
        ####
        allowed = set(_as_tuple(constrained) or ())
        result = tuple(value for value in base if value in allowed)
        if not result:
            raise ValueError(
                f"Constraint for {dimension!r} has no overlap with the configured search space."
            )
        ####
        return result
    ####


    def apply(self, base: SearchSpace) -> SearchSpace:
        return SearchSpace(
            attitude_directions=self._intersect_values(
                base.attitude_directions,
                cast(tuple[AttitudeDirection, ...] | None, self.attitude_direction),
                "attitude_direction",
            ),
            quaternion_orders=self._intersect_values(
                base.quaternion_orders,
                cast(tuple[QuaternionOrder, ...] | None, self.quaternion_order),
                "quaternion_order",
            ),
            euler_orders=self._intersect_values(
                base.euler_orders,
                cast(tuple[str, ...] | None, self.euler_order),
                "euler_order",
            ),
            euler_modes=self._intersect_values(
                base.euler_modes,
                cast(tuple[EulerMode, ...] | None, self.euler_mode),
                "euler_mode",
            ),
            angle_units=self._intersect_values(
                base.angle_units,
                cast(tuple[AngleUnit, ...] | None, self.angle_unit),
                "angle_unit",
            ),
            rate_frames=self._intersect_values(
                base.rate_frames,
                cast(tuple[RateFrame, ...] | None, self.rate_frame),
                "rate_frame",
            ),
            rate_orders=self._intersect_values(
                base.rate_orders,
                cast(tuple[str, ...] | None, self.rate_order),
                "rate_order",
            ),
            rate_units=self._intersect_values(
                base.rate_units,
                cast(tuple[RateUnit, ...] | None, self.rate_unit),
                "rate_unit",
            ),
            rate_signs=self._intersect_values(
                base.rate_signs,
                cast(tuple[Literal[-1, 1], ...] | None, self.rate_sign),
                "rate_sign",
            ),
        )
    ####


    def intersect(self, other: ConventionConstraints) -> ConventionConstraints:
        updates: dict[str, tuple[object, ...] | None] = {}
        for field_name in _CONSTRAINT_FIELDS:
            left = cast(tuple[object, ...] | None, getattr(self, field_name))
            right = cast(tuple[object, ...] | None, getattr(other, field_name))
            if left is None:
                updates[field_name] = right
                continue
            ####
            if right is None:
                updates[field_name] = left
                continue
            ####
            right_values = set(right)
            overlap = tuple(value for value in left if value in right_values)
            if not overlap:
                raise ValueError(f"Hard constraints for {field_name!r} are contradictory.")
            ####
            updates[field_name] = overlap
        ####
        return ConventionConstraints.model_validate(updates)
    ####
####





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




def _validated_payload[T](cls: type[T], value: T | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, cls):
        instance = cast(Any, value)
        return {item.name: getattr(instance, item.name) for item in fields(instance)}
    ####
    if not isinstance(value, Mapping):
        raise TypeError(f"{cls.__name__} requires a mapping or an existing instance.")
    ####
    mapping = cast(Mapping[str, Any], value)
    return {key: item for key, item in mapping.items()}
####




@dataclass(frozen=True, slots=True)
class ConventionPriors(FrozenModel):
    """Relative soft-prior weights; missing values have neutral weight ``1.0``."""

    attitude_direction: dict[AttitudeDirection | str, float] = field(
        default_factory=dict[AttitudeDirection | str, float]
    )
    quaternion_order: dict[QuaternionOrder | str, float] = field(
        default_factory=dict[QuaternionOrder | str, float]
    )
    euler_order: dict[str, float] = field(default_factory=dict[str, float])
    euler_mode: dict[EulerMode | str, float] = field(default_factory=dict[EulerMode | str, float])
    angle_unit: dict[AngleUnit | str, float] = field(default_factory=dict[AngleUnit | str, float])
    rate_frame: dict[RateFrame | str, float] = field(default_factory=dict[RateFrame | str, float])
    rate_order: dict[str, float] = field(default_factory=dict[str, float])
    rate_unit: dict[RateUnit | str, float] = field(default_factory=dict[RateUnit | str, float])
    rate_sign: dict[Literal[-1, 1] | str, float] = field(
        default_factory=dict[Literal[-1, 1] | str, float]
    )

    @classmethod
    def model_validate(cls, value: Self | Mapping[str, Any]) -> Self:
        payload = _validated_payload(cls, value)
        converters: tuple[tuple[str, Callable[[Any], Any], str], ...] = (
            (
                "attitude_direction",
                lambda item: _enum_value(AttitudeDirection, item),
                "attitude_direction",
            ),
            (
                "quaternion_order",
                lambda item: _enum_value(QuaternionOrder, item),
                "quaternion_order",
            ),
            ("euler_order", lambda item: _normalize_euler_orders((str(item),))[0], "euler_order"),
            ("euler_mode", lambda item: _enum_value(EulerMode, item), "euler_mode"),
            ("angle_unit", lambda item: _enum_value(AngleUnit, item), "angle_unit"),
            ("rate_frame", lambda item: _enum_value(RateFrame, item), "rate_frame"),
            ("rate_order", lambda item: _normalize_rate_orders((str(item),))[0], "rate_order"),
            ("rate_unit", lambda item: _enum_value(RateUnit, item), "rate_unit"),
            ("rate_sign", lambda item: _normalize_signs((cast(int | str, item),))[0], "rate_sign"),
        )
        for name, converter, label in converters:
            if payload.get(name) is not None:
                payload[name] = _prior_mapping(payload[name], key_converter=converter, label=label)
            ####
        ####
        return cls(**payload)
    ####


    def configured_dimensions(self) -> tuple[str, ...]:
        return tuple(field_name for field_name in _PRIOR_FIELDS if bool(getattr(self, field_name)))
    ####
####





@dataclass(frozen=True, slots=True)
class DomainPrior(FrozenModel):
    """Class-level provenance prior applied only after trajectory contenders exist."""

    class_weights: dict[str, float] = field(default_factory=dict[str, float])
    context: dict[str, str] = field(default_factory=dict[str, str])
    strength: float = 1.0

    @classmethod
    def model_validate(cls, value: Self | Mapping[str, Any]) -> Self:
        payload = _validated_payload(cls, value)
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
    context: dict[str, str] = field(default_factory=dict[str, str])
    class_weights: dict[str, float] = field(default_factory=dict[str, float])
    representative_preferences: ConventionPriors = field(default_factory=ConventionPriors)
    strength: float = 1.0

    @classmethod
    def model_validate(cls, value: Self | Mapping[str, Any]) -> Self:
        payload = _validated_payload(cls, value)
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
        if not isinstance(cast(Any, self.representative_preferences), ConventionPriors):
            object.__setattr__(
                self,
                "representative_preferences",
                ConventionPriors.model_validate(self.representative_preferences),
            )
        ####
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
        payload = _validated_payload(cls, value)
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





@dataclass(frozen=True, slots=True)
class ConventionHypothesis(FrozenModel):
    attitude_direction: AttitudeDirection | str
    rate_frame: RateFrame | str
    rate_order: str
    rate_unit: RateUnit | str
    quaternion_order: QuaternionOrder | str | None = None
    euler_order: str | None = None
    euler_mode: EulerMode | str | None = None
    angle_unit: AngleUnit | str | None = None
    rate_sign: Literal[-1, 1] | int = 1

    @classmethod
    def model_validate(cls, value: Self | Mapping[str, Any]) -> Self:
        payload = _validated_payload(cls, value)
        for name, enum_type in (
            ("attitude_direction", AttitudeDirection),
            ("rate_frame", RateFrame),
            ("rate_unit", RateUnit),
            ("quaternion_order", QuaternionOrder),
            ("euler_mode", EulerMode),
            ("angle_unit", AngleUnit),
        ):
            if payload.get(name) is not None:
                payload[name] = _enum_value(enum_type, payload[name])
            ####
        ####
        payload["rate_order"] = _normalize_rate_orders((payload["rate_order"],))[0]
        payload["rate_sign"] = _normalize_signs((payload.get("rate_sign", 1),))[0]
        if payload.get("euler_order") is not None:
            payload["euler_order"] = _normalize_euler_orders((payload["euler_order"],))[0]
        ####
        return cls(**payload)
    ####


    def __post_init__(self) -> None:
        if self.quaternion_order is not None and not isinstance(
            self.quaternion_order, QuaternionOrder
        ):
            raise TypeError("quaternion_order must be a QuaternionOrder.")
        ####

        quaternion = self.quaternion_order is not None
        euler = any(
            value is not None for value in (self.euler_order, self.euler_mode, self.angle_unit)
        )
        if quaternion == euler:
            raise ValueError("A hypothesis must define exactly one attitude representation.")
        ####
        if euler and any(
            value is None for value in (self.euler_order, self.euler_mode, self.angle_unit)
        ):
            raise ValueError("Euler hypotheses require order, mode, and angle unit.")
        ####
    ####


    def identifier(self) -> str:
        if self.quaternion_order is not None:
            attitude = f"quat:{self.quaternion_order}:{self.attitude_direction}"
        else:
            attitude = (
                f"euler:{self.euler_mode}:{self.euler_order}:{self.angle_unit}:"
                f"{self.attitude_direction}"
            )
        ####
        return (
            f"{attitude}|rate:{self.rate_frame}:{self.rate_order}:{self.rate_unit}:"
            f"sign={self.rate_sign:+d}"
        )
    ####


    def structural_dual(self) -> Self:
        """Return the exact direction--frame--global-sign dual hypothesis."""

        direction = (
            AttitudeDirection.WORLD_TO_BODY
            if self.attitude_direction is AttitudeDirection.BODY_TO_WORLD
            else AttitudeDirection.BODY_TO_WORLD
        )
        frame = RateFrame.WORLD if self.rate_frame is RateFrame.BODY else RateFrame.BODY
        return self.model_copy(
            update={
                "attitude_direction": direction,
                "rate_frame": frame,
                "rate_sign": -self.rate_sign,
            }
        )
    ####


    def structural_equivalence_id(self) -> str:
        """Stable identifier for the direction--frame--sign equivalence class."""

        return min(self.identifier(), self.structural_dual().identifier())
    ####
####





@dataclass(frozen=True, slots=True)
class LagMetric(FrozenModel):
    window_size: int
    window_count: int
    residual_rmse_rad: float
    residual_median_rad: float
    residual_p95_rad: float
    observed_rotation_rms_rad: float
####




@dataclass(frozen=True, slots=True)
class ScaleRisk(FrozenModel):
    """Raw geometric losses for one overlapping window scale."""

    window_size: int
    trimmed_mse_rad2: float
    untrimmed_mse_rad2: float
    weight: float
####




@dataclass(frozen=True, slots=True)
class HeuristicFeatures(FrozenModel):
    """Representation-only features; never interpret these as probabilities.

    Trajectory risk and domain context are intentionally excluded from this vector.
    A calibrated classifier may later consume the representation feature vector, but
    this package does not label heuristic scores as probabilities.
    """

    schema_penalty: float = 0.0
    wrap_score: float = 0.0
    domain_score: float = 0.0
    branch_penalty: float = 0.0
    singularity_penalty: float = 0.0
    physical_penalty: float = 0.0
    coarse_risk: float = 0.0
    wrap_gain_raw: float = 0.0
    inferred_wrap_count: int = 0
    wrap_unit_status: WrapUnitStatus = WrapUnitStatus.NOT_APPLICABLE
    representation_score: float = 0.0
    representation_feature_vector: Mapping[str, float] = field(default_factory=dict[str, float])
    classifier_probability: float | None = None
####




@dataclass(frozen=True, slots=True)
class CandidateResult(FrozenModel):
    hypothesis: ConventionHypothesis
    criterion: float
    data_criterion: float
    fit_score: float
    normalized_rmse: float
    multiscale_rmse_rad: float
    raw_geometric_risk: float
    untrimmed_geometric_risk: float
    scale_risks: tuple[ScaleRisk, ...]
    interval_rmse_rad: float
    interval_median_rad: float
    interval_p95_rad: float
    interval_max_rad: float
    observed_step_rms_rad: float
    valid_interval_count: int
    near_alias_fraction: float
    hard_alias_fraction: float
    rate_change_rms_angle_rad: float
    lag_metrics: tuple[LagMetric, ...]
    rank: int = 0
    data_rank: int = 0
    relative_support: float = 0.0
    data_relative_support: float = 0.0
    delta_criterion: float = 0.0
    data_delta_criterion: float = 0.0
    prior_relative_weight: float = 1.0
    prior_penalty: float = 0.0
    inferred_sample_period_s: float | None = None
    gimbal_lock_fraction: float | None = None
    equivalence_class_id: str = ""
    exact_equivalent_hypothesis_ids: tuple[str, ...] = ()
    euler_range_penalty_rad2: float = 0.0
    euler_branch_switch_count: int = 0
    euler_singularity_margin: float | None = None
    euler_branch_cost_rad2: float = 0.0
    heuristic_features: HeuristicFeatures = HeuristicFeatures()
    warnings: tuple[str, ...] = ()
    class_data_risk: float = 0.0
    empirical_gap_to_next_class: float | None = None
    partial_lower_bound: float = 0.0
    exact_pruned: bool = False
####




@dataclass(frozen=True, slots=True)
class DimensionAssessment(FrozenModel):
    status: DimensionStatus
    selected_value: str | None
    allowed_values: tuple[str, ...]
    evidence_near_tie_values: tuple[str, ...]
    posterior_near_tie_values: tuple[str, ...]
    relative_prior_weights: Mapping[str, float] | None = None
####




@dataclass(frozen=True, slots=True)
class HypothesisSpaceSummary(FrozenModel):
    base_hypothesis_count: int
    constrained_hypothesis_count: int
    reduction_fraction: float
    fixed_dimensions: Mapping[str, str]
    searched_dimensions: Mapping[str, tuple[str, ...]]
    not_applicable_dimensions: tuple[str, ...]
    relative_prior_weights: Mapping[str, Mapping[str, float]]
####




@dataclass(frozen=True, slots=True)
class OverallDiagnostics(FrozenModel):
    confidence: float
    confidence_label: ConfidenceLabel
    best_relative_support: float
    best_data_relative_support: float
    near_tie_count: int
    near_tie_hypothesis_ids: tuple[str, ...]
    unresolved_dimensions: Mapping[str, tuple[str, ...]]
    posterior_near_tie_count: int
    posterior_near_tie_hypothesis_ids: tuple[str, ...]
    posterior_unresolved_dimensions: Mapping[str, tuple[str, ...]]
    dimension_assessments: Mapping[str, DimensionAssessment]
    effective_excitation_rank: float
    excitation_singular_values: tuple[float, float, float]
    total_observed_motion_rad: float
    observed_step_rms_rad: float
    near_alias_fraction: float
    hard_alias_fraction: float
    time_source: Literal["timestamps", "provided_sample_period", "inferred_sample_period"]
    warnings: tuple[str, ...]
    noise_model: str = "isotropic_rotation_vector_gaussian"
    estimated_noise_sigma_rad: float = 0.0
    noise_residual_count: int = 0
    noise_lag1_autocorrelation: float = 0.0
    evidence_sufficient: bool = False
    evidence_intervals_used: int = 0
    evidence_fraction: float = 0.0
    evidence_log_likelihood_ratio: float = 0.0
    exact_equivalence_class_count: int = 0
    best_equivalence_class_hypothesis_ids: tuple[str, ...] = ()
    rate_scale_period_gauge_active: bool = False
    raw_excitation_rank: int = 0
    raw_excitation_eigenvalues: tuple[float, float, float] = (0.0, 0.0, 0.0)
    raw_excitation_condition_ratio: float = 0.0
####




@dataclass(frozen=True, slots=True)
class InferenceReport(FrozenModel):
    representation: AttitudeRepresentation
    sample_count: int
    hypothesis_count: int
    hypothesis_space: HypothesisSpaceSummary
    best: CandidateResult
    rankings: tuple[CandidateResult, ...]
    diagnostics: OverallDiagnostics
    result_label: ResultLabel = ResultLabel.TRAJECTORY_CLASS
    evidence_contender_class_ids: tuple[str, ...] = ()
    data_winning_class_ids: tuple[str, ...] = ()
    empirical_gap_to_next_class: float | None = None
    operational_class_id: str = ""
    operational_hypothesis_id: str = ""
    operational_reason: str = "trajectory"
    domain_prior_configured: bool = False
    domain_prior_abstained: bool = False
    domain_prior_supported_class_ids: tuple[str, ...] = ()
    preference_profile_id: str = ""
    preference_profile_source: str = ""
    preference_profile_configured: bool = False
    preference_profile_abstained: bool = False
    preference_profile_supported_class_ids: tuple[str, ...] = ()
    representative_selection_reason: str = "canonical-policy"
####


