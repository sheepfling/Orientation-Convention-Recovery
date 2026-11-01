"""Shared typed enums, validation helpers, and Pydantic model behavior."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, replace
from enum import Enum, StrEnum
from math import pi
from typing import Any, Literal, Self, TypeVar, cast

from pydantic import TypeAdapter

__all__ = [
    "CONSTRAINT_FIELDS",
    "EULER_ORDERS",
    "PRIOR_FIELDS",
    "RATE_ORDERS",
    "AngleUnit",
    "AttitudeDirection",
    "AttitudeRepresentation",
    "ConfidenceLabel",
    "DimensionStatus",
    "EulerMode",
    "FrozenModel",
    "QuaternionOrder",
    "RateFrame",
    "RateUnit",
    "ResultLabel",
    "WrapUnitStatus",
    "angle_unit_scale",
    "as_tuple",
    "enum_tuple",
    "enum_value",
    "normalize_euler_orders",
    "normalize_rate_orders",
    "normalize_signs",
    "validate_nonempty_unique",
    "validated_payload",
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
CONSTRAINT_FIELDS: tuple[str, ...] = (
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
PRIOR_FIELDS: tuple[str, ...] = CONSTRAINT_FIELDS
T = TypeVar("T")
EnumT = TypeVar("EnumT", bound=Enum)


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


def confidence_label_for_score(value: float) -> ConfidenceLabel:
    """Map the uncalibrated diagnostic score to its deterministic display band."""

    if value >= 0.75:
        return ConfidenceLabel.HIGH
    ####
    if value >= 0.50:
        return ConfidenceLabel.MODERATE
    ####
    if value >= 0.20:
        return ConfidenceLabel.LOW
    ####
    return ConfidenceLabel.INDETERMINATE
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


def as_tuple(value: T | Sequence[T] | set[T] | None) -> tuple[T, ...] | None:
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


def enum_value(enum_type: type[EnumT], value: EnumT | str) -> EnumT:
    return value if isinstance(value, enum_type) else enum_type(value)
####


def enum_tuple(
    enum_type: type[EnumT],
    value: EnumT | str | Sequence[EnumT | str],
) -> tuple[EnumT, ...]:
    raw = as_tuple(value)
    if raw is None:
        raise ValueError("A non-empty value is required.")
    ####
    return tuple(enum_value(enum_type, item) for item in raw)
####


def validate_nonempty_unique(value: tuple[object, ...], label: str) -> None:
    if not value:
        raise ValueError(f"{label} may not be empty.")
    ####
    if len(set(value)) != len(value):
        raise ValueError(f"{label} may not contain duplicates.")
    ####
####


def normalize_euler_orders(value: Sequence[str]) -> tuple[str, ...]:
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


def normalize_rate_orders(value: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(order).lower() for order in value)
    for order in normalized:
        if len(order) != 3 or set(order) != {"p", "q", "r"}:
            raise ValueError(f"Rate order {order!r} is not a p/q/r permutation.")
        ####
    ####
    return normalized
####


def normalize_signs(value: Sequence[int | str]) -> tuple[Literal[-1, 1], ...]:
    signs = tuple(int(item) for item in value)
    for sign in signs:
        if sign not in (-1, 1):
            raise ValueError("Rate signs must be either -1 or 1.")
        ####
    ####
    return cast(tuple[Literal[-1, 1], ...], signs)
####


def validated_payload(cls: type[T], value: T | Mapping[str, Any]) -> dict[str, Any]:
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
