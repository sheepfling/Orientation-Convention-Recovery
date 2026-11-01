"""Typed convention search space and hard-constraint contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from enum import Enum
from typing import Any, Literal, Self, TypeVar, cast

from pydantic.dataclasses import dataclass

from .model_common import (
    CONSTRAINT_FIELDS,
    EULER_ORDERS,
    RATE_ORDERS,
    AngleUnit,
    AttitudeDirection,
    EulerMode,
    FrozenModel,
    QuaternionOrder,
    RateFrame,
    RateUnit,
    as_tuple,
    enum_tuple,
    normalize_euler_orders,
    normalize_rate_orders,
    normalize_signs,
    validate_nonempty_unique,
    validated_payload,
)

__all__ = ["ConventionConstraints", "SearchSpace"]

T = TypeVar("T")


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
        payload = validated_payload(cls, value)
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
                payload[name] = enum_tuple(enum_type, payload[name])
            ####
        ####
        if "euler_orders" in payload:
            payload["euler_orders"] = normalize_euler_orders(tuple(payload["euler_orders"]))
        ####
        if "rate_orders" in payload:
            payload["rate_orders"] = normalize_rate_orders(tuple(payload["rate_orders"]))
        ####
        if "rate_signs" in payload:
            payload["rate_signs"] = normalize_signs(tuple(payload["rate_signs"]))
        ####
        return cls(**payload)
    ####

    def __post_init__(self) -> None:
        for item in fields(self):
            validate_nonempty_unique(tuple(getattr(self, item.name)), "Search-space fields")
        ####
        normalize_euler_orders(tuple(str(item) for item in self.euler_orders))
        normalize_rate_orders(tuple(str(item) for item in self.rate_orders))
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
        normalize_signs(tuple(self.rate_signs))
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
        payload = validated_payload(cls, value)
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
                payload[name] = enum_tuple(enum_type, payload[name])
            ####
        ####
        if payload.get("euler_order") is not None:
            payload["euler_order"] = normalize_euler_orders(
                tuple(str(item) for item in as_tuple(payload["euler_order"]) or ())
            )
        ####
        if payload.get("rate_order") is not None:
            payload["rate_order"] = normalize_rate_orders(
                tuple(str(item) for item in as_tuple(payload["rate_order"]) or ())
            )
        ####
        if payload.get("rate_sign") is not None:
            payload["rate_sign"] = normalize_signs(
                tuple(cast(int | str, item) for item in as_tuple(payload["rate_sign"]) or ())
            )
        ####
        return cls(**payload)
    ####

    def __post_init__(self) -> None:
        for field_name in CONSTRAINT_FIELDS:
            value = getattr(self, field_name)
            if value is not None:
                validate_nonempty_unique(tuple(as_tuple(value) or ()), "Constraint fields")
            ####
        ####
    ####

    @staticmethod
    def _intersect_values(
        base: tuple[T, ...],
        constrained: tuple[T, ...] | None,
        dimension: str,
    ) -> tuple[T, ...]:
        if constrained is None:
            return base
        ####
        allowed = set(as_tuple(constrained) or ())
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
        for field_name in CONSTRAINT_FIELDS:
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
