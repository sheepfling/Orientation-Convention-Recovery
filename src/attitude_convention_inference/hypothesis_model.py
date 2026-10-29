"""One fully specified orientation and angular-rate convention."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic.dataclasses import dataclass

from .model_common import (
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

__all__ = ["ConventionHypothesis"]


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
        return cls(**validated_payload(cls, value))
    ####

    def __post_init__(self) -> None:
        for name, enum_type in (
            ("attitude_direction", AttitudeDirection),
            ("rate_frame", RateFrame),
            ("rate_unit", RateUnit),
            ("quaternion_order", QuaternionOrder),
            ("euler_mode", EulerMode),
            ("angle_unit", AngleUnit),
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, enum_value(enum_type, value))
            ####
        ####
        object.__setattr__(self, "rate_order", normalize_rate_orders((self.rate_order,))[0])
        object.__setattr__(self, "rate_sign", normalize_signs((self.rate_sign,))[0])
        if self.euler_order is not None:
            object.__setattr__(self, "euler_order", normalize_euler_orders((self.euler_order,))[0])
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
