"""Translate declared attitude and rate conventions into canonical quantities."""

from __future__ import annotations

from math import pi

from scipy.spatial.transform import Rotation

from .math_utils import FloatArray, rotation_from_euler, rotation_from_quat
from .models import (
    AngleUnit,
    AttitudeDirection,
    AttitudeRepresentation,
    ConventionHypothesis,
    EulerMode,
    QuaternionOrder,
    RateUnit,
    angle_unit_scale,
)
from .trajectory import TrajectoryData

__all__ = ["decode_attitudes", "decode_rates"]


def decode_attitudes(data: TrajectoryData, hypothesis: ConventionHypothesis) -> Rotation:
    """Decode stored attitude samples as active body-to-world rotations."""

    if data.orientation is None:
        raise ValueError("No attitude stream is available for this operation.")
    ####
    if data.representation is AttitudeRepresentation.QUATERNION:
        if hypothesis.quaternion_order is None:
            raise ValueError("Quaternion data require a quaternion-order hypothesis.")
        ####
        scalar_first = hypothesis.quaternion_order is QuaternionOrder.SCALAR_FIRST
        decoded = rotation_from_quat(data.orientation, scalar_first=scalar_first)
    else:
        if hypothesis.euler_order is None or hypothesis.euler_mode is None:
            raise ValueError("Euler data require Euler order and mode hypotheses.")
        ####
        angle_unit = hypothesis.angle_unit
        if angle_unit is None:
            raise ValueError("Euler data require an angle-unit hypothesis.")
        ####
        sequence = (
            hypothesis.euler_order.upper()
            if hypothesis.euler_mode is EulerMode.INTRINSIC
            else hypothesis.euler_order.lower()
        )
        decoded = rotation_from_euler(
            sequence,
            data.orientation * angle_unit_scale(AngleUnit(angle_unit)),
        )
    ####

    if hypothesis.attitude_direction is AttitudeDirection.WORLD_TO_BODY:
        decoded = decoded.inv()
    ####
    return decoded
####


def decode_rates(data: TrajectoryData, hypothesis: ConventionHypothesis) -> FloatArray:
    """Decode stored angular rates into body-order radians per second."""

    if data.rates is None:
        raise ValueError("No angular-rate stream is available for this operation.")
    ####
    positions = {component: index for index, component in enumerate(hypothesis.rate_order)}
    canonical = data.rates[:, [positions["p"], positions["q"], positions["r"]]].copy()
    if hypothesis.rate_unit is RateUnit.DEGREES_PER_SECOND:
        canonical *= pi / 180.0
    ####
    canonical *= float(hypothesis.rate_sign)
    return canonical
####
