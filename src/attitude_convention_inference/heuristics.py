"""Representation-only heuristics.

These diagnostics describe Euler ranges, branch behavior, gimbal proximity, and
visible angle wrapping.  They are intentionally separate from trajectory-fit
evidence and must not decide a convention on their own.
"""

from __future__ import annotations

from dataclasses import dataclass

from numpy import (
    abs,
    amin,
    argmin,
    cos,
    diff,
    float64,
    int8,
    maximum,
    mean,
    newaxis,
    pi,
    rint,
    sign,
    sin,
    square,
    stack,
    where,
    zeros,
)
from numpy import (
    sum as np_sum,
)

from .core import TrajectoryData
from .math_utils import FloatArray
from .models import (
    AngleUnit,
    AttitudeRepresentation,
    ConventionHypothesis,
    InferenceConfig,
    WrapUnitStatus,
    angle_unit_scale,
)

__all__ = [
    "EulerHeuristicDiagnostics",
    "WrapUnitDiagnostics",
    "euler_heuristic_diagnostics",
    "gimbal_lock_fraction",
    "wrap_unit_diagnostics",
]


@dataclass(frozen=True, slots=True)
class WrapUnitDiagnostics:
    """Diagnostics produced while testing a candidate rate scale."""

    gain_raw: float
    wrap_count: int
    status: WrapUnitStatus
    score: float
####


@dataclass(frozen=True, slots=True)
class EulerHeuristicDiagnostics:
    """Named diagnostics for Euler range, branch, and singularity behavior."""

    range_penalty_rad2: float
    branch_switch_count: int
    singularity_margin: float | None
    branch_cost_rad2: float
####


def gimbal_lock_fraction(
    data: TrajectoryData,
    hypothesis: ConventionHypothesis,
    config: InferenceConfig,
) -> float | None:
    """Return the fraction of Euler samples near a singularity surface."""

    if data.representation is not AttitudeRepresentation.EULER:
        return None
    ####
    if data.orientation is None or hypothesis.angle_unit is None:
        return None
    ####
    angle_unit = AngleUnit(hypothesis.angle_unit)
    middle = data.orientation[:, 1] * angle_unit_scale(angle_unit)
    proper_euler = hypothesis.euler_order is not None and (
        hypothesis.euler_order[0] == hypothesis.euler_order[2]
    )
    singular_measure = sin(middle) if proper_euler else cos(middle)
    return float(mean(abs(singular_measure) < config.gimbal_cosine_threshold))
####


def _wrap_angle(values: FloatArray) -> FloatArray:
    return (values + pi) % (2.0 * pi) - pi
####


def wrap_unit_diagnostics(
    data: TrajectoryData,
    hypothesis: ConventionHypothesis,
) -> WrapUnitDiagnostics:
    """Score visible angle-period crossings, abstaining when none are present."""

    if data.representation is not AttitudeRepresentation.EULER or hypothesis.angle_unit is None:
        return WrapUnitDiagnostics(0.0, 0, WrapUnitStatus.NOT_APPLICABLE, 0.0)
    ####
    angle_unit = AngleUnit(hypothesis.angle_unit)
    if data.orientation is None:
        return WrapUnitDiagnostics(0.0, 0, WrapUnitStatus.NOT_APPLICABLE, 0.0)
    ####
    periods = {
        AngleUnit.RADIANS: 2.0 * pi,
        AngleUnit.DEGREES: 360.0,
        AngleUnit.TURNS: 1.0,
        AngleUnit.GRADS: 400.0,
    }
    period = periods[angle_unit]
    deltas = diff(data.orientation, axis=0)
    corrections = where(
        abs(deltas) > 0.5 * period,
        -sign(deltas) * period,
        0.0,
    )
    corrected = deltas + corrections
    gain = float(np_sum(maximum(abs(deltas) - abs(corrected), 0.0)))
    wrap_count = int(rint(np_sum(abs(corrections) / period)))
    status = (
        WrapUnitStatus.UNIT_UNDETERMINED_BY_WRAP
        if gain <= 1.0e-12
        else WrapUnitStatus.UNIT_SUGGESTED_BY_WRAP
    )
    return WrapUnitDiagnostics(
        gain_raw=gain,
        wrap_count=wrap_count,
        status=status,
        score=-gain + 0.25 * wrap_count,
    )
####


def euler_heuristic_diagnostics(
    data: TrajectoryData,
    hypothesis: ConventionHypothesis,
) -> EulerHeuristicDiagnostics:
    """Compute Euler branch/range diagnostics without treating them as evidence."""

    if data.representation is not AttitudeRepresentation.EULER:
        return EulerHeuristicDiagnostics(0.0, 0, None, 0.0)
    ####
    if hypothesis.euler_order is None or hypothesis.angle_unit is None:
        return EulerHeuristicDiagnostics(0.0, 0, None, 0.0)
    ####
    if data.orientation is None:
        return EulerHeuristicDiagnostics(0.0, 0, None, 0.0)
    ####

    angles = data.orientation.copy()
    angles *= angle_unit_scale(AngleUnit(hypothesis.angle_unit))
    tait_bryan = hypothesis.euler_order[0] != hypothesis.euler_order[2]
    middle_lower, middle_upper = (-0.5 * pi, 0.5 * pi) if tait_bryan else (0.0, pi)
    outer_violation = maximum(abs(angles[:, (0, 2)]) - pi, 0.0)
    middle_violation = maximum(
        maximum(middle_lower - angles[:, 1], angles[:, 1] - middle_upper),
        0.0,
    )
    range_penalty = float(mean(np_sum(square(outer_violation), axis=1) + square(middle_violation)))

    singular_measure = sin(angles[:, 1]) if not tait_bryan else cos(angles[:, 1])
    singularity_margin = float(amin(abs(singular_measure)))

    alternative = angles.copy()
    alternative[:, 0] += pi
    alternative[:, 2] += pi
    alternative[:, 1] = pi - angles[:, 1] if tait_bryan else -angles[:, 1]
    branches = stack((angles, alternative), axis=1)

    branch_switch_penalty = 0.25
    sample_count = data.sample_count
    costs = zeros((sample_count, 2), dtype=float64)
    predecessors = zeros((sample_count, 2), dtype=int8)
    for index in range(1, sample_count):
        transition = np_sum(
            square(
                _wrap_angle(branches[index, newaxis, :, :] - branches[index - 1, :, newaxis, :])
            ),
            axis=2,
        )
        for state in range(2):
            options = costs[index - 1] + transition[:, state]
            options[1 - state] += branch_switch_penalty
            predecessor = int(argmin(options))
            predecessors[index, state] = predecessor
            costs[index, state] = options[predecessor]
        ####
    ####

    state = int(argmin(costs[-1]))
    switches = 0
    for index in range(sample_count - 1, 0, -1):
        previous = int(predecessors[index, state])
        switches += int(previous != state)
        state = previous
    ####
    return EulerHeuristicDiagnostics(
        range_penalty_rad2=range_penalty,
        branch_switch_count=switches,
        singularity_margin=singularity_margin,
        branch_cost_rad2=float(amin(costs[-1])),
    )
####
