"""Opt-in diagnostics for sensitivity to rate-channel calibration assumptions."""

from __future__ import annotations

from itertools import product
from math import log
from typing import Literal, cast

from numpy import abs as np_abs
from numpy import asarray, bool_, count_nonzero, float64
from numpy.typing import NDArray

from .coercion import ConfigInput, coerce_config
from .inference import infer_trajectory_conventions
from .models import InferenceReport
from .report_models import (
    RateInputChannelRailHits,
    RateInputRailLimits,
    RatePolarityByInputAxis,
    RatePolarityProfileFit,
    RatePolaritySensitivityReport,
    RateRailInspectionReport,
)
from .trajectory import TrajectoryData

__all__ = ["assess_rate_polarity_sensitivity", "inspect_rate_rails"]

BoolArray = NDArray[bool_]


def _longest_true_run(values: BoolArray) -> int:
    """Return the length of the longest consecutive true run."""

    longest_run = 0
    current_run = 0
    for value in values:
        if bool(value):
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
        ####
    ####
    return longest_run
####


def _relative_log_risk_delta(
    profile_risk: float,
    reference_risk: float,
    risk_floor: float,
) -> float:
    """Compare two nonnegative risks without forming an overflow-prone ratio."""

    return log(max(profile_risk, risk_floor)) - log(max(reference_risk, risk_floor))
####


def assess_rate_polarity_sensitivity(
    trajectory: TrajectoryData,
    config: ConfigInput = None,
) -> RatePolaritySensitivityReport:
    """Compare convention fits under all eight raw rate-column polarity patterns.

    Each pattern multiplies the three input rate columns by a hypothetical
    ``(-1, +1)`` sign vector before the configured convention search. The
    operation is performed on a new trajectory for every pattern; ``trajectory``
    is never changed. This opt-in diagnostic runs inference eight times and does
    not estimate a physical calibration, modify the input, or change a result
    returned by :func:`infer_trajectory_conventions`.

    A lower-risk non-identity pattern means the fit is sensitive to a hypothetical
    per-channel sign correction. It is not proof of a sensor-polarity defect:
    polarity patterns may trade off with attitude/rate convention hypotheses,
    and multiple patterns can remain within the configured relative-fit margin.
    """

    orientation = trajectory.orientation
    rates = trajectory.rates
    if orientation is None or rates is None:
        raise ValueError("rate-polarity sensitivity requires both orientation and rate samples")
    ####

    resolved_config = coerce_config(config)
    raw_profiles: list[tuple[RatePolarityByInputAxis, InferenceReport, float, float]] = []
    for raw_signs in product((-1, 1), repeat=3):
        signs = cast(RatePolarityByInputAxis, raw_signs)
        corrected_trajectory = TrajectoryData(
            representation=trajectory.representation,
            orientation=orientation,
            rates=asarray(rates * asarray(signs, dtype=float64), dtype=float64),
            time_s=trajectory.time_s,
        )
        profile_report = infer_trajectory_conventions(corrected_trajectory, resolved_config)
        if not profile_report.rankings:
            raise RuntimeError("convention search returned no data-winning rate-polarity fit")
        ####
        best_data_winner = min(
            profile_report.rankings,
            key=lambda candidate: candidate.class_data_risk,
        )
        raw_profiles.append(
            (
                signs,
                profile_report,
                best_data_winner.class_data_risk,
                best_data_winner.normalized_rmse,
            )
        )
    ####

    best_profile_data_risk = min(profile[2] for profile in raw_profiles)
    identity_profile = next(profile for profile in raw_profiles if profile[0] == (1, 1, 1))
    identity_profile_data_risk = identity_profile[2]
    risk_floor = resolved_config.residual_floor_rad**2
    profile_fits = tuple(
        RatePolarityProfileFit(
            polarity_by_input_axis=signs,
            best_class_data_risk=profile_risk,
            relative_log_risk_delta_from_best=_relative_log_risk_delta(
                profile_risk,
                best_profile_data_risk,
                risk_floor,
            ),
            relative_log_risk_delta_vs_identity=_relative_log_risk_delta(
                profile_risk,
                identity_profile_data_risk,
                risk_floor,
            ),
            within_best_fit_margin=(
                _relative_log_risk_delta(profile_risk, best_profile_data_risk, risk_floor)
                <= resolved_config.near_tie_delta_criterion
            ),
            normalized_rmse=normalized_rmse,
            data_winning_class_ids=profile_report.data_winning_class_ids,
            evidence_contender_class_ids=profile_report.evidence_contender_class_ids,
            decision=profile_report.decision,
            warnings=profile_report.diagnostics.warnings,
        )
        for signs, profile_report, profile_risk, normalized_rmse in raw_profiles
    )
    return RatePolaritySensitivityReport(
        sample_count=trajectory.sample_count,
        near_tie_delta_criterion=resolved_config.near_tie_delta_criterion,
        identity_profile_data_risk=identity_profile_data_risk,
        best_profile_data_risk=best_profile_data_risk,
        profiles=profile_fits,
    )
####


def inspect_rate_rails(
    trajectory: TrajectoryData,
    rail_limits: RateInputRailLimits,
) -> RateRailInspectionReport:
    """Count samples at caller-supplied rails for each raw rate input column.

    The supplied limits are compared in the trajectory's raw input units. This
    function does not infer unknown hardware limits, convert units, or claim
    that a sample at a declared rail was caused by clipping.
    """

    rates = trajectory.rates
    if rates is None:
        raise ValueError("rate rail inspection requires rate samples")
    ####

    channels = tuple(
        _inspect_rate_channel_rails(
            rates,
            sample_count=trajectory.sample_count,
            input_axis_index=axis_index,
            lower_rail_value=rail_limits.lower_by_input_axis[axis_index],
            upper_rail_value=rail_limits.upper_by_input_axis[axis_index],
            tolerance=rail_limits.absolute_tolerance,
        )
        for axis_index in (0, 1, 2)
    )
    return RateRailInspectionReport(
        sample_count=trajectory.sample_count,
        rail_limits=rail_limits,
        channels=channels,
    )
####


def _inspect_rate_channel_rails(
    rates: NDArray[float64],
    *,
    sample_count: int,
    input_axis_index: int,
    lower_rail_value: float,
    upper_rail_value: float,
    tolerance: float,
) -> RateInputChannelRailHits:
    """Summarize observed hits and consecutive runs for one raw rate column."""

    channel_values = rates[:, input_axis_index]
    lower_hits = np_abs(channel_values - lower_rail_value) <= tolerance
    upper_hits = np_abs(channel_values - upper_rail_value) <= tolerance
    lower_count = int(count_nonzero(lower_hits))
    upper_count = int(count_nonzero(upper_hits))
    return RateInputChannelRailHits(
        input_axis_index=cast(Literal[0, 1, 2], input_axis_index),
        lower_rail_value=lower_rail_value,
        upper_rail_value=upper_rail_value,
        lower_rail_hit_count=lower_count,
        upper_rail_hit_count=upper_count,
        lower_rail_hit_fraction=lower_count / sample_count,
        upper_rail_hit_fraction=upper_count / sample_count,
        longest_lower_rail_run=_longest_true_run(lower_hits),
        longest_upper_rail_run=_longest_true_run(upper_hits),
    )
####
