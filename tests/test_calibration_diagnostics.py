from __future__ import annotations

import pytest
from numpy import arange, array, float64, ones
from pydantic import ValidationError

from attitude_convention_inference import (
    AttitudeDirection,
    AttitudeRepresentation,
    QuaternionOrder,
    RateFrame,
    RateInputRailLimits,
    RatePolaritySensitivityReport,
    RateRailInspectionReport,
    RateUnit,
    TrajectoryData,
    assess_rate_polarity_sensitivity,
    inspect_rate_rails,
)
from attitude_convention_inference.synthetic import (
    SyntheticConvention,
    encode_synthetic_trajectory,
    rich_three_axis_rates,
)


def _p_flipped_synthetic_trajectory() -> TrajectoryData:
    time_s = arange(161, dtype=float64) * 0.025
    canonical_rates = rich_three_axis_rates(time_s)
    trajectory = encode_synthetic_trajectory(
        time_s,
        canonical_rates,
        SyntheticConvention(
            representation=AttitudeRepresentation.QUATERNION,
            attitude_direction=AttitudeDirection.BODY_TO_WORLD,
            rate_frame=RateFrame.BODY,
            quaternion_order=QuaternionOrder.SCALAR_LAST,
        ),
    )
    rates = trajectory.rates
    if rates is None:
        raise RuntimeError("synthetic trajectory did not contain rates")
    ####
    observed_rates = rates * array((-1.0, 1.0, 1.0), dtype=float64)
    return TrajectoryData(
        representation=trajectory.representation,
        orientation=trajectory.orientation,
        rates=observed_rates,
        time_s=trajectory.time_s,
    )
####


def test_rate_polarity_sensitivity_compares_all_patterns_without_mutating_input() -> None:
    trajectory = _p_flipped_synthetic_trajectory()
    rates = trajectory.rates
    assert rates is not None
    original_rates = rates.copy()

    report = assess_rate_polarity_sensitivity(trajectory)

    profiles_by_signs = {profile.polarity_by_input_axis: profile for profile in report.profiles}
    assert set(profiles_by_signs) == {
        (-1, -1, -1),
        (-1, -1, 1),
        (-1, 1, -1),
        (-1, 1, 1),
        (1, -1, -1),
        (1, -1, 1),
        (1, 1, -1),
        (1, 1, 1),
    }
    assert report.sample_count == trajectory.sample_count
    assert report.identity_profile_data_risk == profiles_by_signs[(1, 1, 1)].best_class_data_risk
    assert report.best_profile_data_risk == min(
        profile.best_class_data_risk for profile in report.profiles
    )

    p_axis_correction = profiles_by_signs[(-1, 1, 1)]
    assert p_axis_correction.relative_log_risk_delta_vs_identity < -2.0
    assert p_axis_correction.within_best_fit_margin
    best_fit_profiles = tuple(
        profile for profile in report.profiles if profile.within_best_fit_margin
    )
    assert len(best_fit_profiles) > 1
    assert all(profile.data_winning_class_ids for profile in best_fit_profiles)
    assert rates.flags.writeable is False
    assert (rates == original_rates).all()
    assert len(report.model_dump_json()) > 0

    incomplete_payload = report.model_dump()
    incomplete_payload["profiles"] = incomplete_payload["profiles"][:-1]
    with pytest.raises(ValidationError, match="all eight sign patterns"):
        RatePolaritySensitivityReport.model_validate(incomplete_payload)
    ####

    identity_risk_payload = report.model_dump()
    identity_risk_payload["identity_profile_data_risk"] = report.identity_profile_data_risk + 1.0
    with pytest.raises(ValidationError, match="must match the identity profile fit"):
        RatePolaritySensitivityReport.model_validate(identity_risk_payload)
    ####

    best_risk_payload = report.model_dump()
    best_risk_payload["best_profile_data_risk"] = report.best_profile_data_risk + 1.0
    with pytest.raises(ValidationError, match="must match the best profile fit"):
        RatePolaritySensitivityReport.model_validate(best_risk_payload)
    ####
####


def test_rate_polarity_sensitivity_requires_orientation_and_rates() -> None:
    trajectory = TrajectoryData(
        representation=AttitudeRepresentation.QUATERNION,
        orientation=ones((3, 4), dtype=float64),
        rates=None,
    )

    with pytest.raises(ValueError, match="requires both orientation and rate samples"):
        assess_rate_polarity_sensitivity(trajectory)
    ####
####


def test_rate_rail_inspection_counts_known_input_rails_and_runs() -> None:
    rates = array(
        (
            (-1.0, 0.0, -4.0),
            (0.0, -2.0, -4.0),
            (2.0, -2.0, 0.0),
            (2.0, -2.0, 1.0),
            (2.0, 0.0, 1.0),
            (0.0, 3.0, 0.0),
        ),
        dtype=float64,
    )
    trajectory = TrajectoryData(
        representation=AttitudeRepresentation.QUATERNION,
        orientation=None,
        rates=rates,
    )
    rail_limits = RateInputRailLimits(
        input_rate_unit=RateUnit.DEGREES_PER_SECOND,
        lower_by_input_axis=(-1.0, -2.0, -4.0),
        upper_by_input_axis=(2.0, 3.0, 1.0),
    )

    report = inspect_rate_rails(trajectory, rail_limits)

    assert report.sample_count == 6
    assert report.rail_limits.input_rate_unit is RateUnit.DEGREES_PER_SECOND
    assert report.channels[0].lower_rail_hit_count == 1
    assert report.channels[0].upper_rail_hit_count == 3
    assert report.channels[0].upper_rail_hit_fraction == 0.5
    assert report.channels[0].longest_upper_rail_run == 3
    assert report.channels[1].lower_rail_hit_count == 3
    assert report.channels[1].longest_lower_rail_run == 3
    assert report.channels[1].upper_rail_hit_count == 1
    assert report.channels[2].lower_rail_hit_count == 2
    assert report.channels[2].upper_rail_hit_count == 2
    assert report.channels[2].longest_upper_rail_run == 2
    assert RateRailInspectionReport.model_validate(report.model_dump()) == report
####


def test_rate_rail_contract_rejects_overlapping_tolerance_and_missing_rates() -> None:
    with pytest.raises(ValidationError, match="tolerance regions must not overlap"):
        RateInputRailLimits(
            input_rate_unit=RateUnit.RADIANS_PER_SECOND,
            lower_by_input_axis=(-0.1, -1.0, -1.0),
            upper_by_input_axis=(0.1, 1.0, 1.0),
            absolute_tolerance=0.1,
        )
    ####

    trajectory = TrajectoryData(
        representation=AttitudeRepresentation.QUATERNION,
        orientation=ones((3, 4), dtype=float64),
        rates=None,
    )
    limits = RateInputRailLimits(
        input_rate_unit=RateUnit.RADIANS_PER_SECOND,
        lower_by_input_axis=(-1.0, -1.0, -1.0),
        upper_by_input_axis=(1.0, 1.0, 1.0),
    )
    with pytest.raises(ValueError, match="requires rate samples"):
        inspect_rate_rails(trajectory, limits)
    ####
####
