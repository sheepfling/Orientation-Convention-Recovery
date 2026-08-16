from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from attitude_convention_inference import (
    AttitudeDirection,
    AttitudeRepresentation,
    ConventionPriors,
    DomainPrior,
    InferenceConfig,
    PreferenceProfile,
    QuaternionOrder,
    RateFrame,
    RateUnit,
    SearchSpace,
    TrajectoryData,
    infer_conventions,
    should_prune_by_exact_lower_bound,
)
from attitude_convention_inference.synthetic import (
    SyntheticConvention,
    encode_synthetic_trajectory,
)


def _ambiguous_config() -> InferenceConfig:
    return InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=tuple(AttitudeDirection),
            quaternion_orders=(QuaternionOrder.SCALAR_LAST,),
            rate_frames=tuple(RateFrame),
            rate_orders=("pqr",),
            rate_units=(RateUnit.RADIANS_PER_SECOND,),
            rate_signs=(-1, 1),
        )
    )
####


def _ambiguous_data() -> TrajectoryData:
    time_s = np.arange(61, dtype=np.float64) * 0.02
    truth = SyntheticConvention(
        representation=AttitudeRepresentation.QUATERNION,
        attitude_direction=AttitudeDirection.BODY_TO_WORLD,
        quaternion_order=QuaternionOrder.SCALAR_LAST,
        rate_frame=RateFrame.BODY,
        rate_order="pqr",
    )
    return encode_synthetic_trajectory(time_s, np.zeros((time_s.size, 3)), truth)
####


def test_domain_prior_abstains_when_no_contender_has_support() -> None:
    data = _ambiguous_data()
    config = _ambiguous_config()
    baseline = infer_conventions(data, config)

    report = infer_conventions(
        data,
        config,
        domain_prior=DomainPrior(class_weights={"class-not-observed": 100.0}),
    )

    assert report.evidence_contender_class_ids == baseline.evidence_contender_class_ids
    assert report.domain_prior_configured
    assert report.domain_prior_abstained
    assert report.operational_reason == "canonical-policy"
    assert report.operational_class_id in report.evidence_contender_class_ids
####


def test_domain_prior_selects_only_from_evidence_contenders() -> None:
    data = _ambiguous_data()
    config = _ambiguous_config()
    baseline = infer_conventions(data, config)
    target = baseline.evidence_contender_class_ids[-1]

    report = infer_conventions(
        data,
        config,
        domain_prior=DomainPrior(class_weights={target: 2.0}),
    )

    assert report.operational_class_id == target
    assert report.operational_reason == "domain-prior"
    assert report.domain_prior_supported_class_ids == (target,)
####


def test_preference_profile_selects_representative_inside_exact_class() -> None:
    data = _ambiguous_data()
    config = _ambiguous_config()
    profile = PreferenceProfile(
        profile_id="body-rate-policy-v1",
        source="test-policy",
        representative_preferences=ConventionPriors(
            attitude_direction={"world_to_body": 4.0},
            rate_frame={"world": 4.0},
            rate_sign={-1: 4.0},
        ),
    )

    report = infer_conventions(data, config, preference_profile=profile)

    assert report.preference_profile_id == "body-rate-policy-v1"
    assert report.preference_profile_source == "test-policy"
    assert report.preference_profile_configured
    assert report.representative_selection_reason == "preference-profile"
    assert "world_to_body" in report.operational_hypothesis_id
    assert "world" in report.operational_hypothesis_id
    assert "sign=-1" in report.operational_hypothesis_id
    assert len(report.best.exact_equivalent_hypothesis_ids) > 1
####


def test_preference_profile_json_round_trip_and_class_selection() -> None:
    data = _ambiguous_data()
    baseline = infer_conventions(data, _ambiguous_config())
    target = baseline.evidence_contender_class_ids[-1]
    config = _ambiguous_config().model_copy(
        update={
            "preference_profile": PreferenceProfile.model_validate(
                {
                    "profile_id": "class-policy-v1",
                    "source": "test-policy",
                    "class_weights": {target: 3.0},
                    "representative_preferences": {"rate_frame": {"body": 2.0}},
                }
            )
        }
    )

    report = infer_conventions(data, config)
    restored = InferenceConfig.model_validate(config.model_dump(mode="json"))

    assert restored == config
    assert report.operational_class_id == target
    assert report.operational_reason == "preference-profile"
    assert report.preference_profile_supported_class_ids == (target,)
####


def test_preference_profile_abstains_when_class_support_is_external() -> None:
    report = infer_conventions(
        _ambiguous_data(),
        _ambiguous_config(),
        preference_profile=PreferenceProfile(
            profile_id="unmatched-policy-v1",
            class_weights={"class-not-observed": 10.0},
        ),
    )

    assert report.preference_profile_abstained
    assert report.preference_profile_supported_class_ids == ()
    assert report.operational_reason == "canonical-policy"
####


def test_raw_excitation_diagnostic_survives_signed_channel_relabeling() -> None:
    data = _ambiguous_data()
    if data.rates is None:
        raise AssertionError("ambiguous trajectory must contain rates")
    ####
    permuted = replace(data, rates=data.rates[:, (2, 0, 1)] * np.asarray((-1.0, 1.0, -1.0)))
    config = _ambiguous_config()

    first = infer_conventions(data, config)
    second = infer_conventions(permuted, config)

    assert first.diagnostics.raw_excitation_rank == second.diagnostics.raw_excitation_rank
    assert first.diagnostics.raw_excitation_condition_ratio == pytest.approx(
        second.diagnostics.raw_excitation_condition_ratio,
        abs=1.0e-12,
    )
####


def test_exact_lower_bound_pruning_is_strict_and_separate_from_evidence() -> None:
    assert not should_prune_by_exact_lower_bound(1.0, 1.0)
    assert should_prune_by_exact_lower_bound(1.01, 1.0)
    with pytest.raises(ValueError):
        should_prune_by_exact_lower_bound(float("nan"), 1.0)
    ####
####


def test_partial_streams_return_explicit_availability_labels() -> None:
    time_s = np.arange(21, dtype=np.float64) * 0.1
    orientation = np.tile((0.0, 0.0, 0.0, 1.0), (time_s.size, 1))
    rates = np.zeros((time_s.size, 3), dtype=np.float64)

    attitude_report = infer_conventions(
        orientation,
        None,
        representation="quaternion",
        time_s=time_s,
    )
    rate_report = infer_conventions(
        None,
        rates,
        representation="quaternion",
        time_s=time_s,
    )

    assert attitude_report.result_label.value == "attitude-heuristic"
    assert rate_report.result_label.value == "rate-non-ID"
    assert attitude_report.best.heuristic_features.classifier_probability is None
####
