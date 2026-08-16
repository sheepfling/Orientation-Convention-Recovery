from __future__ import annotations

from math import isclose

import pytest

from attitude_convention_inference import (
    DEFAULT_PROFILE_ID,
    InferenceConfig,
    PreferenceProfile,
    WeightedPreferenceProfile,
    available_preference_profiles,
    compose_preference_profiles,
    load_preference_profile,
)


def test_neutral_profile_is_packaged_and_unconfigured() -> None:
    assert DEFAULT_PROFILE_ID in available_preference_profiles()

    profile = load_preference_profile()

    assert isinstance(profile, PreferenceProfile)
    assert profile.profile_id == DEFAULT_PROFILE_ID
    assert profile.source == "package-default"
    assert not profile.configured()
####


def test_curated_profile_is_packaged_with_inspectable_preferences() -> None:
    assert "common-engineering-v1" in available_preference_profiles()

    profile = load_preference_profile("common-engineering-v1")

    assert profile.configured()
    assert profile.source == "package-curated-policy"
    assert profile.representative_preferences.quaternion_order["scalar_last"] == 2.0
    assert profile.context["evidence_kind"] == "curated-policy"
####


def test_unknown_profile_has_actionable_error() -> None:
    with pytest.raises(ValueError, match="available profiles"):
        load_preference_profile("does-not-exist")
    ####
####


def test_mapping_defaults_use_the_neutral_profile() -> None:
    config = InferenceConfig.model_validate({"preference_profile": {}})

    assert config.preference_profile.profile_id == DEFAULT_PROFILE_ID
    assert config.preference_profile.source == "package-default"
    assert not config.preference_profile.configured()
####


def test_profile_composition_uses_weighted_geometric_preferences() -> None:
    neutral = load_preference_profile("neutral-v1")
    common = load_preference_profile("common-engineering-v1")

    composed = compose_preference_profiles(
        (
            WeightedPreferenceProfile(profile=common, weight=3.0),
            WeightedPreferenceProfile(profile=neutral, weight=1.0),
        ),
        profile_id="common-engineering-blended-v1",
        context={"consumer": "test"},
    )

    assert composed.profile_id == "common-engineering-blended-v1"
    assert composed.context["composition_method"] == "weighted-geometric"
    assert "common-engineering-v1=3" in composed.context["composition_members"]
    assert isclose(
        composed.representative_preferences.quaternion_order["scalar_last"],
        2.0**0.75,
    )
    assert composed.representative_preferences.rate_frame["body"] > 1.0
####


def test_profile_composition_preserves_explicit_zero_class_weights() -> None:
    included = PreferenceProfile(profile_id="included", class_weights={"class-a": 4.0})
    excluded = PreferenceProfile(profile_id="excluded", class_weights={"class-a": 0.0})

    composed = compose_preference_profiles(
        (
            WeightedPreferenceProfile(profile=included, weight=1.0),
            WeightedPreferenceProfile(profile=excluded, weight=1.0),
        ),
        profile_id="zero-preserving-v1",
    )

    assert composed.class_weights["class-a"] == 0.0
####


def test_profile_composition_rejects_empty_inputs_and_invalid_weights() -> None:
    with pytest.raises(ValueError, match="At least one profile"):
        compose_preference_profiles((), profile_id="empty-v1")
    ####
    with pytest.raises(ValueError, match="positive and finite"):
        WeightedPreferenceProfile(profile=load_preference_profile(), weight=0.0)
    ####
####
