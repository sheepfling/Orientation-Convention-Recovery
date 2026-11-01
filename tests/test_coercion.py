from __future__ import annotations

from attitude_convention_inference import (
    AttitudeDirection,
    ConventionConstraints,
    ConventionPriors,
    DomainPrior,
    InferenceConfig,
    PreferenceProfile,
    RateFrame,
)
from attitude_convention_inference.coercion import (
    coerce_config,
    coerce_constraints,
    coerce_domain_prior,
    coerce_preference_profile,
    coerce_priors,
    merge_priors,
)


def test_coercers_preserve_absent_and_already_typed_inputs() -> None:
    config = InferenceConfig()
    constraints = ConventionConstraints.model_validate({"rate_frame": "body"})
    priors = ConventionPriors.model_validate({"attitude_direction": {"body_to_world": 2.0}})
    domain_prior = DomainPrior.model_validate({"class_weights": {"class-a": 2.0}})
    profile = PreferenceProfile.model_validate({"profile_id": "flight-test"})

    assert coerce_config(None) == InferenceConfig()
    assert coerce_constraints(None) is None
    assert coerce_priors(None) is None
    assert coerce_domain_prior(None) is None
    assert coerce_preference_profile(None) is None
    assert coerce_config(config) is config
    assert coerce_constraints(constraints) is constraints
    assert coerce_priors(priors) is priors
    assert coerce_domain_prior(domain_prior) is domain_prior
    assert coerce_preference_profile(profile) is profile
####


def test_coercers_validate_mappings_and_merge_prior_overrides() -> None:
    config = coerce_config({"sample_period_s": 0.02})
    constraints = coerce_constraints({"rate_frame": "body"})
    priors = coerce_priors({"rate_frame": {"body": 2.0}})
    domain_prior = coerce_domain_prior({"class_weights": {"class-a": 3.0}})
    profile = coerce_preference_profile({"profile_id": "flight-test"})

    assert config.sample_period_s == 0.02
    assert constraints is not None
    assert constraints.rate_frame == (RateFrame.BODY,)
    assert priors is not None
    assert priors.rate_frame[RateFrame.BODY] == 2.0
    assert domain_prior is not None
    assert domain_prior.class_weights["class-a"] == 3.0
    assert profile is not None
    assert profile.profile_id == "flight-test"

    merged = merge_priors(
        ConventionPriors.model_validate(
            {"attitude_direction": {"body_to_world": 2.0}, "rate_frame": {"body": 3.0}}
        ),
        ConventionPriors.model_validate(
            {"attitude_direction": {"world_to_body": 4.0}, "rate_frame": {"body": 5.0}}
        ),
    )

    assert merged.attitude_direction == {
        AttitudeDirection.BODY_TO_WORLD: 2.0,
        AttitudeDirection.WORLD_TO_BODY: 4.0,
    }
    assert merged.rate_frame[RateFrame.BODY] == 5.0
####
