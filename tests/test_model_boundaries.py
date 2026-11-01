"""Keep moved model contracts importable and identical across module surfaces."""

from __future__ import annotations

from json import loads

from attitude_convention_inference.hypothesis_model import (
    ConventionHypothesis as HypothesisContract,
)
from attitude_convention_inference.model_common import (
    AttitudeDirection as CommonAttitudeDirection,
)
from attitude_convention_inference.models import (
    AttitudeDirection,
    CandidateResult,
    ConventionConstraints,
    ConventionHypothesis,
    ConventionPriors,
    InferenceConfig,
    InferenceReport,
    SearchSpace,
)
from attitude_convention_inference.policy_models import (
    ConventionPriors as PolicyPriors,
)
from attitude_convention_inference.policy_models import InferenceConfig as PolicyConfig
from attitude_convention_inference.report_models import CandidateResult as CandidateContract
from attitude_convention_inference.report_models import InferenceReport as ReportContract
from attitude_convention_inference.search_models import (
    ConventionConstraints as SearchConstraints,
)
from attitude_convention_inference.search_models import SearchSpace as SearchSpaceContract


def test_search_contract_reexports_preserve_identity_and_serialization() -> None:
    assert AttitudeDirection is CommonAttitudeDirection
    assert SearchSpace is SearchSpaceContract
    assert ConventionConstraints is SearchConstraints

    search = SearchSpace.model_validate(
        {
            "attitude_directions": ["body_to_world"],
            "rate_orders": ["pqr", "rqp"],
        }
    )
    constraint = ConventionConstraints.model_validate({"rate_order": "rqp"})
    selected = constraint.apply(search)

    assert selected.attitude_directions == (AttitudeDirection.BODY_TO_WORLD,)
    assert selected.rate_orders == ("rqp",)
    assert SearchSpace.model_validate(loads(search.model_dump_json())) == search
    assert "properties" in SearchSpace.model_json_schema()
####


def test_policy_hypothesis_and_report_reexports_preserve_identity() -> None:
    assert ConventionPriors is PolicyPriors
    assert InferenceConfig is PolicyConfig
    assert ConventionHypothesis is HypothesisContract
    assert CandidateResult is CandidateContract
    assert InferenceReport is ReportContract

    config = InferenceConfig.model_validate(
        {
            "search_space": {"rate_orders": ["pqr"]},
            "priors": {"rate_order": {"pqr": 2.0}},
        }
    )
    hypothesis = ConventionHypothesis.model_validate(
        {
            "attitude_direction": "body_to_world",
            "quaternion_order": "scalar_last",
            "rate_frame": "body",
            "rate_order": "pqr",
            "rate_unit": "radians_per_second",
        }
    )

    assert InferenceConfig.model_validate(loads(config.model_dump_json())) == config
    assert ConventionHypothesis.model_validate(loads(hypothesis.model_dump_json())) == hypothesis
    assert "properties" in InferenceReport.model_json_schema()
####
