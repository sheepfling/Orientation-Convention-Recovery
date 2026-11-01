"""Frozen public models own their nested maps and remain JSON-serializable."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import cast

import pytest
from numpy import arange, float64
from pydantic import TypeAdapter

from attitude_convention_inference import (
    AttitudeRepresentation,
    CohortFusionResult,
    ConventionHypothesis,
    ConventionPriors,
    EvidencePacket,
    HypothesisSpaceSummary,
    InferenceConfig,
    InferenceReport,
    PreferenceProfile,
    QuaternionOrder,
    TrajectoryData,
    infer_conventions,
)
from attitude_convention_inference.models import OverallDiagnostics
from scripts.generate_validation_corpus import analytic_motion


def test_configuration_and_profile_copy_and_freeze_nested_maps() -> None:
    weights = {"pqr": 2.0}
    context = {"origin": "lab"}
    priors = ConventionPriors(rate_order=weights)
    profile = PreferenceProfile(context=context, representative_preferences=priors)
    config = InferenceConfig(priors=priors, preference_profile=profile)

    weights["pqr"] = 7.0
    context["origin"] = "changed"
    assert config.priors.rate_order["pqr"] == 2.0
    assert config.preference_profile.context["origin"] == "lab"
    assert isinstance(config.priors.rate_order, MappingProxyType)
    assert isinstance(config.preference_profile.context, MappingProxyType)
    with pytest.raises(TypeError):
        cast(dict[str, float], priors.rate_order)["pqr"] = 9.0
    ####

    replacement = {"origin": "updated"}
    copied = profile.model_copy(update={"context": replacement})
    replacement["origin"] = "changed again"
    assert copied.context["origin"] == "updated"
    assert isinstance(copied.context, MappingProxyType)
    assert PreferenceProfile.model_validate(json.loads(profile.model_dump_json())) == profile
    assert InferenceConfig.model_validate(json.loads(config.model_dump_json())) == config
    assert profile.model_json_schema()["properties"]["context"]["type"] == "object"
    dumped = profile.model_dump(mode="python")
    dumped["context"]["origin"] = "edited dump"
    assert profile.context["origin"] == "lab"

    with pytest.raises(ValueError, match="Rate order"):
        TypeAdapter(InferenceConfig).validate_python({"priors": {"rate_order": {"invalid": 2.0}}})
    ####
    hypothesis = ConventionHypothesis(
        attitude_direction="body_to_world",
        quaternion_order="scalar_last",
        rate_frame="body",
        rate_order="pqr",
        rate_unit="radians_per_second",
    )
    assert hypothesis.quaternion_order is QuaternionOrder.SCALAR_LAST
    assert (
        ConventionHypothesis.model_validate(json.loads(hypothesis.model_dump_json())) == hypothesis
    )
####


def test_evidence_packet_and_cohort_result_freeze_maps_without_losing_json() -> None:
    evidence = {"A": 0.0}
    prior = {"A": 1.0}
    provenance = {"origin": "sensor"}
    packet = EvidencePacket(
        packet_id="one",
        quotient_space_id="qspace",
        class_ids=("A",),
        log_evidence=evidence,
        local_prior=prior,
        provenance=provenance,
    )
    evidence["A"] = -5.0
    prior["A"] = 0.25
    provenance["origin"] = "changed"
    assert packet.log_evidence["A"] == 0.0
    assert packet.local_prior["A"] == 1.0
    assert packet.provenance["origin"] == "sensor"
    assert isinstance(packet.log_evidence, MappingProxyType)
    with pytest.raises(TypeError):
        cast(dict[str, float], packet.log_evidence)["A"] = -1.0
    ####
    assert EvidencePacket.model_validate(json.loads(packet.model_dump_json())) == packet
    assert packet.model_json_schema()["properties"]["log_evidence"]["type"] == "object"

    example = Path(__file__).resolve().parents[1] / "examples/cohort_fusion/fusion_result.json"
    fused = CohortFusionResult.model_validate(json.loads(example.read_text(encoding="utf-8")))
    assert isinstance(fused.fused_posterior, MappingProxyType)
    assert CohortFusionResult.model_validate(json.loads(fused.model_dump_json())) == fused
####


def test_inference_report_freezes_nested_maps_and_round_trips() -> None:
    source = {"rate_order": {"pqr": 2.0}}
    summary = HypothesisSpaceSummary(
        base_hypothesis_count=2,
        constrained_hypothesis_count=2,
        reduction_fraction=0.0,
        fixed_dimensions={},
        searched_dimensions={"rate_order": ("pqr", "rqp")},
        not_applicable_dimensions=(),
        relative_prior_weights=source,
    )
    source["rate_order"]["pqr"] = 9.0
    assert summary.relative_prior_weights["rate_order"]["pqr"] == 2.0
    assert isinstance(summary.relative_prior_weights["rate_order"], MappingProxyType)
    with pytest.raises(TypeError):
        cast(dict[str, float], summary.relative_prior_weights["rate_order"])["pqr"] = 3.0
    ####
    assert HypothesisSpaceSummary.model_validate(json.loads(summary.model_dump_json())) == summary
    assert summary.model_json_schema()["properties"]["relative_prior_weights"]["type"] == "object"

    motion = analytic_motion(arange(31, dtype=float64) * 0.025)
    data = TrajectoryData(
        representation=AttitudeRepresentation.QUATERNION,
        orientation=motion.quaternion_xyzw,
        rates=motion.body_rates_rad_s,
        time_s=motion.time_s,
    )
    report = infer_conventions(data)
    assert isinstance(report.hypothesis_space.relative_prior_weights, MappingProxyType)
    assert isinstance(report.diagnostics.dimension_assessments, MappingProxyType)
    assert isinstance(
        report.best.heuristic_features.representation_feature_vector, MappingProxyType
    )
    diagnostics_schema = report.diagnostics.model_json_schema()
    confidence_schema = diagnostics_schema["properties"]["confidence"]
    assert confidence_schema["minimum"] == 0.0
    assert confidence_schema["maximum"] == 1.0
    assert "not a probability of correctness" in confidence_schema["description"]
    for invalid_confidence in (-0.01, 1.01, float("nan"), float("inf")):
        invalid_payload = report.diagnostics.model_dump()
        invalid_payload["confidence"] = invalid_confidence
        with pytest.raises(ValueError):
            OverallDiagnostics.model_validate(invalid_payload)
        ####
    ####
    mismatched_label_payload = report.diagnostics.model_dump()
    mismatched_label_payload["confidence"] = 0.1
    mismatched_label_payload["confidence_label"] = "high"
    with pytest.raises(ValueError, match="confidence_label must match"):
        OverallDiagnostics.model_validate(mismatched_label_payload)
    ####
    assert InferenceReport.model_validate(json.loads(report.model_dump_json())) == report
####
