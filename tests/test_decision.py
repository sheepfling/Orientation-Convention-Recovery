from __future__ import annotations

import json

import pytest

from attitude_convention_inference import (
    InferenceDecision,
    InferenceDecisionReason,
    InferenceDecisionStatus,
)


def test_recommends_only_the_sole_sufficient_evidence_contender() -> None:
    decision = InferenceDecision.from_evidence(
        evidence_sufficient=True,
        evidence_contender_class_ids=("class-a",),
    )

    assert decision.status is InferenceDecisionStatus.RECOMMENDATION
    assert decision.reason is InferenceDecisionReason.UNIQUE_SUFFICIENT_EVIDENCE
    assert decision.recommended_class_id == "class-a"
    assert decision.evidence_contender_class_ids == ("class-a",)
    assert InferenceDecision.model_validate(json.loads(decision.model_dump_json())) == decision
####


@pytest.mark.parametrize(
    ("evidence_sufficient", "class_ids", "expected_reason"),
    (
        (False, ("class-a",), InferenceDecisionReason.EVIDENCE_INSUFFICIENT),
        (
            True,
            ("class-a", "class-b"),
            InferenceDecisionReason.MULTIPLE_EVIDENCE_CONTENDERS,
        ),
        (True, (), InferenceDecisionReason.NO_EVIDENCE_CONTENDERS),
    ),
)
def test_abstains_with_a_specific_reason(
    evidence_sufficient: bool,
    class_ids: tuple[str, ...],
    expected_reason: InferenceDecisionReason,
) -> None:
    decision = InferenceDecision.from_evidence(
        evidence_sufficient=evidence_sufficient,
        evidence_contender_class_ids=class_ids,
    )

    assert decision.status is InferenceDecisionStatus.ABSTENTION
    assert decision.reason is expected_reason
    assert decision.recommended_class_id is None
    assert decision.evidence_contender_class_ids == class_ids
####


def test_decision_rejects_inconsistent_recommendation() -> None:
    with pytest.raises(ValueError, match="sole evidence contender"):
        InferenceDecision(
            status=InferenceDecisionStatus.RECOMMENDATION,
            reason=InferenceDecisionReason.UNIQUE_SUFFICIENT_EVIDENCE,
            recommended_class_id="class-a",
            evidence_contender_class_ids=("class-a", "class-b"),
        )
    ####
####
