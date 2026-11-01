"""Evidence-gated convention recommendations with explicit abstention."""

from __future__ import annotations

from enum import StrEnum

from pydantic.dataclasses import dataclass

from .model_common import FrozenModel

__all__ = ["InferenceDecision", "InferenceDecisionReason", "InferenceDecisionStatus"]


class InferenceDecisionStatus(StrEnum):
    RECOMMENDATION = "recommendation"
    ABSTENTION = "abstention"
####


class InferenceDecisionReason(StrEnum):
    UNIQUE_SUFFICIENT_EVIDENCE = "unique_sufficient_evidence"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    MULTIPLE_EVIDENCE_CONTENDERS = "multiple_evidence_contenders"
    NO_EVIDENCE_CONTENDERS = "no_evidence_contenders"
####


@dataclass(frozen=True, slots=True)
class InferenceDecision(FrozenModel):
    """Evidence-only recommendation, or a named reason for abstaining."""

    status: InferenceDecisionStatus
    reason: InferenceDecisionReason
    recommended_class_id: str | None
    evidence_contender_class_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        class_ids = tuple(self.evidence_contender_class_ids)
        if any(not class_id.strip() for class_id in class_ids):
            raise ValueError("evidence contender class IDs must be non-empty strings")
        ####
        if len(class_ids) != len(set(class_ids)):
            raise ValueError("evidence contender class IDs must be unique")
        ####
        object.__setattr__(self, "evidence_contender_class_ids", class_ids)

        if self.status is InferenceDecisionStatus.RECOMMENDATION:
            if self.reason is not InferenceDecisionReason.UNIQUE_SUFFICIENT_EVIDENCE:
                raise ValueError("a recommendation requires a unique sufficient-evidence reason")
            ####
            if self.recommended_class_id is None:
                raise ValueError("a recommendation requires recommended_class_id")
            ####
            if class_ids != (self.recommended_class_id,):
                raise ValueError("a recommendation must be the sole evidence contender")
            ####
        else:
            if self.reason is InferenceDecisionReason.UNIQUE_SUFFICIENT_EVIDENCE:
                raise ValueError("an abstention cannot use the recommendation reason")
            ####
            if self.recommended_class_id is not None:
                raise ValueError("an abstention cannot include recommended_class_id")
            ####
            if (
                self.reason is InferenceDecisionReason.MULTIPLE_EVIDENCE_CONTENDERS
                and len(class_ids) < 2
            ):
                raise ValueError("multiple-contender abstention requires multiple class IDs")
            ####
            if self.reason is InferenceDecisionReason.NO_EVIDENCE_CONTENDERS and class_ids:
                raise ValueError("no-contender abstention requires an empty class-ID tuple")
            ####
        ####
    ####

    @classmethod
    def from_evidence(
        cls,
        *,
        evidence_sufficient: bool,
        evidence_contender_class_ids: tuple[str, ...],
    ) -> InferenceDecision:
        """Recommend only a sole data-evidence contender after sufficient evidence."""

        class_ids = evidence_contender_class_ids
        if not evidence_sufficient:
            return cls(
                status=InferenceDecisionStatus.ABSTENTION,
                reason=InferenceDecisionReason.EVIDENCE_INSUFFICIENT,
                recommended_class_id=None,
                evidence_contender_class_ids=class_ids,
            )
        ####
        if not class_ids:
            return cls(
                status=InferenceDecisionStatus.ABSTENTION,
                reason=InferenceDecisionReason.NO_EVIDENCE_CONTENDERS,
                recommended_class_id=None,
                evidence_contender_class_ids=class_ids,
            )
        ####
        if len(class_ids) > 1:
            return cls(
                status=InferenceDecisionStatus.ABSTENTION,
                reason=InferenceDecisionReason.MULTIPLE_EVIDENCE_CONTENDERS,
                recommended_class_id=None,
                evidence_contender_class_ids=class_ids,
            )
        ####
        return cls(
            status=InferenceDecisionStatus.RECOMMENDATION,
            reason=InferenceDecisionReason.UNIQUE_SUFFICIENT_EVIDENCE,
            recommended_class_id=next(iter(class_ids)),
            evidence_contender_class_ids=class_ids,
        )
    ####
####
