"""Assemble the serialized inference report from ranked evidence and policy."""

from __future__ import annotations

from .assessments import _hypothesis_space_summary
from .core import TrajectoryData
from .models import ConventionHypothesis, InferenceConfig, InferenceReport, OverallDiagnostics
from .scoring import ScoredCandidate as _ScoredCandidate
from .selection import _evidence_contender_classes, _result_label, _select_operational_result

__all__ = ["_build_inference_report"]


def _build_inference_report(
    data: TrajectoryData,
    hypotheses: tuple[ConventionHypothesis, ...],
    ranked: tuple[_ScoredCandidate, ...],
    config: InferenceConfig,
    diagnostics: OverallDiagnostics,
) -> InferenceReport:
    rankings = tuple(item.result for item in ranked)
    hypothesis_space = _hypothesis_space_summary(data.representation, config)
    contender_class_ids = _evidence_contender_classes(ranked, config)
    operational = _select_operational_result(ranked, config)
    data_minimum = min(item.result.class_data_risk for item in ranked)
    data_winning_class_ids = tuple(
        dict.fromkeys(
            item.result.equivalence_class_id
            for item in ranked
            if item.result.class_data_risk == data_minimum
        )
    )
    operational_item = next(
        item for item in ranked if item.result.hypothesis.identifier() == operational.hypothesis_id
    )
    return InferenceReport(
        representation=data.representation,
        sample_count=data.sample_count,
        hypothesis_count=len(hypotheses),
        hypothesis_space=hypothesis_space,
        best=operational_item.result,
        rankings=rankings,
        diagnostics=diagnostics,
        result_label=_result_label(data, config),
        evidence_contender_class_ids=contender_class_ids,
        data_winning_class_ids=data_winning_class_ids,
        empirical_gap_to_next_class=operational_item.result.empirical_gap_to_next_class,
        operational_class_id=operational.class_id,
        operational_hypothesis_id=operational.hypothesis_id,
        operational_reason=operational.reason,
        domain_prior_configured=config.domain_prior.configured(),
        domain_prior_abstained=operational.domain_prior_abstained,
        domain_prior_supported_class_ids=operational.domain_prior_supported_class_ids,
        preference_profile_id=config.preference_profile.profile_id,
        preference_profile_source=config.preference_profile.source,
        preference_profile_configured=config.preference_profile.configured(),
        preference_profile_abstained=operational.preference_profile_abstained,
        preference_profile_supported_class_ids=operational.preference_profile_supported_class_ids,
        representative_selection_reason=operational.representative_reason,
    )
####
