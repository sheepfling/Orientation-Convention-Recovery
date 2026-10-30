"""Measure class recovery and abstention on independently labeled CSV trajectories."""

from __future__ import annotations

import argparse
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self, cast

from numpy import __version__ as numpy_version
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator
from scipy import __version__ as scipy_version

from attitude_convention_inference import (
    AttitudeRepresentation,
    ConventionHypothesis,
    InferenceConfig,
    InferenceDecision,
    InferenceDecisionStatus,
    __version__,
    infer_csv,
)


class MeasurementConditions(BaseModel):
    """Typed measurement perturbations recorded for one validation trajectory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    motion_profile_id: str | None = Field(default=None, min_length=1)
    timestamp_jitter_amplitude_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    dropout_probability: float | None = Field(default=None, ge=0.0, lt=1.0, allow_inf_nan=False)
    dropped_sample_count: int | None = Field(default=0, ge=0)
    orientation_latency_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    rate_latency_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    orientation_noise_sigma_rad: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    rate_noise_sigma_rad_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    orientation_noise_ar1_coefficient: float = Field(
        default=0.0, gt=-1.0, lt=1.0, allow_inf_nan=False
    )
    rate_noise_ar1_coefficient: float = Field(default=0.0, gt=-1.0, lt=1.0, allow_inf_nan=False)
    noise_seed: int | None = Field(default=None, ge=0)
    sample_count: int | None = Field(default=None, gt=0)
    observed_sample_count: int | None = Field(default=None, gt=0)
    sample_period_s: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_sampling_pair(self) -> Self:
        if (self.sample_count is None) != (self.sample_period_s is None):
            raise ValueError("sample_count and sample_period_s must be provided together")
        ####
        if self.observed_sample_count is not None:
            if self.sample_count is None:
                raise ValueError("observed_sample_count requires sample_count and sample_period_s")
            ####
            if self.dropped_sample_count is None:
                raise ValueError("dropped_sample_count is required with observed_sample_count")
            ####
            if self.observed_sample_count + self.dropped_sample_count != self.sample_count:
                raise ValueError(
                    "observed_sample_count plus dropped_sample_count must equal sample_count"
                )
            ####
        ####
        if (
            self.sample_period_s is not None
            and self.timestamp_jitter_amplitude_s >= self.sample_period_s / 2.0
        ):
            raise ValueError("timestamp jitter amplitude must be less than half the sample period")
        ####
        return self
    ####

    @computed_field
    @property
    def duration_s(self) -> float | None:
        if self.sample_count is None or self.sample_period_s is None:
            return None
        ####
        return (self.sample_count - 1) * self.sample_period_s
    ####
####


class ValidationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    csv: Path
    label_source: str = Field(min_length=1)
    representation: AttitudeRepresentation
    truth: ConventionHypothesis
    config: InferenceConfig = Field(default_factory=InferenceConfig)
    orientation_columns: tuple[str, ...] | None = None
    rate_columns: tuple[str, ...] | None = None
    time_column: str | None = None
    measurement_conditions: MeasurementConditions = Field(default_factory=MeasurementConditions)

    @field_validator("truth", mode="before")
    @classmethod
    def _parse_truth(cls, value: object) -> ConventionHypothesis:
        if isinstance(value, ConventionHypothesis):
            return value
        ####
        if not isinstance(value, dict):
            raise ValueError("truth must be a convention hypothesis object")
        ####
        return ConventionHypothesis.model_validate(cast(dict[str, Any], value))
    ####

    @field_validator("config", mode="before")
    @classmethod
    def _parse_config(cls, value: object) -> InferenceConfig:
        if isinstance(value, InferenceConfig):
            return value
        ####
        if not isinstance(value, dict):
            raise ValueError("config must be an inference configuration object")
        ####
        return InferenceConfig.model_validate(cast(dict[str, Any], value))
    ####
####


class ValidationManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    cases: tuple[ValidationCase, ...] = Field(min_length=1)
####


class EvaluationEnvironment(BaseModel):
    """Runtime versions needed to interpret a generated assessment reproducibly."""

    model_config = ConfigDict(frozen=True)

    python: str
    platform: str
    numpy: str
    scipy: str
    package: str
####


class CaseAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    label_source: str
    representation: AttitudeRepresentation
    measurement_conditions: MeasurementConditions
    sample_count: int
    hypothesis_count: int
    truth_class_id: str
    evidence_contender_class_ids: tuple[str, ...]
    data_winning_class_ids: tuple[str, ...]
    data_winner_correct: bool
    operational_class_id: str
    operational_correct: bool
    decision: InferenceDecision
    recommendation_correct: bool | None
    outcome: Literal["resolved_correct", "ambiguous_contains_truth", "evidence_miss"]
    confidence_label: str
    evidence_sufficient: bool
####


class ValidationSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_count: int
    resolved_correct: int
    ambiguous_contains_truth: int
    evidence_miss: int
    data_winner_miss: int
    operational_miss: int
    evidence_insufficient: int
    incorrect_evidence_sufficient: int
    recommendation_count: int
    abstention_count: int
    recommendation_miss: int

    @computed_field
    @property
    def recommendation_coverage(self) -> float:
        return self.recommendation_count / self.case_count
    ####

    @computed_field
    @property
    def recommendation_accuracy(self) -> float | None:
        if self.recommendation_count == 0:
            return None
        ####
        return (self.recommendation_count - self.recommendation_miss) / self.recommendation_count
    ####
####


class ConditionSummary(BaseModel):
    """Recovery and diagnostic counts for one representation/measurement condition."""

    model_config = ConfigDict(frozen=True)

    representation: AttitudeRepresentation
    measurement_conditions: MeasurementConditions
    case_count: int = Field(gt=0)
    observed_sample_count_min: int
    observed_sample_count_max: int
    observed_sample_count_mean: float
    dropped_sample_count_min: int
    dropped_sample_count_max: int
    dropped_sample_count_mean: float
    resolved_correct: int
    ambiguous_contains_truth: int
    evidence_miss: int
    data_winner_miss: int
    operational_miss: int
    evidence_insufficient: int
    incorrect_evidence_sufficient: int
    indeterminate_confidence: int
    recommendation_count: int
    abstention_count: int
    recommendation_miss: int

    @computed_field
    @property
    def recommendation_coverage(self) -> float:
        return self.recommendation_count / self.case_count
    ####

    @computed_field
    @property
    def recommendation_accuracy(self) -> float | None:
        if self.recommendation_count == 0:
            return None
        ####
        return (self.recommendation_count - self.recommendation_miss) / self.recommendation_count
    ####

    @computed_field
    @property
    def resolved_correct_rate(self) -> float:
        return self.resolved_correct / self.case_count
    ####

    @computed_field
    @property
    def ambiguous_contains_truth_rate(self) -> float:
        return self.ambiguous_contains_truth / self.case_count
    ####

    @computed_field
    @property
    def evidence_miss_rate(self) -> float:
        return self.evidence_miss / self.case_count
    ####

    @computed_field
    @property
    def data_winner_miss_rate(self) -> float:
        return self.data_winner_miss / self.case_count
    ####

    @computed_field
    @property
    def operational_miss_rate(self) -> float:
        return self.operational_miss / self.case_count
    ####

    @computed_field
    @property
    def evidence_insufficient_rate(self) -> float:
        return self.evidence_insufficient / self.case_count
    ####

    @computed_field
    @property
    def incorrect_evidence_sufficient_rate(self) -> float:
        return self.incorrect_evidence_sufficient / self.case_count
    ####

    @computed_field
    @property
    def indeterminate_confidence_rate(self) -> float:
        return self.indeterminate_confidence / self.case_count
    ####

    @computed_field
    @property
    def recommendation_miss_rate(self) -> float:
        return self.recommendation_miss / self.case_count
    ####
####


class CorpusAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    environment: EvaluationEnvironment
    summary: ValidationSummary
    cases: tuple[CaseAssessment, ...]
    condition_summaries: tuple[ConditionSummary, ...]
####


@dataclass(frozen=True, slots=True)
class _ConditionGroupKey:
    """Named dimensions that define one comparable acquisition condition."""

    representation: AttitudeRepresentation
    motion_profile_id: str | None
    timestamp_jitter_amplitude_s: float
    dropout_probability: float | None
    orientation_latency_s: float
    rate_latency_s: float
    orientation_noise_sigma_rad: float
    rate_noise_sigma_rad_s: float
    orientation_noise_ar1_coefficient: float
    rate_noise_ar1_coefficient: float
    sample_count: int | None
    sample_period_s: float | None

    def ordering_key(
        self,
    ) -> tuple[str, str, float, float, float, float, float, float, float, float, int, float]:
        """Return a stable total ordering for serialized summaries."""

        return (
            self.representation.value,
            self.motion_profile_id or "",
            self.timestamp_jitter_amplitude_s,
            self.dropout_probability if self.dropout_probability is not None else -1.0,
            self.orientation_latency_s,
            self.rate_latency_s,
            self.orientation_noise_sigma_rad,
            self.rate_noise_sigma_rad_s,
            self.orientation_noise_ar1_coefficient,
            self.rate_noise_ar1_coefficient,
            self.sample_count if self.sample_count is not None else -1,
            self.sample_period_s if self.sample_period_s is not None else -1.0,
        )
    ####
####


def _condition_group_key(assessment: CaseAssessment) -> _ConditionGroupKey:
    conditions = assessment.measurement_conditions
    return _ConditionGroupKey(
        representation=assessment.representation,
        motion_profile_id=conditions.motion_profile_id,
        timestamp_jitter_amplitude_s=conditions.timestamp_jitter_amplitude_s,
        dropout_probability=conditions.dropout_probability,
        orientation_latency_s=conditions.orientation_latency_s,
        rate_latency_s=conditions.rate_latency_s,
        orientation_noise_sigma_rad=conditions.orientation_noise_sigma_rad,
        rate_noise_sigma_rad_s=conditions.rate_noise_sigma_rad_s,
        orientation_noise_ar1_coefficient=conditions.orientation_noise_ar1_coefficient,
        rate_noise_ar1_coefficient=conditions.rate_noise_ar1_coefficient,
        sample_count=conditions.sample_count,
        sample_period_s=conditions.sample_period_s,
    )
####


def _condition_summaries(assessments: list[CaseAssessment]) -> tuple[ConditionSummary, ...]:
    groups: dict[_ConditionGroupKey, list[CaseAssessment]] = {}
    for assessment in assessments:
        key = _condition_group_key(assessment)
        groups.setdefault(key, []).append(assessment)
    ####
    summaries: list[ConditionSummary] = []
    for key, cases in sorted(
        groups.items(),
        key=lambda item: item[0].ordering_key(),
    ):
        representative = cases[0]
        dropped_counts = [item.measurement_conditions.dropped_sample_count or 0 for item in cases]
        observed_counts = [item.sample_count for item in cases]
        summaries.append(
            ConditionSummary(
                representation=key.representation,
                measurement_conditions=representative.measurement_conditions.model_copy(
                    update={
                        "noise_seed": None,
                        "dropped_sample_count": (
                            dropped_counts[0]
                            if min(dropped_counts) == max(dropped_counts)
                            else None
                        ),
                        "observed_sample_count": (
                            observed_counts[0]
                            if representative.measurement_conditions.sample_count is not None
                            and min(observed_counts) == max(observed_counts)
                            else None
                        ),
                    }
                ),
                case_count=len(cases),
                observed_sample_count_min=min(observed_counts),
                observed_sample_count_max=max(observed_counts),
                observed_sample_count_mean=sum(observed_counts) / len(observed_counts),
                dropped_sample_count_min=min(dropped_counts),
                dropped_sample_count_max=max(dropped_counts),
                dropped_sample_count_mean=sum(dropped_counts) / len(dropped_counts),
                resolved_correct=sum(item.outcome == "resolved_correct" for item in cases),
                ambiguous_contains_truth=sum(
                    item.outcome == "ambiguous_contains_truth" for item in cases
                ),
                evidence_miss=sum(item.outcome == "evidence_miss" for item in cases),
                data_winner_miss=sum(not item.data_winner_correct for item in cases),
                operational_miss=sum(not item.operational_correct for item in cases),
                evidence_insufficient=sum(not item.evidence_sufficient for item in cases),
                incorrect_evidence_sufficient=sum(
                    item.outcome == "evidence_miss" and item.evidence_sufficient for item in cases
                ),
                indeterminate_confidence=sum(
                    item.confidence_label == "indeterminate" for item in cases
                ),
                recommendation_count=sum(
                    item.decision.status is InferenceDecisionStatus.RECOMMENDATION for item in cases
                ),
                abstention_count=sum(
                    item.decision.status is InferenceDecisionStatus.ABSTENTION for item in cases
                ),
                recommendation_miss=sum(item.recommendation_correct is False for item in cases),
            )
        )
    ####
    return tuple(summaries)
####


def validate_corpus(manifest_path: Path) -> CorpusAssessment:
    manifest = ValidationManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    case_ids = [case.case_id for case in manifest.cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Corpus case_id values must be unique")
    ####

    assessments: list[CaseAssessment] = []
    for case in manifest.cases:
        csv_path = case.csv if case.csv.is_absolute() else manifest_path.parent / case.csv
        report = infer_csv(
            csv_path,
            representation=case.representation,
            orientation_columns=case.orientation_columns,
            rate_columns=case.rate_columns,
            time_column=case.time_column,
            config=case.config,
        )
        if (
            case.measurement_conditions.observed_sample_count is not None
            and case.measurement_conditions.observed_sample_count != report.sample_count
        ):
            raise ValueError(
                f"{case.case_id}: observed_sample_count in the manifest does not match the CSV"
            )
        ####
        if (case.representation is AttitudeRepresentation.QUATERNION) != (
            case.truth.quaternion_order is not None
        ):
            raise ValueError(f"{case.case_id}: truth representation does not match the CSV")
        ####
        truth_id = case.truth.identifier()
        truth_candidates = [
            item for item in report.rankings if item.hypothesis.identifier() == truth_id
        ]
        if len(truth_candidates) != 1:
            raise ValueError(
                f"{case.case_id}: independently labeled truth is outside the configured "
                "hypothesis search; do not count this case as a recovery result"
            )
        ####
        truth_class_id = truth_candidates[0].equivalence_class_id
        contenders = report.evidence_contender_class_ids
        decision = report.decision
        if truth_class_id not in contenders:
            outcome: Literal["resolved_correct", "ambiguous_contains_truth", "evidence_miss"] = (
                "evidence_miss"
            )
        elif len(contenders) == 1:
            outcome = "resolved_correct"
        else:
            outcome = "ambiguous_contains_truth"
        ####
        assessments.append(
            CaseAssessment(
                case_id=case.case_id,
                label_source=case.label_source,
                representation=case.representation,
                measurement_conditions=case.measurement_conditions,
                sample_count=report.sample_count,
                hypothesis_count=report.hypothesis_count,
                truth_class_id=truth_class_id,
                evidence_contender_class_ids=contenders,
                data_winning_class_ids=report.data_winning_class_ids,
                data_winner_correct=truth_class_id in report.data_winning_class_ids,
                operational_class_id=report.operational_class_id,
                operational_correct=truth_class_id == report.operational_class_id,
                decision=decision,
                recommendation_correct=(
                    None
                    if decision.recommended_class_id is None
                    else truth_class_id == decision.recommended_class_id
                ),
                outcome=outcome,
                confidence_label=report.diagnostics.confidence_label.value,
                evidence_sufficient=report.diagnostics.evidence_sufficient,
            )
        )
    ####

    return CorpusAssessment(
        environment=EvaluationEnvironment(
            python=sys.version,
            platform=platform.platform(),
            numpy=numpy_version,
            scipy=scipy_version,
            package=__version__,
        ),
        summary=ValidationSummary(
            case_count=len(assessments),
            resolved_correct=sum(item.outcome == "resolved_correct" for item in assessments),
            ambiguous_contains_truth=sum(
                item.outcome == "ambiguous_contains_truth" for item in assessments
            ),
            evidence_miss=sum(item.outcome == "evidence_miss" for item in assessments),
            data_winner_miss=sum(not item.data_winner_correct for item in assessments),
            operational_miss=sum(not item.operational_correct for item in assessments),
            evidence_insufficient=sum(not item.evidence_sufficient for item in assessments),
            incorrect_evidence_sufficient=sum(
                item.outcome == "evidence_miss" and item.evidence_sufficient for item in assessments
            ),
            recommendation_count=sum(
                item.decision.status is InferenceDecisionStatus.RECOMMENDATION
                for item in assessments
            ),
            abstention_count=sum(
                item.decision.status is InferenceDecisionStatus.ABSTENTION for item in assessments
            ),
            recommendation_miss=sum(item.recommendation_correct is False for item in assessments),
        ),
        cases=tuple(assessments),
        condition_summaries=_condition_summaries(assessments),
    )
####


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest", type=Path, help="JSON manifest with independently labeled cases"
    )
    parser.add_argument("--output", type=Path, help="write the JSON assessment to a file")
    parser.add_argument(
        "--fail-on-miss",
        action="store_true",
        help="exit nonzero if evidence contenders or the operational class miss truth",
    )
    args = parser.parse_args()
    result = validate_corpus(args.manifest)
    output = result.model_dump_json(indent=2) + "\n"
    if args.output is None:
        print(output, end="")
    else:
        args.output.write_text(output, encoding="utf-8")
    ####
    if args.fail_on_miss and (result.summary.evidence_miss or result.summary.operational_miss):
        raise SystemExit(1)
    ####
####


if __name__ == "__main__":
    main()
####
