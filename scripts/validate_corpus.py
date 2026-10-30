"""Measure class recovery and abstention on independently labeled CSV trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from attitude_convention_inference import (
    AttitudeRepresentation,
    ConventionHypothesis,
    InferenceConfig,
    infer_csv,
)


class MeasurementConditions(BaseModel):
    """Typed measurement perturbations recorded for one validation trajectory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp_jitter_amplitude_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    dropped_sample_count: int = Field(default=0, ge=0)
    orientation_noise_sigma_rad: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    rate_noise_sigma_rad_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    noise_seed: int | None = Field(default=None, ge=0)
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


class CaseAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    label_source: str
    measurement_conditions: MeasurementConditions
    sample_count: int
    hypothesis_count: int
    truth_class_id: str
    evidence_contender_class_ids: tuple[str, ...]
    data_winning_class_ids: tuple[str, ...]
    operational_class_id: str
    outcome: Literal["resolved_correct", "ambiguous_contains_truth", "miss"]
    confidence_label: str
    evidence_sufficient: bool
####


class ValidationSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_count: int
    resolved_correct: int
    ambiguous_contains_truth: int
    miss: int
    incorrect_evidence_sufficient: int
####


class CorpusAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    summary: ValidationSummary
    cases: tuple[CaseAssessment, ...]
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
        if truth_class_id not in contenders:
            outcome: Literal["resolved_correct", "ambiguous_contains_truth", "miss"] = "miss"
        elif len(contenders) == 1:
            outcome = "resolved_correct"
        else:
            outcome = "ambiguous_contains_truth"
        ####
        assessments.append(
            CaseAssessment(
                case_id=case.case_id,
                label_source=case.label_source,
                measurement_conditions=case.measurement_conditions,
                sample_count=report.sample_count,
                hypothesis_count=report.hypothesis_count,
                truth_class_id=truth_class_id,
                evidence_contender_class_ids=contenders,
                data_winning_class_ids=report.data_winning_class_ids,
                operational_class_id=report.operational_class_id,
                outcome=outcome,
                confidence_label=report.diagnostics.confidence_label.value,
                evidence_sufficient=report.diagnostics.evidence_sufficient,
            )
        )
    ####

    return CorpusAssessment(
        summary=ValidationSummary(
            case_count=len(assessments),
            resolved_correct=sum(item.outcome == "resolved_correct" for item in assessments),
            ambiguous_contains_truth=sum(
                item.outcome == "ambiguous_contains_truth" for item in assessments
            ),
            miss=sum(item.outcome == "miss" for item in assessments),
            incorrect_evidence_sufficient=sum(
                item.outcome == "miss" and item.evidence_sufficient for item in assessments
            ),
        ),
        cases=tuple(assessments),
    )
####


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest", type=Path, help="JSON manifest with independently labeled cases"
    )
    parser.add_argument("--output", type=Path, help="write the JSON assessment to a file")
    parser.add_argument(
        "--fail-on-miss", action="store_true", help="exit nonzero if any case misses"
    )
    args = parser.parse_args()
    result = validate_corpus(args.manifest)
    output = result.model_dump_json(indent=2) + "\n"
    if args.output is None:
        print(output, end="")
    else:
        args.output.write_text(output, encoding="utf-8")
    ####
    if args.fail_on_miss and result.summary.miss:
        raise SystemExit(1)
    ####
####


if __name__ == "__main__":
    main()
####
