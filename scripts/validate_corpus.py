"""Measure class recovery and abstention on independently labeled CSV trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from math import exp, isclose, isfinite, log, sqrt
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self, cast

from numpy import __version__ as numpy_version
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)
from scipy import __version__ as scipy_version

import attitude_convention_inference as inference_package
from attitude_convention_inference import (
    AttitudeRepresentation,
    CandidateResult,
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
    motion_seed: int | None = Field(default=None, ge=0)
    timestamp_jitter_amplitude_s: float | None = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
        description="Controlled timestamp-jitter amplitude; null means unreported.",
    )
    timestamp_clock_rate_error_ppm: float | None = Field(
        default=None,
        gt=-1_000_000.0,
        allow_inf_nan=False,
        description=(
            "Known timestamp-clock scale error in parts per million; positive means logged "
            "elapsed time runs faster than physical elapsed time, and null means unreported."
        ),
    )
    dropout_probability: float | None = Field(
        default=None,
        ge=0.0,
        lt=1.0,
        allow_inf_nan=False,
        description="Controlled whole-row dropout probability; null means unreported.",
    )
    dropped_sample_count: int | None = Field(
        default=0,
        ge=0,
        description="Controlled dropped-row count; null means the reference count is unknown.",
    )
    orientation_latency_s: float | None = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
        description="Controlled orientation-channel latency; null means unreported.",
    )
    rate_latency_s: float | None = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
        description="Controlled rate-channel latency; null means unreported.",
    )
    rate_sensor_calibration_id: str | None = Field(
        default=None,
        min_length=1,
        description="Stable identifier for a known rate-channel calibration profile.",
    )
    rate_sensor_misalignment_angle_deg: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description="Known rate-channel axis rotation angle in degrees; null means unreported.",
    )
    rate_sensor_misalignment_axis: tuple[float, float, float] | None = None
    rate_sensor_bias_rad_s: tuple[float, float, float] | None = None
    rate_sensor_scale_by_axis: tuple[float, float, float] | None = None
    rate_sensor_polarity_by_axis: tuple[int, int, int] | None = None
    rate_sensor_clip_abs_rad_s: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
        description="Known absolute rate-channel clipping limit in rad/s; null means unreported.",
    )
    orientation_noise_sigma_rad: float | None = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
        description="Known orientation-noise scale; null means unmeasured.",
    )
    rate_noise_sigma_rad_s: float | None = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
        description="Known rate-noise scale; null means unmeasured.",
    )
    orientation_noise_ar1_coefficient: float | None = Field(
        default=0.0,
        gt=-1.0,
        lt=1.0,
        allow_inf_nan=False,
        description="Known orientation-noise AR(1) coefficient; null means unmeasured.",
    )
    rate_noise_ar1_coefficient: float | None = Field(
        default=0.0,
        gt=-1.0,
        lt=1.0,
        allow_inf_nan=False,
        description="Known rate-noise AR(1) coefficient; null means unmeasured.",
    )
    orientation_noise_correlation_time_s: float | None = Field(
        default=None, gt=0.0, allow_inf_nan=False
    )
    rate_noise_correlation_time_s: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    rate_noise_frame: Literal["truth-rate-frame", "body", "world"] = Field(
        default="truth-rate-frame",
        description=(
            "Coordinate frame used to inject rate noise; truth-rate-frame preserves historical "
            "per-convention injection, while body/world identify one shared vector frame."
        ),
    )
    noise_seed: int | None = Field(default=None, ge=0)
    sample_count: int | None = Field(default=None, gt=0)
    observed_sample_count: int | None = Field(default=None, gt=0)
    sample_period_s: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_sampling_pair(self) -> Self:
        calibration_parameters = (
            self.rate_sensor_misalignment_angle_deg,
            self.rate_sensor_misalignment_axis,
            self.rate_sensor_bias_rad_s,
            self.rate_sensor_scale_by_axis,
            self.rate_sensor_polarity_by_axis,
        )
        if self.rate_sensor_calibration_id is not None and any(
            parameter is None for parameter in calibration_parameters
        ):
            raise ValueError(
                "a named rate-sensor calibration profile requires all core calibration parameters"
            )
        ####
        if self.rate_sensor_misalignment_angle_deg is not None and not isfinite(
            self.rate_sensor_misalignment_angle_deg
        ):
            raise ValueError("rate-sensor misalignment angle must be finite")
        ####
        for name, values in (
            ("misalignment axis", self.rate_sensor_misalignment_axis),
            ("bias", self.rate_sensor_bias_rad_s),
            ("scale", self.rate_sensor_scale_by_axis),
        ):
            if values is not None and any(not isfinite(value) for value in values):
                raise ValueError(f"rate-sensor {name} values must be finite")
            ####
        ####
        if self.rate_sensor_misalignment_axis is not None:
            axis_norm = sqrt(
                sum(component * component for component in self.rate_sensor_misalignment_axis)
            )
            if not isfinite(axis_norm) or axis_norm <= 0.0:
                raise ValueError("rate-sensor misalignment axis must have a finite non-zero norm")
            ####
        ####
        if self.rate_sensor_scale_by_axis is not None and any(
            scale <= 0.0 for scale in self.rate_sensor_scale_by_axis
        ):
            raise ValueError("rate-sensor axis scales must be positive")
        ####
        if self.rate_sensor_polarity_by_axis is not None and any(
            polarity not in (-1, 1) for polarity in self.rate_sensor_polarity_by_axis
        ):
            raise ValueError("rate-sensor axis polarities must be -1 or +1")
        ####
        if (self.sample_count is None) != (self.sample_period_s is None):
            raise ValueError("sample_count and sample_period_s must be provided together")
        ####
        if self.observed_sample_count is not None:
            if self.sample_count is None:
                raise ValueError("observed_sample_count requires sample_count and sample_period_s")
            ####
            if (
                self.dropped_sample_count is not None
                and self.observed_sample_count + self.dropped_sample_count != self.sample_count
            ):
                raise ValueError(
                    "observed_sample_count plus dropped_sample_count must equal sample_count"
                )
            ####
        ####
        if (
            self.sample_period_s is not None
            and self.timestamp_jitter_amplitude_s is not None
            and self.timestamp_jitter_amplitude_s >= self.sample_period_s / 2.0
        ):
            raise ValueError("timestamp jitter amplitude must be less than half the sample period")
        ####
        for name, time_constant_s, ar1_coefficient in (
            (
                "orientation",
                self.orientation_noise_correlation_time_s,
                self.orientation_noise_ar1_coefficient,
            ),
            (
                "rate",
                self.rate_noise_correlation_time_s,
                self.rate_noise_ar1_coefficient,
            ),
        ):
            if time_constant_s is None:
                continue
            ####
            if self.sample_period_s is None:
                raise ValueError(
                    f"{name} noise correlation time requires sample_count and sample_period_s"
                )
            ####
            if ar1_coefficient is None:
                raise ValueError(f"{name} noise AR(1) coefficient is required with a time constant")
            ####
            expected_ar1_coefficient = exp(-self.sample_period_s / time_constant_s)
            if not isclose(
                ar1_coefficient,
                expected_ar1_coefficient,
                rel_tol=1.0e-9,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    f"{name} noise AR(1) coefficient must equal exp(-sample_period_s / "
                    "correlation_time_s)"
                )
            ####
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


class SourceArtifactFingerprint(BaseModel):
    """Identity for one external source file used to make a corpus case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file_name: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("file_name")
    @classmethod
    def _validate_relative_name(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
            raise ValueError("file_name must be a normalized relative path")
        ####
        return path.as_posix()
    ####
####


class DatasetProvenance(BaseModel):
    """Typed source, convention, timing, and transformation record for external data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str = Field(min_length=1)
    citation: str = Field(min_length=1)
    source_url: AnyHttpUrl
    license: str = Field(min_length=1)
    sequence: str = Field(min_length=1)
    orientation_convention_basis: str = Field(min_length=1)
    rate_convention_basis: str = Field(min_length=1)
    timing_contract: str = Field(min_length=1)
    source_artifacts: tuple[SourceArtifactFingerprint, ...] = Field(min_length=1)
    transformations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_unique_artifacts(self) -> Self:
        file_names = tuple(item.file_name for item in self.source_artifacts)
        if len(set(file_names)) != len(file_names):
            raise ValueError("source_artifacts file_name values must be unique")
        ####
        return self
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
    source_provenance: DatasetProvenance | None = None

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

    schema_version: Literal[1, 2, 3, 4, 5]
    cases: tuple[ValidationCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_metadata_schema(self) -> Self:
        has_provenance = any(case.source_provenance is not None for case in self.cases)
        calibration_profiles: dict[str, tuple[object, ...]] = {}
        has_rate_calibration_metadata = False
        for case in self.cases:
            conditions = case.measurement_conditions
            calibration_values = (
                conditions.rate_sensor_misalignment_angle_deg,
                conditions.rate_sensor_misalignment_axis,
                conditions.rate_sensor_bias_rad_s,
                conditions.rate_sensor_scale_by_axis,
                conditions.rate_sensor_polarity_by_axis,
                conditions.rate_sensor_clip_abs_rad_s,
            )
            has_rate_calibration_metadata |= (
                conditions.rate_sensor_calibration_id is not None
                or any(value is not None for value in calibration_values)
            )
            calibration_id = conditions.rate_sensor_calibration_id
            if calibration_id is None:
                continue
            ####
            previous_values = calibration_profiles.get(calibration_id)
            if previous_values is not None and previous_values != calibration_values:
                raise ValueError("each rate_sensor_calibration_id must identify one parameter set")
            ####
            calibration_profiles[calibration_id] = calibration_values
        ####
        if self.schema_version == 1 and has_provenance:
            raise ValueError(
                "source_provenance requires validation manifest schema_version 2 or later"
            )
        ####
        if self.schema_version < 5 and has_rate_calibration_metadata:
            raise ValueError(
                "rate-sensor calibration metadata requires validation manifest schema_version 5"
            )
        ####
        return self
    ####
####


class EvaluationEnvironment(BaseModel):
    """Runtime versions needed to interpret a generated assessment reproducibly."""

    model_config = ConfigDict(frozen=True)

    python: str
    platform: str
    numpy: str
    scipy: str
    package: str
    implementation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
####


class InferenceDiagnostics(BaseModel):
    """Typed inference diagnostics preserved for one evaluated trajectory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    estimated_noise_sigma_rad: float = Field(ge=0.0, allow_inf_nan=False)
    noise_residual_count: int = Field(ge=0)
    noise_lag1_autocorrelation: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    evidence_intervals_used: int = Field(ge=0)
    evidence_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    evidence_log_likelihood_ratio: float = Field(allow_inf_nan=False)
    warnings: tuple[str, ...]
####


class CaseAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    label_source: str
    source_provenance: DatasetProvenance | None = None
    representation: AttitudeRepresentation
    measurement_conditions: MeasurementConditions
    sample_count: int
    hypothesis_count: int
    truth_class_id: str
    evidence_contender_class_ids: tuple[str, ...]
    nearest_alternative_class_id: str | None = Field(
        default=None,
        description="Closest non-winning quotient class by floor-adjusted geometric risk.",
    )
    nearest_alternative_log_risk_delta: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "Log floor-adjusted geometric-risk ratio between the nearest alternative and "
            "the best class; this is a relative-fit margin, not a likelihood ratio."
        ),
    )
    data_winning_class_ids: tuple[str, ...]
    data_winner_correct: bool
    operational_class_id: str
    operational_correct: bool
    decision: InferenceDecision
    recommendation_correct: bool | None
    outcome: Literal["resolved_correct", "ambiguous_contains_truth", "evidence_miss"]
    confidence_label: str
    evidence_sufficient: bool
    inference_diagnostics: InferenceDiagnostics
####


@dataclass(frozen=True, slots=True)
class _ValidatedCaseResult:
    assessment_json: str
    source_sha256: str | None
####


@dataclass(frozen=True, slots=True)
class _NearestAlternativeRisk:
    """Named geometric-risk margin to the nearest distinct quotient class."""

    class_id: str | None
    log_risk_delta: float | None
####


def _nearest_alternative_risk(
    rankings: tuple[CandidateResult, ...],
    residual_floor_rad: float,
) -> _NearestAlternativeRisk:
    """Measure the second class's floor-adjusted log-risk delta from the best class."""

    representatives: dict[str, CandidateResult] = {}
    for result in rankings:
        representatives.setdefault(result.equivalence_class_id, result)
    ####
    risk_floor = residual_floor_rad**2
    ordered = sorted(
        representatives.values(),
        key=lambda result: (
            max(result.class_data_risk, risk_floor),
            result.equivalence_class_id,
        ),
    )
    if len(ordered) < 2:
        return _NearestAlternativeRisk(class_id=None, log_risk_delta=None)
    ####
    best_log_risk = log(max(ordered[0].class_data_risk, risk_floor))
    alternative_log_risk = log(max(ordered[1].class_data_risk, risk_floor))
    return _NearestAlternativeRisk(
        class_id=ordered[1].equivalence_class_id,
        log_risk_delta=alternative_log_risk - best_log_risk,
    )
####


class ValidationCheckpoint(BaseModel):
    """Completed case results bound to one manifest and runtime environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2, 3, 4, 5, 6, 7, 8, 9] = 9
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment: EvaluationEnvironment
    completed_cases: tuple[CaseAssessment, ...]
    source_sha256_by_case: dict[str, str]
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
    """Recovery counts for one truth class, representation, and measurement condition."""

    model_config = ConfigDict(frozen=True)

    representation: AttitudeRepresentation
    truth_class_id: str
    measurement_conditions: MeasurementConditions
    case_count: int = Field(gt=0)
    observed_sample_count_min: int
    observed_sample_count_max: int
    observed_sample_count_mean: float
    dropped_sample_count_min: int | None
    dropped_sample_count_max: int | None
    dropped_sample_count_mean: float | None
    resolved_correct: int
    ambiguous_contains_truth: int
    evidence_miss: int
    data_winner_miss: int
    operational_miss: int
    evidence_insufficient: int
    incorrect_evidence_sufficient: int
    indeterminate_confidence: int
    mean_noise_lag1_autocorrelation: float
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

    schema_version: Literal[5, 6, 7, 8, 9, 10] = 10
    environment: EvaluationEnvironment
    summary: ValidationSummary
    cases: tuple[CaseAssessment, ...]
    condition_summaries: tuple[ConditionSummary, ...]
####


@dataclass(frozen=True, slots=True)
class _ConditionGroupKey:
    """Truth target and acquisition dimensions for one comparable report cell."""

    representation: AttitudeRepresentation
    truth_class_id: str
    motion_profile_id: str | None
    motion_seed: int | None
    timestamp_jitter_amplitude_s: float | None
    timestamp_clock_rate_error_ppm: float | None
    dropout_probability: float | None
    orientation_latency_s: float | None
    rate_latency_s: float | None
    rate_sensor_calibration_key: str
    orientation_noise_sigma_rad: float | None
    rate_noise_sigma_rad_s: float | None
    orientation_noise_ar1_coefficient: float | None
    rate_noise_ar1_coefficient: float | None
    orientation_noise_correlation_time_s: float | None
    rate_noise_correlation_time_s: float | None
    sample_count: int | None
    sample_period_s: float | None

    def ordering_key(
        self,
    ) -> tuple[
        str,
        str,
        str,
        int,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        int,
        float,
        str,
    ]:
        """Return a stable total ordering for serialized summaries."""

        return (
            self.representation.value,
            self.truth_class_id,
            self.motion_profile_id or "",
            self.motion_seed if self.motion_seed is not None else -1,
            (
                self.timestamp_jitter_amplitude_s
                if self.timestamp_jitter_amplitude_s is not None
                else -1.0
            ),
            (
                self.timestamp_clock_rate_error_ppm
                if self.timestamp_clock_rate_error_ppm is not None
                else -1_000_001.0
            ),
            self.dropout_probability if self.dropout_probability is not None else -1.0,
            self.orientation_latency_s if self.orientation_latency_s is not None else -1.0,
            self.rate_latency_s if self.rate_latency_s is not None else -1.0,
            (
                self.orientation_noise_sigma_rad
                if self.orientation_noise_sigma_rad is not None
                else -1.0
            ),
            self.rate_noise_sigma_rad_s if self.rate_noise_sigma_rad_s is not None else -1.0,
            (
                self.orientation_noise_ar1_coefficient
                if self.orientation_noise_ar1_coefficient is not None
                else -2.0
            ),
            (
                self.rate_noise_ar1_coefficient
                if self.rate_noise_ar1_coefficient is not None
                else -2.0
            ),
            (
                self.orientation_noise_correlation_time_s
                if self.orientation_noise_correlation_time_s is not None
                else -1.0
            ),
            (
                self.rate_noise_correlation_time_s
                if self.rate_noise_correlation_time_s is not None
                else -1.0
            ),
            self.sample_count if self.sample_count is not None else -1,
            self.sample_period_s if self.sample_period_s is not None else -1.0,
            self.rate_sensor_calibration_key,
        )
    ####
####


def _condition_group_key(assessment: CaseAssessment) -> _ConditionGroupKey:
    conditions = assessment.measurement_conditions
    rate_sensor_calibration_key = json.dumps(
        (
            conditions.rate_sensor_calibration_id,
            conditions.rate_sensor_misalignment_angle_deg,
            conditions.rate_sensor_misalignment_axis,
            conditions.rate_sensor_bias_rad_s,
            conditions.rate_sensor_scale_by_axis,
            conditions.rate_sensor_polarity_by_axis,
            conditions.rate_sensor_clip_abs_rad_s,
        ),
        separators=(",", ":"),
    )
    return _ConditionGroupKey(
        representation=assessment.representation,
        truth_class_id=assessment.truth_class_id,
        motion_profile_id=conditions.motion_profile_id,
        motion_seed=conditions.motion_seed,
        timestamp_jitter_amplitude_s=conditions.timestamp_jitter_amplitude_s,
        timestamp_clock_rate_error_ppm=conditions.timestamp_clock_rate_error_ppm,
        dropout_probability=conditions.dropout_probability,
        orientation_latency_s=conditions.orientation_latency_s,
        rate_latency_s=conditions.rate_latency_s,
        rate_sensor_calibration_key=rate_sensor_calibration_key,
        orientation_noise_sigma_rad=conditions.orientation_noise_sigma_rad,
        rate_noise_sigma_rad_s=conditions.rate_noise_sigma_rad_s,
        orientation_noise_ar1_coefficient=conditions.orientation_noise_ar1_coefficient,
        rate_noise_ar1_coefficient=conditions.rate_noise_ar1_coefficient,
        orientation_noise_correlation_time_s=conditions.orientation_noise_correlation_time_s,
        rate_noise_correlation_time_s=conditions.rate_noise_correlation_time_s,
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
        dropped_counts = [item.measurement_conditions.dropped_sample_count for item in cases]
        dropped_counts_known = all(count is not None for count in dropped_counts)
        known_dropped_counts = [count for count in dropped_counts if count is not None]
        dropped_count_min = min(known_dropped_counts) if dropped_counts_known else None
        dropped_count_max = max(known_dropped_counts) if dropped_counts_known else None
        observed_counts = [item.sample_count for item in cases]
        summaries.append(
            ConditionSummary(
                representation=key.representation,
                truth_class_id=key.truth_class_id,
                measurement_conditions=representative.measurement_conditions.model_copy(
                    update={
                        "noise_seed": None,
                        "dropped_sample_count": (
                            dropped_count_min if dropped_count_min == dropped_count_max else None
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
                dropped_sample_count_min=dropped_count_min,
                dropped_sample_count_max=dropped_count_max,
                dropped_sample_count_mean=(
                    sum(known_dropped_counts) / len(known_dropped_counts)
                    if dropped_counts_known
                    else None
                ),
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
                mean_noise_lag1_autocorrelation=sum(
                    item.inference_diagnostics.noise_lag1_autocorrelation for item in cases
                )
                / len(cases),
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


def _evaluation_environment() -> EvaluationEnvironment:
    return EvaluationEnvironment(
        python=sys.version,
        platform=platform.platform(),
        numpy=numpy_version,
        scipy=scipy_version,
        package=__version__,
        implementation_sha256=_implementation_sha256(),
    )
####


def _implementation_sha256() -> str:
    package_directory = Path(inference_package.__file__).resolve().parent
    if not package_directory.is_dir():
        raise RuntimeError("Cannot locate the imported inference package source")
    ####
    validator_path = Path(__file__).resolve()
    package_paths = sorted(
        (
            path
            for path in package_directory.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        ),
        key=lambda path: path.as_posix(),
    )
    digest = hashlib.sha256()
    digest.update(b"validator/validate_corpus.py\0")
    digest.update(bytes.fromhex(_sha256_file(validator_path)))
    for path in package_paths:
        relative_path = path.relative_to(package_directory).as_posix()
        digest.update(b"\0package/")
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    ####
    return digest.hexdigest()
####


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
        ####
    ####
    return digest.hexdigest()
####


def _checkpoint_from_results(
    manifest_sha256: str,
    environment: EvaluationEnvironment,
    manifest_case_ids: tuple[str, ...],
    completed_by_id: dict[str, CaseAssessment],
    source_sha256_by_case: dict[str, str],
) -> ValidationCheckpoint:
    return ValidationCheckpoint(
        manifest_sha256=manifest_sha256,
        environment=environment,
        completed_cases=tuple(
            completed_by_id[case_id] for case_id in manifest_case_ids if case_id in completed_by_id
        ),
        source_sha256_by_case={
            case_id: source_sha256_by_case[case_id]
            for case_id in manifest_case_ids
            if case_id in completed_by_id
        },
    )
####


def _write_checkpoint(path: Path, checkpoint: ValidationCheckpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write('{"schema_version":')
            json.dump(checkpoint.schema_version, stream)
            stream.write(',"manifest_sha256":')
            json.dump(checkpoint.manifest_sha256, stream)
            stream.write(',"environment":')
            json.dump(
                checkpoint.environment.model_dump(mode="json"),
                stream,
                separators=(",", ":"),
            )
            stream.write(',"completed_cases":[')
            for index, case in enumerate(checkpoint.completed_cases):
                if index:
                    stream.write(",")
                ####
                json.dump(
                    case.model_dump(
                        mode="json",
                        exclude={"measurement_conditions": {"duration_s"}},
                    ),
                    stream,
                    separators=(",", ":"),
                )
            ####
            stream.write('],"source_sha256_by_case":')
            json.dump(checkpoint.source_sha256_by_case, stream, separators=(",", ":"))
            stream.write("}")
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        ####
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        ####
    ####
####


def _load_checkpoint(
    path: Path,
    *,
    manifest_sha256: str,
    environment: EvaluationEnvironment,
    case_csv_paths: dict[str, Path],
) -> ValidationCheckpoint:
    checkpoint = ValidationCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
    if checkpoint.schema_version < 7:
        raise ValueError(
            "Checkpoint predates nearest-alternative risk margins; use a new checkpoint path "
            "to recompute completed cases"
        )
    ####
    if checkpoint.manifest_sha256 != manifest_sha256:
        raise ValueError("Checkpoint manifest hash does not match the input manifest")
    ####
    if checkpoint.environment != environment:
        raise ValueError("Checkpoint runtime environment does not match the current process")
    ####
    completed_by_id = {case.case_id: case for case in checkpoint.completed_cases}
    if len(completed_by_id) != len(checkpoint.completed_cases):
        raise ValueError("Checkpoint contains duplicate case IDs")
    ####
    unknown_ids = set(completed_by_id) - set(case_csv_paths)
    if unknown_ids:
        raise ValueError("Checkpoint contains case IDs that are not in the input manifest")
    ####
    if set(checkpoint.source_sha256_by_case) != set(completed_by_id):
        raise ValueError("Checkpoint source hashes do not match its completed cases")
    ####
    for case_id, expected_sha256 in checkpoint.source_sha256_by_case.items():
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("Checkpoint contains an invalid source CSV hash")
        ####
        if _sha256_file(case_csv_paths[case_id]) != expected_sha256:
            raise ValueError(f"Checkpoint source CSV hash does not match: {case_id}")
        ####
    ####
    return checkpoint
####


def _validate_case(
    case_json: str,
    csv_path: Path,
    *,
    verify_source_stability: bool,
) -> _ValidatedCaseResult:
    case = ValidationCase.model_validate_json(case_json)
    source_sha256 = _sha256_file(csv_path) if verify_source_stability else None
    report = infer_csv(
        csv_path,
        representation=case.representation,
        orientation_columns=case.orientation_columns,
        rate_columns=case.rate_columns,
        time_column=case.time_column,
        config=case.config,
    )
    if verify_source_stability and _sha256_file(csv_path) != source_sha256:
        raise ValueError(f"{case.case_id}: CSV changed during validation")
    ####
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
    nearest_alternative_risk = _nearest_alternative_risk(
        report.rankings,
        case.config.residual_floor_rad,
    )
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
    assessment = CaseAssessment(
        case_id=case.case_id,
        label_source=case.label_source,
        source_provenance=case.source_provenance,
        representation=case.representation,
        measurement_conditions=case.measurement_conditions,
        sample_count=report.sample_count,
        hypothesis_count=report.hypothesis_count,
        truth_class_id=truth_class_id,
        evidence_contender_class_ids=contenders,
        nearest_alternative_class_id=nearest_alternative_risk.class_id,
        nearest_alternative_log_risk_delta=nearest_alternative_risk.log_risk_delta,
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
        inference_diagnostics=InferenceDiagnostics(
            confidence=report.diagnostics.confidence,
            estimated_noise_sigma_rad=report.diagnostics.estimated_noise_sigma_rad,
            noise_residual_count=report.diagnostics.noise_residual_count,
            noise_lag1_autocorrelation=report.diagnostics.noise_lag1_autocorrelation,
            evidence_intervals_used=report.diagnostics.evidence_intervals_used,
            evidence_fraction=report.diagnostics.evidence_fraction,
            evidence_log_likelihood_ratio=report.diagnostics.evidence_log_likelihood_ratio,
            warnings=report.diagnostics.warnings,
        ),
    )
    return _ValidatedCaseResult(
        assessment_json=assessment.model_dump_json(
            exclude={"measurement_conditions": {"duration_s"}}
        ),
        source_sha256=source_sha256,
    )
####


def _validation_case_json(case: ValidationCase) -> str:
    return case.model_dump_json(exclude={"measurement_conditions": {"duration_s"}})
####


def validate_corpus(
    manifest_path: Path,
    *,
    show_progress: bool = False,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    checkpoint_every: int = 100,
    max_workers: int = 1,
) -> CorpusAssessment:
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1")
    ####
    if type(max_workers) is not int or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    ####
    if resume and checkpoint_path is None:
        raise ValueError("resume requires checkpoint_path")
    ####
    if checkpoint_path is not None and checkpoint_path.resolve() == manifest_path.resolve():
        raise ValueError("Checkpoint path must not overwrite the input manifest")
    ####
    manifest_bytes = manifest_path.read_bytes()
    manifest = ValidationManifest.model_validate_json(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    case_ids = [case.case_id for case in manifest.cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Corpus case_id values must be unique")
    ####
    case_csv_paths = {
        case.case_id: case.csv if case.csv.is_absolute() else manifest_path.parent / case.csv
        for case in manifest.cases
    }
    if checkpoint_path is not None:
        input_paths = {
            manifest_path.resolve(),
            *(path.resolve() for path in case_csv_paths.values()),
        }
        if checkpoint_path.resolve() in input_paths:
            raise ValueError("Checkpoint path must not overwrite the manifest or a source CSV")
        ####
    ####

    environment = _evaluation_environment()
    completed_by_id: dict[str, CaseAssessment] = {}
    source_sha256_by_case: dict[str, str] = {}
    if checkpoint_path is not None and checkpoint_path.exists():
        if not resume:
            raise FileExistsError(
                "Checkpoint already exists; pass resume=True or choose a new path: "
                f"{checkpoint_path}"
            )
        ####
        checkpoint = _load_checkpoint(
            checkpoint_path,
            manifest_sha256=manifest_sha256,
            environment=environment,
            case_csv_paths=case_csv_paths,
        )
        completed_by_id = {case.case_id: case for case in checkpoint.completed_cases}
        source_sha256_by_case = checkpoint.source_sha256_by_case.copy()
    elif resume:
        raise FileNotFoundError(f"Cannot resume; checkpoint does not exist: {checkpoint_path}")
    ####

    total_cases = len(manifest.cases)
    started_at = time.perf_counter()
    processed_this_run = 0
    cases_since_checkpoint = 0
    if show_progress and completed_by_id:
        print(
            f"Resuming from {len(completed_by_id)}/{total_cases} checkpointed cases",
            file=sys.stderr,
            flush=True,
        )
    ####

    def record_result(result: _ValidatedCaseResult) -> None:
        nonlocal processed_this_run, cases_since_checkpoint

        assessment = CaseAssessment.model_validate_json(result.assessment_json)
        completed_by_id[assessment.case_id] = assessment
        if checkpoint_path is not None:
            if result.source_sha256 is None:
                raise RuntimeError("Checkpointed case validation omitted its source hash")
            ####
            source_sha256_by_case[assessment.case_id] = result.source_sha256
        ####
        processed_this_run += 1
        cases_since_checkpoint += 1
        completed_count = len(completed_by_id)
        if checkpoint_path is not None and cases_since_checkpoint >= checkpoint_every:
            _write_checkpoint(
                checkpoint_path,
                _checkpoint_from_results(
                    manifest_sha256,
                    environment,
                    tuple(case_ids),
                    completed_by_id,
                    source_sha256_by_case,
                ),
            )
            cases_since_checkpoint = 0
        ####
        if show_progress and (completed_count % 100 == 0 or completed_count == total_cases):
            elapsed_s = time.perf_counter() - started_at
            cases_per_second = processed_this_run / elapsed_s if elapsed_s > 0.0 else 0.0
            estimated_remaining_s = (
                (total_cases - completed_count) / cases_per_second
                if cases_per_second > 0.0
                else 0.0
            )
            print(
                f"Validated {completed_count}/{total_cases} cases "
                f"({cases_per_second:.2f} cases/s; ETA {estimated_remaining_s:.0f}s)",
                file=sys.stderr,
                flush=True,
            )
        ####
    ####

    uncompleted_cases = tuple(
        case for case in manifest.cases if case.case_id not in completed_by_id
    )
    worker_count = min(max_workers, len(uncompleted_cases))
    if worker_count <= 1:
        for case in uncompleted_cases:
            record_result(
                _validate_case(
                    _validation_case_json(case),
                    case_csv_paths[case.case_id],
                    verify_source_stability=checkpoint_path is not None,
                )
            )
        ####
    else:
        try:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                case_index = 0
                pending: dict[Future[_ValidatedCaseResult], str] = {}

                def submit_next_case() -> None:
                    nonlocal case_index

                    if case_index >= len(uncompleted_cases):
                        return
                    ####
                    case = uncompleted_cases[case_index]
                    case_index += 1
                    pending[
                        executor.submit(
                            _validate_case,
                            _validation_case_json(case),
                            case_csv_paths[case.case_id],
                            verify_source_stability=checkpoint_path is not None,
                        )
                    ] = case.case_id
                ####

                for _ in range(worker_count):
                    submit_next_case()
                ####
                while pending:
                    completed_futures, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed_futures:
                        pending.pop(future)
                        record_result(future.result())
                        submit_next_case()
                    ####
                ####
            ####
        except BrokenProcessPool:
            remaining_cases = tuple(
                case for case in uncompleted_cases if case.case_id not in completed_by_id
            )
            print(
                "Warning: a validation worker exited unexpectedly; retrying "
                f"{len(remaining_cases)} unfinished case(s) sequentially",
                file=sys.stderr,
                flush=True,
            )
            for case in remaining_cases:
                record_result(
                    _validate_case(
                        _validation_case_json(case),
                        case_csv_paths[case.case_id],
                        verify_source_stability=checkpoint_path is not None,
                    )
                )
            ####
        ####
    ####

    if len(completed_by_id) != total_cases:
        raise RuntimeError("Validation finished without an assessment for every manifest case")
    ####
    if checkpoint_path is not None:
        _write_checkpoint(
            checkpoint_path,
            _checkpoint_from_results(
                manifest_sha256,
                environment,
                tuple(case_ids),
                completed_by_id,
                source_sha256_by_case,
            ),
        )
    ####
    assessments = [completed_by_id[case_id] for case_id in case_ids]
    return CorpusAssessment(
        environment=environment,
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
        "--checkpoint",
        type=Path,
        help="atomically save completed case results for later resumption",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="save checkpoint after this many newly completed cases (default: 100)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="case-level worker processes (default: 1)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from the existing --checkpoint file",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="report completed cases and an estimated remaining time to stderr",
    )
    parser.add_argument(
        "--fail-on-miss",
        action="store_true",
        help="exit nonzero if evidence contenders or the operational class miss truth",
    )
    args = parser.parse_args()
    if (
        args.checkpoint is not None
        and args.output is not None
        and args.checkpoint.resolve() == args.output.resolve()
    ):
        parser.error("--checkpoint and --output must be different paths")
    ####
    result = validate_corpus(
        args.manifest,
        show_progress=args.progress,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        max_workers=args.workers,
    )
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
