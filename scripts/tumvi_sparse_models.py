"""Typed JSON contracts for sparse TUM-VI validation results."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from attitude_convention_inference import ConventionHypothesis
from scripts.validate_corpus import DatasetProvenance

__all__ = ["TUMVI_TRUTH", "SparseCadenceAssessment", "SparseTUMVIStudyReport"]

TUMVI_TRUTH = ConventionHypothesis.model_validate(
    {
        "attitude_direction": "body_to_world",
        "quaternion_order": "scalar_last",
        "rate_frame": "body",
        "rate_order": "pqr",
        "rate_unit": "radians_per_second",
        "rate_sign": 1,
    }
)


class SparseCadenceAssessment(BaseModel):
    """Typed recovery outcome for one requested native-pose sampling cadence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_pose_cadence_s: float = Field(gt=0.0, allow_inf_nan=False)
    orientation_anchor_count: int = Field(ge=3)
    dense_rate_sample_count: int = Field(ge=2)
    actual_anchor_span_s: float = Field(ge=0.0, allow_inf_nan=False)
    mean_actual_anchor_cadence_s: float = Field(gt=0.0, allow_inf_nan=False)
    maximum_grid_offset_s: float = Field(ge=0.0, allow_inf_nan=False)
    labeled_hypothesis_rank: int | None = Field(default=None, ge=1)
    labeled_data_rank: int | None = Field(default=None, ge=1)
    label_in_evidence_contenders: bool
    label_is_data_winner: bool
    recommendation: Literal["matched_label", "different_class", "abstained"]
    operational_class_id: str
    operational_reason: str
    evidence_contender_class_count: int = Field(ge=0)
####


class SparseTUMVIStudyReport(BaseModel):
    """Reproducible outcomes and provenance for sparse TUM-VI inference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    study_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    package_version: str = Field(min_length=1)
    observation_model: Literal["sparse_orientation_anchors"]
    sequence: str = Field(min_length=1)
    common_origin_timestamp_ns: int
    requested_start_offset_ns: int = Field(ge=0)
    requested_duration_ns: int = Field(gt=0)
    truth: ConventionHypothesis
    source_provenance: DatasetProvenance
    lineage_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    upstream_lineage_provenance: DatasetProvenance | None = None
    unmeasured_conditions: tuple[
        Literal[
            "orientation_noise",
            "rate_noise",
            "orientation_latency",
            "rate_latency",
            "timestamp_jitter",
            "dropout",
            "timestamp_clock_rate_error",
        ],
        ...,
    ]
    cadence_assessments: tuple[SparseCadenceAssessment, ...] = Field(min_length=1)
####
