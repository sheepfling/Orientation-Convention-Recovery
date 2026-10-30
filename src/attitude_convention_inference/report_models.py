"""Typed candidate, diagnostic, and inference-report contracts."""

from __future__ import annotations

from dataclasses import field
from typing import Literal

from pydantic.dataclasses import dataclass

from .decision import InferenceDecision
from .hypothesis_model import ConventionHypothesis
from .immutable import ReadOnlyMapping, freeze_mapping, freeze_nested_mapping
from .model_common import (
    AttitudeRepresentation,
    ConfidenceLabel,
    DimensionStatus,
    FrozenModel,
    ResultLabel,
    WrapUnitStatus,
)

__all__ = [
    "CandidateResult",
    "DimensionAssessment",
    "HeuristicFeatures",
    "HypothesisSpaceSummary",
    "InferenceReport",
    "LagMetric",
    "OverallDiagnostics",
    "ScaleRisk",
]


@dataclass(frozen=True, slots=True)
class LagMetric(FrozenModel):
    window_size: int
    window_count: int
    residual_rmse_rad: float
    residual_median_rad: float
    residual_p95_rad: float
    observed_rotation_rms_rad: float
####


@dataclass(frozen=True, slots=True)
class ScaleRisk(FrozenModel):
    """Raw geometric losses for one overlapping window scale."""

    window_size: int
    trimmed_mse_rad2: float
    untrimmed_mse_rad2: float
    weight: float
####


@dataclass(frozen=True, slots=True)
class HeuristicFeatures(FrozenModel):
    """Representation-only features; never interpret these as probabilities.

    Trajectory risk and domain context are intentionally excluded from this vector.
    A calibrated classifier may later consume the representation feature vector, but
    this package does not label heuristic scores as probabilities.
    """

    schema_penalty: float = 0.0
    wrap_score: float = 0.0
    domain_score: float = 0.0
    branch_penalty: float = 0.0
    singularity_penalty: float = 0.0
    physical_penalty: float = 0.0
    coarse_risk: float = 0.0
    wrap_gain_raw: float = 0.0
    inferred_wrap_count: int = 0
    wrap_unit_status: WrapUnitStatus = WrapUnitStatus.NOT_APPLICABLE
    representation_score: float = 0.0
    representation_feature_vector: ReadOnlyMapping[str, float] = field(
        default_factory=dict[str, float]
    )
    classifier_probability: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "representation_feature_vector",
            freeze_mapping(self.representation_feature_vector),
        )
    ####
####


@dataclass(frozen=True, slots=True)
class CandidateResult(FrozenModel):
    hypothesis: ConventionHypothesis
    criterion: float
    data_criterion: float
    fit_score: float
    normalized_rmse: float
    multiscale_rmse_rad: float
    raw_geometric_risk: float
    untrimmed_geometric_risk: float
    scale_risks: tuple[ScaleRisk, ...]
    interval_rmse_rad: float
    interval_median_rad: float
    interval_p95_rad: float
    interval_max_rad: float
    observed_step_rms_rad: float
    valid_interval_count: int
    near_alias_fraction: float
    hard_alias_fraction: float
    rate_change_rms_angle_rad: float
    lag_metrics: tuple[LagMetric, ...]
    rank: int = 0
    data_rank: int = 0
    relative_support: float = 0.0
    data_relative_support: float = 0.0
    delta_criterion: float = 0.0
    data_delta_criterion: float = 0.0
    prior_relative_weight: float = 1.0
    prior_penalty: float = 0.0
    inferred_sample_period_s: float | None = None
    gimbal_lock_fraction: float | None = None
    equivalence_class_id: str = ""
    exact_equivalent_hypothesis_ids: tuple[str, ...] = ()
    euler_range_penalty_rad2: float = 0.0
    euler_branch_switch_count: int = 0
    euler_singularity_margin: float | None = None
    euler_branch_cost_rad2: float = 0.0
    heuristic_features: HeuristicFeatures = HeuristicFeatures()
    warnings: tuple[str, ...] = ()
    class_data_risk: float = 0.0
    empirical_gap_to_next_class: float | None = None
    partial_lower_bound: float = 0.0
    exact_pruned: bool = False
####


@dataclass(frozen=True, slots=True)
class DimensionAssessment(FrozenModel):
    status: DimensionStatus
    selected_value: str | None
    allowed_values: tuple[str, ...]
    evidence_near_tie_values: tuple[str, ...]
    posterior_near_tie_values: tuple[str, ...]
    relative_prior_weights: ReadOnlyMapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.relative_prior_weights is not None:
            object.__setattr__(
                self, "relative_prior_weights", freeze_mapping(self.relative_prior_weights)
            )
        ####
    ####
####


@dataclass(frozen=True, slots=True)
class HypothesisSpaceSummary(FrozenModel):
    base_hypothesis_count: int
    constrained_hypothesis_count: int
    reduction_fraction: float
    fixed_dimensions: ReadOnlyMapping[str, str]
    searched_dimensions: ReadOnlyMapping[str, tuple[str, ...]]
    not_applicable_dimensions: tuple[str, ...]
    relative_prior_weights: ReadOnlyMapping[str, ReadOnlyMapping[str, float]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fixed_dimensions", freeze_mapping(self.fixed_dimensions))
        object.__setattr__(self, "searched_dimensions", freeze_mapping(self.searched_dimensions))
        object.__setattr__(
            self, "relative_prior_weights", freeze_nested_mapping(self.relative_prior_weights)
        )
    ####
####


@dataclass(frozen=True, slots=True)
class OverallDiagnostics(FrozenModel):
    confidence: float
    confidence_label: ConfidenceLabel
    best_relative_support: float
    best_data_relative_support: float
    near_tie_count: int
    near_tie_hypothesis_ids: tuple[str, ...]
    unresolved_dimensions: ReadOnlyMapping[str, tuple[str, ...]]
    posterior_near_tie_count: int
    posterior_near_tie_hypothesis_ids: tuple[str, ...]
    posterior_unresolved_dimensions: ReadOnlyMapping[str, tuple[str, ...]]
    dimension_assessments: ReadOnlyMapping[str, DimensionAssessment]
    effective_excitation_rank: float
    excitation_singular_values: tuple[float, float, float]
    total_observed_motion_rad: float
    observed_step_rms_rad: float
    near_alias_fraction: float
    hard_alias_fraction: float
    time_source: Literal["timestamps", "provided_sample_period", "inferred_sample_period"]
    warnings: tuple[str, ...]
    noise_model: str = "isotropic_rotation_vector_gaussian"
    estimated_noise_sigma_rad: float = 0.0
    noise_residual_count: int = 0
    noise_lag1_autocorrelation: float = 0.0
    evidence_sufficient: bool = False
    evidence_intervals_used: int = 0
    evidence_fraction: float = 0.0
    evidence_log_likelihood_ratio: float = 0.0
    exact_equivalence_class_count: int = 0
    best_equivalence_class_hypothesis_ids: tuple[str, ...] = ()
    rate_scale_period_gauge_active: bool = False
    raw_excitation_rank: int = 0
    raw_excitation_eigenvalues: tuple[float, float, float] = (0.0, 0.0, 0.0)
    raw_excitation_condition_ratio: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "unresolved_dimensions", freeze_mapping(self.unresolved_dimensions)
        )
        object.__setattr__(
            self,
            "posterior_unresolved_dimensions",
            freeze_mapping(self.posterior_unresolved_dimensions),
        )
        object.__setattr__(
            self, "dimension_assessments", freeze_mapping(self.dimension_assessments)
        )
    ####
####


@dataclass(frozen=True, slots=True)
class InferenceReport(FrozenModel):
    representation: AttitudeRepresentation
    sample_count: int
    hypothesis_count: int
    hypothesis_space: HypothesisSpaceSummary
    best: CandidateResult
    rankings: tuple[CandidateResult, ...]
    diagnostics: OverallDiagnostics
    result_label: ResultLabel = ResultLabel.TRAJECTORY_CLASS
    evidence_contender_class_ids: tuple[str, ...] = ()
    data_winning_class_ids: tuple[str, ...] = ()
    empirical_gap_to_next_class: float | None = None
    operational_class_id: str = ""
    operational_hypothesis_id: str = ""
    operational_reason: str = "trajectory"
    domain_prior_configured: bool = False
    domain_prior_abstained: bool = False
    domain_prior_supported_class_ids: tuple[str, ...] = ()
    preference_profile_id: str = ""
    preference_profile_source: str = ""
    preference_profile_configured: bool = False
    preference_profile_abstained: bool = False
    preference_profile_supported_class_ids: tuple[str, ...] = ()
    representative_selection_reason: str = "canonical-policy"

    @property
    def decision(self) -> InferenceDecision:
        """Return a data-evidence recommendation without changing the report schema."""

        return InferenceDecision.from_evidence(
            evidence_sufficient=self.diagnostics.evidence_sufficient,
            evidence_contender_class_ids=self.evidence_contender_class_ids,
        )
    ####
####
