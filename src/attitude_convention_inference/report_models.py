"""Typed candidate, diagnostic, and inference-report contracts."""

from __future__ import annotations

from dataclasses import field
from math import isclose, isfinite
from typing import Annotated, Literal, TypeAlias, TypedDict

from pydantic import Field
from pydantic.dataclasses import dataclass

from .decision import InferenceDecision
from .hypothesis_model import ConventionHypothesis
from .immutable import ReadOnlyMapping, freeze_mapping, freeze_nested_mapping
from .model_common import (
    AttitudeRepresentation,
    ConfidenceLabel,
    DimensionStatus,
    FrozenModel,
    RateUnit,
    ResultLabel,
    WrapUnitStatus,
    confidence_label_for_score,
)

__all__ = [
    "CandidateResult",
    "DimensionAssessment",
    "HeuristicFeatures",
    "HypothesisSpaceSummary",
    "InferenceReport",
    "IntervalDiagnosticRow",
    "LagMetric",
    "OverallDiagnostics",
    "RateInputChannelRailHits",
    "RateInputRailLimits",
    "RatePolarityProfileFit",
    "RatePolaritySensitivityReport",
    "RateRailInspectionReport",
    "ScaleRisk",
]

RatePolarityByInputAxis: TypeAlias = tuple[Literal[-1, 1], Literal[-1, 1], Literal[-1, 1]]


class _IntervalDiagnosticRequiredFields(TypedDict):
    interval: int
    dt_s: float
    omega_x_rad_s: float
    omega_y_rad_s: float
    omega_z_rad_s: float
    observed_delta_qx: float
    observed_delta_qy: float
    observed_delta_qz: float
    observed_delta_qw: float
    predicted_delta_qx: float
    predicted_delta_qy: float
    predicted_delta_qz: float
    predicted_delta_qw: float
    error_rotvec_x_rad: float
    error_rotvec_y_rad: float
    error_rotvec_z_rad: float
    error_angle_rad: float
    unwrapped_rate_angle_rad: float
    near_alias: bool
    hard_alias: bool
####


class IntervalDiagnosticRow(_IntervalDiagnosticRequiredFields, total=False):
    """One adjacent-sample kinematic comparison, with angles in radians.

    Timestamp fields are present only when the input trajectory includes times.
    """

    time_start_s: float
    time_end_s: float
####


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
class RatePolarityProfileFit(FrozenModel):
    """Best convention fit after a hypothetical sign correction to raw rate columns."""

    polarity_by_input_axis: Annotated[
        RatePolarityByInputAxis,
        Field(description="Hypothetical sign multiplier for the three raw rate input columns."),
    ]
    best_class_data_risk: Annotated[
        float,
        Field(
            ge=0.0,
            allow_inf_nan=False,
            description="Minimum class-level geometric risk after this hypothetical sign change.",
        ),
    ]
    relative_log_risk_delta_from_best: Annotated[
        float,
        Field(
            allow_inf_nan=False,
            description=(
                "Log of the floor-adjusted class data risk relative to the best sign profile; "
                "not a likelihood ratio."
            ),
        ),
    ]
    relative_log_risk_delta_vs_identity: Annotated[
        float,
        Field(
            allow_inf_nan=False,
            description=(
                "Log of the floor-adjusted class data risk relative to the identity sign profile; "
                "negative values fit better."
            ),
        ),
    ]
    within_best_fit_margin: Annotated[
        bool,
        Field(description="Whether this profile is within the configured margin of the best fit."),
    ]
    normalized_rmse: Annotated[
        float,
        Field(ge=0.0, allow_inf_nan=False, description="Normalized RMSE of the best data fit."),
    ]
    data_winning_class_ids: Annotated[
        tuple[str, ...],
        Field(description="Convention classes with minimum geometric risk for this profile."),
    ]
    evidence_contender_class_ids: Annotated[
        tuple[str, ...],
        Field(description="Convention classes retained by the configured relative-fit screen."),
    ]
    decision: Annotated[
        InferenceDecision,
        Field(description="Decision for this hypothetical profile, not a sensor correction."),
    ]
    warnings: Annotated[
        tuple[str, ...],
        Field(description="Inference warnings for the hypothetical profile."),
    ]
####


@dataclass(frozen=True, slots=True)
class RatePolaritySensitivityReport(FrozenModel):
    """Opt-in comparison of all per-input-axis rate sign patterns.

    This is a sensitivity analysis, not a sensor calibration estimate. Several
    sign patterns can fit equally well because rate polarity can trade off with
    the convention hypotheses. The assessment never modifies input data or
    recommends applying a correction.
    """

    sample_count: Annotated[int, Field(gt=0, description="Number of trajectory samples assessed.")]
    near_tie_delta_criterion: Annotated[
        float,
        Field(
            ge=0.0,
            allow_inf_nan=False,
            description="Configured log-risk margin used to mark profiles near the best fit.",
        ),
    ]
    identity_profile_data_risk: Annotated[
        float,
        Field(ge=0.0, allow_inf_nan=False, description="Best class data risk with no sign change."),
    ]
    best_profile_data_risk: Annotated[
        float,
        Field(
            ge=0.0,
            allow_inf_nan=False,
            description="Lowest class data risk across all eight profiles.",
        ),
    ]
    profiles: Annotated[
        tuple[RatePolarityProfileFit, ...],
        Field(description="One fit assessment for every per-axis sign pattern."),
    ]

    def __post_init__(self) -> None:
        expected_profiles = {
            tuple(signs)
            for signs in (
                (-1, -1, -1),
                (-1, -1, 1),
                (-1, 1, -1),
                (-1, 1, 1),
                (1, -1, -1),
                (1, -1, 1),
                (1, 1, -1),
                (1, 1, 1),
            )
        }
        observed_profiles = {profile.polarity_by_input_axis for profile in self.profiles}
        if len(self.profiles) != len(expected_profiles) or observed_profiles != expected_profiles:
            raise ValueError(
                "a rate-polarity sensitivity report must contain all eight sign patterns"
            )
        ####
        identity_profile = next(
            profile for profile in self.profiles if profile.polarity_by_input_axis == (1, 1, 1)
        )
        if identity_profile.best_class_data_risk != self.identity_profile_data_risk:
            raise ValueError("identity_profile_data_risk must match the identity profile fit")
        ####
        if self.best_profile_data_risk != min(
            profile.best_class_data_risk for profile in self.profiles
        ):
            raise ValueError("best_profile_data_risk must match the best profile fit")
        ####
    ####
####


@dataclass(frozen=True, slots=True)
class RateInputRailLimits(FrozenModel):
    """Caller-supplied lower/upper rail values for raw rate input columns.

    Rail values and tolerance use ``input_rate_unit``; they are not converted
    to canonical radians per second by this inspection contract.
    """

    input_rate_unit: Annotated[
        RateUnit,
        Field(description="Unit of the input samples and supplied lower/upper rail values."),
    ]
    lower_by_input_axis: Annotated[
        tuple[float, float, float],
        Field(description="Lower rail values for raw input columns 0, 1, and 2."),
    ]
    upper_by_input_axis: Annotated[
        tuple[float, float, float],
        Field(description="Upper rail values for raw input columns 0, 1, and 2."),
    ]
    absolute_tolerance: Annotated[
        float,
        Field(
            ge=0.0,
            allow_inf_nan=False,
            description="Absolute match tolerance in the declared input rate unit.",
        ),
    ] = 0.0

    def __post_init__(self) -> None:
        values = (*self.lower_by_input_axis, *self.upper_by_input_axis)
        if not all(isfinite(value) for value in values):
            raise ValueError("rate input rail values must be finite")
        ####
        for axis_index, (lower, upper) in enumerate(
            zip(self.lower_by_input_axis, self.upper_by_input_axis, strict=True)
        ):
            if lower >= upper:
                raise ValueError(f"input axis {axis_index} lower rail must be below its upper rail")
            ####
            if 2.0 * self.absolute_tolerance >= upper - lower:
                raise ValueError(f"input axis {axis_index} rail tolerance regions must not overlap")
            ####
        ####
    ####
####


@dataclass(frozen=True, slots=True)
class RateInputChannelRailHits(FrozenModel):
    """Observed sample counts at the declared rails for one raw rate column."""

    input_axis_index: Annotated[
        Literal[0, 1, 2], Field(description="Zero-based raw rate input column index.")
    ]
    lower_rail_value: Annotated[
        float, Field(allow_inf_nan=False, description="Caller-declared lower rail in input units.")
    ]
    upper_rail_value: Annotated[
        float, Field(allow_inf_nan=False, description="Caller-declared upper rail in input units.")
    ]
    lower_rail_hit_count: Annotated[
        int, Field(ge=0, description="Samples within tolerance of the declared lower rail.")
    ]
    upper_rail_hit_count: Annotated[
        int, Field(ge=0, description="Samples within tolerance of the declared upper rail.")
    ]
    lower_rail_hit_fraction: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            allow_inf_nan=False,
            description="Lower-hit count / sample count.",
        ),
    ]
    upper_rail_hit_fraction: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            allow_inf_nan=False,
            description="Upper-hit count / sample count.",
        ),
    ]
    longest_lower_rail_run: Annotated[
        int, Field(ge=0, description="Longest consecutive run at the lower rail.")
    ]
    longest_upper_rail_run: Annotated[
        int, Field(ge=0, description="Longest consecutive run at the upper rail.")
    ]
####


@dataclass(frozen=True, slots=True)
class RateRailInspectionReport(FrozenModel):
    """Measured hits at caller-declared rate rails; not a clipping inference."""

    sample_count: Annotated[int, Field(gt=0, description="Number of rate samples inspected.")]
    rail_limits: Annotated[
        RateInputRailLimits,
        Field(description="Caller-supplied input rails and the unit/tolerance used for matching."),
    ]
    channels: Annotated[
        tuple[RateInputChannelRailHits, ...],
        Field(description="Rail-hit summaries in raw input-column order."),
    ]

    def __post_init__(self) -> None:
        if tuple(channel.input_axis_index for channel in self.channels) != (0, 1, 2):
            raise ValueError("rate rail report must contain input axes 0, 1, and 2 in order")
        ####
        for channel in self.channels:
            if channel.lower_rail_hit_count + channel.upper_rail_hit_count > self.sample_count:
                raise ValueError("lower and upper rail hit counts exceed the sample count")
            ####
            if (
                channel.longest_lower_rail_run > channel.lower_rail_hit_count
                or channel.longest_upper_rail_run > channel.upper_rail_hit_count
            ):
                raise ValueError("longest rail-hit run exceeds its rail-hit count")
            ####
            expected_lower_fraction = channel.lower_rail_hit_count / self.sample_count
            expected_upper_fraction = channel.upper_rail_hit_count / self.sample_count
            if not isclose(
                channel.lower_rail_hit_fraction,
                expected_lower_fraction,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ) or not isclose(
                channel.upper_rail_hit_fraction,
                expected_upper_fraction,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError("rail-hit fractions must match their counts and sample count")
            ####
        ####
        expected_lower_rails = self.rail_limits.lower_by_input_axis
        expected_upper_rails = self.rail_limits.upper_by_input_axis
        if any(
            channel.lower_rail_value != expected_lower_rails[channel.input_axis_index]
            or channel.upper_rail_value != expected_upper_rails[channel.input_axis_index]
            for channel in self.channels
        ):
            raise ValueError("channel rail values must match the supplied rail limits")
        ####
    ####
####


@dataclass(frozen=True, slots=True)
class OverallDiagnostics(FrozenModel):
    confidence: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            allow_inf_nan=False,
            description=(
                "Uncalibrated composite heuristic score in [0, 1]; "
                "not a probability of correctness."
            ),
        ),
    ]
    confidence_label: Annotated[
        ConfidenceLabel,
        Field(
            description=("Deterministic band of the heuristic score, not a calibrated probability.")
        ),
    ]
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
    # Estimated from interval residual vectors, not raw sensor samples.
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
        if self.confidence_label is not confidence_label_for_score(self.confidence):
            raise ValueError("confidence_label must match the heuristic confidence score band")
        ####
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
    observation_model: Annotated[
        Literal["aligned_samples", "sparse_orientation_anchors"],
        Field(description="Sampling relationship between the orientation and rate observations."),
    ] = "aligned_samples"
    rate_sample_count: Annotated[
        int | None,
        Field(
            ge=2,
            description=(
                "Number of dense rate samples for sparse-anchor input; null for aligned input."
            ),
        ),
    ] = None

    def __post_init__(self) -> None:
        if self.observation_model == "aligned_samples" and self.rate_sample_count is not None:
            raise ValueError("rate_sample_count is only set for sparse-orientation reports")
        ####
        if self.observation_model == "sparse_orientation_anchors" and (
            self.rate_sample_count is None or self.rate_sample_count < 2
        ):
            raise ValueError("sparse-orientation reports require at least two rate samples")
        ####
    ####

    @property
    def decision(self) -> InferenceDecision:
        """Return a data-evidence recommendation without changing the report schema."""

        return InferenceDecision.from_evidence(
            evidence_sufficient=self.diagnostics.evidence_sufficient,
            evidence_contender_class_ids=self.evidence_contender_class_ids,
        )
    ####
####
