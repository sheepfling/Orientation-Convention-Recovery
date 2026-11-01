"""Measure convention recovery under seeded, out-of-model rate-sensor defects."""

from __future__ import annotations

import argparse
from enum import StrEnum
from math import exp, expm1, isfinite, sqrt
from typing import Literal
from unittest.mock import patch

from numpy import arange, array, bool_, clip, deg2rad, float64, zeros
from numpy.random import Generator, SeedSequence, default_rng
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

import attitude_convention_inference.evidence as _evidence
from attitude_convention_inference import (
    AttitudeDirection,
    AttitudeRepresentation,
    ConventionHypothesis,
    InferenceConfig,
    InferenceDecisionReason,
    InferenceDecisionStatus,
    QuaternionOrder,
    RateFrame,
    RateUnit,
    SearchSpace,
    TrajectoryData,
    infer_conventions,
)
from attitude_convention_inference.math_utils import (
    rotation_apply,
    rotation_from_quat,
    rotation_from_rotvec,
    rotation_product,
    rotation_quaternions,
)
from attitude_convention_inference.version import __version__
from scripts.generate_validation_corpus import (
    DEFAULT_NOISE_SEED_START,
    NOISE_PROFILES,
    SAMPLED_RATE_INTEGRATION_STEP_S,
    SAMPLED_RATE_PROFILES,
    MotionProfile,
    SyntheticMotion,
    maneuver_burst_rate_motion,
    piecewise_linear_rate_motion,
    sampled_rate_motion,
)

__all__ = [
    "DEFAULT_ACQUISITION_PERTURBATION",
    "DEFAULT_MIXED_TIMING_CALIBRATION_CONDITIONS",
    "DEFAULT_NOISE_SEEDS",
    "DEFAULT_PERTURBATION_SCREEN_NOISE_SEEDS",
    "DEFAULT_SENSOR_MISMATCH_CONDITIONS",
    "HIGH_NOISE_OU_JITTER_DROPOUT",
    "LOW_NOISE_OU_JITTER_DROPOUT",
    "LOW_NOISE_SCREEN_SEEDS",
    "MODERATE_NOISE_OU_JITTER_DROPOUT",
    "AcquisitionPerturbation",
    "AcquisitionProfile",
    "SensorMismatchCase",
    "SensorMismatchCondition",
    "SensorMismatchConditionSummary",
    "SensorMismatchSweep",
    "run_sensor_mismatch_sweep",
]

FloatArray = NDArray[float64]
BoolArray = NDArray[bool_]
DEFAULT_SAMPLE_COUNT = 151
DEFAULT_SAMPLE_PERIOD_S = 0.02
DEFAULT_MOTION_SEEDS = (701, 702, 703)
DEFAULT_MOTION_PROFILE = MotionProfile.SAMPLED_RATE_NONANALYTIC
DEFAULT_NOISE_SEEDS = (DEFAULT_NOISE_SEED_START,)
LOW_NOISE_SCREEN_SEEDS = (20314101, 20314102, 20314103)
DEFAULT_PERTURBATION_SCREEN_NOISE_SEEDS = LOW_NOISE_SCREEN_SEEDS
_MISALIGNMENT_AXIS = (0.3, -0.5, 0.8)
_RATE_ORDERS = ("pqr", "prq", "qpr", "qrp", "rpq", "rqp")


class AcquisitionProfile(StrEnum):
    """Named measurement-noise and acquisition-artifact bundles for validation."""

    CLEAN = "clean"
    LOW_OU_JITTER_DROPOUT = "low-ou-jitter-dropout"
    MODERATE_OU_JITTER_DROPOUT = "moderate-ou-jitter-dropout"
    HIGH_OU_JITTER_DROPOUT = "high-ou-jitter-dropout"
####


class AcquisitionPerturbation(BaseModel):
    """Explicit sensor-noise and acquisition settings shared by one sweep."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    orientation_noise_sigma_rad: float = 0.0
    rate_noise_sigma_rad_s: float = 0.0
    orientation_noise_correlation_time_s: float | None = None
    rate_noise_correlation_time_s: float | None = None
    timestamp_jitter_amplitude_s: float = 0.0
    dropout_probability: float = 0.0

    @model_validator(mode="after")
    def _validate_acquisition_parameters(self) -> AcquisitionPerturbation:
        nonnegative_values = (
            self.orientation_noise_sigma_rad,
            self.rate_noise_sigma_rad_s,
            self.timestamp_jitter_amplitude_s,
        )
        if any(not isfinite(value) or value < 0.0 for value in nonnegative_values):
            raise ValueError("noise scales and jitter amplitude must be finite and non-negative")
        ####
        if not isfinite(self.dropout_probability) or not 0.0 <= self.dropout_probability < 1.0:
            raise ValueError("dropout_probability must be finite and in [0, 1)")
        ####
        for name, correlation_time_s in (
            ("orientation", self.orientation_noise_correlation_time_s),
            ("rate", self.rate_noise_correlation_time_s),
        ):
            if correlation_time_s is not None and (
                not isfinite(correlation_time_s) or correlation_time_s <= 0.0
            ):
                raise ValueError(f"{name} noise correlation time must be positive and finite")
            ####
        ####
        return self
    ####
####


DEFAULT_ACQUISITION_PERTURBATION = AcquisitionPerturbation()


def _acquisition_for_noise_tier(
    noise_tier: Literal["low", "moderate", "high"],
) -> AcquisitionPerturbation:
    noise_profile = next(
        (profile for profile in NOISE_PROFILES if profile.name == noise_tier),
        None,
    )
    if noise_profile is None:
        raise RuntimeError(f"Missing {noise_tier!r} noise profile")
    ####
    return AcquisitionPerturbation(
        orientation_noise_sigma_rad=noise_profile.orientation_sigma_rad,
        rate_noise_sigma_rad_s=noise_profile.rate_sigma_rad_s,
        orientation_noise_correlation_time_s=0.05,
        rate_noise_correlation_time_s=0.1,
        timestamp_jitter_amplitude_s=0.002,
        dropout_probability=0.1,
    )
####


LOW_NOISE_OU_JITTER_DROPOUT = _acquisition_for_noise_tier("low")
MODERATE_NOISE_OU_JITTER_DROPOUT = _acquisition_for_noise_tier("moderate")
HIGH_NOISE_OU_JITTER_DROPOUT = _acquisition_for_noise_tier("high")


class SensorMismatchCondition(BaseModel):
    """One fixed corruption applied to rates while ground-truth attitudes stay unchanged."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition_id: str = Field(min_length=1)
    misalignment_angle_deg: float = 0.0
    misalignment_axis: tuple[float, float, float] = _MISALIGNMENT_AXIS
    orientation_latency_s: float = 0.0
    rate_latency_s: float = 0.0
    bias_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale_by_axis: tuple[float, float, float] = (1.0, 1.0, 1.0)
    polarity_by_axis: tuple[int, int, int] = (1, 1, 1)
    clip_abs_rad_s: float | None = None

    @model_validator(mode="after")
    def _validate_physical_parameters(self) -> SensorMismatchCondition:
        values = (
            self.misalignment_angle_deg,
            *self.misalignment_axis,
            self.orientation_latency_s,
            self.rate_latency_s,
            *self.bias_rad_s,
            *self.scale_by_axis,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("sensor mismatch values must be finite")
        ####
        axis_norm = sqrt(sum(component * component for component in self.misalignment_axis))
        if not isfinite(axis_norm) or axis_norm <= 0.0:
            raise ValueError("misalignment_axis must have a finite, non-zero norm")
        ####
        if self.orientation_latency_s < 0.0 or self.rate_latency_s < 0.0:
            raise ValueError("channel latencies must be non-negative")
        ####
        if any(scale <= 0.0 for scale in self.scale_by_axis):
            raise ValueError("scale_by_axis values must be positive")
        ####
        if any(polarity not in (-1, 1) for polarity in self.polarity_by_axis):
            raise ValueError("polarity_by_axis values must be -1 or +1")
        ####
        if self.clip_abs_rad_s is not None and (
            not isfinite(self.clip_abs_rad_s) or self.clip_abs_rad_s <= 0.0
        ):
            raise ValueError("clip_abs_rad_s must be positive and finite when provided")
        ####
        return self
    ####

    def apply_to_rates(
        self,
        true_body_rates_rad_s: FloatArray,
        *,
        measurement_noise_rad_s: FloatArray | None = None,
    ) -> FloatArray:
        """Apply calibration, optional sensor-axis noise, and saturation."""

        axis = array(self.misalignment_axis, dtype=float64)
        axis /= sqrt(float((axis * axis).sum()))
        rotation = rotation_from_rotvec(deg2rad(self.misalignment_angle_deg) * axis)
        measured = rotation_apply(rotation, array(true_body_rates_rad_s, dtype=float64, copy=True))
        measured += array(self.bias_rad_s, dtype=float64)
        measured *= array(self.scale_by_axis, dtype=float64)
        measured *= array(self.polarity_by_axis, dtype=float64)
        if measurement_noise_rad_s is not None:
            noise = array(measurement_noise_rad_s, dtype=float64)
            if noise.shape != measured.shape:
                raise ValueError("measurement_noise_rad_s must match the rate array shape")
            ####
            measured += noise
        ####
        if self.clip_abs_rad_s is not None:
            measured = clip(measured, -self.clip_abs_rad_s, self.clip_abs_rad_s)
        ####
        return array(measured, dtype=float64)
    ####
####


def _default_conditions() -> tuple[SensorMismatchCondition, ...]:
    conditions: list[SensorMismatchCondition] = [SensorMismatchCondition(condition_id="clean")]
    conditions.extend(
        SensorMismatchCondition(
            condition_id=f"axis-rotation-{angle_deg:03d}deg",
            misalignment_angle_deg=angle_deg,
        )
        for angle_deg in (5, 15, 30, 45, 60, 90, 120, 150)
    )
    conditions.extend(
        SensorMismatchCondition(
            condition_id=f"bias-{round(bias_rad_s * 1000):03d}mrad-s",
            bias_rad_s=(bias_rad_s, -0.7 * bias_rad_s, 0.4 * bias_rad_s),
        )
        for bias_rad_s in (0.05, 0.15, 0.3, 0.6)
    )
    conditions.extend(
        SensorMismatchCondition(
            condition_id=f"axis-scale-{index:02d}",
            scale_by_axis=scale,
        )
        for index, scale in enumerate(
            ((1.3, 0.8, 1.1), (2.0, 0.5, 1.0), (0.2, 1.5, 0.6)),
            start=1,
        )
    )
    conditions.extend(
        SensorMismatchCondition(
            condition_id=f"axis-polarity-{index:02d}",
            polarity_by_axis=polarity,
        )
        for index, polarity in enumerate(
            ((-1, 1, 1), (1, -1, 1), (1, 1, -1), (-1, -1, 1)),
            start=1,
        )
    )
    conditions.extend(
        SensorMismatchCondition(
            condition_id=f"clip-{round(limit_rad_s * 100):03d}cents-rad-s",
            clip_abs_rad_s=limit_rad_s,
        )
        for limit_rad_s in (0.5, 0.25)
    )
    return tuple(conditions)
####


DEFAULT_SENSOR_MISMATCH_CONDITIONS = _default_conditions()

DEFAULT_MIXED_TIMING_CALIBRATION_CONDITIONS = (
    SensorMismatchCondition(
        condition_id="mixed-misalignment-bias-channel-latency",
        misalignment_angle_deg=30.0,
        orientation_latency_s=0.013,
        rate_latency_s=0.027,
        bias_rad_s=(0.05, -0.035, 0.02),
    ),
    SensorMismatchCondition(
        condition_id="mixed-misalignment-scale-rate-latency",
        misalignment_angle_deg=60.0,
        rate_latency_s=0.027,
        scale_by_axis=(1.3, 0.8, 1.1),
    ),
    SensorMismatchCondition(
        condition_id="mixed-axis-polarity-orientation-latency",
        orientation_latency_s=0.013,
        polarity_by_axis=(-1, 1, -1),
        clip_abs_rad_s=0.5,
    ),
)


class SensorMismatchCase(BaseModel):
    """Inference result and truth comparison for one motion/defect pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    motion_seed: int
    noise_seed: int
    retained_sample_count: int
    condition: SensorMismatchCondition
    truth_class_id: str
    data_winning_class_ids: tuple[str, ...]
    evidence_contender_class_ids: tuple[str, ...]
    recommended_class_id: str | None
    recommendation_correct: bool | None
    truth_supported_by_evidence: bool
    dependence_guard_triggered: bool
    evidence_sufficient: bool
    decision_status: InferenceDecisionStatus
    decision_reason: InferenceDecisionReason
    normalized_rmse: float
    residual_sigma_rad: float
    residual_lag1_correlation: float
    heuristic_confidence: float
####


class SensorMismatchConditionSummary(BaseModel):
    """Aggregated recovery outcomes for one defect across motion and noise seeds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition_id: str
    trial_count: int
    recommendation_count: int
    correct_recommendation_count: int
    wrong_recommendation_count: int
    abstention_count: int
    truth_supported_count: int
    dependence_guard_count: int
####


class SensorMismatchSweep(BaseModel):
    """Reproducible out-of-model sensor stress summary with per-case outcomes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    generator: str
    package_version: str
    motion_profile: MotionProfile
    acquisition_profile: AcquisitionProfile
    acquisition: AcquisitionPerturbation
    sample_count: int
    sample_period_s: float
    integration_step_s: float
    persistence_guard_correlation_limit: float
    persistence_guard_residual_floor_rad: float
    motion_seeds: tuple[int, ...]
    noise_seeds: tuple[int, ...]
    case_count: int
    recommendation_count: int
    correct_recommendation_count: int
    wrong_recommendation_count: int
    abstention_count: int
    truth_supported_count: int
    dependence_guard_count: int
    conditions: tuple[SensorMismatchConditionSummary, ...]
    cases: tuple[SensorMismatchCase, ...]
####


def _inference_config() -> InferenceConfig:
    return InferenceConfig(
        search_space=SearchSpace(
            attitude_directions=(AttitudeDirection.BODY_TO_WORLD,),
            quaternion_orders=(QuaternionOrder.SCALAR_LAST,),
            rate_frames=tuple(RateFrame),
            rate_orders=_RATE_ORDERS,
            rate_units=(RateUnit.RADIANS_PER_SECOND,),
            rate_signs=(-1, 1),
        ),
        window_sizes=(1, 2, 4, 8),
    )
####


def _sample_motion(
    motion_profile: MotionProfile,
    sample_times_s: FloatArray,
    motion_seed: int,
) -> SyntheticMotion:
    if motion_profile is MotionProfile.SAMPLED_RATE_NONANALYTIC:
        return sampled_rate_motion(
            sample_times_s,
            seed=motion_seed,
            reference_time_s=0.0,
            integration_step_s=SAMPLED_RATE_INTEGRATION_STEP_S,
        )
    ####
    if motion_profile is MotionProfile.SAMPLED_RATE_PIECEWISE_LINEAR:
        return piecewise_linear_rate_motion(
            sample_times_s,
            seed=motion_seed,
            reference_time_s=0.0,
            integration_step_s=SAMPLED_RATE_INTEGRATION_STEP_S,
        )
    ####
    if motion_profile is MotionProfile.SAMPLED_RATE_MANEUVER_BURSTS:
        return maneuver_burst_rate_motion(
            sample_times_s,
            seed=motion_seed,
            reference_time_s=0.0,
            integration_step_s=SAMPLED_RATE_INTEGRATION_STEP_S,
        )
    ####
    raise ValueError(f"Unsupported sampled-rate motion profile: {motion_profile.value}")
####


def _sample_vector_noise(
    random: Generator,
    sample_times_s: FloatArray,
    sigma: float,
    correlation_time_s: float | None,
) -> FloatArray:
    sample_count = sample_times_s.size
    if correlation_time_s is None:
        return array(random.normal(0.0, sigma, (sample_count, 3)), dtype=float64)
    ####
    noise = zeros((sample_count, 3), dtype=float64)
    noise[0] = random.normal(0.0, sigma, 3)
    for sample_index in range(1, sample_count):
        delta_time_s = float(sample_times_s[sample_index] - sample_times_s[sample_index - 1])
        decay = exp(-delta_time_s / correlation_time_s)
        innovation_sigma = sigma * sqrt(-expm1(-2.0 * delta_time_s / correlation_time_s))
        noise[sample_index] = decay * noise[sample_index - 1] + random.normal(
            0.0, innovation_sigma, 3
        )
    ####
    return noise
####


def _apply_orientation_noise(
    quaternion_xyzw: FloatArray,
    rotation_vectors_rad: FloatArray,
) -> FloatArray:
    noise_rotations = rotation_from_rotvec(rotation_vectors_rad)
    true_rotations = rotation_from_quat(quaternion_xyzw)
    return rotation_quaternions(rotation_product(noise_rotations, true_rotations))
####


def _retained_sample_mask(
    sample_count: int,
    dropout_probability: float,
    random: Generator,
) -> BoolArray:
    retained = array(random.random(sample_count) >= dropout_probability, dtype=bool_)
    retained[0] = True
    retained[-1] = True
    return retained
####


def _evaluate_case(
    motion_seed: int,
    noise_seed: int,
    condition: SensorMismatchCondition,
    motion_profile: MotionProfile,
    acquisition: AcquisitionPerturbation,
    sample_times_s: FloatArray,
    config: InferenceConfig,
) -> SensorMismatchCase:
    orientation_seed, rate_seed, jitter_seed, dropout_seed = SeedSequence(noise_seed).spawn(4)
    orientation_random = default_rng(orientation_seed)
    rate_random = default_rng(rate_seed)
    jitter_random = default_rng(jitter_seed)
    dropout_random = default_rng(dropout_seed)
    if acquisition.timestamp_jitter_amplitude_s == 0.0:
        timestamp_jitter = zeros(sample_times_s.size, dtype=float64)
    else:
        timestamp_jitter = array(
            jitter_random.uniform(
                -acquisition.timestamp_jitter_amplitude_s,
                acquisition.timestamp_jitter_amplitude_s,
                sample_times_s.size,
            ),
            dtype=float64,
        )
        timestamp_jitter[0] = 0.0
    ####
    observed_times_s = sample_times_s + timestamp_jitter
    orientation_times_s = observed_times_s - condition.orientation_latency_s
    rate_times_s = observed_times_s - condition.rate_latency_s
    orientation_motion = _sample_motion(
        motion_profile,
        orientation_times_s,
        motion_seed,
    )
    rate_motion = (
        orientation_motion
        if condition.orientation_latency_s == condition.rate_latency_s
        else _sample_motion(motion_profile, rate_times_s, motion_seed)
    )
    if acquisition.orientation_noise_sigma_rad == 0.0:
        orientation = orientation_motion.quaternion_xyzw
    else:
        orientation_noise = _sample_vector_noise(
            orientation_random,
            orientation_times_s,
            acquisition.orientation_noise_sigma_rad,
            acquisition.orientation_noise_correlation_time_s,
        )
        orientation = _apply_orientation_noise(
            orientation_motion.quaternion_xyzw,
            orientation_noise,
        )
    ####
    rate_noise = (
        None
        if acquisition.rate_noise_sigma_rad_s == 0.0
        else _sample_vector_noise(
            rate_random,
            rate_times_s,
            acquisition.rate_noise_sigma_rad_s,
            acquisition.rate_noise_correlation_time_s,
        )
    )
    measured_rates = condition.apply_to_rates(
        rate_motion.body_rates_rad_s,
        measurement_noise_rad_s=rate_noise,
    )
    retained = _retained_sample_mask(
        sample_times_s.size,
        acquisition.dropout_probability,
        dropout_random,
    )
    retained_sample_count = int(retained.sum())
    if retained_sample_count < 21:
        raise ValueError(f"Dropout with noise seed {noise_seed} retained fewer than 21 samples")
    ####
    data = TrajectoryData(
        representation=AttitudeRepresentation.QUATERNION,
        orientation=orientation[retained],
        rates=measured_rates[retained],
        time_s=observed_times_s[retained],
    )
    report = infer_conventions(data, config)
    truth = ConventionHypothesis(
        attitude_direction=AttitudeDirection.BODY_TO_WORLD,
        rate_frame=RateFrame.BODY,
        rate_order="pqr",
        rate_unit=RateUnit.RADIANS_PER_SECOND,
        quaternion_order=QuaternionOrder.SCALAR_LAST,
        rate_sign=1,
    )
    truth_class_ids = tuple(
        candidate.equivalence_class_id
        for candidate in report.rankings
        if candidate.hypothesis == truth
    )
    if len(truth_class_ids) != 1:
        raise RuntimeError("the configured search did not yield one exact truth class")
    ####
    truth_class_id = truth_class_ids[0]
    recommended_class_id = report.decision.recommended_class_id
    recommendation_correct = (
        None if recommended_class_id is None else recommended_class_id == truth_class_id
    )
    dependence_guard_triggered = any(
        "independent-increment evidence is withheld" in warning
        for warning in report.diagnostics.warnings
    )
    return SensorMismatchCase(
        motion_seed=motion_seed,
        noise_seed=noise_seed,
        retained_sample_count=retained_sample_count,
        condition=condition,
        truth_class_id=truth_class_id,
        data_winning_class_ids=report.data_winning_class_ids,
        evidence_contender_class_ids=report.evidence_contender_class_ids,
        recommended_class_id=recommended_class_id,
        recommendation_correct=recommendation_correct,
        truth_supported_by_evidence=truth_class_id in report.evidence_contender_class_ids,
        dependence_guard_triggered=dependence_guard_triggered,
        evidence_sufficient=report.diagnostics.evidence_sufficient,
        decision_status=report.decision.status,
        decision_reason=report.decision.reason,
        normalized_rmse=report.best.normalized_rmse,
        residual_sigma_rad=report.diagnostics.estimated_noise_sigma_rad,
        residual_lag1_correlation=report.diagnostics.noise_lag1_autocorrelation,
        heuristic_confidence=report.diagnostics.confidence,
    )
####


def _condition_summary(
    condition: SensorMismatchCondition,
    cases: tuple[SensorMismatchCase, ...],
) -> SensorMismatchConditionSummary:
    recommendations = tuple(case for case in cases if case.recommendation_correct is not None)
    return SensorMismatchConditionSummary(
        condition_id=condition.condition_id,
        trial_count=len(cases),
        recommendation_count=len(recommendations),
        correct_recommendation_count=sum(case.recommendation_correct is True for case in cases),
        wrong_recommendation_count=sum(case.recommendation_correct is False for case in cases),
        abstention_count=sum(case.recommendation_correct is None for case in cases),
        truth_supported_count=sum(case.truth_supported_by_evidence for case in cases),
        dependence_guard_count=sum(case.dependence_guard_triggered for case in cases),
    )
####


def _acquisition_for_profile(profile: AcquisitionProfile) -> AcquisitionPerturbation:
    if profile is AcquisitionProfile.CLEAN:
        return DEFAULT_ACQUISITION_PERTURBATION
    ####
    if profile is AcquisitionProfile.LOW_OU_JITTER_DROPOUT:
        return LOW_NOISE_OU_JITTER_DROPOUT
    ####
    if profile is AcquisitionProfile.MODERATE_OU_JITTER_DROPOUT:
        return MODERATE_NOISE_OU_JITTER_DROPOUT
    ####
    if profile is AcquisitionProfile.HIGH_OU_JITTER_DROPOUT:
        return HIGH_NOISE_OU_JITTER_DROPOUT
    ####
    raise ValueError(f"Unsupported acquisition profile: {profile.value}")
####


def run_sensor_mismatch_sweep(
    motion_seeds: tuple[int, ...] = DEFAULT_MOTION_SEEDS,
    *,
    conditions: tuple[SensorMismatchCondition, ...] = DEFAULT_SENSOR_MISMATCH_CONDITIONS,
    motion_profile: MotionProfile = DEFAULT_MOTION_PROFILE,
    acquisition_profile: AcquisitionProfile = AcquisitionProfile.CLEAN,
    noise_seeds: tuple[int, ...] = DEFAULT_NOISE_SEEDS,
    persistence_guard_correlation_limit: float | None = None,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    sample_period_s: float = DEFAULT_SAMPLE_PERIOD_S,
) -> SensorMismatchSweep:
    """Evaluate defects over paired motion/noise seeds and an acquisition profile."""

    if not motion_seeds or len(motion_seeds) != len(set(motion_seeds)):
        raise ValueError("motion_seeds must be non-empty and unique")
    ####
    if any(seed < 0 for seed in motion_seeds):
        raise ValueError("motion_seeds must be non-negative")
    ####
    if not noise_seeds or len(noise_seeds) != len(set(noise_seeds)):
        raise ValueError("noise_seeds must be non-empty and unique")
    ####
    if any(seed < 0 for seed in noise_seeds):
        raise ValueError("noise_seeds must be non-negative")
    ####
    if not conditions or len({condition.condition_id for condition in conditions}) != len(
        conditions
    ):
        raise ValueError("conditions must be non-empty with unique condition_id values")
    ####
    if sample_count < 3:
        raise ValueError("sample_count must be at least three")
    ####
    if not isfinite(sample_period_s) or sample_period_s <= 0.0:
        raise ValueError("sample_period_s must be positive and finite")
    ####
    if motion_profile not in SAMPLED_RATE_PROFILES:
        supported = ", ".join(profile.value for profile in SAMPLED_RATE_PROFILES)
        raise ValueError(f"motion_profile must be one of: {supported}")
    ####
    acquisition = _acquisition_for_profile(acquisition_profile)
    if acquisition.timestamp_jitter_amplitude_s >= sample_period_s / 2.0:
        raise ValueError("timestamp_jitter_amplitude_s must be less than half the sample period")
    ####
    # This repo-local harness records the package's current fail-safe threshold.
    package_guard_limit = _evidence.PERSISTENT_RESIDUAL_CORRELATION_LIMIT
    resolved_guard_limit = (
        package_guard_limit
        if persistence_guard_correlation_limit is None
        else persistence_guard_correlation_limit
    )
    if not isfinite(resolved_guard_limit) or resolved_guard_limit <= 0.0:
        raise ValueError("persistence_guard_correlation_limit must be positive and finite")
    ####
    sample_times_s = array(
        arange(sample_count, dtype=float64) * sample_period_s,
        dtype=float64,
    )
    config = _inference_config()
    with patch.object(
        _evidence,
        "PERSISTENT_RESIDUAL_CORRELATION_LIMIT",
        resolved_guard_limit,
    ):
        cases = tuple(
            _evaluate_case(
                motion_seed,
                noise_seed,
                condition,
                motion_profile,
                acquisition,
                sample_times_s,
                config,
            )
            for motion_seed in motion_seeds
            for condition in conditions
            for noise_seed in noise_seeds
        )
    ####
    condition_summaries = tuple(
        _condition_summary(
            condition,
            tuple(case for case in cases if case.condition.condition_id == condition.condition_id),
        )
        for condition in conditions
    )
    recommendations = tuple(case for case in cases if case.recommendation_correct is not None)
    return SensorMismatchSweep(
        generator=f"{motion_profile.value} rates and fixed-grid quaternion integration",
        package_version=__version__,
        motion_profile=motion_profile,
        acquisition_profile=acquisition_profile,
        acquisition=acquisition,
        sample_count=sample_count,
        sample_period_s=sample_period_s,
        integration_step_s=SAMPLED_RATE_INTEGRATION_STEP_S,
        persistence_guard_correlation_limit=resolved_guard_limit,
        persistence_guard_residual_floor_rad=config.residual_floor_rad,
        motion_seeds=motion_seeds,
        noise_seeds=noise_seeds,
        case_count=len(cases),
        recommendation_count=len(recommendations),
        correct_recommendation_count=sum(case.recommendation_correct is True for case in cases),
        wrong_recommendation_count=sum(case.recommendation_correct is False for case in cases),
        abstention_count=sum(case.recommendation_correct is None for case in cases),
        truth_supported_count=sum(case.truth_supported_by_evidence for case in cases),
        dependence_guard_count=sum(case.dependence_guard_triggered for case in cases),
        conditions=condition_summaries,
        cases=cases,
    )
####


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_MOTION_SEEDS,
        help="independent rate-motion seeds (default: 701 702 703)",
    )
    parser.add_argument(
        "--motion-profile",
        choices=tuple(profile.value for profile in SAMPLED_RATE_PROFILES),
        default=DEFAULT_MOTION_PROFILE.value,
        help="independent rate-motion family",
    )
    parser.add_argument(
        "--acquisition-profile",
        choices=tuple(profile.value for profile in AcquisitionProfile),
        default=AcquisitionProfile.CLEAN.value,
        help="clean controls or low/moderate/high correlated noise with jitter and row dropout",
    )
    parser.add_argument(
        "--noise-seeds",
        type=int,
        nargs="+",
        help="paired measurement-noise seeds (defaults depend on acquisition profile)",
    )
    parser.add_argument(
        "--include-mixed-timing-calibration",
        action="store_true",
        help="include combined channel latency and sensor calibration defects",
    )
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--sample-period-s", type=float, default=DEFAULT_SAMPLE_PERIOD_S)
    parser.add_argument(
        "--persistence-guard-correlation-limit",
        type=float,
        help="temporary guard override for this validation run only (e.g. 1.1 disables it)",
    )
    arguments = parser.parse_args()
    conditions = DEFAULT_SENSOR_MISMATCH_CONDITIONS
    if arguments.include_mixed_timing_calibration:
        conditions += DEFAULT_MIXED_TIMING_CALIBRATION_CONDITIONS
    ####
    acquisition_profile = AcquisitionProfile(arguments.acquisition_profile)
    noise_seeds = (
        tuple(arguments.noise_seeds)
        if arguments.noise_seeds is not None
        else (
            DEFAULT_NOISE_SEEDS
            if acquisition_profile is AcquisitionProfile.CLEAN
            else DEFAULT_PERTURBATION_SCREEN_NOISE_SEEDS
        )
    )
    report = run_sensor_mismatch_sweep(
        tuple(arguments.motion_seeds),
        conditions=conditions,
        motion_profile=MotionProfile(arguments.motion_profile),
        acquisition_profile=acquisition_profile,
        noise_seeds=noise_seeds,
        persistence_guard_correlation_limit=arguments.persistence_guard_correlation_limit,
        sample_count=arguments.sample_count,
        sample_period_s=arguments.sample_period_s,
    )
    print(report.model_dump_json(indent=2))
####


if __name__ == "__main__":
    main()
####
