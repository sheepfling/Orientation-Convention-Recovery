"""Measure sparse-orientation/dense-rate recovery across seeded synthetic conditions."""

from __future__ import annotations

import argparse
from enum import StrEnum
from math import isclose, isfinite
from pathlib import Path
from statistics import median

from numpy import arange, array, bool_, count_nonzero, float64, int64
from numpy.random import SeedSequence, default_rng
from pydantic import BaseModel, ConfigDict, Field, model_validator

from attitude_convention_inference import (
    AttitudeDirection,
    AttitudeRepresentation,
    ConventionHypothesis,
    InferenceDecisionStatus,
    QuaternionOrder,
    RateFrame,
    RateUnit,
    SparseOrientationRateData,
    infer_sparse_orientation_conventions,
)
from attitude_convention_inference.math_utils import (
    FloatArray,
    rotation_from_quat,
    rotation_from_rotvec,
    rotation_product,
    rotation_quaternions,
)
from attitude_convention_inference.version import __version__
from scripts.generate_validation_corpus import (
    NO_NOISE_PROFILE,
    NOISE_PROFILES,
    SAMPLED_RATE_INTEGRATION_STEP_S,
    SAMPLED_RATE_PROFILES,
    MotionProfile,
    NoiseProfile,
    SyntheticMotion,
    maneuver_burst_rate_motion,
    piecewise_linear_rate_motion,
    sampled_rate_motion,
)

__all__ = [
    "DEFAULT_SPARSE_ANCHOR_CADENCES_S",
    "DEFAULT_SPARSE_MOTION_SEEDS",
    "DEFAULT_SPARSE_TIMING_CONDITIONS",
    "SparseAnchorCase",
    "SparseAnchorNoiseSpec",
    "SparseAnchorNoiseTier",
    "SparseAnchorStudyConfig",
    "SparseAnchorStudyReport",
    "SparseAnchorSummary",
    "SparseAnchorTimingCondition",
    "run_sparse_anchor_study",
    "summarize_sparse_anchor_cases",
]

DEFAULT_SPARSE_ANCHOR_CADENCES_S = (0.2, 0.4, 0.8, 1.6, 3.2)
DEFAULT_SPARSE_MOTION_SEEDS = (840_101, 840_102, 840_103, 840_104, 840_105)
DEFAULT_SPARSE_NOISE_SEED = 840_201
DEFAULT_SPARSE_DURATION_S = 16.0
DEFAULT_RATE_PERIOD_S = 0.01
_STUDY_LIMITATION = (
    "Controlled synthetic sensitivity study only; not field accuracy or a calibrated error rate. "
    "Decision thresholds and policy were not tuned."
)


class SparseAnchorNoiseTier(StrEnum):
    """Independent orientation/rate Gaussian noise levels used by this study."""

    NONE = "none"
    MODERATE = "moderate"
####


class SparseAnchorNoiseSpec(BaseModel):
    """Per-axis marginal standard deviations for one paired noise condition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tier: SparseAnchorNoiseTier
    orientation_sigma_rad: float
    rate_sigma_rad_s: float
####


class SparseAnchorTimingCondition(BaseModel):
    """One named anchor, timestamp, channel-latency, and clock-scale condition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition_id: str = Field(min_length=1)
    anchor_drop_probability: float = Field(default=0.0, ge=0.0, lt=1.0)
    anchor_timestamp_jitter_s: float = Field(default=0.0, ge=0.0)
    rate_timestamp_jitter_s: float = Field(default=0.0, ge=0.0)
    orientation_value_latency_s: float = Field(default=0.0, ge=0.0)
    rate_value_latency_s: float = Field(default=0.0, ge=0.0)
    clock_rate_error_ppm: float = 0.0

    @model_validator(mode="after")
    def _validate_finite_timing_scales(self) -> SparseAnchorTimingCondition:
        values = (
            self.anchor_drop_probability,
            self.anchor_timestamp_jitter_s,
            self.rate_timestamp_jitter_s,
            self.orientation_value_latency_s,
            self.rate_value_latency_s,
            self.clock_rate_error_ppm,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("timing perturbations must be finite")
        ####
        if self.clock_rate_error_ppm <= -1_000_000.0:
            raise ValueError("clock_rate_error_ppm must be greater than -1,000,000")
        ####
        return self
    ####
####


DEFAULT_SPARSE_TIMING_CONDITIONS = (
    SparseAnchorTimingCondition(condition_id="regular-exact"),
    SparseAnchorTimingCondition(
        condition_id="irregular-anchor-dropout-20pct",
        anchor_drop_probability=0.2,
    ),
    SparseAnchorTimingCondition(
        condition_id="independent-timestamp-jitter",
        anchor_timestamp_jitter_s=0.01,
        rate_timestamp_jitter_s=0.002,
    ),
    SparseAnchorTimingCondition(
        condition_id="irregular-anchor-dropout-and-jitter",
        anchor_drop_probability=0.2,
        anchor_timestamp_jitter_s=0.01,
        rate_timestamp_jitter_s=0.002,
    ),
    SparseAnchorTimingCondition(
        condition_id="relative-channel-value-latency",
        orientation_value_latency_s=0.01,
        rate_value_latency_s=0.025,
    ),
    SparseAnchorTimingCondition(
        condition_id="shared-clock-rate-error-plus-1000ppm",
        clock_rate_error_ppm=1_000.0,
    ),
)


class SparseAnchorStudyConfig(BaseModel):
    """Validated, repeatable dimensions of one sparse-anchor stress study."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    duration_s: float = Field(default=DEFAULT_SPARSE_DURATION_S, gt=0.0)
    rate_period_s: float = Field(default=DEFAULT_RATE_PERIOD_S, gt=0.0)
    anchor_cadences_s: tuple[float, ...] = DEFAULT_SPARSE_ANCHOR_CADENCES_S
    motion_profiles: tuple[MotionProfile, ...] = SAMPLED_RATE_PROFILES
    motion_seeds: tuple[int, ...] = DEFAULT_SPARSE_MOTION_SEEDS
    noise_tiers: tuple[SparseAnchorNoiseTier, ...] = (
        SparseAnchorNoiseTier.NONE,
        SparseAnchorNoiseTier.MODERATE,
    )
    timing_conditions: tuple[SparseAnchorTimingCondition, ...] = DEFAULT_SPARSE_TIMING_CONDITIONS
    noise_seed: int = DEFAULT_SPARSE_NOISE_SEED

    @model_validator(mode="after")
    def _validate_study_grid(self) -> SparseAnchorStudyConfig:
        if not isfinite(self.duration_s) or not isfinite(self.rate_period_s):
            raise ValueError("duration_s and rate_period_s must be finite")
        ####
        rate_intervals = self.duration_s / self.rate_period_s
        if not isclose(rate_intervals, round(rate_intervals), rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("duration_s must be an integer multiple of rate_period_s")
        ####
        if not self.anchor_cadences_s:
            raise ValueError("anchor_cadences_s must not be empty")
        ####
        if tuple(sorted(set(self.anchor_cadences_s))) != self.anchor_cadences_s:
            raise ValueError("anchor_cadences_s must be unique and strictly increasing")
        ####
        for cadence_s in self.anchor_cadences_s:
            if not isfinite(cadence_s) or cadence_s <= 0.0:
                raise ValueError("anchor cadences must be positive and finite")
            ####
            cadence_intervals = cadence_s / self.rate_period_s
            duration_intervals = self.duration_s / cadence_s
            if not isclose(
                cadence_intervals,
                round(cadence_intervals),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise ValueError("each anchor cadence must be a multiple of rate_period_s")
            ####
            if round(duration_intervals) < 2 or not isclose(
                duration_intervals,
                round(duration_intervals),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise ValueError(
                    "duration_s must contain at least two whole anchor intervals at each cadence"
                )
            ####
            base_cadence_ratio = cadence_s / self.anchor_cadences_s[0]
            if not isclose(
                base_cadence_ratio,
                round(base_cadence_ratio),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise ValueError("anchor cadences must be nested integer multiples")
            ####
        ####
        if not self.motion_profiles or len(set(self.motion_profiles)) != len(self.motion_profiles):
            raise ValueError("motion_profiles must be non-empty and unique")
        ####
        if any(profile not in SAMPLED_RATE_PROFILES for profile in self.motion_profiles):
            raise ValueError("motion_profiles must use independently labeled sampled-rate families")
        ####
        if not self.motion_seeds or len(set(self.motion_seeds)) != len(self.motion_seeds):
            raise ValueError("motion_seeds must be non-empty and unique")
        ####
        if any(seed < 0 for seed in self.motion_seeds) or self.noise_seed < 0:
            raise ValueError("random seeds must be non-negative")
        ####
        if not self.noise_tiers or len(set(self.noise_tiers)) != len(self.noise_tiers):
            raise ValueError("noise_tiers must be non-empty and unique")
        ####
        condition_ids = tuple(condition.condition_id for condition in self.timing_conditions)
        if not condition_ids or len(set(condition_ids)) != len(condition_ids):
            raise ValueError("timing_conditions must have unique non-empty condition IDs")
        ####
        minimum_anchor_cadence_s = self.anchor_cadences_s[0]
        for condition in self.timing_conditions:
            if condition.anchor_timestamp_jitter_s >= minimum_anchor_cadence_s / 2.0:
                raise ValueError(
                    "anchor timestamp jitter must be less than half the minimum anchor cadence"
                )
            ####
            if condition.rate_timestamp_jitter_s >= self.rate_period_s / 2.0:
                raise ValueError("rate timestamp jitter must be less than half the rate period")
            ####
        ####
        return self
    ####
####


class SparseAnchorCase(BaseModel):
    """Truth-retention and decision outcome for one motion/acquisition cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    motion_profile: MotionProfile
    motion_seed: int
    noise_tier: SparseAnchorNoiseTier
    noise_seed: int
    timing_condition_id: str
    anchor_cadence_s: float
    anchor_count: int
    rate_sample_count: int
    truth_class_id: str = Field(min_length=1)
    data_winning_class_ids: tuple[str, ...]
    evidence_contender_class_ids: tuple[str, ...]
    recommended_class_id: str | None
    evidence_sufficient: bool
    decision_status: InferenceDecisionStatus
    normalized_rmse: float

    @property
    def data_winner_retains_truth(self) -> bool:
        return self.truth_class_id in self.data_winning_class_ids
    ####

    @property
    def evidence_retains_truth(self) -> bool:
        return self.truth_class_id in self.evidence_contender_class_ids
    ####

    @property
    def recommendation_correct(self) -> bool | None:
        if self.recommended_class_id is None:
            return None
        ####
        return self.recommended_class_id == self.truth_class_id
    ####
####


class SparseAnchorSummary(BaseModel):
    """Aggregated outcomes for one motion, noise, timing, and cadence condition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    motion_profile: MotionProfile
    noise_tier: SparseAnchorNoiseTier
    timing_condition_id: str
    anchor_cadence_s: float
    trial_count: int
    minimum_anchor_count: int
    median_anchor_count: float
    maximum_anchor_count: int
    data_winner_truth_count: int
    evidence_truth_count: int
    evidence_sufficient_count: int
    correct_recommendation_count: int
    wrong_recommendation_count: int
    abstention_count: int
    median_normalized_rmse: float
####


class SparseAnchorStudyReport(BaseModel):
    """Versioned full results and study limitations for reproducibility."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 3
    package_version: str
    generator: str
    limitation: str = _STUDY_LIMITATION
    config: SparseAnchorStudyConfig
    noise_specs: tuple[SparseAnchorNoiseSpec, ...]
    summaries: tuple[SparseAnchorSummary, ...]
    cases: tuple[SparseAnchorCase, ...]
####


def _sample_motion(
    profile: MotionProfile,
    time_s: FloatArray,
    motion_seed: int,
) -> SyntheticMotion:
    """Use the independent fixed-grid generator for one sampled-rate motion."""

    if profile is MotionProfile.SAMPLED_RATE_NONANALYTIC:
        return sampled_rate_motion(
            time_s,
            seed=motion_seed,
            reference_time_s=0.0,
            integration_step_s=SAMPLED_RATE_INTEGRATION_STEP_S,
        )
    ####
    if profile is MotionProfile.SAMPLED_RATE_PIECEWISE_LINEAR:
        return piecewise_linear_rate_motion(
            time_s,
            seed=motion_seed,
            reference_time_s=0.0,
            integration_step_s=SAMPLED_RATE_INTEGRATION_STEP_S,
        )
    ####
    if profile is MotionProfile.SAMPLED_RATE_MANEUVER_BURSTS:
        return maneuver_burst_rate_motion(
            time_s,
            seed=motion_seed,
            reference_time_s=0.0,
            integration_step_s=SAMPLED_RATE_INTEGRATION_STEP_S,
        )
    ####
    raise ValueError(f"Unsupported sampled-rate motion profile: {profile.value}")
####


def _noise_profile(tier: SparseAnchorNoiseTier) -> NoiseProfile:
    if tier is SparseAnchorNoiseTier.NONE:
        return NO_NOISE_PROFILE
    ####
    if tier is SparseAnchorNoiseTier.MODERATE:
        return next(profile for profile in NOISE_PROFILES if profile.name == "moderate")
    ####
    raise ValueError(f"Unsupported noise tier: {tier.value}")
####


def _evaluate_sparse_case(
    *,
    profile: MotionProfile,
    motion_seed: int,
    noise_tier: SparseAnchorNoiseTier,
    timing_condition: SparseAnchorTimingCondition,
    config: SparseAnchorStudyConfig,
    rate_times_s: FloatArray,
    base_anchor_stride: int,
) -> tuple[SparseAnchorCase, ...]:
    """Evaluate nested cadences with paired motion, measurement noise, and timing errors."""

    rate_source_times_s = rate_times_s - timing_condition.rate_value_latency_s
    rate_motion = _sample_motion(profile, rate_source_times_s, motion_seed)
    (
        orientation_seed,
        rate_seed,
        anchor_timing_seed,
        rate_timing_seed,
        dropout_seed,
    ) = SeedSequence((config.noise_seed, motion_seed)).spawn(5)
    orientation_random = default_rng(orientation_seed)
    rate_random = default_rng(rate_seed)
    anchor_timing_random = default_rng(anchor_timing_seed)
    rate_timing_random = default_rng(rate_timing_seed)
    dropout_random = default_rng(dropout_seed)
    base_anchor_count = round(config.duration_s / config.anchor_cadences_s[0]) + 1
    orientation_standard_noise = orientation_random.normal(
        0.0,
        1.0,
        (base_anchor_count, 3),
    )
    rate_standard_noise = rate_random.normal(0.0, 1.0, rate_motion.body_rates_rad_s.shape)
    anchor_timestamp_jitter = anchor_timing_random.uniform(
        -timing_condition.anchor_timestamp_jitter_s,
        timing_condition.anchor_timestamp_jitter_s,
        base_anchor_count,
    )
    anchor_timestamp_jitter[0] = 0.0
    anchor_timestamp_jitter[-1] = 0.0
    rate_timestamp_jitter = rate_timing_random.uniform(
        -timing_condition.rate_timestamp_jitter_s,
        timing_condition.rate_timestamp_jitter_s,
        rate_times_s.size,
    )
    rate_timestamp_jitter[0] = 0.0
    rate_timestamp_jitter[-1] = 0.0
    retained_base_anchors = array(
        dropout_random.random(base_anchor_count) >= timing_condition.anchor_drop_probability,
        dtype=bool_,
    )
    retained_base_anchors[0] = True
    retained_base_anchors[-1] = True
    tier_noise = _noise_profile(noise_tier)
    anchor_base_indices = arange(base_anchor_count, dtype=int64) * base_anchor_stride
    base_anchor_times_s = rate_times_s[anchor_base_indices]
    orientation_source_times_s = base_anchor_times_s - timing_condition.orientation_value_latency_s
    orientation_motion = _sample_motion(profile, orientation_source_times_s, motion_seed)
    noisy_anchor_quaternions = rotation_quaternions(
        rotation_product(
            rotation_from_rotvec(orientation_standard_noise * tier_noise.orientation_sigma_rad),
            rotation_from_quat(orientation_motion.quaternion_xyzw),
        )
    )
    noisy_rates = array(
        rate_motion.body_rates_rad_s + rate_standard_noise * tier_noise.rate_sigma_rad_s,
        dtype=float64,
    )
    timestamp_scale = 1.0 + timing_condition.clock_rate_error_ppm * 1.0e-6
    recorded_rate_times_s = array(
        (rate_times_s + rate_timestamp_jitter) * timestamp_scale,
        dtype=float64,
    )
    truth = ConventionHypothesis(
        attitude_direction=AttitudeDirection.BODY_TO_WORLD,
        quaternion_order=QuaternionOrder.SCALAR_LAST,
        rate_frame=RateFrame.BODY,
        rate_order="pqr",
        rate_unit=RateUnit.RADIANS_PER_SECOND,
        rate_sign=1,
    )
    cases: list[SparseAnchorCase] = []
    for cadence_s in config.anchor_cadences_s:
        anchor_stride = round(cadence_s / config.rate_period_s)
        nested_stride = anchor_stride // base_anchor_stride
        candidate_base_indices = arange(base_anchor_count, dtype=int64)[::nested_stride]
        retained_at_cadence = retained_base_anchors[candidate_base_indices]
        if count_nonzero(retained_at_cadence) < 3:
            retained_at_cadence[candidate_base_indices.size // 2] = True
        ####
        anchor_indices = candidate_base_indices[retained_at_cadence]
        anchor_times_s = array(
            (base_anchor_times_s[anchor_indices] + anchor_timestamp_jitter[anchor_indices])
            * timestamp_scale,
            dtype=float64,
        )
        data = SparseOrientationRateData(
            representation=AttitudeRepresentation.QUATERNION,
            orientation_anchors=noisy_anchor_quaternions[anchor_indices],
            orientation_anchor_times_s=anchor_times_s,
            rates=noisy_rates,
            rate_times_s=recorded_rate_times_s,
        )
        report = infer_sparse_orientation_conventions(data)
        truth_classes = tuple(
            result.equivalence_class_id for result in report.rankings if result.hypothesis == truth
        )
        if len(truth_classes) != 1:
            raise RuntimeError(
                "The default quaternion search did not yield exactly one truth class"
            )
        ####
        truth_class_id = truth_classes[0]
        recommended_class_id = report.decision.recommended_class_id
        cases.append(
            SparseAnchorCase(
                motion_profile=profile,
                motion_seed=motion_seed,
                noise_tier=noise_tier,
                noise_seed=config.noise_seed,
                timing_condition_id=timing_condition.condition_id,
                anchor_cadence_s=cadence_s,
                anchor_count=data.anchor_count,
                rate_sample_count=data.rate_sample_count,
                truth_class_id=truth_class_id,
                data_winning_class_ids=report.data_winning_class_ids,
                evidence_contender_class_ids=report.evidence_contender_class_ids,
                recommended_class_id=recommended_class_id,
                evidence_sufficient=report.diagnostics.evidence_sufficient,
                decision_status=report.decision.status,
                normalized_rmse=report.best.normalized_rmse,
            )
        )
    ####
    return tuple(cases)
####


def summarize_sparse_anchor_cases(
    cases: tuple[SparseAnchorCase, ...],
) -> tuple[SparseAnchorSummary, ...]:
    """Aggregate per-case truth retention and recommendation outcomes."""

    grouped: dict[
        tuple[MotionProfile, SparseAnchorNoiseTier, str, float], list[SparseAnchorCase]
    ] = {}
    for case in cases:
        key = (
            case.motion_profile,
            case.noise_tier,
            case.timing_condition_id,
            case.anchor_cadence_s,
        )
        grouped.setdefault(key, []).append(case)
    ####
    summaries: list[SparseAnchorSummary] = []
    for (profile, noise_tier, timing_condition_id, cadence_s), group in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0].value,
            item[0][1].value,
            item[0][2],
            item[0][3],
        ),
    ):
        summaries.append(
            SparseAnchorSummary(
                motion_profile=profile,
                noise_tier=noise_tier,
                timing_condition_id=timing_condition_id,
                anchor_cadence_s=cadence_s,
                trial_count=len(group),
                minimum_anchor_count=min(case.anchor_count for case in group),
                median_anchor_count=float(median(case.anchor_count for case in group)),
                maximum_anchor_count=max(case.anchor_count for case in group),
                data_winner_truth_count=sum(case.data_winner_retains_truth for case in group),
                evidence_truth_count=sum(case.evidence_retains_truth for case in group),
                evidence_sufficient_count=sum(case.evidence_sufficient for case in group),
                correct_recommendation_count=sum(
                    case.recommendation_correct is True for case in group
                ),
                wrong_recommendation_count=sum(
                    case.recommendation_correct is False for case in group
                ),
                abstention_count=sum(case.recommendation_correct is None for case in group),
                median_normalized_rmse=float(median(case.normalized_rmse for case in group)),
            )
        )
    ####
    return tuple(summaries)
####


def run_sparse_anchor_study(
    config: SparseAnchorStudyConfig | None = None,
) -> SparseAnchorStudyReport:
    """Run a paired synthetic study without changing inference or decision policy."""

    resolved = config or SparseAnchorStudyConfig()
    rate_sample_count = round(resolved.duration_s / resolved.rate_period_s) + 1
    rate_times_s = array(arange(rate_sample_count, dtype=float64) * resolved.rate_period_s)
    base_anchor_stride = round(resolved.anchor_cadences_s[0] / resolved.rate_period_s)
    cases = tuple(
        case
        for profile in resolved.motion_profiles
        for motion_seed in resolved.motion_seeds
        for noise_tier in resolved.noise_tiers
        for timing_condition in resolved.timing_conditions
        for case in _evaluate_sparse_case(
            profile=profile,
            motion_seed=motion_seed,
            noise_tier=noise_tier,
            timing_condition=timing_condition,
            config=resolved,
            rate_times_s=rate_times_s,
            base_anchor_stride=base_anchor_stride,
        )
    )
    return SparseAnchorStudyReport(
        package_version=__version__,
        generator=(
            "scripts/generate_validation_corpus.py fixed-grid sampled-rate motion; "
            "independent Gaussian measurement noise, anchor dropout, and channel timestamp jitter"
        ),
        config=resolved,
        noise_specs=tuple(
            SparseAnchorNoiseSpec(
                tier=tier,
                orientation_sigma_rad=_noise_profile(tier).orientation_sigma_rad,
                rate_sigma_rad_s=_noise_profile(tier).rate_sigma_rad_s,
            )
            for tier in resolved.noise_tiers
        ),
        summaries=summarize_sparse_anchor_cases(cases),
        cases=cases,
    )
####


def _format_summary(report: SparseAnchorStudyReport) -> str:
    rows = [
        "profile | noise | timing | cadence_s | anchors min/med/max | trials | winner | "
        "evidence | sufficient | "
        "correct | wrong | abstain | median_rmse",
        "--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---:",
    ]
    for item in report.summaries:
        rows.append(
            f"{item.motion_profile.value} | {item.noise_tier.value} | "
            f"{item.timing_condition_id} | {item.anchor_cadence_s:g} | "
            f"{item.minimum_anchor_count}/{item.median_anchor_count:g}/"
            f"{item.maximum_anchor_count} | {item.trial_count} | "
            f"{item.data_winner_truth_count} | "
            f"{item.evidence_truth_count} | {item.evidence_sufficient_count} | "
            f"{item.correct_recommendation_count} | {item.wrong_recommendation_count} | "
            f"{item.abstention_count} | {item.median_normalized_rmse:.6g}"
        )
    ####
    return "\n".join(rows)
####


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional path for the full JSON report")
    parser.add_argument("--duration-s", type=float, default=DEFAULT_SPARSE_DURATION_S)
    parser.add_argument("--rate-period-s", type=float, default=DEFAULT_RATE_PERIOD_S)
    parser.add_argument(
        "--anchor-cadences-s",
        type=float,
        nargs="+",
        default=DEFAULT_SPARSE_ANCHOR_CADENCES_S,
    )
    parser.add_argument("--motion-seeds", type=int, nargs="+", default=DEFAULT_SPARSE_MOTION_SEEDS)
    parser.add_argument(
        "--motion-profiles",
        choices=tuple(profile.value for profile in SAMPLED_RATE_PROFILES),
        nargs="+",
        default=tuple(profile.value for profile in SAMPLED_RATE_PROFILES),
    )
    parser.add_argument(
        "--noise-tiers",
        choices=tuple(tier.value for tier in SparseAnchorNoiseTier),
        nargs="+",
        default=tuple(tier.value for tier in SparseAnchorNoiseTier),
    )
    parser.add_argument("--noise-seed", type=int, default=DEFAULT_SPARSE_NOISE_SEED)
    args = parser.parse_args()
    config = SparseAnchorStudyConfig(
        duration_s=args.duration_s,
        rate_period_s=args.rate_period_s,
        anchor_cadences_s=tuple(args.anchor_cadences_s),
        motion_profiles=tuple(MotionProfile(value) for value in args.motion_profiles),
        motion_seeds=tuple(args.motion_seeds),
        noise_tiers=tuple(SparseAnchorNoiseTier(value) for value in args.noise_tiers),
        noise_seed=args.noise_seed,
    )
    report = run_sparse_anchor_study(config)
    print(_format_summary(report))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote report: {args.output}")
    ####
####


if __name__ == "__main__":
    main()
####
