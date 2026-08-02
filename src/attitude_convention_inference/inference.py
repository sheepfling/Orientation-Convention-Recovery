from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import exp, log, sqrt
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from .core import (
    FloatArray,
    TrajectoryData,
    compose_increment_windows,
    decode_attitudes,
    decode_rates,
    integrate_interval_increments,
    observed_interval_increments,
    observed_window_increments,
    resolve_interval_durations,
)
from .models import (
    AngleUnit,
    AttitudeRepresentation,
    CandidateResult,
    ConfidenceLabel,
    ConventionHypothesis,
    DimensionAssessment,
    DimensionStatus,
    HypothesisSpaceSummary,
    InferenceConfig,
    InferenceReport,
    LagMetric,
    OverallDiagnostics,
    SearchSpace,
)

_DIMENSIONS: tuple[str, ...] = (
    "attitude_direction",
    "quaternion_order",
    "euler_order",
    "euler_mode",
    "angle_unit",
    "rate_frame",
    "rate_order",
    "rate_unit",
    "rate_sign",
)
_SEARCH_FIELD_BY_DIMENSION: dict[str, str] = {
    "attitude_direction": "attitude_directions",
    "quaternion_order": "quaternion_orders",
    "euler_order": "euler_orders",
    "euler_mode": "euler_modes",
    "angle_unit": "angle_units",
    "rate_frame": "rate_frames",
    "rate_order": "rate_orders",
    "rate_unit": "rate_units",
    "rate_sign": "rate_signs",
}


@dataclass(frozen=True, slots=True)
class _ScoredCandidate:
    result: CandidateResult
    attitudes: Rotation
    rates_rad_s: FloatArray
    interval_durations_s: FloatArray
    predicted_intervals: Rotation
    observed_intervals: Rotation
    interval_error_rotvecs: FloatArray
    path_angles: FloatArray


####


def _relevant_dimensions(representation: AttitudeRepresentation) -> tuple[str, ...]:
    attitude_dimensions = (
        ("quaternion_order",)
        if representation is AttitudeRepresentation.QUATERNION
        else ("euler_order", "euler_mode", "angle_unit")
    )
    return (
        "attitude_direction",
        *attitude_dimensions,
        "rate_frame",
        "rate_order",
        "rate_unit",
        "rate_sign",
    )


####


def _search_values(search: SearchSpace, dimension: str) -> tuple[object, ...]:
    return tuple(getattr(search, _SEARCH_FIELD_BY_DIMENSION[dimension]))


####


def _hypothesis_count(
    representation: AttitudeRepresentation,
    search: SearchSpace,
) -> int:
    count = 1
    for dimension in _relevant_dimensions(representation):
        count *= len(_search_values(search, dimension))
    return count


####


def generate_hypotheses(
    representation: AttitudeRepresentation,
    config: InferenceConfig,
) -> tuple[ConventionHypothesis, ...]:
    search = config.resolved_search_space()
    rate_options = tuple(
        product(search.rate_frames, search.rate_orders, search.rate_units, search.rate_signs)
    )
    hypotheses: list[ConventionHypothesis] = []

    if representation is AttitudeRepresentation.QUATERNION:
        for direction, quaternion_order in product(
            search.attitude_directions,
            search.quaternion_orders,
        ):
            for rate_frame, rate_order, rate_unit, rate_sign in rate_options:
                hypotheses.append(
                    ConventionHypothesis(
                        attitude_direction=direction,
                        quaternion_order=quaternion_order,
                        rate_frame=rate_frame,
                        rate_order=rate_order,
                        rate_unit=rate_unit,
                        rate_sign=rate_sign,
                    )
                )
    else:
        for direction, euler_order, euler_mode, angle_unit in product(
            search.attitude_directions,
            search.euler_orders,
            search.euler_modes,
            search.angle_units,
        ):
            for rate_frame, rate_order, rate_unit, rate_sign in rate_options:
                hypotheses.append(
                    ConventionHypothesis(
                        attitude_direction=direction,
                        euler_order=euler_order,
                        euler_mode=euler_mode,
                        angle_unit=angle_unit,
                        rate_frame=rate_frame,
                        rate_order=rate_order,
                        rate_unit=rate_unit,
                        rate_sign=rate_sign,
                    )
                )

    return tuple(hypotheses)


####


def _trimmed_mean_square(values: FloatArray, trim_fraction: float) -> float:
    if values.size == 0:
        return float("inf")
    keep_count = max(1, int(np.ceil(values.size * (1.0 - trim_fraction))))
    squared = np.sort(np.square(values))[:keep_count]
    return float(np.mean(squared))


####


def _gimbal_lock_fraction(
    data: TrajectoryData,
    hypothesis: ConventionHypothesis,
    config: InferenceConfig,
) -> float | None:
    if data.representation is not AttitudeRepresentation.EULER:
        return None
    if hypothesis.angle_unit is None:
        return None
    middle = data.orientation[:, 1]
    if hypothesis.angle_unit is AngleUnit.DEGREES:
        middle = np.deg2rad(middle)
    return float(np.mean(np.abs(np.cos(middle)) < config.gimbal_cosine_threshold))


####


def _score_hypothesis(
    data: TrajectoryData,
    hypothesis: ConventionHypothesis,
    config: InferenceConfig,
) -> _ScoredCandidate:
    attitudes = decode_attitudes(data, hypothesis)
    rates_rad_s = decode_rates(data, hypothesis)
    durations, inferred_period, _, time_warnings = resolve_interval_durations(
        data,
        attitudes,
        rates_rad_s,
        hypothesis.rate_frame,
        config,
    )
    predicted, path_angles, rate_change_angles = integrate_interval_increments(
        rates_rad_s,
        durations,
        hypothesis.rate_frame,
        config.integration_substeps,
    )
    observed = observed_interval_increments(attitudes, hypothesis.rate_frame)
    interval_errors = (predicted.inv() * observed).as_rotvec()
    interval_error_angles = np.linalg.norm(interval_errors, axis=1)
    observed_interval_angles = observed.magnitude()

    valid_windows = tuple(
        window for window in config.window_sizes if window <= data.sample_count - 1
    )
    lag_metrics: list[LagMetric] = []
    scaled_error_angles: list[FloatArray] = []
    scaled_observed_angles: list[FloatArray] = []

    for window_size in valid_windows:
        predicted_windows, starts = compose_increment_windows(
            predicted,
            window_size,
            hypothesis.rate_frame,
        )
        observed_windows = observed_window_increments(
            attitudes,
            starts,
            window_size,
            hypothesis.rate_frame,
        )
        errors = (predicted_windows.inv() * observed_windows).magnitude()
        observed_angles = observed_windows.magnitude()
        scale = sqrt(float(window_size))
        scaled_error_angles.append(np.asarray(errors, dtype=np.float64) / scale)
        scaled_observed_angles.append(np.asarray(observed_angles, dtype=np.float64) / scale)
        lag_metrics.append(
            LagMetric(
                window_size=window_size,
                window_count=int(errors.size),
                residual_rmse_rad=float(np.sqrt(np.mean(np.square(errors)))),
                residual_median_rad=float(np.median(errors)),
                residual_p95_rad=float(np.quantile(errors, 0.95)),
                observed_rotation_rms_rad=float(
                    np.sqrt(np.mean(np.square(observed_angles)))
                ),
            )
        )

    multiscale_errors = np.concatenate(scaled_error_angles)
    multiscale_observed = np.concatenate(scaled_observed_angles)
    robust_mse = _trimmed_mean_square(multiscale_errors, config.trim_fraction)
    robust_mse = max(robust_mse, config.residual_floor_rad**2)
    multiscale_rmse = sqrt(robust_mse)
    observed_scale = max(
        float(np.sqrt(np.mean(np.square(multiscale_observed)))),
        config.residual_floor_rad,
    )
    normalized_rmse = multiscale_rmse / observed_scale
    fit_score = 100.0 * exp(-0.5 * normalized_rmse**2)

    effective_observation_count = max(3, 3 * (data.sample_count - 1))
    parameter_count = 1 if inferred_period is not None else 0
    data_criterion = (
        effective_observation_count * log(robust_mse)
        + parameter_count * log(effective_observation_count)
    )

    near_alias_fraction = float(np.mean(path_angles >= config.near_alias_step_rad))
    hard_alias_fraction = float(np.mean(path_angles >= config.hard_alias_step_rad))
    rate_change_rms = float(np.sqrt(np.mean(np.square(rate_change_angles))))
    warnings = list(time_warnings)
    if hard_alias_fraction > 0.0:
        warnings.append(
            "At least one sample interval contains an unwrapped rate integral of pi "
            "radians or more."
        )
    elif near_alias_fraction > 0.0:
        warnings.append("At least one sample interval is close to rotational aliasing.")
    if rate_change_rms >= config.high_rate_change_angle_rad:
        warnings.append(
            "Angular rate changes substantially within a sample interval; interpolation "
            "error may matter."
        )

    gimbal_fraction = _gimbal_lock_fraction(data, hypothesis, config)
    if gimbal_fraction is not None and gimbal_fraction > 0.0:
        warnings.append("Some Euler samples are near this hypothesis's gimbal-lock surface.")

    result = CandidateResult(
        hypothesis=hypothesis,
        criterion=float(data_criterion),
        data_criterion=float(data_criterion),
        fit_score=float(fit_score),
        normalized_rmse=float(normalized_rmse),
        multiscale_rmse_rad=float(multiscale_rmse),
        interval_rmse_rad=float(np.sqrt(np.mean(np.square(interval_error_angles)))),
        interval_median_rad=float(np.median(interval_error_angles)),
        interval_p95_rad=float(np.quantile(interval_error_angles, 0.95)),
        interval_max_rad=float(np.max(interval_error_angles)),
        observed_step_rms_rad=float(
            np.sqrt(np.mean(np.square(observed_interval_angles)))
        ),
        inferred_sample_period_s=inferred_period,
        valid_interval_count=data.sample_count - 1,
        near_alias_fraction=near_alias_fraction,
        hard_alias_fraction=hard_alias_fraction,
        rate_change_rms_angle_rad=rate_change_rms,
        gimbal_lock_fraction=gimbal_fraction,
        lag_metrics=tuple(lag_metrics),
        warnings=tuple(warnings),
    )
    return _ScoredCandidate(
        result=result,
        attitudes=attitudes,
        rates_rad_s=rates_rad_s,
        interval_durations_s=durations,
        predicted_intervals=predicted,
        observed_intervals=observed,
        interval_error_rotvecs=np.asarray(interval_errors, dtype=np.float64),
        path_angles=path_angles,
    )


####


def _hypothesis_value(hypothesis: ConventionHypothesis, dimension: str) -> object | None:
    return getattr(hypothesis, dimension)


####


def _prior_log_weight(
    hypothesis: ConventionHypothesis,
    config: InferenceConfig,
) -> float:
    total = 0.0
    for dimension in _DIMENSIONS:
        value = _hypothesis_value(hypothesis, dimension)
        if value is None:
            continue
        weights = getattr(config.priors, dimension)
        total += log(float(weights.get(value, 1.0)))
    return total


####


def _support_weights(criteria: np.ndarray) -> np.ndarray:
    minimum = float(np.min(criteria))
    raw = np.exp(-0.5 * np.clip(criteria - minimum, 0.0, 1400.0))
    return raw / float(np.sum(raw))


####


def _replace_result(
    item: _ScoredCandidate,
    result: CandidateResult,
) -> _ScoredCandidate:
    return _ScoredCandidate(
        result=result,
        attitudes=item.attitudes,
        rates_rad_s=item.rates_rad_s,
        interval_durations_s=item.interval_durations_s,
        predicted_intervals=item.predicted_intervals,
        observed_intervals=item.observed_intervals,
        interval_error_rotvecs=item.interval_error_rotvecs,
        path_angles=item.path_angles,
    )


####


def _assign_ranks_and_support(
    scored: Iterable[_ScoredCandidate],
    config: InferenceConfig,
) -> tuple[_ScoredCandidate, ...]:
    candidates = tuple(scored)
    if not candidates:
        raise ValueError("No convention hypothesis could be scored.")

    data_criteria = np.asarray(
        [item.result.data_criterion for item in candidates],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(data_criteria)):
        raise ValueError("No finite convention hypothesis could be scored.")

    data_supports = _support_weights(data_criteria)
    data_minimum = float(np.min(data_criteria))
    data_order = np.argsort(data_criteria)
    data_rank_by_index = {
        int(candidate_index): rank
        for rank, candidate_index in enumerate(data_order, start=1)
    }

    prior_logs = np.asarray(
        [_prior_log_weight(item.result.hypothesis, config) for item in candidates],
        dtype=np.float64,
    )
    maximum_prior_log = float(np.max(prior_logs))
    prior_penalties = 2.0 * (maximum_prior_log - prior_logs)
    posterior_criteria = data_criteria + prior_penalties
    posterior_supports = _support_weights(posterior_criteria)
    posterior_minimum = float(np.min(posterior_criteria))
    posterior_order = np.argsort(posterior_criteria)

    ranked: list[_ScoredCandidate] = []
    for rank, candidate_index_raw in enumerate(posterior_order, start=1):
        candidate_index = int(candidate_index_raw)
        item = candidates[candidate_index]
        updated_result = item.result.model_copy(
            update={
                "rank": rank,
                "data_rank": data_rank_by_index[candidate_index],
                "relative_support": float(posterior_supports[candidate_index]),
                "data_relative_support": float(data_supports[candidate_index]),
                "delta_criterion": float(
                    posterior_criteria[candidate_index] - posterior_minimum
                ),
                "data_delta_criterion": float(data_criteria[candidate_index] - data_minimum),
                "criterion": float(posterior_criteria[candidate_index]),
                "prior_relative_weight": float(
                    exp(prior_logs[candidate_index] - maximum_prior_log)
                ),
                "prior_penalty": float(prior_penalties[candidate_index]),
            }
        )
        ranked.append(_replace_result(item, updated_result))
    return tuple(ranked)


####


def _effective_excitation_rank(
    rates_rad_s: FloatArray,
    durations_s: FloatArray,
) -> tuple[float, tuple[float, float, float]]:
    integrated = 0.5 * (rates_rad_s[:-1] + rates_rad_s[1:]) * durations_s[:, np.newaxis]
    singular_values = np.linalg.svd(integrated, compute_uv=False)
    padded = np.zeros(3, dtype=np.float64)
    padded[: singular_values.size] = singular_values
    energy = np.square(padded)
    total = float(np.sum(energy))
    if total <= 1.0e-30:
        return 0.0, (0.0, 0.0, 0.0)
    probabilities = energy / total
    nonzero = probabilities > 1.0e-15
    entropy = -float(np.sum(probabilities[nonzero] * np.log(probabilities[nonzero])))
    effective_rank = exp(entropy)
    return float(effective_rank), tuple(float(value) for value in padded)


####


def _hypothesis_dimensions(hypothesis: ConventionHypothesis) -> dict[str, str]:
    values = hypothesis.model_dump(mode="json")
    return {key: str(value) for key, value in values.items() if value is not None}


####


def _unresolved_dimensions(
    near_ties: tuple[_ScoredCandidate, ...],
) -> dict[str, tuple[str, ...]]:
    gathered: dict[str, set[str]] = {}
    for item in near_ties:
        for key, value in _hypothesis_dimensions(item.result.hypothesis).items():
            gathered.setdefault(key, set()).add(value)
    return {
        key: tuple(sorted(values))
        for key, values in gathered.items()
        if len(values) > 1
    }


####


def _relative_prior_weights(
    dimension: str,
    allowed_values: tuple[object, ...],
    config: InferenceConfig,
) -> dict[str, float]:
    configured = getattr(config.priors, dimension)
    if not configured:
        return {}
    weights = {str(value): float(configured.get(value, 1.0)) for value in allowed_values}
    maximum = max(weights.values())
    return {key: weight / maximum for key, weight in weights.items()}


####


def _hypothesis_space_summary(
    representation: AttitudeRepresentation,
    config: InferenceConfig,
) -> HypothesisSpaceSummary:
    base = config.search_space
    constrained = config.resolved_search_space()
    relevant = set(_relevant_dimensions(representation))
    fixed: dict[str, str] = {}
    searched: dict[str, tuple[str, ...]] = {}
    prior_weights: dict[str, dict[str, float]] = {}

    for dimension in _DIMENSIONS:
        if dimension not in relevant:
            continue
        values = _search_values(constrained, dimension)
        string_values = tuple(str(value) for value in values)
        if len(values) == 1:
            fixed[dimension] = string_values[0]
        else:
            searched[dimension] = string_values
        relative = _relative_prior_weights(dimension, values, config)
        if relative:
            prior_weights[dimension] = relative

    base_count = _hypothesis_count(representation, base)
    constrained_count = _hypothesis_count(representation, constrained)
    return HypothesisSpaceSummary(
        base_hypothesis_count=base_count,
        constrained_hypothesis_count=constrained_count,
        reduction_fraction=1.0 - constrained_count / base_count,
        fixed_dimensions=fixed,
        searched_dimensions=searched,
        not_applicable_dimensions=tuple(
            dimension for dimension in _DIMENSIONS if dimension not in relevant
        ),
        relative_prior_weights=prior_weights,
    )


####


def _dimension_values(
    candidates: tuple[_ScoredCandidate, ...],
    dimension: str,
) -> tuple[str, ...]:
    values = {
        str(value)
        for item in candidates
        if (value := _hypothesis_value(item.result.hypothesis, dimension)) is not None
    }
    return tuple(sorted(values))


####


def _dimension_assessments(
    representation: AttitudeRepresentation,
    best: _ScoredCandidate,
    evidence_near_ties: tuple[_ScoredCandidate, ...],
    posterior_near_ties: tuple[_ScoredCandidate, ...],
    config: InferenceConfig,
) -> dict[str, DimensionAssessment]:
    search = config.resolved_search_space()
    relevant = set(_relevant_dimensions(representation))
    assessments: dict[str, DimensionAssessment] = {}

    for dimension in _DIMENSIONS:
        if dimension not in relevant:
            assessments[dimension] = DimensionAssessment(
                status=DimensionStatus.NOT_APPLICABLE,
                selected_value=None,
                allowed_values=(),
                evidence_near_tie_values=(),
                posterior_near_tie_values=(),
            )
            continue

        allowed_raw = _search_values(search, dimension)
        allowed = tuple(str(value) for value in allowed_raw)
        selected_raw = _hypothesis_value(best.result.hypothesis, dimension)
        selected = None if selected_raw is None else str(selected_raw)
        evidence_values = _dimension_values(evidence_near_ties, dimension)
        posterior_values = _dimension_values(posterior_near_ties, dimension)

        if len(allowed) == 1:
            status = DimensionStatus.FIXED
        elif len(evidence_values) > 1:
            status = DimensionStatus.AMBIGUOUS
        else:
            status = DimensionStatus.INFERRED

        assessments[dimension] = DimensionAssessment(
            status=status,
            selected_value=selected,
            allowed_values=allowed,
            evidence_near_tie_values=evidence_values,
            posterior_near_tie_values=posterior_values,
            relative_prior_weights=_relative_prior_weights(
                dimension,
                allowed_raw,
                config,
            ),
        )
    return assessments


####


def _confidence_label(value: float) -> ConfidenceLabel:
    if value >= 0.75:
        return ConfidenceLabel.HIGH
    if value >= 0.50:
        return ConfidenceLabel.MODERATE
    if value >= 0.20:
        return ConfidenceLabel.LOW
    return ConfidenceLabel.INDETERMINATE


####


def _build_overall_diagnostics(
    data: TrajectoryData,
    ranked: tuple[_ScoredCandidate, ...],
    config: InferenceConfig,
) -> OverallDiagnostics:
    best = ranked[0]
    evidence_near_ties = tuple(
        sorted(
            (
                item
                for item in ranked
                if item.result.data_delta_criterion <= config.near_tie_delta_criterion
            ),
            key=lambda item: item.result.data_rank,
        )
    )
    posterior_near_ties = tuple(
        item
        for item in ranked
        if item.result.delta_criterion <= config.near_tie_delta_criterion
    )
    unresolved = _unresolved_dimensions(evidence_near_ties)
    posterior_unresolved = _unresolved_dimensions(posterior_near_ties)
    observed_angles = best.observed_intervals.magnitude()
    total_motion = float(np.sum(observed_angles))
    step_rms = float(np.sqrt(np.mean(np.square(observed_angles))))
    effective_rank, singular_values = _effective_excitation_rank(
        best.rates_rad_s,
        best.interval_durations_s,
    )

    fit_factor = exp(-0.5 * (best.result.normalized_rmse / 0.25) ** 2)
    uniqueness_factor = min(1.0, best.result.data_relative_support / 0.90)
    step_factor = min(1.0, step_rms / config.slow_step_threshold_rad)
    total_factor = min(1.0, total_motion / config.slow_total_motion_threshold_rad)
    motion_factor = sqrt(step_factor * total_factor)
    alias_factor = max(
        0.02,
        1.0
        - best.result.hard_alias_fraction
        - 0.5
        * (best.result.near_alias_fraction - best.result.hard_alias_fraction),
    )
    integration_factor = min(
        1.0,
        config.high_rate_change_angle_rad
        / max(config.high_rate_change_angle_rad, best.result.rate_change_rms_angle_rad),
    )
    excitation_factor = 0.30 + 0.70 * min(1.0, max(0.0, (effective_rank - 1.0) / 1.5))
    sample_factor = min(1.0, (data.sample_count - 1) / 20.0) ** 0.25

    confidence = float(
        np.clip(
            fit_factor
            * uniqueness_factor
            * motion_factor
            * alias_factor
            * integration_factor
            * excitation_factor
            * sample_factor,
            0.0,
            1.0,
        )
    )

    if unresolved:
        confidence = min(confidence, 0.49)
    if best.result.hard_alias_fraction > 0.0:
        confidence = min(confidence, 0.35)
    if (
        step_rms < config.slow_step_threshold_rad
        and total_motion < config.slow_total_motion_threshold_rad
    ):
        confidence = min(confidence, 0.35)

    warnings: list[str] = []
    if len(evidence_near_ties) > 1:
        warnings.append(
            "Multiple convention hypotheses are statistically indistinguishable from "
            "the trajectory evidence at the configured threshold."
        )
    if len(evidence_near_ties) > len(posterior_near_ties):
        warnings.append(
            "Soft priors select among evidence-near-tied hypotheses; the underlying "
            "empirical ambiguity remains."
        )
    if best.result.data_rank != 1:
        warnings.append(
            "Soft priors changed the top-ranked convention from the data-only winner."
        )
    if step_rms < config.slow_step_threshold_rad:
        warnings.append(
            "Per-sample attitude motion is small, so convention evidence is noise-sensitive."
        )
    if total_motion < config.slow_total_motion_threshold_rad:
        warnings.append("The trajectory contains little total rotational excitation.")
    if effective_rank < 1.35:
        warnings.append(
            "Motion is nearly coaxial; frame and unused-axis permutations may be "
            "fundamentally unidentifiable."
        )
    elif effective_rank < 2.0:
        warnings.append("Rotational excitation does not strongly cover all three axes.")
    if best.result.hard_alias_fraction > 0.0:
        warnings.append(
            "One or more rate integrals exceed pi radians per sample; orientation "
            "samples cannot uniquely reveal wrap count."
        )
    elif best.result.near_alias_fraction > 0.0:
        warnings.append("One or more intervals approach the pi-radian alias boundary.")
    if best.result.rate_change_rms_angle_rad >= config.high_rate_change_angle_rad:
        warnings.append(
            "Within-step rate variation is large relative to the sampling interval."
        )
    if best.result.normalized_rmse > 0.50:
        warnings.append(
            "Even the best convention has a large residual relative to the observed motion."
        )
    if data.time_s is None and config.sample_period_s is None:
        warnings.append(
            "Sample period was inferred separately for each hypothesis; absolute rate "
            "scale and time scale are not independently identifiable."
        )
    if best.result.gimbal_lock_fraction is not None and best.result.gimbal_lock_fraction > 0.0:
        warnings.append("The winning Euler hypothesis passes near gimbal lock.")

    time_source = (
        "timestamps"
        if data.time_s is not None
        else "provided_sample_period"
        if config.sample_period_s is not None
        else "inferred_sample_period"
    )
    return OverallDiagnostics(
        confidence=confidence,
        confidence_label=_confidence_label(confidence),
        best_relative_support=best.result.relative_support,
        best_data_relative_support=best.result.data_relative_support,
        near_tie_count=len(evidence_near_ties),
        near_tie_hypothesis_ids=tuple(
            item.result.hypothesis.identifier()
            for item in evidence_near_ties[: config.top_k]
        ),
        unresolved_dimensions=unresolved,
        posterior_near_tie_count=len(posterior_near_ties),
        posterior_near_tie_hypothesis_ids=tuple(
            item.result.hypothesis.identifier()
            for item in posterior_near_ties[: config.top_k]
        ),
        posterior_unresolved_dimensions=posterior_unresolved,
        dimension_assessments=_dimension_assessments(
            data.representation,
            best,
            evidence_near_ties,
            posterior_near_ties,
            config,
        ),
        effective_excitation_rank=effective_rank,
        excitation_singular_values=singular_values,
        total_observed_motion_rad=total_motion,
        observed_step_rms_rad=step_rms,
        near_alias_fraction=best.result.near_alias_fraction,
        hard_alias_fraction=best.result.hard_alias_fraction,
        time_source=time_source,
        warnings=tuple(warnings),
    )


####


def _infer_trajectory_conventions(
    data: TrajectoryData,
    config: InferenceConfig | None = None,
) -> InferenceReport:
    resolved_config = config or InferenceConfig()
    hypotheses = generate_hypotheses(data.representation, resolved_config)
    scored: list[_ScoredCandidate] = []
    failures: list[str] = []

    for hypothesis in hypotheses:
        try:
            scored.append(_score_hypothesis(data, hypothesis, resolved_config))
        except (ValueError, FloatingPointError) as error:
            failures.append(f"{hypothesis.identifier()}: {error}")

    if not scored:
        details = "\n".join(failures[:10])
        raise ValueError(f"All hypotheses failed to score.\n{details}")

    ranked = _assign_ranks_and_support(scored, resolved_config)
    diagnostics = _build_overall_diagnostics(data, ranked, resolved_config)
    rankings = tuple(item.result for item in ranked)
    hypothesis_space = _hypothesis_space_summary(data.representation, resolved_config)
    return InferenceReport(
        representation=data.representation,
        sample_count=data.sample_count,
        hypothesis_count=len(hypotheses),
        hypothesis_space=hypothesis_space,
        best=rankings[0],
        rankings=rankings,
        diagnostics=diagnostics,
    )


####


def best_interval_diagnostics(
    data: TrajectoryData,
    report: InferenceReport,
    config: InferenceConfig | None = None,
) -> tuple[dict[str, float | int | bool], ...]:
    resolved_config = config or InferenceConfig()
    scored = _score_hypothesis(data, report.best.hypothesis, resolved_config)
    predicted_quaternions = scored.predicted_intervals.as_quat()
    observed_quaternions = scored.observed_intervals.as_quat()
    errors = scored.interval_error_rotvecs
    rows: list[dict[str, float | int | bool]] = []

    for index in range(data.sample_count - 1):
        row: dict[str, float | int | bool] = {
            "interval": index,
            "dt_s": float(scored.interval_durations_s[index]),
            "omega_x_rad_s": float(
                0.5 * (scored.rates_rad_s[index, 0] + scored.rates_rad_s[index + 1, 0])
            ),
            "omega_y_rad_s": float(
                0.5 * (scored.rates_rad_s[index, 1] + scored.rates_rad_s[index + 1, 1])
            ),
            "omega_z_rad_s": float(
                0.5 * (scored.rates_rad_s[index, 2] + scored.rates_rad_s[index + 1, 2])
            ),
            "observed_delta_qx": float(observed_quaternions[index, 0]),
            "observed_delta_qy": float(observed_quaternions[index, 1]),
            "observed_delta_qz": float(observed_quaternions[index, 2]),
            "observed_delta_qw": float(observed_quaternions[index, 3]),
            "predicted_delta_qx": float(predicted_quaternions[index, 0]),
            "predicted_delta_qy": float(predicted_quaternions[index, 1]),
            "predicted_delta_qz": float(predicted_quaternions[index, 2]),
            "predicted_delta_qw": float(predicted_quaternions[index, 3]),
            "error_rotvec_x_rad": float(errors[index, 0]),
            "error_rotvec_y_rad": float(errors[index, 1]),
            "error_rotvec_z_rad": float(errors[index, 2]),
            "error_angle_rad": float(np.linalg.norm(errors[index])),
            "unwrapped_rate_angle_rad": float(scored.path_angles[index]),
            "near_alias": bool(scored.path_angles[index] >= resolved_config.near_alias_step_rad),
            "hard_alias": bool(scored.path_angles[index] >= resolved_config.hard_alias_step_rad),
        }
        if data.time_s is not None:
            row["time_start_s"] = float(data.time_s[index])
            row["time_end_s"] = float(data.time_s[index + 1])
        rows.append(row)
    return tuple(rows)


####
