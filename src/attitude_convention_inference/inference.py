"""Evidence-preserving scoring, ranking, policy selection, and report assembly."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import exp, isclose, log, sqrt
from typing import Any

from numpy import (
    all as np_all,
)
from numpy import (
    amax,
    amin,
    argmax,
    asarray,
    clip,
    diff,
    float64,
    intp,
    mean,
    median,
    ones,
    quantile,
    square,
    zeros,
)
from numpy import (
    isfinite as np_isfinite,
)
from numpy import (
    sqrt as np_sqrt,
)
from numpy import (
    sum as np_sum,
)
from numpy.linalg import norm
from numpy.typing import NDArray
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
from .heuristics import euler_heuristic_diagnostics as _euler_heuristic_diagnostics
from .heuristics import gimbal_lock_fraction as _gimbal_lock_fraction
from .heuristics import wrap_unit_diagnostics as _wrap_unit_diagnostics
from .hypotheses import DIMENSIONS as _DIMENSIONS
from .hypotheses import generate_hypotheses
from .hypotheses import hypothesis_count as _hypothesis_count
from .hypotheses import relevant_dimensions as _relevant_dimensions
from .hypotheses import search_values as _search_values
from .math_utils import rotation_magnitudes as _rotation_magnitudes
from .math_utils import rotation_quaternions as _rotation_quaternions
from .math_utils import rotation_rotvecs as _rotation_rotvecs
from .math_utils import support_weights as _support_weights
from .math_utils import trimmed_mean_square as _trimmed_mean_square
from .models import (
    AttitudeDirection,
    AttitudeRepresentation,
    CandidateResult,
    ConfidenceLabel,
    ConventionHypothesis,
    DimensionAssessment,
    DimensionStatus,
    HeuristicFeatures,
    HypothesisSpaceSummary,
    InferenceConfig,
    InferenceReport,
    LagMetric,
    OverallDiagnostics,
    QuaternionOrder,
    RateFrame,
    RateUnit,
    ResultLabel,
    ScaleRisk,
)
from .statistics import effective_excitation_rank as _effective_excitation_rank
from .statistics import estimate_noise_process
from .statistics import fixed_noise_log_likelihood as _fixed_noise_log_likelihood
from .statistics import raw_rate_excitation as _raw_rate_excitation

__all__ = [
    "best_interval_diagnostics",
    "estimate_noise_process",
    "generate_hypotheses",
    "infer_trajectory_conventions",
]


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




@dataclass(frozen=True, slots=True)
class _EvidenceSummary:
    """Statistical evidence and sufficiency metrics for a candidate hypothesis."""

    noise_sigma_rad: float
    noise_residual_count: int
    noise_lag1_autocorrelation: float
    sufficient: bool
    intervals_used: int
    fraction: float
    log_likelihood_ratio: float
####




@dataclass(frozen=True, slots=True)
class _OperationalSelection:
    """Operational class and hypothesis selection with policy explanations."""

    class_id: str
    hypothesis_id: str
    reason: str
    domain_prior_abstained: bool
    domain_prior_supported_class_ids: tuple[str, ...]
    preference_profile_abstained: bool
    preference_profile_supported_class_ids: tuple[str, ...]
    representative_reason: str
####




def _score_hypothesis(
    data: TrajectoryData,
    hypothesis: ConventionHypothesis,
    config: InferenceConfig,
    *,
    attitudes: Rotation | None = None,
    rates_rad_s: FloatArray | None = None,
) -> _ScoredCandidate:
    attitudes = decode_attitudes(data, hypothesis) if attitudes is None else attitudes
    rates_rad_s = decode_rates(data, hypothesis) if rates_rad_s is None else rates_rad_s
    frame = RateFrame(hypothesis.rate_frame)
    duration_resolution = resolve_interval_durations(
        data,
        attitudes,
        rates_rad_s,
        frame,
        config,
    )
    durations = duration_resolution.durations_s
    inferred_period = duration_resolution.inferred_period_s
    time_warnings = duration_resolution.warnings
    integrated = integrate_interval_increments(
        rates_rad_s,
        duration_resolution.durations_s,
        frame,
        config.integration_substeps,
    )
    predicted = integrated.predicted_increments
    path_angles = integrated.path_angles_rad
    rate_change_angles = integrated.rate_change_angles_rad
    observed = observed_interval_increments(attitudes, frame)
    interval_errors = _rotation_rotvecs(predicted.inv() * observed)
    interval_error_angles = norm(interval_errors, axis=1)
    observed_interval_angles = _rotation_magnitudes(observed)

    valid_windows = tuple(
        window for window in config.window_sizes if window <= data.sample_count - 1
    )
    lag_metrics: list[LagMetric] = []
    scale_risks: list[ScaleRisk] = []

    for window_size in valid_windows:
        composition = compose_increment_windows(
            predicted,
            window_size,
            frame,
        )
        observed_windows = observed_window_increments(
            attitudes,
            composition.starts,
            window_size,
            frame,
        )
        errors = _rotation_magnitudes(composition.window_increments.inv() * observed_windows)
        observed_angles = _rotation_magnitudes(observed_windows)
        scale = sqrt(float(window_size))
        scaled_errors = asarray(errors, dtype=float64) / scale
        scale_risks.append(
            ScaleRisk(
                window_size=window_size,
                trimmed_mse_rad2=_trimmed_mean_square(scaled_errors, config.trim_fraction),
                untrimmed_mse_rad2=float(mean(square(scaled_errors))),
                weight=1.0 / len(valid_windows),
            )
        )
        lag_metrics.append(
            LagMetric(
                window_size=window_size,
                window_count=int(errors.size),
                residual_rmse_rad=float(np_sqrt(mean(square(errors)))),
                residual_median_rad=float(median(errors)),
                residual_p95_rad=float(quantile(errors, 0.95)),
                observed_rotation_rms_rad=float(np_sqrt(mean(square(observed_angles)))),
            )
        )
    ####

    raw_geometric_risk = float(sum(item.weight * item.trimmed_mse_rad2 for item in scale_risks))
    untrimmed_geometric_risk = float(
        sum(item.weight * item.untrimmed_mse_rad2 for item in scale_risks)
    )
    multiscale_rmse = sqrt(raw_geometric_risk)
    # The criterion is only a monotone contrast transform for tie thresholds and
    # support display. Candidate ordering is the raw geometric risk above.
    data_criterion = log(max(raw_geometric_risk, config.residual_floor_rad**2))

    near_alias_fraction = float(mean(path_angles >= config.near_alias_step_rad))
    hard_alias_fraction = float(mean(path_angles >= config.hard_alias_step_rad))
    rate_change_rms = float(np_sqrt(mean(square(rate_change_angles))))
    warnings = list(time_warnings)
    if hard_alias_fraction > 0.0:
        warnings.append(
            "At least one sample interval contains an unwrapped rate integral of pi "
            "radians or more."
        )
    elif near_alias_fraction > 0.0:
        warnings.append("At least one sample interval is close to rotational aliasing.")
    ####
    if rate_change_rms >= config.high_rate_change_angle_rad:
        warnings.append(
            "Angular rate changes substantially within a sample interval; interpolation "
            "error may matter."
        )
    ####

    gimbal_fraction = _gimbal_lock_fraction(data, hypothesis, config)
    euler_diagnostics = _euler_heuristic_diagnostics(data, hypothesis)
    euler_range_penalty = euler_diagnostics.range_penalty_rad2
    euler_branch_switches = euler_diagnostics.branch_switch_count
    euler_singularity_margin = euler_diagnostics.singularity_margin
    euler_branch_cost = euler_diagnostics.branch_cost_rad2
    wrap_diagnostics = _wrap_unit_diagnostics(data, hypothesis)
    wrap_gain = wrap_diagnostics.gain_raw
    wrap_count = wrap_diagnostics.wrap_count
    wrap_status = wrap_diagnostics.status
    wrap_score = wrap_diagnostics.score
    if gimbal_fraction is not None and gimbal_fraction > 0.0:
        warnings.append("Some Euler samples are near this hypothesis's gimbal-lock surface.")
    ####
    if euler_range_penalty > 0.0:
        warnings.append(
            "Some Euler samples fall outside this hypothesis's principal output ranges; "
            "the represented rotations may still be valid under an unwrapped branch."
        )
    ####
    heuristic_features = HeuristicFeatures(
        branch_penalty=euler_range_penalty,
        singularity_penalty=(
            0.0 if euler_singularity_margin is None else max(0.0, 1.0 - euler_singularity_margin)
        ),
        coarse_risk=0.0,
        wrap_score=wrap_score,
        wrap_gain_raw=wrap_gain,
        inferred_wrap_count=wrap_count,
        wrap_unit_status=wrap_status,
        representation_score=float(wrap_score + euler_range_penalty + euler_branch_cost),
        representation_feature_vector={
            "wrap_score": float(wrap_score),
            "wrap_gain_raw": float(wrap_gain),
            "range_penalty_rad2": float(euler_range_penalty),
            "branch_cost_rad2": float(euler_branch_cost),
            "singularity_margin": (
                0.0 if euler_singularity_margin is None else float(euler_singularity_margin)
            ),
        },
        classifier_probability=None,
    )

    result = CandidateResult(
        hypothesis=hypothesis,
        criterion=float(data_criterion),
        data_criterion=float(data_criterion),
        fit_score=0.0,
        normalized_rmse=float(multiscale_rmse),
        multiscale_rmse_rad=float(multiscale_rmse),
        raw_geometric_risk=raw_geometric_risk,
        untrimmed_geometric_risk=untrimmed_geometric_risk,
        scale_risks=tuple(scale_risks),
        interval_rmse_rad=float(np_sqrt(mean(square(interval_error_angles)))),
        interval_median_rad=float(median(interval_error_angles)),
        interval_p95_rad=float(quantile(interval_error_angles, 0.95)),
        interval_max_rad=float(amax(interval_error_angles)),
        observed_step_rms_rad=float(np_sqrt(mean(square(observed_interval_angles)))),
        inferred_sample_period_s=inferred_period,
        valid_interval_count=data.sample_count - 1,
        near_alias_fraction=near_alias_fraction,
        hard_alias_fraction=hard_alias_fraction,
        rate_change_rms_angle_rad=rate_change_rms,
        gimbal_lock_fraction=gimbal_fraction,
        euler_range_penalty_rad2=euler_range_penalty,
        euler_branch_switch_count=euler_branch_switches,
        euler_singularity_margin=euler_singularity_margin,
        euler_branch_cost_rad2=euler_branch_cost,
        heuristic_features=heuristic_features,
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
        interval_error_rotvecs=asarray(interval_errors, dtype=float64),
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
        ####
        weights = getattr(config.priors, dimension)
        total += log(float(weights.get(value, 1.0)))
    ####
    return total
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




def _rate_unit_partner(hypothesis: ConventionHypothesis) -> ConventionHypothesis:
    rate_unit = (
        RateUnit.DEGREES_PER_SECOND
        if hypothesis.rate_unit is RateUnit.RADIANS_PER_SECOND
        else RateUnit.RADIANS_PER_SECOND
    )
    return hypothesis.model_copy(update={"rate_unit": rate_unit})
####




def _canonical_key(hypothesis: ConventionHypothesis) -> tuple[str, ...]:
    """Return the stable minimum-intervention key used only after evidence ties."""

    return (
        "0" if hypothesis.attitude_direction is AttitudeDirection.BODY_TO_WORLD else "1",
        "0" if hypothesis.rate_frame is RateFrame.BODY else "1",
        "0" if hypothesis.rate_sign == 1 else "1",
        "0" if hypothesis.rate_unit is RateUnit.RADIANS_PER_SECOND else "1",
        "0" if hypothesis.rate_order == "pqr" else "1",
        "0"
        if hypothesis.quaternion_order is QuaternionOrder.SCALAR_LAST
        else "1"
        if hypothesis.quaternion_order is QuaternionOrder.SCALAR_FIRST
        else "0",
        hypothesis.identifier(),
    )
####




def _period_is_active_bound(result: CandidateResult, config: InferenceConfig) -> bool:
    if result.inferred_sample_period_s is None:
        return False
    ####
    lower, upper = config.inferred_sample_period_bounds_s
    tolerance = max(1.0e-12, (upper - lower) * 1.0e-6)
    return (
        abs(result.inferred_sample_period_s - lower) <= tolerance
        or abs(result.inferred_sample_period_s - upper) <= tolerance
    )
####




def _annotate_exact_equivalences(
    scored: Iterable[_ScoredCandidate],
    data: TrajectoryData,
    config: InferenceConfig,
) -> tuple[_ScoredCandidate, ...]:
    """Collapse exact observational symmetries before numeric ranking.

    The direction--frame--global-sign dual is exact for every trajectory.  Missing
    timestamps add the rate-unit/sample-period scale gauge unless a configured period
    bound is active.  Members receive one shared raw risk so floating-point noise cannot
    manufacture a winner inside either class.
    """

    candidates = tuple(scored)
    identifiers = {
        item.result.hypothesis.identifier(): index for index, item in enumerate(candidates)
    }
    neighbours = [set([index]) for index in range(len(candidates))]
    scale_gauge = data.time_s is None and config.sample_period_s is None

    for index, item in enumerate(candidates):
        partner = item.result.hypothesis.structural_dual().identifier()
        if partner in identifiers:
            neighbours[index].add(identifiers[partner])
        ####
        if scale_gauge and not _period_is_active_bound(item.result, config):
            unit_partner = _rate_unit_partner(item.result.hypothesis).identifier()
            partner_index = identifiers.get(unit_partner)
            if partner_index is not None and not _period_is_active_bound(
                candidates[partner_index].result, config
            ):
                neighbours[index].add(partner_index)
            ####
        ####
    ####

    groups: list[tuple[int, ...]] = []
    unseen = set(range(len(candidates)))
    while unseen:
        pending = [unseen.pop()]
        group: set[int] = set()
        while pending:
            index = pending.pop()
            if index in group:
                continue
            ####
            group.add(index)
            linked = neighbours[index]
            for other_index, other_neighbours in enumerate(neighbours):
                if index in other_neighbours:
                    linked = linked | {other_index}
                ####
            ####
            for other_index in linked:
                if other_index in unseen:
                    unseen.remove(other_index)
                    pending.append(other_index)
                ####
            ####
        ####
        groups.append(tuple(sorted(group)))
    ####

    updated = list(candidates)
    for group_indices in groups:
        members = [candidates[index] for index in group_indices]
        identifiers_in_group = tuple(
            sorted(item.result.hypothesis.identifier() for item in members)
        )
        class_id = identifiers_in_group[0]
        shared_risk = float(mean([item.result.raw_geometric_risk for item in members]))
        shared_untrimmed_risk = float(
            mean([item.result.untrimmed_geometric_risk for item in members])
        )
        shared_rmse = sqrt(shared_risk)
        shared_criterion = log(max(shared_risk, config.residual_floor_rad**2))
        for index in group_indices:
            result = candidates[index].result.model_copy(
                update={
                    "raw_geometric_risk": shared_risk,
                    "untrimmed_geometric_risk": shared_untrimmed_risk,
                    "multiscale_rmse_rad": shared_rmse,
                    "data_criterion": shared_criterion,
                    "equivalence_class_id": class_id,
                    "exact_equivalent_hypothesis_ids": identifiers_in_group,
                }
            )
            updated[index] = _replace_result(candidates[index], result)
        ####
    ####
    return tuple(updated)
####




def _result_label(data: TrajectoryData, config: InferenceConfig) -> ResultLabel:
    if data.orientation is None and data.rates is None:
        return ResultLabel.NO_DATA
    ####
    if data.rates is None:
        return ResultLabel.ATTITUDE_HEURISTIC
    ####
    if data.orientation is None:
        return ResultLabel.RATE_NON_ID
    ####
    if data.time_s is None and config.sample_period_s is None:
        return ResultLabel.SCALE_GAUGE
    ####
    return ResultLabel.TRAJECTORY_QUOTIENT
####




def _evidence_contender_classes(
    ranked: tuple[_ScoredCandidate, ...],
    config: InferenceConfig,
) -> tuple[str, ...]:
    """Return distinct quotient classes within the configured data-risk margin."""

    representatives: dict[str, _ScoredCandidate] = {}
    for item in ranked:
        representatives.setdefault(item.result.equivalence_class_id, item)
    ####
    ordered = sorted(
        representatives.values(),
        key=lambda item: (
            item.result.class_data_risk,
            item.result.equivalence_class_id,
            _canonical_key(item.result.hypothesis),
        ),
    )
    if not ordered:
        return ()
    ####
    minimum = max(ordered[0].result.class_data_risk, config.residual_floor_rad**2)
    minimum_criterion = log(minimum)
    return tuple(
        item.result.equivalence_class_id
        for item in ordered
        if log(max(item.result.class_data_risk, config.residual_floor_rad**2)) - minimum_criterion
        <= config.near_tie_delta_criterion
    )
####




def _select_operational_result(
    ranked: tuple[_ScoredCandidate, ...],
    config: InferenceConfig,
) -> _OperationalSelection:
    """Select an operational class only after preserving data contenders."""

    contender_ids = _evidence_contender_classes(ranked, config)
    by_class: dict[str, list[_ScoredCandidate]] = {}
    for item in ranked:
        by_class.setdefault(item.result.equivalence_class_id, []).append(item)
    ####
    domain_prior = config.domain_prior
    profile = config.preference_profile
    domain_supported = tuple(
        class_id
        for class_id in contender_ids
        if float(domain_prior.class_weights.get(class_id, 0.0)) > 0.0
    )
    profile_supported = tuple(
        class_id
        for class_id in contender_ids
        if float(profile.class_weights.get(class_id, 0.0)) > 0.0
    )
    effective_class_scores: dict[str, float] = {}
    for class_id in contender_ids:
        if class_id in profile.class_weights:
            effective_weight = float(profile.class_weights[class_id])
        elif class_id in domain_prior.class_weights:
            effective_weight = float(domain_prior.class_weights[class_id])
        else:
            effective_weight = 0.0
        ####
        if effective_weight <= 0.0:
            continue
        ####
        score = 0.0
        if class_id in domain_prior.class_weights:
            domain_weight = float(domain_prior.class_weights[class_id])
            if domain_weight > 0.0:
                score += float(domain_prior.strength) * log(domain_weight)
            ####
        ####
        if class_id in profile.class_weights:
            profile_weight = float(profile.class_weights[class_id])
            if profile_weight > 0.0:
                score += float(profile.strength) * log(profile_weight)
            ####
        ####
        effective_class_scores[class_id] = score
    ####
    supported = tuple(class_id for class_id in contender_ids if class_id in effective_class_scores)
    abstained = False
    profile_abstained = False
    if len(contender_ids) == 1:
        selected_class = contender_ids[0]
        selected_members = by_class[selected_class]
        reason = (
            "exact-gauge-class"
            if len(selected_members[0].result.exact_equivalent_hypothesis_ids) > 1
            else "trajectory-unique"
        )
    elif supported:
        selected_class = max(
            supported,
            key=lambda class_id: (
                effective_class_scores[class_id],
                class_id,
            ),
        )
        reason = "preference-profile" if profile.class_configured() else "domain-prior"
        selected_members = by_class[selected_class]
    else:
        selected_class = (
            contender_ids[0] if contender_ids else ranked[0].result.equivalence_class_id
        )
        reason = "canonical-policy"
        abstained = domain_prior.configured()
        profile_abstained = profile.class_configured()
        selected_members = by_class[selected_class]
    ####
    canonical = min(selected_members, key=lambda item: _canonical_key(item.result.hypothesis))
    operational = canonical
    representative_reason = "canonical-policy"
    if profile.representative_configured():

        def preference_log_weight(item: _ScoredCandidate) -> float:
            total = 0.0
            hypothesis = item.result.hypothesis
            for dimension in _DIMENSIONS:
                value = _hypothesis_value(hypothesis, dimension)
                if value is None:
                    continue
                ####
                weights = getattr(profile.representative_preferences, dimension)
                total += float(profile.strength) * log(float(weights.get(value, 1.0)))
            ####
            return total
        ####


        preference_scores = {
            item.result.hypothesis.identifier(): preference_log_weight(item)
            for item in selected_members
        }
        maximum = max(preference_scores.values())
        preferred_members = [
            item
            for item in selected_members
            if isclose(
                preference_scores[item.result.hypothesis.identifier()],
                maximum,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ]
        operational = min(
            preferred_members,
            key=lambda item: _canonical_key(item.result.hypothesis),
        )
        if operational.result.hypothesis.identifier() != canonical.result.hypothesis.identifier():
            representative_reason = "preference-profile"
        ####
    ####
    return _OperationalSelection(
        class_id=selected_class,
        hypothesis_id=operational.result.hypothesis.identifier(),
        reason=reason,
        domain_prior_abstained=abstained,
        domain_prior_supported_class_ids=domain_supported,
        preference_profile_abstained=profile_abstained,
        preference_profile_supported_class_ids=profile_supported,
        representative_reason=representative_reason,
    )
####




def _score_partial_hypothesis(
    data: TrajectoryData,
    hypothesis: ConventionHypothesis,
    config: InferenceConfig,
) -> _ScoredCandidate:
    """Create an explicit heuristic/non-identifying result for partial streams."""

    sample_count = data.sample_count
    interval_count = sample_count - 1
    if data.orientation is not None:
        attitudes = decode_attitudes(data, hypothesis)
        euler_diagnostics = _euler_heuristic_diagnostics(data, hypothesis)
        euler_range_penalty = euler_diagnostics.range_penalty_rad2
        euler_branch_switches = euler_diagnostics.branch_switch_count
        euler_singularity_margin = euler_diagnostics.singularity_margin
        euler_branch_cost = euler_diagnostics.branch_cost_rad2
        wrap_diagnostics = _wrap_unit_diagnostics(data, hypothesis)
        wrap_gain = wrap_diagnostics.gain_raw
        wrap_count = wrap_diagnostics.wrap_count
        wrap_status = wrap_diagnostics.status
        wrap_score = wrap_diagnostics.score
        representation_score = wrap_score + euler_range_penalty + euler_branch_cost
        rates = zeros((sample_count, 3), dtype=float64)
        gimbal_fraction = _gimbal_lock_fraction(data, hypothesis, config)
        heuristic_features = HeuristicFeatures(
            wrap_score=wrap_score,
            wrap_gain_raw=wrap_gain,
            inferred_wrap_count=wrap_count,
            wrap_unit_status=wrap_status,
            branch_penalty=euler_range_penalty,
            singularity_penalty=(
                0.0
                if euler_singularity_margin is None
                else max(0.0, 1.0 - euler_singularity_margin)
            ),
            representation_score=float(representation_score),
            representation_feature_vector={
                "wrap_score": float(wrap_score),
                "range_penalty_rad2": float(euler_range_penalty),
                "branch_cost_rad2": float(euler_branch_cost),
            },
        )
    else:
        attitudes = Rotation.identity(sample_count)
        rates = decode_rates(data, hypothesis)
        euler_range_penalty = 0.0
        euler_branch_switches = 0
        euler_singularity_margin = None
        euler_branch_cost = 0.0
        gimbal_fraction = None
        representation_score = 0.0
        heuristic_features = HeuristicFeatures()
    ####

    durations = (
        diff(data.time_s) if data.time_s is not None else ones(interval_count, dtype=float64)
    )
    predicted = Rotation.identity(interval_count)
    observed = Rotation.identity(interval_count)
    errors = zeros((interval_count, 3), dtype=float64)
    score = max(float(representation_score), 0.0)
    result = CandidateResult(
        hypothesis=hypothesis,
        criterion=log(max(score, config.residual_floor_rad**2)),
        data_criterion=log(max(score, config.residual_floor_rad**2)),
        fit_score=0.0,
        normalized_rmse=0.0,
        multiscale_rmse_rad=sqrt(score),
        raw_geometric_risk=score,
        untrimmed_geometric_risk=score,
        scale_risks=(),
        interval_rmse_rad=0.0,
        interval_median_rad=0.0,
        interval_p95_rad=0.0,
        interval_max_rad=0.0,
        observed_step_rms_rad=0.0,
        valid_interval_count=interval_count,
        near_alias_fraction=0.0,
        hard_alias_fraction=0.0,
        rate_change_rms_angle_rad=0.0,
        lag_metrics=(),
        inferred_sample_period_s=None,
        gimbal_lock_fraction=gimbal_fraction,
        euler_range_penalty_rad2=euler_range_penalty,
        euler_branch_switch_count=euler_branch_switches,
        euler_singularity_margin=euler_singularity_margin,
        euler_branch_cost_rad2=euler_branch_cost,
        heuristic_features=heuristic_features,
    )
    return _ScoredCandidate(
        result=result,
        attitudes=attitudes,
        rates_rad_s=rates,
        interval_durations_s=durations,
        predicted_intervals=predicted,
        observed_intervals=observed,
        interval_error_rotvecs=errors,
        path_angles=zeros(interval_count, dtype=float64),
    )
####




def _ranks_with_exact_ties(order: NDArray[Any], values: NDArray[Any]) -> dict[int, int]:
    ranks: dict[int, int] = {}
    previous: float | None = None
    current_rank = 0
    for position, candidate_index_raw in enumerate(order, start=1):
        candidate_index = int(candidate_index_raw)
        value = float(values[candidate_index])
        if previous is None or value != previous:
            current_rank = position
            previous = value
        ####
        ranks[candidate_index] = current_rank
    ####
    return ranks
####




def _assign_ranks_and_support(
    scored: Iterable[_ScoredCandidate],
    data: TrajectoryData,
    config: InferenceConfig,
) -> tuple[_ScoredCandidate, ...]:
    candidates = _annotate_exact_equivalences(scored, data, config)
    if not candidates:
        raise ValueError("No convention hypothesis could be scored.")
    ####

    class_risks: dict[str, float] = {}
    class_keys: dict[str, tuple[str, ...]] = {}
    for item in candidates:
        class_id = item.result.equivalence_class_id
        class_risks[class_id] = min(
            class_risks.get(class_id, float("inf")),
            float(item.result.raw_geometric_risk),
        )
        class_keys.setdefault(class_id, _canonical_key(item.result.hypothesis))
    ####
    ordered_classes = sorted(
        class_risks,
        key=lambda class_id: (class_risks[class_id], class_id, class_keys[class_id]),
    )
    distinct_risks = sorted(set(class_risks.values()))
    next_class_risk = {
        class_id: next(
            (risk for risk in distinct_risks if risk > class_risks[class_id]),
            None,
        )
        for class_id in ordered_classes
    }

    def _class_gap(item: _ScoredCandidate) -> float | None:
        next_risk = next_class_risk[item.result.equivalence_class_id]
        if next_risk is None:
            return None
        ####
        return max(0.0, next_risk - class_risks[item.result.equivalence_class_id])
    ####


    candidates = tuple(
        _replace_result(
            item,
            item.result.model_copy(
                update={
                    "class_data_risk": class_risks[item.result.equivalence_class_id],
                    "empirical_gap_to_next_class": _class_gap(item),
                }
            ),
        )
        for item in candidates
    )

    display_scale = max(
        float(median([item.result.observed_step_rms_rad for item in candidates])),
        config.residual_floor_rad,
    )
    candidates = tuple(
        _replace_result(
            item,
            item.result.model_copy(
                update={
                    "normalized_rmse": item.result.multiscale_rmse_rad / display_scale,
                    "fit_score": 100.0
                    * exp(-0.5 * (item.result.multiscale_rmse_rad / display_scale) ** 2),
                }
            ),
        )
        for item in candidates
    )

    data_criteria: FloatArray = asarray(
        [item.result.data_criterion for item in candidates],
        dtype=float64,
    )
    if not np_all(np_isfinite(data_criteria)):
        raise ValueError("No finite convention hypothesis could be scored.")
    ####

    data_supports = _support_weights(data_criteria)
    data_minimum = float(amin(data_criteria))
    raw_risks: FloatArray = asarray(
        [item.result.raw_geometric_risk for item in candidates], dtype=float64
    )

    def _data_sort_key(index: int) -> tuple[float, str, tuple[str, ...]]:
        return (
            float(raw_risks[index]),
            candidates[index].result.equivalence_class_id,
            _canonical_key(candidates[index].result.hypothesis),
        )
    ####


    data_order: NDArray[intp] = asarray(
        sorted(range(len(candidates)), key=_data_sort_key),
        dtype=intp,
    )
    data_rank_by_index = _ranks_with_exact_ties(data_order, raw_risks)

    prior_logs = asarray(
        [_prior_log_weight(item.result.hypothesis, config) for item in candidates],
        dtype=float64,
    )
    maximum_prior_log = float(amax(prior_logs))
    prior_penalties = 2.0 * (maximum_prior_log - prior_logs)
    posterior_criteria: FloatArray = data_criteria + prior_penalties
    posterior_supports = _support_weights(posterior_criteria)
    posterior_minimum = float(amin(posterior_criteria))

    def _posterior_sort_key(index: int) -> tuple[float, str, tuple[str, ...]]:
        return (
            float(posterior_criteria[index]),
            candidates[index].result.equivalence_class_id,
            _canonical_key(candidates[index].result.hypothesis),
        )
    ####


    posterior_order: NDArray[intp] = asarray(
        sorted(range(len(candidates)), key=_posterior_sort_key),
        dtype=intp,
    )

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
                "delta_criterion": float(posterior_criteria[candidate_index] - posterior_minimum),
                "data_delta_criterion": float(data_criteria[candidate_index] - data_minimum),
                "criterion": float(posterior_criteria[candidate_index]),
                "prior_relative_weight": float(
                    exp(prior_logs[candidate_index] - maximum_prior_log)
                ),
                "prior_penalty": float(prior_penalties[candidate_index]),
            }
        )
        ranked.append(_replace_result(item, updated_result))
    ####
    return tuple(ranked)
####




def _evidence_summary(
    ranked: tuple[_ScoredCandidate, ...],
    config: InferenceConfig,
) -> _EvidenceSummary:
    """Estimate noise and find the earliest stable evidence prefix."""

    representatives: dict[str, _ScoredCandidate] = {}
    for item in sorted(ranked, key=lambda candidate: candidate.result.data_rank):
        representatives.setdefault(item.result.equivalence_class_id, item)
    ####
    data_ranked = tuple(representatives.values())
    data_winner = data_ranked[0]
    noise = estimate_noise_process(
        data_winner.interval_error_rotvecs,
        config.trim_fraction,
    )
    interval_count = data_winner.interval_error_rotvecs.shape[0]
    if data_winner.result.observed_step_rms_rad < config.slow_step_threshold_rad:
        return _EvidenceSummary(
            noise.sigma_rad,
            noise.retained_count,
            noise.lag1_autocorrelation,
            False,
            interval_count,
            1.0,
            0.0,
        )
    ####
    if len(data_ranked) < 2:
        return _EvidenceSummary(
            noise.sigma_rad,
            noise.retained_count,
            noise.lag1_autocorrelation,
            False,
            interval_count,
            1.0,
            0.0,
        )
    ####

    final_scores = asarray(
        [
            _fixed_noise_log_likelihood(item.interval_error_rotvecs, noise.sigma_rad)
            for item in data_ranked
        ],
        dtype=float64,
    )
    final_delta = float(final_scores[0] - amax(final_scores[1:]))
    minimum = min(config.evidence_min_intervals, interval_count)
    stable_count = 0
    earliest = interval_count
    sufficient = False
    for prefix in range(minimum, interval_count + 1):
        prefix_scores = asarray(
            [
                _fixed_noise_log_likelihood(item.interval_error_rotvecs[:prefix], noise.sigma_rad)
                for item in data_ranked
            ],
            dtype=float64,
        )
        winner_index = int(argmax(prefix_scores))
        winner_delta = float(prefix_scores[0] - amax(prefix_scores[1:]))
        if winner_index == 0 and winner_delta >= config.evidence_log_likelihood_ratio_threshold:
            stable_count += 1
        else:
            stable_count = 0
        ####
        if stable_count >= config.evidence_stability_windows:
            sufficient = True
            earliest = prefix
            break
        ####
    ####
    fraction = float(earliest / interval_count) if interval_count else 1.0
    return _EvidenceSummary(
        noise.sigma_rad,
        noise.retained_count,
        noise.lag1_autocorrelation,
        sufficient,
        earliest,
        fraction,
        final_delta,
    )
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
        ####
    ####
    return {key: tuple(sorted(values)) for key, values in gathered.items() if len(values) > 1}
####




def _relative_prior_weights(
    dimension: str,
    allowed_values: tuple[object, ...],
    config: InferenceConfig,
) -> dict[str, float]:
    configured = getattr(config.priors, dimension)
    if not configured:
        return {}
    ####
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
        ####
        values = _search_values(constrained, dimension)
        string_values = tuple(str(value) for value in values)
        if len(values) == 1:
            fixed[dimension] = string_values[0]
        else:
            searched[dimension] = string_values
        ####
        relative = _relative_prior_weights(dimension, values, config)
        if relative:
            prior_weights[dimension] = relative
        ####
    ####

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
        ####

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
        ####

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
    ####
    return assessments
####




def _confidence_label(value: float) -> ConfidenceLabel:
    if value >= 0.75:
        return ConfidenceLabel.HIGH
    ####
    if value >= 0.50:
        return ConfidenceLabel.MODERATE
    ####
    if value >= 0.20:
        return ConfidenceLabel.LOW
    ####
    return ConfidenceLabel.INDETERMINATE
####




def _build_overall_diagnostics(
    data: TrajectoryData,
    ranked: tuple[_ScoredCandidate, ...],
    config: InferenceConfig,
) -> OverallDiagnostics:
    best = ranked[0]
    equivalence_classes = {
        item.result.equivalence_class_id: item.result.exact_equivalent_hypothesis_ids
        for item in ranked
    }
    rate_scale_period_gauge_active = (
        data.time_s is None
        and config.sample_period_s is None
        and len(config.resolved_search_space().rate_units) > 1
    )
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
        item for item in ranked if item.result.delta_criterion <= config.near_tie_delta_criterion
    )
    unresolved = _unresolved_dimensions(evidence_near_ties)
    posterior_unresolved = _unresolved_dimensions(posterior_near_ties)
    observed_angles = _rotation_magnitudes(best.observed_intervals)
    total_motion = float(np_sum(observed_angles))
    step_rms = float(np_sqrt(mean(square(observed_angles))))
    excitation = _effective_excitation_rank(
        best.rates_rad_s,
        best.interval_durations_s,
    )
    raw_excitation = _raw_rate_excitation(
        data,
        config,
    )
    evidence = _evidence_summary(ranked, config)
    effective_rank = excitation.effective_rank
    singular_values = excitation.singular_values
    raw_excitation_rank = raw_excitation.rank
    raw_excitation_eigenvalues = raw_excitation.eigenvalues
    raw_excitation_ratio = raw_excitation.weakest_axis_ratio
    noise_sigma = evidence.noise_sigma_rad
    noise_residual_count = evidence.noise_residual_count
    noise_autocorrelation = evidence.noise_lag1_autocorrelation
    evidence_sufficient = evidence.sufficient
    evidence_intervals_used = evidence.intervals_used
    evidence_fraction = evidence.fraction
    evidence_log_likelihood_ratio = evidence.log_likelihood_ratio

    fit_factor = exp(-0.5 * (best.result.normalized_rmse / 0.25) ** 2)
    uniqueness_factor = min(1.0, best.result.data_relative_support / 0.90)
    step_factor = min(1.0, step_rms / config.slow_step_threshold_rad)
    total_factor = min(1.0, total_motion / config.slow_total_motion_threshold_rad)
    motion_factor = sqrt(step_factor * total_factor)
    alias_factor = max(
        0.02,
        1.0
        - best.result.hard_alias_fraction
        - 0.5 * (best.result.near_alias_fraction - best.result.hard_alias_fraction),
    )
    integration_factor = min(
        1.0,
        config.high_rate_change_angle_rad
        / max(config.high_rate_change_angle_rad, best.result.rate_change_rms_angle_rad),
    )
    excitation_factor = 0.30 + 0.70 * min(1.0, max(0.0, (effective_rank - 1.0) / 1.5))
    sample_factor = min(1.0, (data.sample_count - 1) / 20.0) ** 0.25

    confidence = float(
        clip(
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
    ####
    if len(ranked) > 1 and not evidence_sufficient:
        confidence = min(confidence, 0.49)
    ####
    if best.result.hard_alias_fraction > 0.0:
        confidence = min(confidence, 0.35)
    ####
    if (
        step_rms < config.slow_step_threshold_rad
        and total_motion < config.slow_total_motion_threshold_rad
    ):
        confidence = min(confidence, 0.35)
    ####

    warnings: list[str] = []
    if len(best.result.exact_equivalent_hypothesis_ids) > 1:
        warnings.append(
            "The selected result is an exact observational equivalence class; attitude "
            "direction, rate frame, and global rate sign cannot be separated by this "
            "trajectory alone."
        )
    ####
    if len(evidence_near_ties) > 1:
        warnings.append(
            "Multiple convention hypotheses remain within the configured empirical evidence margin."
        )
    ####
    if len(ranked) > 1 and not evidence_sufficient:
        warnings.append(
            "The leading convention did not achieve a stable evidence threshold before "
            "the end of the trajectory; use the full file or collect more informative motion."
        )
    elif len(ranked) > 1 and evidence_intervals_used < data.sample_count - 1:
        warnings.append(
            f"The leading convention separated after {evidence_intervals_used} of "
            f"{data.sample_count - 1} intervals; later samples remain useful for validation."
        )
    ####
    if len(evidence_near_ties) > len(posterior_near_ties):
        warnings.append(
            "Soft priors select among evidence-near-tied hypotheses; the underlying "
            "empirical ambiguity remains."
        )
    ####
    if best.result.data_rank != 1:
        warnings.append("Soft priors changed the top-ranked convention from the data-only winner.")
    ####
    if step_rms < config.slow_step_threshold_rad:
        warnings.append(
            "Per-sample attitude motion is small, so convention evidence is noise-sensitive."
        )
    ####
    if total_motion < config.slow_total_motion_threshold_rad:
        warnings.append("The trajectory contains little total rotational excitation.")
    ####
    if effective_rank < 1.35:
        warnings.append(
            "Motion is nearly coaxial; frame and unused-axis permutations may be "
            "fundamentally unidentifiable."
        )
    elif effective_rank < 2.0:
        warnings.append("Rotational excitation does not strongly cover all three axes.")
    ####
    if best.result.hard_alias_fraction > 0.0:
        warnings.append(
            "One or more rate integrals exceed pi radians per sample; orientation "
            "samples cannot uniquely reveal wrap count."
        )
    elif best.result.near_alias_fraction > 0.0:
        warnings.append("One or more intervals approach the pi-radian alias boundary.")
    ####
    if best.result.rate_change_rms_angle_rad >= config.high_rate_change_angle_rad:
        warnings.append("Within-step rate variation is large relative to the sampling interval.")
    ####
    if best.result.normalized_rmse > 0.50:
        warnings.append(
            "Even the best convention has a large residual relative to the observed motion."
        )
    ####
    if rate_scale_period_gauge_active:
        warnings.append(
            "Without timestamps, rate-unit scale and sample period form an exact gauge "
            "unless external bounds or metadata anchor one of them."
        )
    ####
    if best.result.gimbal_lock_fraction is not None and best.result.gimbal_lock_fraction > 0.0:
        warnings.append("The winning Euler hypothesis passes near gimbal lock.")
    ####
    if abs(noise_autocorrelation) >= 0.25:
        warnings.append(
            "Residuals show noticeable lag-one correlation; the iid Gaussian evidence "
            "scale should be treated as approximate."
        )
    ####

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
            item.result.hypothesis.identifier() for item in evidence_near_ties[: config.top_k]
        ),
        unresolved_dimensions=unresolved,
        posterior_near_tie_count=len(posterior_near_ties),
        posterior_near_tie_hypothesis_ids=tuple(
            item.result.hypothesis.identifier() for item in posterior_near_ties[: config.top_k]
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
        estimated_noise_sigma_rad=noise_sigma,
        noise_residual_count=noise_residual_count,
        noise_lag1_autocorrelation=noise_autocorrelation,
        evidence_sufficient=evidence_sufficient,
        evidence_intervals_used=evidence_intervals_used,
        evidence_fraction=evidence_fraction,
        evidence_log_likelihood_ratio=evidence_log_likelihood_ratio,
        exact_equivalence_class_count=len(equivalence_classes),
        best_equivalence_class_hypothesis_ids=best.result.exact_equivalent_hypothesis_ids,
        rate_scale_period_gauge_active=rate_scale_period_gauge_active,
        raw_excitation_rank=raw_excitation_rank,
        raw_excitation_eigenvalues=raw_excitation_eigenvalues,
        raw_excitation_condition_ratio=raw_excitation_ratio,
    )
####




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




def infer_trajectory_conventions(
    data: TrajectoryData,
    config: InferenceConfig | None = None,
) -> InferenceReport:
    resolved_config = config or InferenceConfig()
    hypotheses = generate_hypotheses(data.representation, resolved_config)
    if data.orientation is None or data.rates is None:
        partial_scored = tuple(
            _score_partial_hypothesis(data, hypothesis, resolved_config)
            for hypothesis in hypotheses
        )
        ranked = _assign_ranks_and_support(partial_scored, data, resolved_config)
        diagnostics = _build_overall_diagnostics(data, ranked, resolved_config)
        return _build_inference_report(
            data,
            hypotheses,
            ranked,
            resolved_config,
            diagnostics,
        )
    ####

    scored: list[_ScoredCandidate] = []
    failures: list[str] = []
    attitude_cache: dict[tuple[object, ...], Rotation] = {}
    rate_cache: dict[tuple[object, ...], FloatArray] = {}

    for hypothesis in hypotheses:
        try:
            attitude_key = (
                hypothesis.attitude_direction,
                hypothesis.quaternion_order,
                hypothesis.euler_order,
                hypothesis.euler_mode,
                hypothesis.angle_unit,
            )
            if attitude_key not in attitude_cache:
                attitude_cache[attitude_key] = decode_attitudes(data, hypothesis)
            ####

            rate_key = (hypothesis.rate_order, hypothesis.rate_unit, hypothesis.rate_sign)
            if rate_key not in rate_cache:
                rate_cache[rate_key] = decode_rates(data, hypothesis)
            ####

            scored.append(
                _score_hypothesis(
                    data,
                    hypothesis,
                    resolved_config,
                    attitudes=attitude_cache[attitude_key],
                    rates_rad_s=rate_cache[rate_key],
                )
            )
        except (ValueError, FloatingPointError) as error:
            failures.append(f"{hypothesis.identifier()}: {error}")
        ####
    ####

    if not scored:
        details = "\n".join(failures[:10])
        raise ValueError(f"All hypotheses failed to score.\n{details}")
    ####

    ranked = _assign_ranks_and_support(scored, data, resolved_config)
    diagnostics = _build_overall_diagnostics(data, ranked, resolved_config)
    return _build_inference_report(
        data,
        hypotheses,
        ranked,
        resolved_config,
        diagnostics,
    )
####




def best_interval_diagnostics(
    data: TrajectoryData,
    report: InferenceReport,
    config: InferenceConfig | None = None,
) -> tuple[dict[str, float | int | bool], ...]:
    resolved_config = config or InferenceConfig()
    scored = _score_hypothesis(data, report.best.hypothesis, resolved_config)
    predicted_quaternions = _rotation_quaternions(scored.predicted_intervals)
    observed_quaternions = _rotation_quaternions(scored.observed_intervals)
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
            "error_angle_rad": float(norm(errors[index, :])),
            "unwrapped_rate_angle_rad": float(scored.path_angles[index]),
            "near_alias": bool(scored.path_angles[index] >= resolved_config.near_alias_step_rad),
            "hard_alias": bool(scored.path_angles[index] >= resolved_config.hard_alias_step_rad),
        }
        if data.time_s is not None:
            row["time_start_s"] = float(data.time_s[index])
            row["time_end_s"] = float(data.time_s[index + 1])
        ####
        rows.append(row)
    ####
    return tuple(rows)
####


