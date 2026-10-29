"""Overall and per-interval diagnostics from ranked trajectory candidates."""

from __future__ import annotations

from math import exp, sqrt

from numpy import clip, mean, square
from numpy import sqrt as np_sqrt
from numpy import sum as np_sum
from numpy.linalg import norm

from .assessments import _dimension_assessments, _unresolved_dimensions
from .core import TrajectoryData
from .evidence import _evidence_summary
from .math_utils import rotation_magnitudes as _rotation_magnitudes
from .math_utils import rotation_quaternions as _rotation_quaternions
from .models import ConfidenceLabel, InferenceConfig, InferenceReport, OverallDiagnostics
from .scoring import ScoredCandidate as _ScoredCandidate
from .scoring import score_hypothesis as _score_hypothesis
from .scoring import score_partial_hypothesis as _score_partial_hypothesis
from .statistics import effective_excitation_rank as _effective_excitation_rank
from .statistics import raw_rate_excitation as _raw_rate_excitation

__all__ = ["_build_overall_diagnostics", "best_interval_diagnostics"]


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
    detailed = (
        _score_partial_hypothesis(
            data, best.result.hypothesis, config, retain_interval_details=True
        )
        if data.orientation is None or data.rates is None
        else _score_hypothesis(data, best.result.hypothesis, config, retain_interval_details=True)
    )
    details = detailed.interval_details
    if details is None:
        raise RuntimeError("Interval details were not retained for diagnostics")
    ####
    observed_angles = _rotation_magnitudes(details.observed_intervals)
    total_motion = float(np_sum(observed_angles))
    step_rms = float(np_sqrt(mean(square(observed_angles))))
    excitation = _effective_excitation_rank(
        details.rates_rad_s,
        details.interval_durations_s,
    )
    raw_excitation = _raw_rate_excitation(
        data,
        config,
    )
    evidence = _evidence_summary(data, ranked, detailed, config)
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


def best_interval_diagnostics(
    data: TrajectoryData,
    report: InferenceReport,
    config: InferenceConfig | None = None,
) -> tuple[dict[str, float | int | bool], ...]:
    if data.orientation is None or data.rates is None:
        raise ValueError("Interval diagnostics require both orientation and angular rates")
    ####
    resolved_config = config or InferenceConfig()
    scored = _score_hypothesis(
        data, report.best.hypothesis, resolved_config, retain_interval_details=True
    )
    details = scored.interval_details
    if details is None:
        raise RuntimeError("Interval details were not retained for diagnostics")
    ####
    predicted_quaternions = _rotation_quaternions(details.predicted_intervals)
    observed_quaternions = _rotation_quaternions(details.observed_intervals)
    errors = scored.interval_error_rotvecs
    if errors is None:
        raise RuntimeError("Interval residual vectors were not retained for diagnostics")
    ####
    rows: list[dict[str, float | int | bool]] = []

    for index in range(data.sample_count - 1):
        row: dict[str, float | int | bool] = {
            "interval": index,
            "dt_s": float(details.interval_durations_s[index]),
            "omega_x_rad_s": float(
                0.5 * (details.rates_rad_s[index, 0] + details.rates_rad_s[index + 1, 0])
            ),
            "omega_y_rad_s": float(
                0.5 * (details.rates_rad_s[index, 1] + details.rates_rad_s[index + 1, 1])
            ),
            "omega_z_rad_s": float(
                0.5 * (details.rates_rad_s[index, 2] + details.rates_rad_s[index + 1, 2])
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
            "unwrapped_rate_angle_rad": float(details.path_angles[index]),
            "near_alias": bool(details.path_angles[index] >= resolved_config.near_alias_step_rad),
            "hard_alias": bool(details.path_angles[index] >= resolved_config.hard_alias_step_rad),
        }
        if data.time_s is not None:
            row["time_start_s"] = float(data.time_s[index])
            row["time_end_s"] = float(data.time_s[index + 1])
        ####
        rows.append(row)
    ####
    return tuple(rows)
####
