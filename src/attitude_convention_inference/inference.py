"""Orchestrate hypothesis scoring, ranking, diagnostics, and report assembly."""

from __future__ import annotations

from dataclasses import dataclass

from scipy.spatial.transform import Rotation

from .decoding import decode_attitudes, decode_rates
from .diagnostics import _build_overall_diagnostics, best_interval_diagnostics
from .heuristics import EulerHeuristicDiagnosticsCache
from .hypotheses import generate_hypotheses
from .integration import IntegratedIntervalsCache, ObservedIncrementCache
from .math_utils import FloatArray
from .model_common import (
    AngleUnit,
    AttitudeDirection,
    EulerMode,
    QuaternionOrder,
    RateUnit,
)
from .models import InferenceConfig, InferenceReport
from .ranking import _assign_ranks_and_support
from .reporting import _build_inference_report
from .scoring import ScoredCandidate as _ScoredCandidate
from .scoring import score_hypothesis as _score_hypothesis
from .scoring import score_partial_hypothesis as _score_partial_hypothesis
from .statistics import estimate_noise_process
from .trajectory import TrajectoryData

__all__ = [
    "best_interval_diagnostics",
    "estimate_noise_process",
    "generate_hypotheses",
    "infer_trajectory_conventions",
]


@dataclass(frozen=True, slots=True)
class _AttitudeDecodeCacheKey:
    """Fields that determine one decoded attitude series."""

    direction: AttitudeDirection | str
    quaternion_order: QuaternionOrder | str | None
    euler_order: str | None
    euler_mode: EulerMode | str | None
    angle_unit: AngleUnit | str | None
####


@dataclass(frozen=True, slots=True)
class _RateDecodeCacheKey:
    """Fields that determine one decoded angular-rate series."""

    order: str
    unit: RateUnit | str
    sign: int
####


def infer_trajectory_conventions(
    data: TrajectoryData,
    config: InferenceConfig | None = None,
) -> InferenceReport:
    resolved_config = config or InferenceConfig()
    hypotheses = generate_hypotheses(data.representation, resolved_config)
    euler_diagnostics_cache = EulerHeuristicDiagnosticsCache(data)
    if data.orientation is None or data.rates is None:
        partial_scored = tuple(
            _score_partial_hypothesis(
                data,
                hypothesis,
                resolved_config,
                euler_diagnostics_cache=euler_diagnostics_cache,
            )
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
    # Hypotheses are grouped by attitude convention. Retain only the current
    # decoded rotation series instead of every Euler variant on a long log.
    current_attitude_key: _AttitudeDecodeCacheKey | None = None
    current_attitudes: Rotation | None = None
    current_observed_increment_cache: ObservedIncrementCache | None = None
    integrated_intervals_cache = IntegratedIntervalsCache(data)
    rate_cache: dict[_RateDecodeCacheKey, FloatArray] = {}

    for hypothesis in hypotheses:
        try:
            attitude_key = _AttitudeDecodeCacheKey(
                direction=hypothesis.attitude_direction,
                quaternion_order=hypothesis.quaternion_order,
                euler_order=hypothesis.euler_order,
                euler_mode=hypothesis.euler_mode,
                angle_unit=hypothesis.angle_unit,
            )
            if current_attitudes is None or attitude_key != current_attitude_key:
                current_attitudes = decode_attitudes(data, hypothesis)
                current_observed_increment_cache = ObservedIncrementCache(current_attitudes)
                current_attitude_key = attitude_key
            ####

            if current_observed_increment_cache is None:
                raise RuntimeError("Attitude-dependent scoring state was not initialized")
            ####

            rate_key = _RateDecodeCacheKey(
                order=hypothesis.rate_order,
                unit=hypothesis.rate_unit,
                sign=int(hypothesis.rate_sign),
            )
            if rate_key not in rate_cache:
                rate_cache[rate_key] = decode_rates(data, hypothesis)
            ####

            scored.append(
                _score_hypothesis(
                    data,
                    hypothesis,
                    resolved_config,
                    attitudes=current_attitudes,
                    rates_rad_s=rate_cache[rate_key],
                    euler_diagnostics_cache=euler_diagnostics_cache,
                    observed_increment_cache=current_observed_increment_cache,
                    integrated_intervals_cache=integrated_intervals_cache,
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
