"""Orchestrate hypothesis scoring, ranking, diagnostics, and report assembly."""

from __future__ import annotations

from scipy.spatial.transform import Rotation

from .core import FloatArray, TrajectoryData, decode_attitudes, decode_rates
from .diagnostics import _build_overall_diagnostics, best_interval_diagnostics
from .hypotheses import generate_hypotheses
from .models import InferenceConfig, InferenceReport
from .ranking import _assign_ranks_and_support
from .reporting import _build_inference_report
from .scoring import ScoredCandidate as _ScoredCandidate
from .scoring import score_hypothesis as _score_hypothesis
from .scoring import score_partial_hypothesis as _score_partial_hypothesis
from .statistics import estimate_noise_process

__all__ = [
    "best_interval_diagnostics",
    "estimate_noise_process",
    "generate_hypotheses",
    "infer_trajectory_conventions",
]


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
    # Hypotheses are grouped by attitude convention. Retain only the current
    # decoded rotation series instead of every Euler variant on a long log.
    current_attitude_key: tuple[object, ...] | None = None
    current_attitudes: Rotation | None = None
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
            if current_attitudes is None or attitude_key != current_attitude_key:
                current_attitudes = decode_attitudes(data, hypothesis)
                current_attitude_key = attitude_key
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
                    attitudes=current_attitudes,
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
