"""Public convenience entry points for convention inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, overload

from numpy import asarray, float64
from numpy.typing import ArrayLike

from .coercion import (
    ConfigInput,
    ConstraintInput,
    DomainPriorInput,
    PreferenceProfileInput,
    PriorInput,
    coerce_config,
    coerce_constraints,
    coerce_domain_prior,
    coerce_preference_profile,
    coerce_priors,
    merge_priors,
)
from .inference import infer_trajectory_conventions
from .io import CsvLayout, detect_csv_layout, load_csv_trajectory
from .models import (
    AttitudeRepresentation,
    InferenceConfig,
    InferenceReport,
)
from .sparse_inference import infer_sparse_orientation_trajectory
from .trajectory import SparseOrientationRateData, TrajectoryData

__all__ = [
    "infer_attitude_conventions",
    "infer_conventions",
    "infer_csv",
    "infer_sparse_orientation_conventions",
    "inspect_csv",
    "resolve_config",
]


def resolve_config(
    config: ConfigInput = None,
    *,
    constraints: ConstraintInput = None,
    priors: PriorInput = None,
    domain_prior: DomainPriorInput = None,
    preference_profile: PreferenceProfileInput = None,
    sample_period_s: float | None = None,
) -> InferenceConfig:
    """Resolve convenience overrides into one immutable inference configuration."""

    resolved = coerce_config(config)
    constraint_model = coerce_constraints(constraints)
    prior_model = coerce_priors(priors)
    domain_prior_model = coerce_domain_prior(domain_prior)
    preference_profile_model = coerce_preference_profile(preference_profile)

    if constraint_model is not None:
        resolved = replace(
            resolved,
            constraints=resolved.constraints.intersect(constraint_model),
        )
    ####
    if prior_model is not None:
        resolved = replace(
            resolved,
            priors=merge_priors(resolved.priors, prior_model),
        )
    ####
    if domain_prior_model is not None:
        resolved = replace(resolved, domain_prior=domain_prior_model)
    ####
    if preference_profile_model is not None:
        resolved = replace(resolved, preference_profile=preference_profile_model)
    ####
    if sample_period_s is not None:
        resolved = replace(resolved, sample_period_s=sample_period_s)
    ####
    return resolved
####


@overload
def infer_conventions(
    orientation: TrajectoryData,
    rates: InferenceConfig | Mapping[str, Any] | None = None,
    *,
    config: ConfigInput = None,
    constraints: ConstraintInput = None,
    priors: PriorInput = None,
    domain_prior: DomainPriorInput = None,
    preference_profile: PreferenceProfileInput = None,
    sample_period_s: float | None = None,
) -> InferenceReport: ...


@overload
def infer_conventions(
    orientation: SparseOrientationRateData,
    rates: InferenceConfig | Mapping[str, Any] | None = None,
    *,
    config: ConfigInput = None,
    constraints: ConstraintInput = None,
    priors: PriorInput = None,
    domain_prior: DomainPriorInput = None,
    preference_profile: PreferenceProfileInput = None,
) -> InferenceReport: ...


@overload
def infer_conventions(
    orientation: ArrayLike | None,
    rates: ArrayLike | None,
    *,
    representation: AttitudeRepresentation | str,
    time_s: ArrayLike | None = None,
    config: ConfigInput = None,
    constraints: ConstraintInput = None,
    priors: PriorInput = None,
    domain_prior: DomainPriorInput = None,
    preference_profile: PreferenceProfileInput = None,
    sample_period_s: float | None = None,
) -> InferenceReport: ...


def infer_conventions(
    orientation: TrajectoryData | SparseOrientationRateData | ArrayLike | None,
    rates: ArrayLike | InferenceConfig | Mapping[str, Any] | None = None,
    *,
    representation: AttitudeRepresentation | str | None = None,
    time_s: ArrayLike | None = None,
    config: ConfigInput = None,
    constraints: ConstraintInput = None,
    priors: PriorInput = None,
    domain_prior: DomainPriorInput = None,
    preference_profile: PreferenceProfileInput = None,
    sample_period_s: float | None = None,
) -> InferenceReport:
    """Infer attitude and angular-rate conventions from trajectory consistency.

    This is the canonical public entry point. Pass either a :class:`TrajectoryData`
    object or ordinary array-like orientation/rate data. Partial hard constraints and
    soft priors may be dataclass models or plain mappings.
    """

    if isinstance(orientation, (TrajectoryData, SparseOrientationRateData)):
        if representation is not None or time_s is not None:
            model_name = (
                "SparseOrientationRateData"
                if isinstance(orientation, SparseOrientationRateData)
                else "TrajectoryData"
            )
            raise ValueError(f"representation and time_s must not be supplied with {model_name}.")
        ####
        if isinstance(orientation, SparseOrientationRateData) and sample_period_s is not None:
            raise ValueError("sample_period_s must not be supplied with timestamped sparse data.")
        ####
        positional_config: ConfigInput = None
        if rates is not None:
            if isinstance(rates, (InferenceConfig, Mapping)):
                positional_config = rates
            else:
                raise ValueError(
                    "The second positional argument with TrajectoryData must be an InferenceConfig."
                )
            ####
        ####
        if config is not None and positional_config is not None:
            raise ValueError("Configuration was supplied both positionally and by keyword.")
        ####
        data = orientation
        resolved = resolve_config(
            config if config is not None else positional_config,
            constraints=constraints,
            priors=priors,
            domain_prior=domain_prior,
            preference_profile=preference_profile,
            sample_period_s=sample_period_s,
        )
        if isinstance(data, SparseOrientationRateData):
            return infer_sparse_orientation_trajectory(data, resolved)
        ####
        return infer_trajectory_conventions(data, resolved)
    ####

    if isinstance(rates, (InferenceConfig, Mapping)):
        raise ValueError("Array input requires rates to be an array or None.")
    ####
    if representation is None:
        raise ValueError("Array input requires representation='quaternion' or 'euler'.")
    ####

    data = TrajectoryData(
        representation=AttitudeRepresentation(representation),
        orientation=(None if orientation is None else asarray(orientation, dtype=float64)),
        rates=None if rates is None else asarray(rates, dtype=float64),
        time_s=None if time_s is None else asarray(time_s, dtype=float64),
    )
    resolved = resolve_config(
        config,
        constraints=constraints,
        priors=priors,
        domain_prior=domain_prior,
        preference_profile=preference_profile,
        sample_period_s=sample_period_s,
    )
    return infer_trajectory_conventions(data, resolved)
####


def infer_sparse_orientation_conventions(
    data: SparseOrientationRateData,
    config: ConfigInput = None,
    *,
    constraints: ConstraintInput = None,
    priors: PriorInput = None,
    domain_prior: DomainPriorInput = None,
    preference_profile: PreferenceProfileInput = None,
) -> InferenceReport:
    """Infer conventions from sparse orientation anchors and timestamped dense rates.

    Both streams must use the same clock. Rates are integrated over every anchor
    interval, with linear interpolation at anchor times and no extrapolation.
    Sequential evidence is based on alternating anchor intervals so adjacent
    residuals do not reuse an orientation observation.
    """

    resolved = resolve_config(
        config,
        constraints=constraints,
        priors=priors,
        domain_prior=domain_prior,
        preference_profile=preference_profile,
    )
    return infer_sparse_orientation_trajectory(data, resolved)
####


def infer_attitude_conventions(
    orientation: ArrayLike,
    rates: ArrayLike,
    *,
    representation: AttitudeRepresentation | str,
    time_s: ArrayLike | None = None,
    config: ConfigInput = None,
    constraints: ConstraintInput = None,
    priors: PriorInput = None,
    domain_prior: DomainPriorInput = None,
    preference_profile: PreferenceProfileInput = None,
    sample_period_s: float | None = None,
) -> InferenceReport:
    """Compatibility alias for :func:`infer_conventions`."""

    return infer_conventions(
        orientation,
        rates,
        representation=representation,
        time_s=time_s,
        config=config,
        constraints=constraints,
        priors=priors,
        domain_prior=domain_prior,
        preference_profile=preference_profile,
        sample_period_s=sample_period_s,
    )
####


def infer_csv(
    path: str | Path,
    *,
    representation: AttitudeRepresentation | str | None = None,
    orientation_columns: Sequence[str] | None = None,
    rate_columns: Sequence[str] | None = None,
    time_column: str | None = None,
    config: ConfigInput = None,
    constraints: ConstraintInput = None,
    priors: PriorInput = None,
    domain_prior: DomainPriorInput = None,
    preference_profile: PreferenceProfileInput = None,
    sample_period_s: float | None = None,
) -> InferenceReport:
    """Infer conventions from CSV with explicit or automatically detected columns."""

    representation_model = (
        None if representation is None else AttitudeRepresentation(representation)
    )
    layout = detect_csv_layout(
        path,
        representation=representation_model,
        orientation_columns=orientation_columns,
        rate_columns=rate_columns,
        time_column=time_column,
    )
    data = load_csv_trajectory(
        path,
        layout.representation,
        layout.orientation_columns,
        layout.rate_columns,
        layout.time_column,
    )
    return infer_conventions(
        data,
        config,
        constraints=constraints,
        priors=priors,
        domain_prior=domain_prior,
        preference_profile=preference_profile,
        sample_period_s=sample_period_s,
    )
####


def inspect_csv(
    path: str | Path,
    *,
    representation: AttitudeRepresentation | str | None = None,
    orientation_columns: Sequence[str] | None = None,
    rate_columns: Sequence[str] | None = None,
    time_column: str | None = None,
) -> CsvLayout:
    """Resolve the CSV layout without running inference."""

    representation_model = (
        None if representation is None else AttitudeRepresentation(representation)
    )
    return detect_csv_layout(
        path,
        representation=representation_model,
        orientation_columns=orientation_columns,
        rate_columns=rate_columns,
        time_column=time_column,
    )
####
