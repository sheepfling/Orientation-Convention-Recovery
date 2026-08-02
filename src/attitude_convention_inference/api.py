from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, overload

import numpy as np
from numpy.typing import ArrayLike

from .core import TrajectoryData
from .inference import _infer_trajectory_conventions
from .io import CsvLayout, detect_csv_layout, load_csv_trajectory
from .models import (
    AttitudeRepresentation,
    ConventionConstraints,
    ConventionPriors,
    InferenceConfig,
    InferenceReport,
)

ConfigInput = InferenceConfig | Mapping[str, Any] | None
ConstraintInput = ConventionConstraints | Mapping[str, Any] | None
PriorInput = ConventionPriors | Mapping[str, Any] | None


def _coerce_config(value: ConfigInput) -> InferenceConfig:
    if value is None:
        return InferenceConfig()
    if isinstance(value, InferenceConfig):
        return value
    return InferenceConfig.model_validate(value)


####


def _coerce_constraints(value: ConstraintInput) -> ConventionConstraints | None:
    if value is None:
        return None
    if isinstance(value, ConventionConstraints):
        return value
    return ConventionConstraints.model_validate(value)


####


def _coerce_priors(value: PriorInput) -> ConventionPriors | None:
    if value is None:
        return None
    if isinstance(value, ConventionPriors):
        return value
    return ConventionPriors.model_validate(value)


####


def _merge_priors(base: ConventionPriors, override: ConventionPriors) -> ConventionPriors:
    payload = base.model_dump(mode="python")
    override_payload = override.model_dump(mode="python")
    for dimension, weights in override_payload.items():
        payload[dimension] = {**payload[dimension], **weights}
    return ConventionPriors.model_validate(payload)


####


def resolve_config(
    config: ConfigInput = None,
    *,
    constraints: ConstraintInput = None,
    priors: PriorInput = None,
    sample_period_s: float | None = None,
) -> InferenceConfig:
    """Resolve convenience overrides into one immutable inference configuration."""

    resolved = _coerce_config(config)
    constraint_model = _coerce_constraints(constraints)
    prior_model = _coerce_priors(priors)

    if constraint_model is not None:
        resolved = replace(
            resolved,
            constraints=resolved.constraints.intersect(constraint_model),
        )
    if prior_model is not None:
        resolved = replace(
            resolved,
            priors=_merge_priors(resolved.priors, prior_model),
        )
    if sample_period_s is not None:
        resolved = replace(resolved, sample_period_s=sample_period_s)
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
    sample_period_s: float | None = None,
) -> InferenceReport: ...


@overload
def infer_conventions(
    orientation: ArrayLike,
    rates: ArrayLike,
    *,
    representation: AttitudeRepresentation | str,
    time_s: ArrayLike | None = None,
    config: ConfigInput = None,
    constraints: ConstraintInput = None,
    priors: PriorInput = None,
    sample_period_s: float | None = None,
) -> InferenceReport: ...


def infer_conventions(
    orientation: TrajectoryData | ArrayLike,
    rates: ArrayLike | InferenceConfig | Mapping[str, Any] | None = None,
    *,
    representation: AttitudeRepresentation | str | None = None,
    time_s: ArrayLike | None = None,
    config: ConfigInput = None,
    constraints: ConstraintInput = None,
    priors: PriorInput = None,
    sample_period_s: float | None = None,
) -> InferenceReport:
    """Infer attitude and angular-rate conventions from trajectory consistency.

    This is the canonical public entry point. Pass either a :class:`TrajectoryData`
    object or ordinary array-like orientation/rate data. Partial hard constraints and
    soft priors may be dataclass models or plain mappings.
    """

    if isinstance(orientation, TrajectoryData):
        if representation is not None or time_s is not None:
            raise ValueError(
                "representation and time_s must not be supplied with TrajectoryData."
            )
        positional_config: ConfigInput = None
        if rates is not None:
            if isinstance(rates, InferenceConfig) or isinstance(rates, Mapping):
                positional_config = rates
            else:
                raise ValueError(
                    "The second positional argument with TrajectoryData must be an InferenceConfig."
                )
        if config is not None and positional_config is not None:
            raise ValueError("Configuration was supplied both positionally and by keyword.")
        data = orientation
        resolved = resolve_config(
            config if config is not None else positional_config,
            constraints=constraints,
            priors=priors,
            sample_period_s=sample_period_s,
        )
        return _infer_trajectory_conventions(data, resolved)

    if rates is None or isinstance(rates, (InferenceConfig, Mapping)):
        raise ValueError("Array input requires an angular-rate array.")
    if representation is None:
        raise ValueError("Array input requires representation='quaternion' or 'euler'.")

    data = TrajectoryData(
        representation=AttitudeRepresentation(representation),
        orientation=np.asarray(orientation, dtype=np.float64),
        rates=np.asarray(rates, dtype=np.float64),
        time_s=None if time_s is None else np.asarray(time_s, dtype=np.float64),
    )
    resolved = resolve_config(
        config,
        constraints=constraints,
        priors=priors,
        sample_period_s=sample_period_s,
    )
    return _infer_trajectory_conventions(data, resolved)


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
