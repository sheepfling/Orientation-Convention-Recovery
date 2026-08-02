from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum, StrEnum
from math import isfinite, pi
from typing import Any, Literal, Self, TypeVar, cast

EULER_ORDERS: tuple[str, ...] = ("xyz", "xzy", "yxz", "yzx", "zxy", "zyx")
RATE_ORDERS: tuple[str, ...] = ("pqr", "prq", "qpr", "qrp", "rpq", "rqp")
_CONSTRAINT_FIELDS: tuple[str, ...] = (
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
_PRIOR_FIELDS: tuple[str, ...] = _CONSTRAINT_FIELDS

T = TypeVar("T")
EnumT = TypeVar("EnumT", bound=Enum)


class AttitudeRepresentation(StrEnum):
    EULER = "euler"
    QUATERNION = "quaternion"


####


class AttitudeDirection(StrEnum):
    BODY_TO_WORLD = "body_to_world"
    WORLD_TO_BODY = "world_to_body"


####


class QuaternionOrder(StrEnum):
    SCALAR_FIRST = "scalar_first"
    SCALAR_LAST = "scalar_last"


####


class EulerMode(StrEnum):
    INTRINSIC = "intrinsic"
    EXTRINSIC = "extrinsic"


####


class AngleUnit(StrEnum):
    RADIANS = "radians"
    DEGREES = "degrees"


####


class RateFrame(StrEnum):
    BODY = "body"
    WORLD = "world"


####


class RateUnit(StrEnum):
    RADIANS_PER_SECOND = "radians_per_second"
    DEGREES_PER_SECOND = "degrees_per_second"


####


class ConfidenceLabel(StrEnum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INDETERMINATE = "indeterminate"


####


class DimensionStatus(StrEnum):
    FIXED = "fixed"
    INFERRED = "inferred"
    AMBIGUOUS = "ambiguous"
    NOT_APPLICABLE = "not_applicable"


####


def _jsonable(value: object, *, json_mode: bool) -> object:
    if isinstance(value, Enum):
        return value.value if json_mode else value
    if is_dataclass(value):
        return {
            item.name: _jsonable(getattr(value, item.name), json_mode=json_mode)
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        if json_mode:
            return {
                str(key.value if isinstance(key, Enum) else key): _jsonable(
                    item,
                    json_mode=True,
                )
                for key, item in value.items()
            }
        return {
            key: _jsonable(item, json_mode=False)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        converted = tuple(_jsonable(item, json_mode=json_mode) for item in value)
        return list(converted) if json_mode else converted
    if isinstance(value, list):
        return [_jsonable(item, json_mode=json_mode) for item in value]
    return value


####


class FrozenModel:
    """Small immutable-dataclass compatibility layer.

    The public serialization helpers intentionally mirror the subset of Pydantic's
    API used by earlier releases. They keep existing JSON/config workflows working
    without retaining Pydantic as a runtime dependency.
    """

    @classmethod
    def model_validate(cls: type[Self], value: Self | Mapping[str, Any]) -> Self:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"{cls.__name__} requires a mapping or an existing instance.")
        try:
            return cls(**dict(value))  # type: ignore[misc]
        except TypeError as error:
            raise ValueError(str(error)) from error

    ####

    def model_dump(self, *, mode: Literal["python", "json"] = "python") -> dict[str, Any]:
        if mode not in ("python", "json"):
            raise ValueError("mode must be 'python' or 'json'.")
        converted = _jsonable(self, json_mode=mode == "json")
        return cast(dict[str, Any], converted)

    ####

    def model_dump_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent, allow_nan=False)

    ####

    def model_copy(self: Self, *, update: Mapping[str, object] | None = None) -> Self:
        return replace(self, **dict(update or {}))

    ####


####


def _as_tuple(value: T | Sequence[T] | set[T] | None) -> tuple[T, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, Enum)) or not isinstance(value, (Sequence, set)):
        return (cast(T, value),)
    return tuple(value)


####


def _enum_value(enum_type: type[EnumT], value: EnumT | str) -> EnumT:
    return value if isinstance(value, enum_type) else enum_type(value)


####


def _enum_tuple(
    enum_type: type[EnumT],
    value: EnumT | str | Sequence[EnumT | str],
) -> tuple[EnumT, ...]:
    raw = _as_tuple(value)
    if raw is None:
        raise ValueError("A non-empty value is required.")
    return tuple(_enum_value(enum_type, item) for item in raw)


####


def _validate_nonempty_unique(value: tuple[object, ...], label: str) -> None:
    if not value:
        raise ValueError(f"{label} may not be empty.")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} may not contain duplicates.")


####


def _normalize_euler_orders(value: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(order).lower() for order in value)
    for order in normalized:
        if len(order) != 3 or set(order) != {"x", "y", "z"}:
            raise ValueError(f"Euler order {order!r} is not a three-axis Tait-Bryan permutation.")
    return normalized


####


def _normalize_rate_orders(value: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(order).lower() for order in value)
    for order in normalized:
        if len(order) != 3 or set(order) != {"p", "q", "r"}:
            raise ValueError(f"Rate order {order!r} is not a p/q/r permutation.")
    return normalized


####


def _normalize_signs(value: Sequence[int | str]) -> tuple[Literal[-1, 1], ...]:
    signs = tuple(int(item) for item in value)
    for sign in signs:
        if sign not in (-1, 1):
            raise ValueError("Rate signs must be either -1 or 1.")
    return cast(tuple[Literal[-1, 1], ...], signs)


####


@dataclass(frozen=True, slots=True)
class SearchSpace(FrozenModel):
    attitude_directions: tuple[AttitudeDirection, ...] = (
        AttitudeDirection.BODY_TO_WORLD,
        AttitudeDirection.WORLD_TO_BODY,
    )
    quaternion_orders: tuple[QuaternionOrder, ...] = (
        QuaternionOrder.SCALAR_FIRST,
        QuaternionOrder.SCALAR_LAST,
    )
    euler_orders: tuple[str, ...] = EULER_ORDERS
    euler_modes: tuple[EulerMode, ...] = (EulerMode.INTRINSIC, EulerMode.EXTRINSIC)
    angle_units: tuple[AngleUnit, ...] = (AngleUnit.RADIANS,)
    rate_frames: tuple[RateFrame, ...] = (RateFrame.BODY, RateFrame.WORLD)
    rate_orders: tuple[str, ...] = RATE_ORDERS
    rate_units: tuple[RateUnit, ...] = (RateUnit.RADIANS_PER_SECOND,)
    rate_signs: tuple[Literal[-1, 1], ...] = (1,)

    @classmethod
    def model_validate(cls, value: Self | Mapping[str, Any]) -> Self:
        payload = _validated_payload(cls, value)
        enum_fields: tuple[tuple[str, type[Enum]], ...] = (
            ("attitude_directions", AttitudeDirection),
            ("quaternion_orders", QuaternionOrder),
            ("euler_modes", EulerMode),
            ("angle_units", AngleUnit),
            ("rate_frames", RateFrame),
            ("rate_units", RateUnit),
        )
        for name, enum_type in enum_fields:
            if name in payload:
                payload[name] = _enum_tuple(enum_type, payload[name])
        if "euler_orders" in payload:
            payload["euler_orders"] = _normalize_euler_orders(tuple(payload["euler_orders"]))
        if "rate_orders" in payload:
            payload["rate_orders"] = _normalize_rate_orders(tuple(payload["rate_orders"]))
        if "rate_signs" in payload:
            payload["rate_signs"] = _normalize_signs(tuple(payload["rate_signs"]))
        return cls(**payload)

    def __post_init__(self) -> None:
        for item in fields(self):
            _validate_nonempty_unique(tuple(getattr(self, item.name)), "Search-space fields")
        _normalize_euler_orders(tuple(str(item) for item in self.euler_orders))
        _normalize_rate_orders(tuple(str(item) for item in self.rate_orders))
        for name, enum_type in (
            ("attitude_directions", AttitudeDirection),
            ("quaternion_orders", QuaternionOrder),
            ("euler_modes", EulerMode),
            ("angle_units", AngleUnit),
            ("rate_frames", RateFrame),
            ("rate_units", RateUnit),
        ):
            for value in getattr(self, name):
                enum_type(value)
        _normalize_signs(tuple(self.rate_signs))

    ####


####


@dataclass(frozen=True, slots=True)
class ConventionConstraints(FrozenModel):
    """Hard constraints layered on top of :class:`SearchSpace`.

    A scalar locks a dimension, a sequence retains an allowed subset, and ``None``
    leaves the dimension unknown.
    """

    attitude_direction: tuple[AttitudeDirection, ...] | AttitudeDirection | str | None = None
    quaternion_order: tuple[QuaternionOrder, ...] | QuaternionOrder | str | None = None
    euler_order: tuple[str, ...] | str | None = None
    euler_mode: tuple[EulerMode, ...] | EulerMode | str | None = None
    angle_unit: tuple[AngleUnit, ...] | AngleUnit | str | None = None
    rate_frame: tuple[RateFrame, ...] | RateFrame | str | None = None
    rate_order: tuple[str, ...] | str | None = None
    rate_unit: tuple[RateUnit, ...] | RateUnit | str | None = None
    rate_sign: tuple[Literal[-1, 1], ...] | Literal[-1, 1] | str | None = None

    @classmethod
    def model_validate(cls, value: Self | Mapping[str, Any]) -> Self:
        payload = _validated_payload(cls, value)
        enum_fields: tuple[tuple[str, type[Enum]], ...] = (
            ("attitude_direction", AttitudeDirection),
            ("quaternion_order", QuaternionOrder),
            ("euler_mode", EulerMode),
            ("angle_unit", AngleUnit),
            ("rate_frame", RateFrame),
            ("rate_unit", RateUnit),
        )
        for name, enum_type in enum_fields:
            if payload.get(name) is not None:
                payload[name] = _enum_tuple(enum_type, payload[name])
        if payload.get("euler_order") is not None:
            payload["euler_order"] = _normalize_euler_orders(
                tuple(str(item) for item in _as_tuple(payload["euler_order"]) or ())
            )
        if payload.get("rate_order") is not None:
            payload["rate_order"] = _normalize_rate_orders(
                tuple(str(item) for item in _as_tuple(payload["rate_order"]) or ())
            )
        if payload.get("rate_sign") is not None:
            payload["rate_sign"] = _normalize_signs(
                tuple(cast(int | str, item) for item in _as_tuple(payload["rate_sign"]) or ())
            )
        return cls(**payload)

    def __post_init__(self) -> None:
        for field_name in _CONSTRAINT_FIELDS:
            value = getattr(self, field_name)
            if value is not None:
                _validate_nonempty_unique(tuple(_as_tuple(value) or ()), "Constraint fields")

    ####

    @staticmethod
    def _intersect_values(
        base: tuple[T, ...],
        constrained: tuple[T, ...] | None,
        dimension: str,
    ) -> tuple[T, ...]:
        if constrained is None:
            return base
        allowed = set(_as_tuple(constrained) or ())
        result = tuple(value for value in base if value in allowed)
        if not result:
            raise ValueError(
                f"Constraint for {dimension!r} has no overlap with the configured search space."
            )
        return result

    ####

    def apply(self, base: SearchSpace) -> SearchSpace:
        return SearchSpace(
            attitude_directions=self._intersect_values(
                base.attitude_directions,
                cast(tuple[AttitudeDirection, ...] | None, self.attitude_direction),
                "attitude_direction",
            ),
            quaternion_orders=self._intersect_values(
                base.quaternion_orders,
                cast(tuple[QuaternionOrder, ...] | None, self.quaternion_order),
                "quaternion_order",
            ),
            euler_orders=self._intersect_values(
                base.euler_orders,
                cast(tuple[str, ...] | None, self.euler_order),
                "euler_order",
            ),
            euler_modes=self._intersect_values(
                base.euler_modes,
                cast(tuple[EulerMode, ...] | None, self.euler_mode),
                "euler_mode",
            ),
            angle_units=self._intersect_values(
                base.angle_units,
                cast(tuple[AngleUnit, ...] | None, self.angle_unit),
                "angle_unit",
            ),
            rate_frames=self._intersect_values(
                base.rate_frames,
                cast(tuple[RateFrame, ...] | None, self.rate_frame),
                "rate_frame",
            ),
            rate_orders=self._intersect_values(
                base.rate_orders,
                cast(tuple[str, ...] | None, self.rate_order),
                "rate_order",
            ),
            rate_units=self._intersect_values(
                base.rate_units,
                cast(tuple[RateUnit, ...] | None, self.rate_unit),
                "rate_unit",
            ),
            rate_signs=self._intersect_values(
                base.rate_signs,
                cast(tuple[Literal[-1, 1], ...] | None, self.rate_sign),
                "rate_sign",
            ),
        )

    ####

    def intersect(self, other: ConventionConstraints) -> ConventionConstraints:
        updates: dict[str, tuple[object, ...] | None] = {}
        for field_name in _CONSTRAINT_FIELDS:
            left = cast(tuple[object, ...] | None, getattr(self, field_name))
            right = cast(tuple[object, ...] | None, getattr(other, field_name))
            if left is None:
                updates[field_name] = right
                continue
            if right is None:
                updates[field_name] = left
                continue
            right_values = set(right)
            overlap = tuple(value for value in left if value in right_values)
            if not overlap:
                raise ValueError(f"Hard constraints for {field_name!r} are contradictory.")
            updates[field_name] = overlap
        return ConventionConstraints.model_validate(updates)

    ####


####


def _prior_mapping(
    value: Mapping[object, float] | None,
    *,
    key_converter: Any,
    label: str,
) -> dict[Any, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} priors must be a mapping.")
    result: dict[Any, float] = {}
    for raw_key, raw_weight in value.items():
        key = key_converter(raw_key)
        weight = float(raw_weight)
        if not isfinite(weight) or weight <= 0.0:
            raise ValueError(f"Prior weight for {key!r} must be positive and finite.")
        result[key] = weight
    return result


def _validated_payload(cls: type[T], value: T | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, cls):
        return {item.name: getattr(value, item.name) for item in fields(value)}
    if not isinstance(value, Mapping):
        raise TypeError(f"{cls.__name__} requires a mapping or an existing instance.")
    return dict(value)


####


@dataclass(frozen=True, slots=True)
class ConventionPriors(FrozenModel):
    """Relative soft-prior weights; missing values have neutral weight ``1.0``."""

    attitude_direction: Mapping[AttitudeDirection | str, float] = field(default_factory=dict)
    quaternion_order: Mapping[QuaternionOrder | str, float] = field(default_factory=dict)
    euler_order: Mapping[str, float] = field(default_factory=dict)
    euler_mode: Mapping[EulerMode | str, float] = field(default_factory=dict)
    angle_unit: Mapping[AngleUnit | str, float] = field(default_factory=dict)
    rate_frame: Mapping[RateFrame | str, float] = field(default_factory=dict)
    rate_order: Mapping[str, float] = field(default_factory=dict)
    rate_unit: Mapping[RateUnit | str, float] = field(default_factory=dict)
    rate_sign: Mapping[Literal[-1, 1] | str, float] = field(default_factory=dict)

    @classmethod
    def model_validate(cls, value: Self | Mapping[str, Any]) -> Self:
        payload = _validated_payload(cls, value)
        converters: tuple[tuple[str, Any, str], ...] = (
            ("attitude_direction", lambda item: _enum_value(AttitudeDirection, item), "attitude_direction"),
            ("quaternion_order", lambda item: _enum_value(QuaternionOrder, item), "quaternion_order"),
            ("euler_order", lambda item: _normalize_euler_orders((str(item),))[0], "euler_order"),
            ("euler_mode", lambda item: _enum_value(EulerMode, item), "euler_mode"),
            ("angle_unit", lambda item: _enum_value(AngleUnit, item), "angle_unit"),
            ("rate_frame", lambda item: _enum_value(RateFrame, item), "rate_frame"),
            ("rate_order", lambda item: _normalize_rate_orders((str(item),))[0], "rate_order"),
            ("rate_unit", lambda item: _enum_value(RateUnit, item), "rate_unit"),
            ("rate_sign", lambda item: _normalize_signs((cast(int | str, item),))[0], "rate_sign"),
        )
        for name, converter, label in converters:
            if payload.get(name) is not None:
                payload[name] = _prior_mapping(payload[name], key_converter=converter, label=label)
        return cls(**payload)

    ####

    def configured_dimensions(self) -> tuple[str, ...]:
        return tuple(field_name for field_name in _PRIOR_FIELDS if bool(getattr(self, field_name)))

    ####


####


@dataclass(frozen=True, slots=True)
class InferenceConfig(FrozenModel):
    search_space: SearchSpace | Mapping[str, Any] = SearchSpace()
    constraints: ConventionConstraints | Mapping[str, Any] = ConventionConstraints()
    priors: ConventionPriors | Mapping[str, Any] = ConventionPriors()
    sample_period_s: float | None = None
    infer_sample_period_when_missing: bool = True
    inferred_sample_period_bounds_s: tuple[float, float] = (1.0e-6, 10.0)
    integration_substeps: int = 4
    window_sizes: tuple[int, ...] = (1, 2, 4, 8)
    trim_fraction: float = 0.02
    residual_floor_rad: float = 1.0e-5
    near_tie_delta_criterion: float = 2.0
    slow_step_threshold_rad: float = 0.002
    slow_total_motion_threshold_rad: float = 0.10
    near_alias_step_rad: float = 0.80 * pi
    hard_alias_step_rad: float = pi
    high_rate_change_angle_rad: float = 0.50
    gimbal_cosine_threshold: float = 0.05
    top_k: int = 20

    @classmethod
    def model_validate(cls, value: Self | Mapping[str, Any]) -> Self:
        payload = _validated_payload(cls, value)
        if "search_space" in payload:
            payload["search_space"] = SearchSpace.model_validate(payload["search_space"])
        if "constraints" in payload:
            payload["constraints"] = ConventionConstraints.model_validate(payload["constraints"])
        if "priors" in payload:
            payload["priors"] = ConventionPriors.model_validate(payload["priors"])
        if "inferred_sample_period_bounds_s" in payload:
            payload["inferred_sample_period_bounds_s"] = tuple(
                float(value) for value in payload["inferred_sample_period_bounds_s"]
            )
        if "window_sizes" in payload:
            payload["window_sizes"] = tuple(int(value) for value in payload["window_sizes"])
        return cls(**payload)

    def __post_init__(self) -> None:
        if self.sample_period_s is not None and (
            not isfinite(float(self.sample_period_s)) or self.sample_period_s <= 0.0
        ):
            raise ValueError("sample_period_s must be positive and finite.")
        lower, upper = (float(value) for value in self.inferred_sample_period_bounds_s)
        if (
            not isfinite(lower)
            or not isfinite(upper)
            or lower <= 0.0
            or upper <= lower
        ):
            raise ValueError(
                "inferred_sample_period_bounds_s must be finite, positive, and strictly increasing."
            )
        if self.inferred_sample_period_bounds_s != (lower, upper):
            raise ValueError("inferred_sample_period_bounds_s must contain floats.")

        if not 1 <= int(self.integration_substeps) <= 64:
            raise ValueError("integration_substeps must be between 1 and 64.")
        if not isinstance(self.integration_substeps, int):
            raise ValueError("integration_substeps must be an integer.")

        windows = tuple(sorted(set(int(window) for window in self.window_sizes)))
        if not windows or any(window < 1 for window in windows):
            raise ValueError("Window sizes must be positive integers.")
        if tuple(self.window_sizes) != windows:
            raise ValueError("window_sizes must contain unique positive integers.")

        bounded_checks = (
            (
                "trim_fraction",
                self.trim_fraction,
                isfinite(float(self.trim_fraction)) and 0.0 <= self.trim_fraction < 0.25,
            ),
            (
                "gimbal_cosine_threshold",
                self.gimbal_cosine_threshold,
                isfinite(float(self.gimbal_cosine_threshold))
                and 0.0 < self.gimbal_cosine_threshold < 1.0,
            ),
        )
        for name, _, valid in bounded_checks:
            if not valid:
                raise ValueError(f"{name} is outside its valid range.")

        positive_fields = (
            "residual_floor_rad",
            "near_tie_delta_criterion",
            "slow_step_threshold_rad",
            "slow_total_motion_threshold_rad",
            "high_rate_change_angle_rad",
        )
        for name in positive_fields:
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if not isfinite(float(self.near_alias_step_rad)) or not 0.0 < self.near_alias_step_rad < pi:
            raise ValueError("near_alias_step_rad must be between 0 and pi.")
        if not isfinite(float(self.hard_alias_step_rad)) or self.hard_alias_step_rad < pi:
            raise ValueError("hard_alias_step_rad must be at least pi.")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1.")

        self.constraints.apply(self.search_space)

    ####

    def resolved_search_space(self) -> SearchSpace:
        constraints = cast(ConventionConstraints, self.constraints)
        search_space = cast(SearchSpace, self.search_space)
        return constraints.apply(search_space)

    ####


####


@dataclass(frozen=True, slots=True)
class ConventionHypothesis(FrozenModel):
    attitude_direction: AttitudeDirection | str
    rate_frame: RateFrame | str
    rate_order: str
    rate_unit: RateUnit | str
    quaternion_order: QuaternionOrder | str | None = None
    euler_order: str | None = None
    euler_mode: EulerMode | str | None = None
    angle_unit: AngleUnit | str | None = None
    rate_sign: Literal[-1, 1] | int = 1

    @classmethod
    def model_validate(cls, value: Self | Mapping[str, Any]) -> Self:
        payload = _validated_payload(cls, value)
        for name, enum_type in (
            ("attitude_direction", AttitudeDirection),
            ("rate_frame", RateFrame),
            ("rate_unit", RateUnit),
            ("quaternion_order", QuaternionOrder),
            ("euler_mode", EulerMode),
            ("angle_unit", AngleUnit),
        ):
            if payload.get(name) is not None:
                payload[name] = _enum_value(enum_type, payload[name])
        payload["rate_order"] = _normalize_rate_orders((payload["rate_order"],))[0]
        payload["rate_sign"] = _normalize_signs((payload.get("rate_sign", 1),))[0]
        if payload.get("euler_order") is not None:
            payload["euler_order"] = _normalize_euler_orders((payload["euler_order"],))[0]
        return cls(**payload)

    def __post_init__(self) -> None:
        if self.quaternion_order is not None:
            if not isinstance(self.quaternion_order, QuaternionOrder):
                raise TypeError("quaternion_order must be a QuaternionOrder.")

        quaternion = self.quaternion_order is not None
        euler = any(
            value is not None for value in (self.euler_order, self.euler_mode, self.angle_unit)
        )
        if quaternion == euler:
            raise ValueError("A hypothesis must define exactly one attitude representation.")
        if euler and any(
            value is None for value in (self.euler_order, self.euler_mode, self.angle_unit)
        ):
            raise ValueError("Euler hypotheses require order, mode, and angle unit.")

    ####

    def identifier(self) -> str:
        if self.quaternion_order is not None:
            attitude = f"quat:{self.quaternion_order}:{self.attitude_direction}"
        else:
            attitude = (
                f"euler:{self.euler_mode}:{self.euler_order}:{self.angle_unit}:"
                f"{self.attitude_direction}"
            )
        return (
            f"{attitude}|rate:{self.rate_frame}:{self.rate_order}:{self.rate_unit}:"
            f"sign={self.rate_sign:+d}"
        )

    ####


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
class CandidateResult(FrozenModel):
    hypothesis: ConventionHypothesis
    criterion: float
    data_criterion: float
    fit_score: float
    normalized_rmse: float
    multiscale_rmse_rad: float
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
    warnings: tuple[str, ...] = ()


####


@dataclass(frozen=True, slots=True)
class DimensionAssessment(FrozenModel):
    status: DimensionStatus
    selected_value: str | None
    allowed_values: tuple[str, ...]
    evidence_near_tie_values: tuple[str, ...]
    posterior_near_tie_values: tuple[str, ...]
    relative_prior_weights: Mapping[str, float] | None = None

    ####


####


@dataclass(frozen=True, slots=True)
class HypothesisSpaceSummary(FrozenModel):
    base_hypothesis_count: int
    constrained_hypothesis_count: int
    reduction_fraction: float
    fixed_dimensions: Mapping[str, str]
    searched_dimensions: Mapping[str, tuple[str, ...]]
    not_applicable_dimensions: tuple[str, ...]
    relative_prior_weights: Mapping[str, Mapping[str, float]]


####


@dataclass(frozen=True, slots=True)
class OverallDiagnostics(FrozenModel):
    confidence: float
    confidence_label: ConfidenceLabel
    best_relative_support: float
    best_data_relative_support: float
    near_tie_count: int
    near_tie_hypothesis_ids: tuple[str, ...]
    unresolved_dimensions: Mapping[str, tuple[str, ...]]
    posterior_near_tie_count: int
    posterior_near_tie_hypothesis_ids: tuple[str, ...]
    posterior_unresolved_dimensions: Mapping[str, tuple[str, ...]]
    dimension_assessments: Mapping[str, DimensionAssessment]
    effective_excitation_rank: float
    excitation_singular_values: tuple[float, float, float]
    total_observed_motion_rad: float
    observed_step_rms_rad: float
    near_alias_fraction: float
    hard_alias_fraction: float
    time_source: Literal["timestamps", "provided_sample_period", "inferred_sample_period"]
    warnings: tuple[str, ...]

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

    ####


####
