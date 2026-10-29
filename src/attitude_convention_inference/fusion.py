"""Evidence packets and auditable cohort fusion over quotient classes.

The local inference engine answers which quotient classes a trajectory supports.
This module is the product boundary after that run: it serializes prior-separated
class evidence and combines independent packets without changing the local result.
The implementation deliberately uses the package's frozen-dataclass model layer,
so packet exchange uses the same Pydantic-backed model layer as local reports.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from numpy import (
    all as np_all,
)
from numpy import (
    amax,
    argsort,
    asarray,
    dot,
    exp,
    finfo,
    float64,
    full,
    log,
)
from numpy import (
    any as np_any,
)
from numpy import (
    isfinite as np_isfinite,
)
from numpy import (
    sum as np_sum,
)
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass

from .immutable import ReadOnlyMapping, freeze_mapping
from .models import FrozenModel

__all__ = [
    "CalibrationDescriptor",
    "CohortFusionResult",
    "EvidencePacket",
    "EvidenceSemantics",
    "FusionStatus",
    "LineageDescriptor",
    "LogOddsExplanation",
    "build_shared_prototype",
    "coalesce_evidence_packets",
    "fuse_target_with_robust_borrowing",
    "load_evidence_packet",
    "relative_log_evidence_from_posterior",
    "write_fusion_result",
]

if TYPE_CHECKING:
    from .models import InferenceReport
####


class EvidenceSemantics(StrEnum):
    """Meaning of a packet's class-relative evidence values."""

    CALIBRATED_LIKELIHOOD = "calibrated-likelihood"
    CALIBRATED_GENERALIZED = "calibrated-generalized"
    GENERALIZED_WEIGHT = "generalized-weight"
    UNCALIBRATED_RISK = "uncalibrated-risk"
####


class FusionStatus(StrEnum):
    """Top-level status returned by the cohort service."""

    COHORT_POSTERIOR = "cohort-posterior"
    COHORT_WEIGHT_ONLY = "cohort-weight-only"
    COHORT_CONFLICT = "cohort-conflict"
    COHORT_ABSTAIN = "cohort-abstain"
####


def _string_tuple(
    value: Sequence[object] | set[object] | frozenset[object], name: str
) -> tuple[str, ...]:
    values = tuple(str(item) for item in value)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    ####
    return values
####


def _mapping_of_floats(value: Mapping[Any, Any], name: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        number = float(raw_value)
        if not math.isfinite(number):
            raise ValueError(f"{name} values must be finite")
        ####
        result[key] = number
    ####
    return result
####


def _mapping_of_strings(value: Mapping[Any, Any], name: str) -> dict[str, str]:
    return {str(key): str(item) for key, item in value.items()}
####


def _empty_float_mapping() -> dict[str, float]:
    return {}
####


def _empty_string_mapping() -> dict[str, str]:
    return {}
####


@dataclass(frozen=True, slots=True)
class CalibrationDescriptor(FrozenModel):
    """Versioned evidence-to-probability calibration metadata."""

    calibration_id: str | None = None
    validated: bool = False
    validation_domain: str | None = None
    temperature: float | None = None
    effective_sample_definition: str | None = None

    def __post_init__(self) -> None:
        if self.temperature is not None and (
            not math.isfinite(float(self.temperature)) or self.temperature <= 0.0
        ):
            raise ValueError("temperature must be positive and finite")
        ####
    ####
####


@dataclass(frozen=True, slots=True)
class LineageDescriptor(FrozenModel):
    """Atomic source and derivation lineage used for dependence checks."""

    atomic_source_ids: Sequence[str] | set[str] | frozenset[str] = ()
    parent_packet_ids: Sequence[str] | set[str] | frozenset[str] = ()
    dependency_group_ids: Sequence[str] | set[str] | frozenset[str] = ()
    sensor_ids: Sequence[str] | set[str] | frozenset[str] = ()
    time_start_ns: int | None = None
    time_end_ns: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "atomic_source_ids",
            "parent_packet_ids",
            "dependency_group_ids",
            "sensor_ids",
        ):
            value = getattr(self, name)
            if isinstance(value, str):
                value = (value,)
            ####
            object.__setattr__(self, name, _string_tuple(value, name))
        ####
        if (
            self.time_start_ns is not None
            and self.time_end_ns is not None
            and self.time_end_ns <= self.time_start_ns
        ):
            raise ValueError("time_end_ns must be greater than time_start_ns")
        ####
    ####
####


@dataclass(frozen=True, slots=True)
class EvidencePacket(FrozenModel):
    """Immutable local evidence over a versioned quotient-class universe."""

    schema_version: str = "1.0"
    packet_id: str = ""
    quotient_space_id: str = ""
    class_ids: tuple[str, ...] = ()
    log_evidence: ReadOnlyMapping[str, float] = field(default_factory=_empty_float_mapping)
    local_prior: ReadOnlyMapping[str, float] = field(default_factory=_empty_float_mapping)
    contender_set: tuple[str, ...] = ()
    evidence_semantics: EvidenceSemantics = EvidenceSemantics.UNCALIBRATED_RISK
    calibration: CalibrationDescriptor = field(default_factory=CalibrationDescriptor)
    lineage: LineageDescriptor = field(default_factory=LineageDescriptor)
    provenance: ReadOnlyMapping[str, str] = field(default_factory=_empty_string_mapping)
    hard_anchors: tuple[str, ...] = ()
    excitation_rank: int | None = None
    excitation_eigenvalues: tuple[float, ...] = ()
    local_result_reason: str | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        class_ids = _string_tuple(self.class_ids, "class_ids")
        if not class_ids:
            raise ValueError("class_ids must be non-empty")
        ####
        log_evidence = _mapping_of_floats(self.log_evidence, "log_evidence")
        local_prior = _mapping_of_floats(self.local_prior, "local_prior")
        class_set = set(class_ids)
        if set(log_evidence) != class_set:
            raise ValueError("log_evidence keys must exactly match class_ids")
        ####
        if set(local_prior) != class_set:
            raise ValueError("local_prior keys must exactly match class_ids")
        ####
        evidence_values = asarray([log_evidence[key] for key in class_ids])
        if not math.isclose(float(amax(evidence_values)), 0.0, abs_tol=1.0e-9):
            raise ValueError("log_evidence must use max-zero normalization")
        ####
        prior_values = asarray([local_prior[key] for key in class_ids])
        if np_any(prior_values <= 0.0) or not math.isclose(
            float(np_sum(prior_values)), 1.0, abs_tol=1.0e-9
        ):
            raise ValueError("local_prior must be strictly positive and sum to one")
        ####
        contender_set = _string_tuple(self.contender_set, "contender_set")
        if not set(contender_set).issubset(class_set):
            raise ValueError("contender_set must be a subset of class_ids")
        ####
        eigenvalues = tuple(float(value) for value in self.excitation_eigenvalues)
        if any(not math.isfinite(value) or value < 0.0 for value in eigenvalues):
            raise ValueError("excitation_eigenvalues must be finite and nonnegative")
        ####
        if self.excitation_rank is not None and not 0 <= int(self.excitation_rank) <= 3:
            raise ValueError("excitation_rank must lie in [0, 3]")
        ####
        calibration = self.calibration
        if isinstance(calibration, Mapping):
            calibration = CalibrationDescriptor.model_validate(calibration)
        ####
        lineage = self.lineage
        if isinstance(lineage, Mapping):
            lineage = LineageDescriptor.model_validate(lineage)
        ####
        object.__setattr__(self, "class_ids", class_ids)
        object.__setattr__(self, "log_evidence", freeze_mapping(log_evidence))
        object.__setattr__(self, "local_prior", freeze_mapping(local_prior))
        object.__setattr__(self, "contender_set", contender_set)
        object.__setattr__(self, "evidence_semantics", EvidenceSemantics(self.evidence_semantics))
        object.__setattr__(self, "calibration", calibration)
        object.__setattr__(self, "lineage", lineage)
        object.__setattr__(
            self,
            "provenance",
            freeze_mapping(_mapping_of_strings(self.provenance, "provenance")),
        )
        object.__setattr__(self, "hard_anchors", _string_tuple(self.hard_anchors, "hard_anchors"))
        object.__setattr__(self, "excitation_eigenvalues", eigenvalues)
        object.__setattr__(self, "warnings", _string_tuple(self.warnings, "warnings"))
    ####

    @property
    def probability_ready(self) -> bool:
        return (
            self.evidence_semantics
            in {
                EvidenceSemantics.CALIBRATED_LIKELIHOOD,
                EvidenceSemantics.CALIBRATED_GENERALIZED,
            }
            and self.calibration.validated
        )
    ####

    def evidence_vector(self) -> NDArray[float64]:
        return asarray([self.log_evidence[key] for key in self.class_ids], dtype=float64)
    ####

    def local_prior_vector(self) -> NDArray[float64]:
        return asarray([self.local_prior[key] for key in self.class_ids], dtype=float64)
    ####

    def local_posterior(self) -> dict[str, float]:
        posterior = _normalize_log_weights(log(self.local_prior_vector()) + self.evidence_vector())
        return _vector_to_mapping(self.class_ids, posterior)
    ####

    @classmethod
    def from_inference_report(
        cls,
        report: InferenceReport,
        *,
        packet_id: str,
        quotient_space_id: str = "orientation-qspace-v1",
        evidence_semantics: EvidenceSemantics = EvidenceSemantics.UNCALIBRATED_RISK,
        calibration: CalibrationDescriptor | Mapping[str, Any] | None = None,
        lineage: LineageDescriptor | Mapping[str, Any] | None = None,
        provenance: Mapping[str, str] | None = None,
        hard_anchors: Sequence[str] = (),
    ) -> Self:
        """Serialize a local report without collapsing its evidence contenders."""

        class_risks: dict[str, float] = {}
        class_prior_mass: dict[str, float] = {}
        for result in report.rankings:
            class_id = result.equivalence_class_id
            class_risks[class_id] = min(
                class_risks.get(class_id, float("inf")), result.class_data_risk
            )
            class_prior_mass[class_id] = class_prior_mass.get(class_id, 0.0) + max(
                result.prior_relative_weight, 0.0
            )
        ####
        if not class_risks:
            raise ValueError("Inference report contains no quotient-class rankings")
        ####
        class_ids = tuple(sorted(class_risks))
        raw_evidence = {class_id: -class_risks[class_id] for class_id in class_ids}
        maximum = max(raw_evidence.values())
        log_evidence = {class_id: value - maximum for class_id, value in raw_evidence.items()}
        prior_total = sum(class_prior_mass.values())
        if prior_total <= 0.0:
            local_prior = {class_id: 1.0 / len(class_ids) for class_id in class_ids}
        else:
            local_prior = {
                class_id: class_prior_mass.get(class_id, 0.0) / prior_total
                for class_id in class_ids
            }
            floor = finfo(float).tiny
            local_prior = {class_id: max(value, floor) for class_id, value in local_prior.items()}
            normalizer = sum(local_prior.values())
            local_prior = {class_id: value / normalizer for class_id, value in local_prior.items()}
        ####
        diagnostics = report.diagnostics
        return cls(
            packet_id=packet_id,
            quotient_space_id=quotient_space_id,
            class_ids=class_ids,
            log_evidence=log_evidence,
            local_prior=local_prior,
            contender_set=tuple(
                class_id
                for class_id in report.evidence_contender_class_ids
                if class_id in class_risks
            )
            or class_ids,
            evidence_semantics=evidence_semantics,
            calibration=(
                calibration
                if isinstance(calibration, CalibrationDescriptor)
                else CalibrationDescriptor.model_validate(calibration or {})
            ),
            lineage=(
                lineage
                if isinstance(lineage, LineageDescriptor)
                else LineageDescriptor.model_validate(lineage or {})
            ),
            provenance=provenance or {},
            hard_anchors=tuple(hard_anchors),
            excitation_rank=diagnostics.raw_excitation_rank,
            excitation_eigenvalues=diagnostics.raw_excitation_eigenvalues,
            local_result_reason=report.operational_reason,
            warnings=diagnostics.warnings,
        )
    ####
####


@dataclass(frozen=True, slots=True)
class LogOddsExplanation(FrozenModel):
    """Pairwise local/cohort/fused log-odds decomposition."""

    winner_class: str
    runner_up_class: str
    local_log_odds: float
    borrowed_prior_log_odds: float
    fused_log_odds: float
####


@dataclass(frozen=True, slots=True)
class CohortFusionResult(FrozenModel):
    """Auditable result returned beside the target's immutable local packet."""

    target_packet_id: str
    reference_packet_ids: tuple[str, ...]
    quotient_space_id: str
    class_ids: tuple[str, ...]
    model_name: str = "robust-sharedness-mixture"
    local_posterior: ReadOnlyMapping[str, float] = field(default_factory=_empty_float_mapping)
    prototype_posterior: ReadOnlyMapping[str, float] = field(default_factory=_empty_float_mapping)
    borrowed_prior: ReadOnlyMapping[str, float] = field(default_factory=_empty_float_mapping)
    fused_posterior: ReadOnlyMapping[str, float] = field(default_factory=_empty_float_mapping)
    fused_contender_set: tuple[str, ...] = ()
    prior_sharedness: float = 0.0
    posterior_sharedness: float = 0.0
    probability_valid: bool = False
    status: FusionStatus = FusionStatus.COHORT_ABSTAIN
    explanation: LogOddsExplanation = field(
        default_factory=lambda: LogOddsExplanation("", "", 0.0, 0.0, 0.0)
    )
    calibration_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        class_ids = _string_tuple(self.class_ids, "class_ids")
        if not class_ids:
            raise ValueError("class_ids must be non-empty")
        ####
        for name in ("local_posterior", "prototype_posterior", "borrowed_prior", "fused_posterior"):
            mapping = _mapping_of_floats(getattr(self, name), name)
            if set(mapping) != set(class_ids):
                raise ValueError(f"{name} keys must exactly match class_ids")
            ####
            object.__setattr__(self, name, freeze_mapping(mapping))
        ####
        object.__setattr__(self, "class_ids", class_ids)
        object.__setattr__(
            self,
            "reference_packet_ids",
            _string_tuple(self.reference_packet_ids, "reference_packet_ids"),
        )
        contenders = _string_tuple(self.fused_contender_set, "fused_contender_set")
        if not set(contenders).issubset(set(class_ids)):
            raise ValueError("fused_contender_set must be a subset of class_ids")
        ####
        object.__setattr__(self, "fused_contender_set", contenders)
        object.__setattr__(self, "status", FusionStatus(self.status))
        explanation = self.explanation
        if isinstance(explanation, Mapping):
            explanation = LogOddsExplanation.model_validate(explanation)
        ####
        object.__setattr__(self, "explanation", explanation)
        object.__setattr__(
            self,
            "calibration_ids",
            _string_tuple(self.calibration_ids, "calibration_ids"),
        )
        object.__setattr__(self, "warnings", _string_tuple(self.warnings, "warnings"))
        for name in ("prior_sharedness", "posterior_sharedness"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            ####
            object.__setattr__(self, name, value)
        ####
    ####
####


def relative_log_evidence_from_posterior(
    posterior: Mapping[str, float],
    prior: Mapping[str, float],
    class_ids: Sequence[str],
) -> dict[str, float]:
    """Remove a known local prior and return max-zero relative log evidence."""

    posterior_vector = _validated_probability_vector(posterior, class_ids, "posterior")
    prior_vector = _validated_probability_vector(prior, class_ids, "prior")
    if np_any(posterior_vector <= 0.0) or np_any(prior_vector <= 0.0):
        raise ValueError("posterior and prior must have strictly positive support")
    ####
    log_evidence = log(posterior_vector) - log(prior_vector)
    log_evidence -= amax(log_evidence)
    return _vector_to_mapping(class_ids, log_evidence)
####


def build_shared_prototype(
    reference_packets: Sequence[EvidencePacket],
    *,
    baseline_prior: Mapping[str, float] | None = None,
    strict_lineage: bool = True,
    power_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Coalesce independent reference packets into one shared-class posterior."""

    if not reference_packets:
        raise ValueError("at least one reference packet is required")
    ####
    class_ids = _validate_compatibility(reference_packets)
    _validate_lineage(reference_packets, strict=strict_lineage)
    prior_vector = _resolve_baseline_prior(class_ids, baseline_prior)
    log_weights = log(prior_vector)
    for packet in reference_packets:
        power = 1.0 if power_weights is None else power_weights.get(packet.packet_id, 1.0)
        if not math.isfinite(power) or not 0.0 <= power <= 1.0:
            raise ValueError(f"invalid power weight for packet {packet.packet_id!r}")
        ####
        log_weights = log_weights + power * packet.evidence_vector()
    ####
    return _vector_to_mapping(class_ids, _normalize_log_weights(log_weights))
####


def coalesce_evidence_packets(
    reference_packets: Sequence[EvidencePacket],
    *,
    baseline_prior: Mapping[str, float] | None = None,
    strict_lineage: bool = True,
    power_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Named product-boundary alias for shared reference evidence coalescence."""

    return build_shared_prototype(
        reference_packets,
        baseline_prior=baseline_prior,
        strict_lineage=strict_lineage,
        power_weights=power_weights,
    )
####


def fuse_target_with_robust_borrowing(
    target_packet: EvidencePacket,
    reference_packets: Sequence[EvidencePacket],
    *,
    prior_sharedness: float,
    baseline_prior: Mapping[str, float] | None = None,
    strict_lineage: bool = True,
    power_weights: Mapping[str, float] | None = None,
    accept_probability: float = 0.80,
    accept_margin: float = 0.20,
    minimum_sharedness: float = 0.50,
) -> CohortFusionResult:
    """Fuse a target with a reference cohort using a robust sharedness mixture."""

    if not 0.0 <= prior_sharedness <= 1.0:
        raise ValueError("prior_sharedness must lie in [0, 1]")
    ####
    for name, value in {
        "accept_probability": accept_probability,
        "accept_margin": accept_margin,
        "minimum_sharedness": minimum_sharedness,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
        ####
    ####
    all_packets = [*reference_packets, target_packet]
    if len({packet.packet_id for packet in all_packets}) != len(all_packets):
        raise ValueError("packet IDs must be unique within one fusion")
    ####
    class_ids = _validate_compatibility(all_packets)
    _validate_lineage(all_packets, strict=strict_lineage)
    prior_vector = _resolve_baseline_prior(class_ids, baseline_prior)
    prototype_mapping = build_shared_prototype(
        reference_packets,
        baseline_prior=_vector_to_mapping(class_ids, prior_vector),
        strict_lineage=False,
        power_weights=power_weights,
    )
    prototype_vector = _mapping_to_vector(prototype_mapping, class_ids)
    borrowed_prior_vector = (
        prior_sharedness * prototype_vector + (1.0 - prior_sharedness) * prior_vector
    )
    target_evidence = target_packet.evidence_vector()
    fused_vector = _normalize_log_weights(log(borrowed_prior_vector) + target_evidence)
    local_vector = _normalize_log_weights(log(target_packet.local_prior_vector()) + target_evidence)
    relative_likelihood = exp(target_evidence)
    shared_predictive = float(dot(prototype_vector, relative_likelihood))
    baseline_predictive = float(dot(prior_vector, relative_likelihood))
    denominator = (
        prior_sharedness * shared_predictive + (1.0 - prior_sharedness) * baseline_predictive
    )
    posterior_sharedness = (
        prior_sharedness * shared_predictive / denominator if denominator > 0.0 else 0.0
    )
    ranking = argsort(-fused_vector)
    winner_index = int(ranking[0])
    runner_up_index = int(ranking[1]) if len(ranking) > 1 else winner_index
    winner_probability = float(fused_vector[winner_index])
    runner_up_probability = float(fused_vector[runner_up_index])
    margin = winner_probability - runner_up_probability
    local_log_odds = float(target_evidence[winner_index] - target_evidence[runner_up_index])
    borrowed_log_odds = float(
        math.log(borrowed_prior_vector[winner_index])
        - math.log(borrowed_prior_vector[runner_up_index])
    )
    explanation = LogOddsExplanation(
        winner_class=class_ids[winner_index],
        runner_up_class=class_ids[runner_up_index],
        local_log_odds=local_log_odds,
        borrowed_prior_log_odds=borrowed_log_odds,
        fused_log_odds=local_log_odds + borrowed_log_odds,
    )
    probability_valid = all(packet.probability_ready for packet in all_packets)
    warnings: list[str] = []
    if not probability_valid:
        warnings.append(
            "At least one packet is not probability-ready; fused values are evidence weights."
        )
    ####
    if probability_valid:
        if posterior_sharedness < minimum_sharedness:
            status = FusionStatus.COHORT_CONFLICT
        elif winner_probability < accept_probability or margin < accept_margin:
            status = FusionStatus.COHORT_ABSTAIN
        else:
            status = FusionStatus.COHORT_POSTERIOR
        ####
    else:
        status = FusionStatus.COHORT_WEIGHT_ONLY
    ####
    fused_contender_set = tuple(
        class_id
        for class_id, probability in zip(class_ids, fused_vector, strict=True)
        if winner_probability - float(probability) <= accept_margin
    )
    calibration_ids = tuple(
        sorted(
            {
                packet.calibration.calibration_id
                for packet in all_packets
                if packet.calibration.calibration_id is not None
            }
        )
    )
    return CohortFusionResult(
        target_packet_id=target_packet.packet_id,
        reference_packet_ids=tuple(packet.packet_id for packet in reference_packets),
        quotient_space_id=target_packet.quotient_space_id,
        class_ids=class_ids,
        local_posterior=_vector_to_mapping(class_ids, local_vector),
        prototype_posterior=prototype_mapping,
        borrowed_prior=_vector_to_mapping(class_ids, borrowed_prior_vector),
        fused_posterior=_vector_to_mapping(class_ids, fused_vector),
        fused_contender_set=fused_contender_set,
        prior_sharedness=prior_sharedness,
        posterior_sharedness=posterior_sharedness,
        probability_valid=probability_valid,
        status=status,
        explanation=explanation,
        calibration_ids=calibration_ids,
        warnings=tuple(warnings),
    )
####


def _validate_compatibility(packets: Sequence[EvidencePacket]) -> tuple[str, ...]:
    if not packets:
        raise ValueError("at least one packet is required")
    ####
    first = packets[0]
    for packet in packets[1:]:
        if packet.quotient_space_id != first.quotient_space_id:
            raise ValueError(
                "quotient_space_id mismatch: "
                f"{first.quotient_space_id!r} != {packet.quotient_space_id!r}"
            )
        ####
        if packet.class_ids != first.class_ids:
            raise ValueError("class_ids/order mismatch across packets")
        ####
    ####
    return first.class_ids
####


def _validate_lineage(packets: Sequence[EvidencePacket], *, strict: bool) -> None:
    if not strict:
        return
    ####
    for packet in packets:
        if not packet.lineage.atomic_source_ids:
            raise ValueError(
                f"packet {packet.packet_id!r} lacks atomic_source_ids required by strict lineage"
            )
        ####
    ####
    for index, left in enumerate(packets):
        for right in packets[index + 1 :]:
            overlap = set(left.lineage.atomic_source_ids) & set(right.lineage.atomic_source_ids)
            if overlap:
                raise ValueError(
                    "dependent packets share atomic sources: "
                    f"{left.packet_id!r}, {right.packet_id!r}, overlap={sorted(overlap)!r}"
                )
            ####
            group_overlap = set(left.lineage.dependency_group_ids) & set(
                right.lineage.dependency_group_ids
            )
            if group_overlap:
                raise ValueError(
                    "dependent packets share dependency groups: "
                    f"{left.packet_id!r}, {right.packet_id!r}, overlap={sorted(group_overlap)!r}"
                )
            ####
            if (
                left.packet_id in right.lineage.parent_packet_ids
                or right.packet_id in left.lineage.parent_packet_ids
            ):
                raise ValueError(
                    f"packets {left.packet_id!r} and {right.packet_id!r} have "
                    "direct derivation dependence"
                )
            ####
        ####
    ####
####


def _resolve_baseline_prior(
    class_ids: Sequence[str], baseline_prior: Mapping[str, float] | None
) -> NDArray[float64]:
    if baseline_prior is None:
        return full(len(class_ids), 1.0 / len(class_ids), dtype=float64)
    ####
    return _validated_probability_vector(baseline_prior, class_ids, "baseline_prior")
####


def _validated_probability_vector(
    mapping: Mapping[str, float], class_ids: Sequence[str], name: str
) -> NDArray[float64]:
    if set(mapping) != set(class_ids):
        raise ValueError(f"{name} keys must exactly match class_ids")
    ####
    vector = _mapping_to_vector(mapping, class_ids)
    if not np_all(np_isfinite(vector)) or np_any(vector <= 0.0):
        raise ValueError(f"{name} values must be finite and strictly positive")
    ####
    if not math.isclose(float(np_sum(vector)), 1.0, abs_tol=1.0e-9):
        raise ValueError(f"{name} must sum to one")
    ####
    return vector
####


def _mapping_to_vector(mapping: Mapping[str, float], class_ids: Sequence[str]) -> NDArray[float64]:
    return asarray([mapping[class_id] for class_id in class_ids], dtype=float64)
####


def _vector_to_mapping(
    class_ids: Sequence[str], vector: Sequence[float] | NDArray[float64]
) -> dict[str, float]:
    return {class_id: float(value) for class_id, value in zip(class_ids, vector, strict=True)}
####


def _normalize_log_weights(log_weights: NDArray[float64]) -> NDArray[float64]:
    if log_weights.ndim != 1 or not np_all(np_isfinite(log_weights)):
        raise ValueError("log weights must be a finite one-dimensional vector")
    ####
    shifted = log_weights - float(amax(log_weights))
    weights = exp(shifted)
    return weights / float(np_sum(weights))
####


def load_evidence_packet(path: str | Path) -> EvidencePacket:
    """Load and validate one packet JSON document."""

    return EvidencePacket.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
####


def write_fusion_result(path: str | Path, result: CohortFusionResult) -> None:
    """Write one validated fusion product as stable JSON."""

    Path(path).write_text(result.model_dump_json(indent=2), encoding="utf-8")
####


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse orientation evidence packets.")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--prior-sharedness", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-unknown-or-overlapping-lineage", action="store_true")
    parser.add_argument("--accept-probability", type=float, default=0.80)
    parser.add_argument("--accept-margin", type=float, default=0.20)
    parser.add_argument("--minimum-sharedness", type=float, default=0.50)
    return parser.parse_args(argv)
####


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = fuse_target_with_robust_borrowing(
        load_evidence_packet(args.target),
        [load_evidence_packet(path) for path in args.reference],
        prior_sharedness=args.prior_sharedness,
        strict_lineage=not args.allow_unknown_or_overlapping_lineage,
        accept_probability=args.accept_probability,
        accept_margin=args.accept_margin,
        minimum_sharedness=args.minimum_sharedness,
    )
    write_fusion_result(args.output, result)
    return 0
####


if __name__ == "__main__":
    raise SystemExit(main())
####
