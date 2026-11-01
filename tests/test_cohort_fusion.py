from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from attitude_convention_inference import (
    CalibrationDescriptor,
    EvidencePacket,
    EvidenceSemantics,
    FusionStatus,
    LineageDescriptor,
    coalesce_evidence_packets,
    fuse_target_with_robust_borrowing,
    relative_log_evidence_from_posterior,
)

CLASS_IDS = ("C1", "C2", "C3")
UNIFORM = {class_id: 1.0 / 3.0 for class_id in CLASS_IDS}
CALIBRATION = CalibrationDescriptor(
    calibration_id="orientation-risk-temp-v1",
    validated=True,
    validation_domain="synthetic-reference",
    temperature=1.0,
    effective_sample_definition="independent validation blocks",
)


def packet(
    packet_id: str,
    likelihood_ratio: tuple[float, float, float],
    source_id: str,
    *,
    calibrated: bool = True,
    quotient_space_id: str = "orientation-qspace-v1",
) -> EvidencePacket:
    maximum = max(likelihood_ratio)
    log_evidence = {
        class_id: math.log(value / maximum)
        for class_id, value in zip(CLASS_IDS, likelihood_ratio, strict=True)
    }
    return EvidencePacket(
        packet_id=packet_id,
        quotient_space_id=quotient_space_id,
        class_ids=CLASS_IDS,
        log_evidence=log_evidence,
        local_prior=UNIFORM,
        contender_set=CLASS_IDS,
        evidence_semantics=(
            EvidenceSemantics.CALIBRATED_LIKELIHOOD
            if calibrated
            else EvidenceSemantics.UNCALIBRATED_RISK
        ),
        calibration=CALIBRATION if calibrated else CalibrationDescriptor(),
        lineage=LineageDescriptor(atomic_source_ids=frozenset({source_id})),
        excitation_rank=3,
    )
####


def test_document_example_matches_expected_values() -> None:
    result = fuse_target_with_robust_borrowing(
        packet("C", (1.4, 1.0, 0.8), "source-C"),
        [packet("A", (9.0, 1.0, 0.5), "source-A"), packet("B", (6.0, 1.0, 0.5), "source-B")],
        prior_sharedness=0.8,
    )

    assert result.status is FusionStatus.COHORT_POSTERIOR
    assert result.probability_valid
    assert result.prototype_posterior["C1"] == pytest.approx(0.97737557, abs=1.0e-8)
    assert result.local_posterior["C1"] == pytest.approx(0.4375, abs=1.0e-8)
    assert result.fused_posterior["C1"] == pytest.approx(0.89634924, abs=1.0e-8)
    assert result.posterior_sharedness == pytest.approx(0.83903860, abs=1.0e-8)
####


def test_target_conflict_reduces_sharedness_and_abstains() -> None:
    result = fuse_target_with_robust_borrowing(
        packet("C", (0.3, 3.0, 1.0), "source-C"),
        [packet("A", (9.0, 1.0, 0.5), "source-A"), packet("B", (6.0, 1.0, 0.5), "source-B")],
        prior_sharedness=0.8,
    )

    assert result.status is FusionStatus.COHORT_CONFLICT
    assert result.posterior_sharedness == pytest.approx(0.49556771, abs=1.0e-8)
    assert result.fused_posterior["C1"] == pytest.approx(0.44795371, abs=1.0e-8)
    assert result.fused_posterior["C2"] == pytest.approx(0.42836669, abs=1.0e-8)
####


def test_overlap_and_dependency_groups_are_rejected() -> None:
    shared = packet("C", (1.4, 1.0, 0.8), "shared-source")
    with pytest.raises(ValueError, match="dependent packets share atomic sources"):
        fuse_target_with_robust_borrowing(
            shared,
            [packet("A", (9.0, 1.0, 0.5), "shared-source")],
            prior_sharedness=0.8,
        )
    ####

    grouped = LineageDescriptor(
        atomic_source_ids=("source-C",), dependency_group_ids=("derived-group",)
    )
    target = packet("C", (1.4, 1.0, 0.8), "source-C").model_copy(update={"lineage": grouped})
    reference = packet("A", (9.0, 1.0, 0.5), "source-A").model_copy(
        update={
            "lineage": LineageDescriptor(
                atomic_source_ids=("source-A",), dependency_group_ids=("derived-group",)
            )
        }
    )
    with pytest.raises(ValueError, match="share dependency groups"):
        fuse_target_with_robust_borrowing(target, [reference], prior_sharedness=0.8)
    ####
####


def test_shared_sensor_with_overlapping_time_windows_is_rejected() -> None:
    target = packet("C", (1.4, 1.0, 0.8), "source-C").model_copy(
        update={
            "lineage": LineageDescriptor(
                atomic_source_ids=("source-C",),
                sensor_ids=("imu-7",),
                time_start_ns=0,
                time_end_ns=100,
            )
        }
    )
    reference = packet("A", (9.0, 1.0, 0.5), "source-A").model_copy(
        update={
            "lineage": LineageDescriptor(
                atomic_source_ids=("source-A",),
                sensor_ids=("imu-7",),
                time_start_ns=100,
                time_end_ns=200,
            )
        }
    )

    with pytest.raises(ValueError, match="overlapping time windows"):
        fuse_target_with_robust_borrowing(target, [reference], prior_sharedness=0.8)
    ####
####


def test_shared_sensor_requires_time_bounds_under_strict_lineage() -> None:
    target = packet("C", (1.4, 1.0, 0.8), "source-C").model_copy(
        update={
            "lineage": LineageDescriptor(
                atomic_source_ids=("source-C",),
                sensor_ids=("imu-7",),
            )
        }
    )
    reference = packet("A", (9.0, 1.0, 0.5), "source-A").model_copy(
        update={
            "lineage": LineageDescriptor(
                atomic_source_ids=("source-A",),
                sensor_ids=("imu-7",),
            )
        }
    )

    with pytest.raises(ValueError, match="lack complete time bounds"):
        fuse_target_with_robust_borrowing(target, [reference], prior_sharedness=0.8)
    ####
####


def test_unknown_sensor_overlap_can_only_be_overridden_explicitly() -> None:
    target = packet("C", (1.4, 1.0, 0.8), "source-C").model_copy(
        update={
            "lineage": LineageDescriptor(
                atomic_source_ids=("source-C",),
                sensor_ids=("imu-7",),
            )
        }
    )
    reference = packet("A", (9.0, 1.0, 0.5), "source-A").model_copy(
        update={
            "lineage": LineageDescriptor(
                atomic_source_ids=("source-A",),
                sensor_ids=("imu-7",),
            )
        }
    )

    result = fuse_target_with_robust_borrowing(
        target,
        [reference],
        prior_sharedness=0.8,
        strict_lineage=False,
    )

    assert result.target_packet_id == "C"
####


def test_same_sensor_nonoverlapping_time_windows_can_be_fused() -> None:
    target = packet("C", (1.4, 1.0, 0.8), "source-C").model_copy(
        update={
            "lineage": LineageDescriptor(
                atomic_source_ids=("source-C",),
                sensor_ids=("imu-7",),
                time_start_ns=0,
                time_end_ns=100,
            )
        }
    )
    reference = packet("A", (9.0, 1.0, 0.5), "source-A").model_copy(
        update={
            "lineage": LineageDescriptor(
                atomic_source_ids=("source-A",),
                sensor_ids=("imu-7",),
                time_start_ns=101,
                time_end_ns=200,
            )
        }
    )

    result = fuse_target_with_robust_borrowing(target, [reference], prior_sharedness=0.8)

    assert result.target_packet_id == "C"
####


def test_lineage_time_bounds_must_be_provided_together() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        LineageDescriptor(time_start_ns=100)
    ####
####


def test_uncalibrated_packet_returns_weight_only_status() -> None:
    result = fuse_target_with_robust_borrowing(
        packet("C", (1.4, 1.0, 0.8), "source-C", calibrated=False),
        [packet("A", (9.0, 1.0, 0.5), "source-A")],
        prior_sharedness=0.8,
    )
    assert result.status is FusionStatus.COHORT_WEIGHT_ONLY
    assert not result.probability_valid
    assert result.warnings
####


@pytest.mark.parametrize("missing_field", ("calibration_id", "validation_domain"))
def test_validated_calibration_requires_auditable_identity_and_domain(
    missing_field: str,
) -> None:
    payload: dict[str, object] = {
        "calibration_id": "cal-v1",
        "validated": True,
        "validation_domain": "held-out-sensors-v1",
    }
    del payload[missing_field]

    with pytest.raises(ValueError, match=f"requires {missing_field}"):
        CalibrationDescriptor.model_validate(payload)
    ####
####


def test_packet_identifiers_must_be_nonempty() -> None:
    with pytest.raises(ValueError, match="packet_id must be non-empty"):
        packet("", (9.0, 1.0, 0.5), "source-A")
    ####

    with pytest.raises(ValueError, match="quotient_space_id must be non-empty"):
        packet(
            "A",
            (9.0, 1.0, 0.5),
            "source-A",
            quotient_space_id="  ",
        )
    ####
####


def test_unsupported_packet_schema_version_is_rejected() -> None:
    payload: dict[str, Any] = packet(
        "A",
        (9.0, 1.0, 0.5),
        "source-A",
    ).model_dump()
    payload["schema_version"] = "2.0"

    with pytest.raises(ValueError, match="unsupported EvidencePacket schema_version"):
        EvidencePacket.model_validate(payload)
    ####
####


def test_duplicate_packet_ids_are_rejected() -> None:
    duplicate = packet("A", (9.0, 1.0, 0.5), "source-A")

    with pytest.raises(ValueError, match="packet IDs must be unique"):
        fuse_target_with_robust_borrowing(
            duplicate,
            [duplicate],
            prior_sharedness=0.8,
        )
    ####
####


def test_coalescence_rejects_duplicate_packet_ids_with_distinct_sources() -> None:
    first = packet("same-id", (9.0, 1.0, 0.5), "source-A")
    second = packet("same-id", (6.0, 1.0, 0.5), "source-B")

    with pytest.raises(ValueError, match="packet IDs must be unique"):
        coalesce_evidence_packets([first, second])
    ####
####


def test_empty_reference_cohort_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one reference packet"):
        fuse_target_with_robust_borrowing(
            packet("C", (1.4, 1.0, 0.8), "source-C"),
            [],
            prior_sharedness=0.8,
        )
    ####
####


def test_invalid_power_and_sharedness_are_rejected() -> None:
    references = [packet("A", (9.0, 1.0, 0.5), "source-A")]
    with pytest.raises(ValueError, match="invalid power weight"):
        coalesce_evidence_packets(references, power_weights={"A": 1.5})
    ####

    with pytest.raises(ValueError, match="unknown packet IDs"):
        coalesce_evidence_packets(references, power_weights={"unknown": 0.5})
    ####

    with pytest.raises(ValueError, match="prior_sharedness"):
        fuse_target_with_robust_borrowing(
            packet("C", (1.4, 1.0, 0.8), "source-C"),
            references,
            prior_sharedness=1.1,
        )
    ####
####


def test_nonfinite_packet_evidence_and_nonpositive_priors_are_rejected() -> None:
    base = packet("A", (9.0, 1.0, 0.5), "source-A")

    with pytest.raises(ValueError, match="log_evidence values must be finite"):
        base.model_copy(
            update={
                "log_evidence": {"C1": 0.0, "C2": math.nan, "C3": -1.0},
            }
        )
    ####

    with pytest.raises(ValueError, match="local_prior must be strictly positive"):
        base.model_copy(
            update={
                "local_prior": {"C1": 0.0, "C2": 0.5, "C3": 0.5},
            }
        )
    ####
####


def test_coalescence_rejects_class_order_and_baseline_prior_mismatches() -> None:
    reordered = packet("B", (6.0, 1.0, 0.5), "source-B").model_copy(
        update={"class_ids": ("C2", "C1", "C3")}
    )
    with pytest.raises(ValueError, match="class_ids/order mismatch"):
        coalesce_evidence_packets([packet("A", (9.0, 1.0, 0.5), "source-A"), reordered])
    ####

    with pytest.raises(ValueError, match="baseline_prior keys must exactly match"):
        coalesce_evidence_packets(
            [packet("A", (9.0, 1.0, 0.5), "source-A")],
            baseline_prior={"C1": 0.5, "C2": 0.5},
        )
    ####
####


def test_quotient_mismatch_and_prior_removal_are_explicit() -> None:
    with pytest.raises(ValueError, match="quotient_space_id mismatch"):
        fuse_target_with_robust_borrowing(
            packet("C", (1.4, 1.0, 0.8), "source-C"),
            [packet("A", (9.0, 1.0, 0.5), "source-A", quotient_space_id="v2")],
            prior_sharedness=0.8,
        )
    ####

    prior = {"C1": 0.5, "C2": 0.3, "C3": 0.2}
    likelihood = {"C1": 4.0, "C2": 1.0, "C3": 0.5}
    denominator = sum(prior[key] * likelihood[key] for key in CLASS_IDS)
    posterior = {key: prior[key] * likelihood[key] / denominator for key in CLASS_IDS}
    recovered = relative_log_evidence_from_posterior(posterior, prior, CLASS_IDS)
    assert recovered["C1"] == pytest.approx(0.0)
    assert math.exp(recovered["C2"]) == pytest.approx(0.25)
    assert math.exp(recovered["C3"]) == pytest.approx(0.125)
####


def test_coalescence_and_json_contract_round_trip(tmp_path: Path) -> None:
    references = [
        packet("A", (9.0, 1.0, 0.5), "source-A"),
        packet("B", (6.0, 1.0, 0.5), "source-B"),
    ]
    prototype = coalesce_evidence_packets(references)
    assert prototype["C1"] == pytest.approx(0.9773755656)

    encoded = packet("A", (9.0, 1.0, 0.5), "source-A").model_dump_json()
    restored = EvidencePacket.model_validate(json.loads(encoded))
    assert restored == references[0]
    assert (tmp_path / "packet.json").write_text(encoded, encoding="utf-8") > 0
####


def test_packet_from_report_preserves_class_contenders() -> None:
    result_one = SimpleNamespace(
        equivalence_class_id="C1", class_data_risk=0.1, prior_relative_weight=1.0
    )
    result_two = SimpleNamespace(
        equivalence_class_id="C2", class_data_risk=0.2, prior_relative_weight=1.0
    )
    report = SimpleNamespace(
        rankings=(result_one, result_two),
        evidence_contender_class_ids=("C1", "C2"),
        operational_reason="trajectory-tie",
        diagnostics=SimpleNamespace(
            raw_excitation_rank=3,
            raw_excitation_eigenvalues=(3.0, 2.0, 1.0),
            warnings=(),
        ),
    )
    packet_result = EvidencePacket.from_inference_report(cast(Any, report), packet_id="local")
    assert packet_result.contender_set == ("C1", "C2")
    assert packet_result.log_evidence["C1"] == pytest.approx(0.0)
    assert packet_result.excitation_rank == 3
####
