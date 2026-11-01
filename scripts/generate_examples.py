"""Regenerate the checked-in example data files."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np

from attitude_convention_inference import (
    AttitudeDirection,
    AttitudeRepresentation,
    CalibrationDescriptor,
    ConventionConstraints,
    ConventionPriors,
    EvidencePacket,
    EvidenceSemantics,
    InferenceConfig,
    LineageDescriptor,
    PreferenceProfile,
    QuaternionOrder,
    RateFrame,
    fuse_target_with_robust_borrowing,
)
from attitude_convention_inference.synthetic import (
    SyntheticConvention,
    encode_synthetic_trajectory,
    rich_three_axis_rates,
)

CLASS_IDS = ("C1", "C2", "C3")
UNIFORM_PRIOR = {class_id: 1.0 / 3.0 for class_id in CLASS_IDS}
CALIBRATION = CalibrationDescriptor(
    calibration_id="orientation-risk-temp-v1",
    validated=True,
    validation_domain="synthetic-reference",
    temperature=1.0,
    effective_sample_definition="independent validation blocks",
)

DOMAIN_PRIOR_DOCUMENT: dict[str, Any] = {
    "generated": "2026-08-14",
    "semantics": {
        "survey_scores_are": "documentation-support scores",
        "survey_scores_are_not": [
            "calibrated probabilities",
            "market-share estimates",
            "trajectory evidence",
        ],
        "precedence": [
            "documented_schema_or_api",
            "ecosystem_or_protocol",
            "artifact_role",
            "broad_domain",
            "uninformative_fallback",
        ],
        "selection_flow": [
            "trajectory_or_statistical_contender_set",
            "hard_metadata",
            "survey_domain_class_prior",
            "within_class_representative_prior",
            "canonical_policy_fallback",
        ],
        "default_profile_weight": 1.0,
        "unspecified_fields": "remain unrestricted; never infer a mismatch",
    },
    "profiles": [
        {
            "id": "scipy_rotation",
            "realm": "scientific_numeric",
            "ecosystem": "SciPy",
            "artifact_role": "numeric_rotation_array",
            "quaternion": {"order": "xyzw", "action": "active", "scalar_first_optional": True},
            "euler": {
                "sequence": "explicit",
                "intrinsic_extrinsic": "explicit_by_case",
                "angle_unit_default": "rad",
            },
            "rate": None,
            "citation_key": "scipyQuaternion",
        },
        {
            "id": "eigen_quaternion_coefficients",
            "realm": "scientific_numeric",
            "ecosystem": "Eigen",
            "artifact_role": "coefficient_array_or_mapped_memory",
            "quaternion": {"order": "xyzw", "constructor_argument_order": "wxyz"},
            "euler": None,
            "rate": None,
            "citation_key": "eigenQuaternion",
        },
        {
            "id": "matlab_eul2quat",
            "realm": "scientific_numeric",
            "ecosystem": "MATLAB",
            "artifact_role": "numeric_quaternion_output",
            "quaternion": {"order": "wxyz"},
            "euler": {
                "sequence_default": "ZYX_intrinsic",
                "angle_unit_default": "rad",
            },
            "rate": None,
            "citation_key": "matlabEul2Quat",
        },
        {
            "id": "matlab_aerospace_angle2quat",
            "realm": "aerospace",
            "ecosystem": "MATLAB Aerospace Toolbox",
            "artifact_role": "aerospace_attitude_conversion",
            "quaternion": {"order": "wxyz", "action": "passive", "handedness": "right"},
            "euler": {
                "sequence_default": "ZYX_intrinsic",
                "angle_unit_default": "rad",
                "action": "passive",
            },
            "rate": None,
            "citation_key": "matlabAeroAngle2Quat",
        },
        {
            "id": "naif_spice",
            "realm": "nasa_space",
            "ecosystem": "JPL NAIF SPICE",
            "artifact_role": "spice_attitude",
            "quaternion": {"order": "wxyz", "style": "SPICE_Hamilton"},
            "euler": {
                "sequence": "explicit",
                "angle_unit_default": "rad",
                "construction": "moving_axis",
            },
            "rate": None,
            "citation_key": "naifQuaternion",
        },
        {
            "id": "naif_spice_ck",
            "realm": "nasa_space",
            "ecosystem": "JPL NAIF SPICE",
            "artifact_role": "ck_pointing_angular_velocity",
            "quaternion": None,
            "euler": None,
            "rate": {"unit": "rad/s", "expression_frame": "base_reference", "handedness": "right"},
            "citation_key": "naifCK",
        },
        {
            "id": "nasa_pds_quaternion",
            "realm": "nasa_space",
            "ecosystem": "NASA PDS",
            "artifact_role": "pds_quaternion_product",
            "quaternion": {"documented_types": {"SPICE": "wxyz", "Spacecraft_Telemetry": "xyzw"}},
            "euler": None,
            "rate": None,
            "citation_key": "nasaPDSQuaternion",
        },
        {
            "id": "ros_core_imu",
            "realm": "robotics_autonomy",
            "ecosystem": "ROS",
            "artifact_role": "imu_telemetry",
            "quaternion": {"order": "xyzw", "orientation_semantics": "sensor_frame_wrt_world"},
            "euler": {"preferred_rpy": "XYZ_fixed", "angle_unit_default": "rad"},
            "rate": {"unit": "rad/s", "expression_frame": "sensor_body", "handedness": "right"},
            "citation_keys": ["rosrep103", "rosrep145", "rosQuaternion"],
        },
        {
            "id": "mavlink_px4_attitude",
            "realm": "robotics_autonomy",
            "ecosystem": "MAVLink/PX4",
            "artifact_role": "flight_telemetry",
            "quaternion": {"order": "wxyz", "style": "Hamilton"},
            "euler": {"sequence_default": "ZYX_intrinsic", "angle_unit_default": "rad"},
            "rate": {"unit": "rad/s", "expression_frame": "body_FRD"},
            "citation_keys": ["mavlinkCommon", "px4VehicleOdometry"],
        },
        {
            "id": "unity_3d",
            "realm": "game_engine",
            "ecosystem": "Unity",
            "artifact_role": "3d_transform_rigidbody",
            "quaternion": {"order": "xyzw"},
            "euler": {"sequence_default": "ZXY_extrinsic", "angle_unit_default": "deg"},
            "rate": {"unit": "rad/s", "expression_frame": None},
            "citation_keys": ["unityQuaternion", "unityEuler", "unityAngularVelocity"],
        },
        {
            "id": "unreal_engine",
            "realm": "game_engine",
            "ecosystem": "Unreal Engine",
            "artifact_role": "3d_transform_physics",
            "quaternion": {"order": "xyzw"},
            "euler": {
                "sequence_default": "ZYX_intrinsic",
                "angle_unit_default": "deg",
                "sign_conventions": "engine_specific",
            },
            "rate": {"documented_units": ["rad/s", "deg/s"], "explicit_world_rad_s_getter": True},
            "citation_keys": [
                "unrealQuaternion",
                "unrealRotator",
                "unrealAngularVelocity",
                "unrealAngularDegrees",
            ],
        },
        {
            "id": "godot_3d",
            "realm": "game_engine",
            "ecosystem": "Godot",
            "artifact_role": "3d_transform_rigidbody",
            "quaternion": {"order": "xyzw", "style": "Hamilton"},
            "euler": {
                "sequence_default": "YXZ_intrinsic",
                "angle_unit_default": "rad",
            },
            "rate": {"unit": "rad/s", "expression_frame": None},
            "citation_keys": ["godotQuaternion", "godotBasis", "godotAngularVelocity"],
        },
    ],
    "broad_scorecards": {
        "scientific_numeric": {
            "quaternion_order": {
                "xyzw": {"score": 67, "support": 2, "informative_profiles": 3},
                "wxyz": {"score": 33, "support": 1, "informative_profiles": 3},
            },
            "euler_angle_unit": {"rad": {"score": 100, "support": 2, "informative_profiles": 2}},
            "euler_sequence_note": (
                "ZYX is the only fixed default among surveyed scientific interfaces that provide "
                "a default; coverage 1/1, so treat as weak."
            ),
            "rate_frame": "uninformative",
        },
        "robotics_autonomy_ros_vs_mavlink_px4": {
            "quaternion_order": {
                "xyzw": {"score": 50, "support": 1, "informative_profiles": 2},
                "wxyz": {"score": 50, "support": 1, "informative_profiles": 2},
            },
            "euler_angle_unit": {"rad": {"score": 100, "support": 2, "informative_profiles": 2}},
            "euler_sequence_quotient": {
                "ZYX_intrinsic__equiv__XYZ_extrinsic": {
                    "score": 100,
                    "support": 2,
                    "informative_profiles": 2,
                }
            },
            "rate": {"rad_s_body_sensor": {"score": 100, "support": 2, "informative_profiles": 2}},
        },
        "nasa_spacecraft_broad": {
            "quaternion_order": "mixed_by_specification_no_broad_tie_breaker",
            "euler": "SPICE_uses_radians_but_sequence_is_explicit_no_broad_sequence_default",
            "rate": (
                "no_broad_rate_frame_prior; use CK base/reference rad/s only when CK role is known"
            ),
        },
        "major_game_engines": {
            "quaternion_order": {
                "xyzw": {"score": 100, "support": 3, "informative_profiles": 3},
                "wxyz": {"score": 0, "support": 0, "informative_profiles": 3},
            },
            "euler_angle_unit": {
                "deg": {"score": 67, "support": 2, "informative_profiles": 3},
                "rad": {"score": 33, "support": 1, "informative_profiles": 3},
            },
            "euler_sequence_quotient": {
                "YXZ_intrinsic__equiv__ZXY_extrinsic": {
                    "score": 67,
                    "support": 2,
                    "informative_profiles": 3,
                },
                "ZYX_intrinsic__equiv__XYZ_extrinsic": {
                    "score": 33,
                    "support": 1,
                    "informative_profiles": 3,
                },
            },
            "rate": (
                "rad/s is documented by all three surveyed rigid-body APIs; Unreal also exposes "
                "deg/s; broad expression frame unresolved"
            ),
        },
    },
    "engine_rules": {
        "class_prior": "pi_survey(C|d,r) = sum_p rho_p(d,r)*1[C intersects H_p] / Z_p",
        "tempered_prior": (
            "pi_lambda(C|d,r) = (1-lambda)/|T_ev| + "
            "lambda*pi_survey(C|d,r)/sum_{G in T_ev} pi_survey(G|d,r)"
        ),
        "representative_prior": "pi_rep(h|C,d,r) proportional to sum_p rho_p(d,r)*1[h in H_p]",
        "abstain_when": [
            "survey prior has no support inside evidence contender set",
            "broad-domain support is tied or insufficient",
            "required convention dimension is undocumented",
        ],
    },
}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
####


def _write_model(path: Path, model: Any) -> None:
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
####


def _build_partial_config() -> InferenceConfig:
    return InferenceConfig(
        constraints=ConventionConstraints(
            quaternion_order=QuaternionOrder.SCALAR_FIRST,
            rate_order=("pqr", "rqp"),
        ),
        priors=ConventionPriors(
            attitude_direction={
                AttitudeDirection.WORLD_TO_BODY: 3.0,
                AttitudeDirection.BODY_TO_WORLD: 1.0,
            },
            rate_frame={RateFrame.WORLD: 3.0, RateFrame.BODY: 1.0},
        ),
        integration_substeps=4,
        window_sizes=(1, 2, 4, 8),
        top_k=20,
    )
####


def _build_preference_profile() -> PreferenceProfile:
    return PreferenceProfile(
        profile_id="robotics-stack-v1",
        source="replace-with-reviewed-policy-record",
        context={"domain": "robotics", "role": "imu"},
        representative_preferences=ConventionPriors(
            quaternion_order={QuaternionOrder.SCALAR_LAST: 2.0},
            rate_frame={RateFrame.BODY: 2.0},
            rate_unit={"radians_per_second": 2.0},
        ),
        strength=1.0,
    )
####


def _write_synthetic_quaternion_csv(path: Path) -> None:
    """Write the demo trajectory with portable 14-significant-digit values."""

    time_s = np.arange(0.0, 4.0, 0.025)
    rates = rich_three_axis_rates(time_s)
    convention = SyntheticConvention(
        representation=AttitudeRepresentation.QUATERNION,
        attitude_direction=AttitudeDirection.WORLD_TO_BODY,
        quaternion_order=QuaternionOrder.SCALAR_FIRST,
        rate_frame=RateFrame.WORLD,
        rate_order="rqp",
    )
    data = encode_synthetic_trajectory(time_s, rates, convention)
    if data.time_s is None or data.orientation is None or data.rates is None:
        raise RuntimeError("Synthetic quaternion example requires all three data streams")
    ####
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("time", "q0", "q1", "q2", "q3", "w0", "w1", "w2"))
        for index in range(data.sample_count):
            writer.writerow(
                tuple(
                    format(float(value), ".14g")
                    for value in (
                        data.time_s[index],
                        *data.orientation[index],
                        *data.rates[index],
                    )
                )
            )
        ####
    ####
####


def _build_packet(
    packet_id: str,
    likelihood_ratio: tuple[float, float, float],
    source_id: str,
) -> EvidencePacket:
    maximum = max(likelihood_ratio)
    log_evidence = {
        class_id: math.log(value / maximum)
        for class_id, value in zip(CLASS_IDS, likelihood_ratio, strict=True)
    }
    return EvidencePacket(
        packet_id=packet_id,
        quotient_space_id="orientation-qspace-v1",
        class_ids=CLASS_IDS,
        log_evidence=log_evidence,
        local_prior=UNIFORM_PRIOR,
        contender_set=("C2", "C1", "C3"),
        evidence_semantics=EvidenceSemantics.CALIBRATED_LIKELIHOOD,
        calibration=CALIBRATION,
        lineage=LineageDescriptor(atomic_source_ids=(source_id,)),
        provenance={"domain": "example", "artifact_role": "trajectory"},
        excitation_rank=3,
    )
####


def _write_cohort_fusion_examples(example_dir: Path) -> None:
    packet_a = _build_packet("A", (9.0, 1.0, 0.5), "source-A")
    packet_b = _build_packet("B", (6.0, 1.0, 0.5), "source-B")
    packet_c = _build_packet("C", (1.4, 1.0, 0.8), "source-C")
    for packet in (packet_a, packet_b, packet_c):
        _write_model(example_dir / f"packet_{packet.packet_id}.json", packet)
    ####

    result = fuse_target_with_robust_borrowing(
        packet_c,
        [packet_a, packet_b],
        prior_sharedness=0.8,
    )
    _write_model(example_dir / "fusion_result.json", result)
####


def generate_examples(root: Path) -> None:
    examples = root / "examples"
    cohort_fusion = examples / "cohort_fusion"
    cohort_fusion.mkdir(parents=True, exist_ok=True)

    partial_config = _build_partial_config()
    constraint_payload = {
        key: value
        for key, value in partial_config.constraints.model_dump(mode="json").items()
        if value is not None
    }
    prior_payload = {
        key: value for key, value in partial_config.priors.model_dump(mode="json").items() if value
    }
    config_payload = {
        "constraints": constraint_payload,
        "priors": prior_payload,
        "integration_substeps": partial_config.integration_substeps,
        "window_sizes": partial_config.window_sizes,
        "top_k": partial_config.top_k,
    }
    _write_json(examples / "partially_known_config.json", config_payload)
    profile_payload = _build_preference_profile().model_dump(mode="json")
    representative_preferences: object = profile_payload["representative_preferences"]
    if isinstance(representative_preferences, dict):
        profile_payload["representative_preferences"] = {
            key: value
            for key, value in cast(dict[str, object], representative_preferences).items()
            if value
        }
    ####
    _write_json(examples / "preference_profile.json", {"preference_profile": profile_payload})
    _write_synthetic_quaternion_csv(examples / "synthetic_quaternion.csv")
    _write_json(cohort_fusion / "orientation-convention-domain-priors.json", DOMAIN_PRIOR_DOCUMENT)
    _write_cohort_fusion_examples(cohort_fusion)
####


if __name__ == "__main__":
    generate_examples(Path(__file__).resolve().parents[1])
####
