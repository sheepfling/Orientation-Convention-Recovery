"""The release contract gate detects drift without a checked-in schema baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.check_contract_compatibility import (
    ContractSnapshot,
    _compare,
    _normalize_legacy_pep695_syntax,
)


def _alpha_snapshot() -> ContractSnapshot:
    return ContractSnapshot(
        package_file="alpha/src/attitude_convention_inference/__init__.py",
        root_exports=("infer_conventions", "InferenceReport"),
        model_exports=("InferenceReport", "RateFrame"),
        function_signatures={"infer_conventions": "(data, config=None)"},
        class_signatures={"InferenceReport": "(best)"},
        enum_members={"RateFrame": (("BODY", "'body'"), ("WORLD", "'world'"))},
        model_schema_sha256={"InferenceReport": "alpha-schema"},
    )
####


def test_contract_gate_allows_additive_exports() -> None:
    alpha = _alpha_snapshot()
    current = alpha.model_copy(
        update={
            "root_exports": (*alpha.root_exports, "new_helper"),
            "model_exports": (*alpha.model_exports, "NewReport"),
            "model_schema_sha256": {**alpha.model_schema_sha256, "NewReport": "new-schema"},
        }
    )

    assert _compare(alpha, current) == []
####


def test_contract_gate_allows_approved_interval_row_type_refinement() -> None:
    old_signature = (
        "(data: 'TrajectoryData', report: 'InferenceReport', "
        "config: 'InferenceConfig | None' = None) -> "
        "'tuple[dict[str, float | int | bool], ...]'"
    )
    refined_signature = (
        "(data: 'TrajectoryData', report: 'InferenceReport', "
        "config: 'InferenceConfig | None' = None) -> "
        "'tuple[IntervalDiagnosticRow, ...]'"
    )
    alpha = _alpha_snapshot().model_copy(
        update={"function_signatures": {"best_interval_diagnostics": old_signature}}
    )
    current = alpha.model_copy(
        update={"function_signatures": {"best_interval_diagnostics": refined_signature}}
    )

    assert _compare(alpha, current) == []

    changed_call_shape = current.model_copy(
        update={
            "function_signatures": {
                "best_interval_diagnostics": refined_signature.replace(
                    "data: 'TrajectoryData'", "trajectory: 'TrajectoryData'"
                )
            }
        }
    )
    assert _compare(alpha, changed_call_shape) == [
        "function signature changed: best_interval_diagnostics"
    ]
####


def test_contract_gate_allows_sparse_input_and_trailing_report_metadata() -> None:
    old_function = (
        "(orientation: 'TrajectoryData | ArrayLike | None', rates: 'ArrayLike | None' = None) "
        "-> 'InferenceReport'"
    )
    new_function = old_function.replace(
        "TrajectoryData | ArrayLike | None",
        "TrajectoryData | SparseOrientationRateData | ArrayLike | None",
    )
    old_parameters = (("best", "POSITIONAL_OR_KEYWORD", "<required>"),)
    new_parameters = (
        *old_parameters,
        ("observation_model", "POSITIONAL_OR_KEYWORD", "'aligned_samples'"),
        ("rate_sample_count", "POSITIONAL_OR_KEYWORD", "None"),
    )
    alpha = _alpha_snapshot().model_copy(
        update={
            "function_signatures": {"infer_conventions": old_function},
            "class_signatures": {"InferenceReport": repr(old_parameters)},
        }
    )
    current = alpha.model_copy(
        update={
            "function_signatures": {"infer_conventions": new_function},
            "class_signatures": {"InferenceReport": repr(new_parameters)},
        }
    )

    assert _compare(alpha, current) == []

    required_addition = (
        *old_parameters,
        ("observation_model", "POSITIONAL_OR_KEYWORD", "<required>"),
        ("rate_sample_count", "POSITIONAL_OR_KEYWORD", "None"),
    )
    breaking = current.model_copy(
        update={"class_signatures": {"InferenceReport": repr(required_addition)}}
    )
    assert _compare(alpha, breaking) == ["class signature changed: InferenceReport"]
####


def test_contract_gate_allows_only_additive_sparse_report_schema_fields() -> None:
    old_schema: dict[str, Any] = {
        "title": "InferenceReport",
        "type": "object",
        "properties": {"sample_count": {"type": "integer"}},
        "required": ["sample_count"],
    }
    new_schema: dict[str, Any] = {
        **old_schema,
        "properties": {
            **old_schema["properties"],
            "observation_model": {
                "default": "aligned_samples",
                "description": (
                    "Sampling relationship between the orientation and rate observations."
                ),
                "enum": ["aligned_samples", "sparse_orientation_anchors"],
                "title": "Observation Model",
                "type": "string",
            },
            "rate_sample_count": {
                "anyOf": [{"minimum": 2, "type": "integer"}, {"type": "null"}],
                "default": None,
                "description": (
                    "Number of dense rate samples for sparse-anchor input; null for aligned input."
                ),
                "title": "Rate Sample Count",
            },
        },
    }
    alpha = _alpha_snapshot().model_copy(
        update={
            "model_schema_sha256": {"InferenceReport": "alpha-schema"},
            "model_json_schemas": {"InferenceReport": old_schema},
        }
    )
    current = alpha.model_copy(
        update={
            "model_schema_sha256": {"InferenceReport": "new-schema"},
            "model_json_schemas": {"InferenceReport": new_schema},
        }
    )

    assert _compare(alpha, current) == []

    breaking_schema = {
        **new_schema,
        "properties": {
            **new_schema["properties"],
            "sample_count": {"type": "integer", "minimum": 3},
        },
    }
    breaking = current.model_copy(
        update={"model_json_schemas": {"InferenceReport": breaking_schema}}
    )
    assert _compare(alpha, breaking) == ["model JSON schema changed: InferenceReport"]
####


def test_contract_gate_allows_only_bounded_confidence_schema_refinement() -> None:
    old_schema: dict[str, Any] = {
        "title": "OverallDiagnostics",
        "type": "object",
        "properties": {
            "confidence": {"title": "Confidence", "type": "number"},
            "confidence_label": {"$ref": "#/$defs/ConfidenceLabel"},
        },
    }
    new_schema: dict[str, Any] = {
        "title": "OverallDiagnostics",
        "type": "object",
        "properties": {
            "confidence": {
                "title": "Confidence",
                "type": "number",
                "description": (
                    "Uncalibrated composite heuristic score in [0, 1]; "
                    "not a probability of correctness."
                ),
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "confidence_label": {
                "$ref": "#/$defs/ConfidenceLabel",
                "description": (
                    "Deterministic band of the heuristic score, not a calibrated probability."
                ),
            },
        },
    }
    old_report_schema: dict[str, Any] = {
        "properties": {"diagnostics": {"$ref": "#/$defs/OverallDiagnostics"}},
        "$defs": {"OverallDiagnostics": old_schema},
    }
    new_report_schema: dict[str, Any] = {
        "properties": {"diagnostics": {"$ref": "#/$defs/OverallDiagnostics"}},
        "$defs": {"OverallDiagnostics": new_schema},
    }
    alpha = _alpha_snapshot().model_copy(
        update={
            "model_schema_sha256": {
                "OverallDiagnostics": "alpha-schema",
                "InferenceReport": "alpha-report-schema",
            },
            "model_json_schemas": {
                "OverallDiagnostics": old_schema,
                "InferenceReport": old_report_schema,
            },
        }
    )
    current = alpha.model_copy(
        update={
            "model_schema_sha256": {
                "OverallDiagnostics": "refined-schema",
                "InferenceReport": "refined-report-schema",
            },
            "model_json_schemas": {
                "OverallDiagnostics": new_schema,
                "InferenceReport": new_report_schema,
            },
        }
    )

    assert _compare(alpha, current) == []

    invalid_schema = {
        **new_schema,
        "properties": {
            **new_schema["properties"],
            "confidence": {
                **new_schema["properties"]["confidence"],
                "maximum": 1.5,
            },
        },
    }
    invalid = current.model_copy(
        update={
            "model_json_schemas": {
                **current.model_json_schemas,
                "OverallDiagnostics": invalid_schema,
            }
        }
    )
    assert _compare(alpha, invalid) == ["model JSON schema changed: OverallDiagnostics"]
####


def test_contract_gate_reports_removed_or_changed_alpha_contracts() -> None:
    alpha = _alpha_snapshot()
    changed = alpha.model_copy(
        update={
            "root_exports": ("InferenceReport",),
            "model_exports": ("RateFrame",),
            "function_signatures": {"infer_conventions": "(data)"},
            "class_signatures": {"InferenceReport": "(best, extra)"},
            "enum_members": {"RateFrame": (("BODY", "'body'"),)},
            "model_schema_sha256": {"InferenceReport": "changed-schema"},
        }
    )

    assert _compare(alpha, changed) == [
        "package exports removed: infer_conventions",
        "model exports removed: InferenceReport",
        "function signature changed: infer_conventions",
        "class signature changed: InferenceReport",
        "enum members changed: RateFrame",
        "model JSON schema changed: InferenceReport",
    ]
####


def test_legacy_type_syntax_normalization_is_limited_to_probe_source(tmp_path: Path) -> None:
    source_root = tmp_path / "src"
    package_root = source_root / "attitude_convention_inference"
    package_root.mkdir(parents=True)
    coercion_file = package_root / "coercion.py"
    coercion_file.write_text(
        "from __future__ import annotations\n"
        "from typing import Any\n"
        "type ConfigInput = dict[str, Any] | None\n",
        encoding="utf-8",
    )
    models_file = package_root / "models.py"
    models_file.write_text(
        "from __future__ import annotations\n"
        "from enum import Enum\n"
        "from typing import Any, Literal, Self, cast\n"
        "def _as_tuple[T](value: T) -> T: ...\n"
        "def _enum_value[EnumT: Enum](value: EnumT) -> EnumT: ...\n"
        "def _enum_tuple[EnumT: Enum](value: EnumT) -> EnumT: ...\n"
        "def _validated_payload[T](value: T) -> T: ...\n",
        encoding="utf-8",
    )

    _normalize_legacy_pep695_syntax(source_root)

    assert coercion_file.read_text(encoding="utf-8") == (
        "from __future__ import annotations\n"
        "from typing import Any, TypeAlias\n"
        "ConfigInput: TypeAlias = dict[str, Any] | None\n"
    )
    normalized_models = models_file.read_text(encoding="utf-8")
    assert 'T = TypeVar("T")' in normalized_models
    assert 'EnumT = TypeVar("EnumT", bound=Enum)' in normalized_models
    assert "def _as_tuple(" in normalized_models
    assert "def _enum_value(" in normalized_models
    assert "def _enum_tuple(" in normalized_models
    assert "def _validated_payload(" in normalized_models
####
