"""Compare public contracts with the local alpha tag without stored schemas."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import subprocess
import sys
import tarfile
from dataclasses import is_dataclass
from enum import Enum
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import (
    Any,
    NotRequired,
    Required,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

from pydantic import BaseModel, Field

__all__ = ["ContractSnapshot", "_compare", "_normalize_legacy_pep695_syntax", "main"]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAG = "0.0.1.a0"
APPROVED_RETURN_TYPE_REFINEMENTS: dict[str, tuple[str, str]] = {
    "best_interval_diagnostics": (
        "tuple[dict[str, float | int | bool], ...]",
        "tuple[IntervalDiagnosticRow, ...]",
    ),
}
APPROVED_INFERENCE_INPUT_EXTENSIONS = {
    "infer_conventions": (
        "orientation: 'TrajectoryData | ArrayLike | None'",
        "orientation: 'TrajectoryData | SparseOrientationRateData | ArrayLike | None'",
    )
}


class ContractSnapshot(BaseModel):
    """Named export and schema fingerprints from one isolated package import."""

    package_file: str
    root_exports: tuple[str, ...]
    model_exports: tuple[str, ...]
    function_signatures: dict[str, str]
    class_signatures: dict[str, str]
    enum_members: dict[str, tuple[tuple[str, str], ...]]
    model_schema_sha256: dict[str, str]
    model_json_schemas: dict[str, dict[str, Any]] = Field(default_factory=dict)
####


def _constructor_contract(value: type[object]) -> str:
    """Track call shape and defaults, ignoring relocated type annotations."""

    parameters: list[tuple[str, str, str]] = []
    for parameter in inspect.signature(value).parameters.values():
        default = (
            "<required>"
            if parameter.default is inspect.Parameter.empty
            else repr(parameter.default).replace("mappingproxy({})", "{}")
        )
        parameters.append((parameter.name, parameter.kind.name, default))
    ####
    return repr(tuple(parameters))
####


def _typed_dict_contract(value: type[object]) -> str:
    """Capture named field types and requiredness for exported TypedDicts."""

    fields: list[tuple[str, str, bool]] = []
    required_keys = cast(frozenset[str], cast(Any, value).__required_keys__)
    for name, annotation in get_type_hints(value, include_extras=True).items():
        origin = get_origin(annotation)
        required = name in required_keys
        if origin is NotRequired:
            required = False
        elif origin is Required:
            required = True
        ####
        field_type = get_args(annotation)[0] if origin in (NotRequired, Required) else annotation
        fields.append((name, str(field_type), required))
    ####
    return repr(tuple(fields))
####


def _signature_call_and_return(signature: str) -> tuple[str, str]:
    call, separator, return_annotation = signature.rpartition(" -> ")
    if not separator:
        return signature, ""
    ####
    return call, return_annotation.strip().strip("'\"")
####


def _is_approved_return_type_refinement(name: str, old: str, new: str) -> bool:
    allowed = APPROVED_RETURN_TYPE_REFINEMENTS.get(name)
    if allowed is None:
        return False
    ####
    old_call, old_return = _signature_call_and_return(old)
    new_call, new_return = _signature_call_and_return(new)
    return old_call == new_call and (old_return, new_return) == allowed
####


def _is_approved_input_model_extension(name: str, old: str, new: str) -> bool:
    """Allow one additive data-model alternative without changing the call shape."""

    allowed = APPROVED_INFERENCE_INPUT_EXTENSIONS.get(name)
    if allowed is None:
        return False
    ####
    old_call, old_return = _signature_call_and_return(old)
    new_call, new_return = _signature_call_and_return(new)
    old_annotation, new_annotation = allowed
    return (
        old_return == new_return
        and new_annotation in new_call
        and new_call.replace(new_annotation, old_annotation, 1) == old_call
    )
####


def _is_approved_trailing_inference_report_extension(name: str, old: str, new: str) -> bool:
    """Allow only the two optional sparse-observation metadata constructor fields."""

    if name != "InferenceReport":
        return False
    ####
    try:
        old_parameters = ast.literal_eval(old)
        new_parameters = ast.literal_eval(new)
    except (SyntaxError, ValueError):
        return False
    ####
    if not isinstance(old_parameters, tuple) or not isinstance(new_parameters, tuple):
        return False
    ####
    old_parameters = cast(tuple[object, ...], old_parameters)
    new_parameters = cast(tuple[object, ...], new_parameters)
    expected_additions = (
        ("observation_model", "POSITIONAL_OR_KEYWORD", "'aligned_samples'"),
        ("rate_sample_count", "POSITIONAL_OR_KEYWORD", "None"),
    )
    return (
        new_parameters[: len(old_parameters)] == old_parameters
        and new_parameters[len(old_parameters) :] == expected_additions
    )
####


def _schema_object(value: object) -> dict[str, Any] | None:
    """Narrow a decoded JSON value to a string-keyed schema object."""

    if not isinstance(value, dict):
        return None
    ####
    return cast(dict[str, Any], value)
####


def _is_approved_confidence_schema_refinement(
    name: str,
    old_schema: dict[str, Any],
    new_schema: dict[str, Any],
) -> bool:
    """Allow only the explicit [0, 1] heuristic-confidence contract refinement."""

    if name == "OverallDiagnostics":
        old_properties = _schema_object(old_schema.get("properties"))
        new_properties = _schema_object(new_schema.get("properties"))
    elif name == "InferenceReport":
        old_definitions = _schema_object(old_schema.get("$defs"))
        new_definitions = _schema_object(new_schema.get("$defs"))
        if old_definitions is None or new_definitions is None:
            return False
        ####
        old_diagnostics = _schema_object(old_definitions.get("OverallDiagnostics"))
        new_diagnostics = _schema_object(new_definitions.get("OverallDiagnostics"))
        if old_diagnostics is None or new_diagnostics is None:
            return False
        ####
        old_properties = _schema_object(old_diagnostics.get("properties"))
        new_properties = _schema_object(new_diagnostics.get("properties"))
    else:
        return False
    ####
    if old_properties is None or new_properties is None:
        return False
    ####
    old_confidence = _schema_object(old_properties.get("confidence"))
    new_confidence = _schema_object(new_properties.get("confidence"))
    old_label = _schema_object(old_properties.get("confidence_label"))
    new_label = _schema_object(new_properties.get("confidence_label"))
    if old_confidence is None or new_confidence is None or old_label is None or new_label is None:
        return False
    ####
    expected_confidence = {
        **old_confidence,
        "description": (
            "Uncalibrated composite heuristic score in [0, 1]; not a probability of correctness."
        ),
        "minimum": 0.0,
        "maximum": 1.0,
    }
    expected_label = {
        **old_label,
        "description": "Deterministic band of the heuristic score, not a calibrated probability.",
    }
    if new_confidence != expected_confidence or new_label != expected_label:
        return False
    ####
    normalized_new_schema = json.loads(json.dumps(new_schema))
    if name == "OverallDiagnostics":
        normalized_properties = normalized_new_schema["properties"]
    else:
        normalized_properties = normalized_new_schema["$defs"]["OverallDiagnostics"]["properties"]
    ####
    normalized_properties["confidence"] = old_confidence
    normalized_properties["confidence_label"] = old_label
    return normalized_new_schema == old_schema
####


def _is_approved_sparse_report_schema_extension(
    name: str,
    old_schema: dict[str, Any],
    new_schema: dict[str, Any],
) -> bool:
    """Allow only additive stream-sampling metadata on InferenceReport."""

    if name != "InferenceReport":
        return False
    ####
    old_properties = _schema_object(old_schema.get("properties"))
    new_properties = _schema_object(new_schema.get("properties"))
    if old_properties is None or new_properties is None:
        return False
    ####
    if set(new_properties) - set(old_properties) != {"observation_model", "rate_sample_count"}:
        return False
    ####
    expected_observation_model = {
        "default": "aligned_samples",
        "description": "Sampling relationship between the orientation and rate observations.",
        "enum": ["aligned_samples", "sparse_orientation_anchors"],
        "title": "Observation Model",
        "type": "string",
    }
    expected_rate_sample_count = {
        "anyOf": [{"minimum": 2, "type": "integer"}, {"type": "null"}],
        "default": None,
        "description": (
            "Number of dense rate samples for sparse-anchor input; null for aligned input."
        ),
        "title": "Rate Sample Count",
    }
    if (
        new_properties["observation_model"] != expected_observation_model
        or new_properties["rate_sample_count"] != expected_rate_sample_count
    ):
        return False
    ####
    normalized_new_schema = json.loads(json.dumps(new_schema))
    normalized_properties = _schema_object(normalized_new_schema.get("properties"))
    if normalized_properties is None:
        return False
    ####
    normalized_properties.pop("observation_model", None)
    normalized_properties.pop("rate_sample_count", None)
    return normalized_new_schema == old_schema or _is_approved_confidence_schema_refinement(
        "InferenceReport",
        old_schema,
        normalized_new_schema,
    )
####


def _normalize_legacy_pep695_syntax(source_root: Path) -> None:
    """Adapt only the extracted alpha source so Python 3.11 can inspect its API."""

    coercion_file = source_root / "attitude_convention_inference" / "coercion.py"
    source = coercion_file.read_text(encoding="utf-8")
    rewritten_lines: list[str] = []
    changed = False
    for line in source.splitlines():
        if line.startswith("type ") and " = " in line:
            alias, value = line.removeprefix("type ").split(" = ", maxsplit=1)
            rewritten_lines.append(f"{alias}: TypeAlias = {value}")
            changed = True
        else:
            rewritten_lines.append(line)
        ####
    ####
    if changed:
        compatible_source = "\n".join(rewritten_lines) + "\n"
        compatible_source = compatible_source.replace(
            "from typing import Any\n",
            "from typing import Any, TypeAlias\n",
            1,
        )
        coercion_file.write_text(compatible_source, encoding="utf-8")
    ####

    models_file = source_root / "attitude_convention_inference" / "models.py"
    models_source = models_file.read_text(encoding="utf-8")
    generic_signatures = (
        "def _as_tuple[T](",
        "def _enum_value[EnumT: Enum](",
        "def _enum_tuple[EnumT: Enum](",
        "def _validated_payload[T](",
        "def _intersect_values[T](",
    )
    if not any(signature in models_source for signature in generic_signatures):
        return
    ####
    models_source = models_source.replace(
        "from typing import Any, Literal, Self, cast\n",
        "from typing import Any, Literal, Self, TypeVar, cast\n",
        1,
    )
    models_source = models_source.replace(
        "def _as_tuple[T](",
        'T = TypeVar("T")\nEnumT = TypeVar("EnumT", bound=Enum)\n\n\ndef _as_tuple(',
        1,
    )
    models_source = models_source.replace("def _enum_value[EnumT: Enum](", "def _enum_value(", 1)
    models_source = models_source.replace("def _enum_tuple[EnumT: Enum](", "def _enum_tuple(", 1)
    models_source = models_source.replace(
        "def _validated_payload[T](", "def _validated_payload(", 1
    )
    models_source = models_source.replace("def _intersect_values[T](", "def _intersect_values(", 1)
    models_file.write_text(models_source, encoding="utf-8")
####


def _snapshot(source_root: Path) -> ContractSnapshot:
    """Import one source tree in this process and inspect its public surface."""

    sys.path.insert(0, str(source_root.resolve()))
    import attitude_convention_inference as package
    from attitude_convention_inference import models

    package_file = Path(package.__file__ or "").resolve()
    if not package_file.is_relative_to(source_root.resolve()):
        raise RuntimeError(f"Imported package from an unexpected path: {package_file}")
    ####

    root_values: dict[str, object] = {name: getattr(package, name) for name in package.__all__}
    model_values: dict[str, object] = {name: getattr(models, name) for name in models.__all__}
    functions = {
        name: str(inspect.signature(value))
        for name, value in root_values.items()
        if inspect.isfunction(value)
    }
    classes: dict[str, str] = {}
    for name, value in root_values.items():
        if not isinstance(value, type) or issubclass(value, Enum):
            continue
        ####
        class_type = cast("type[object]", value)
        classes[name] = (
            _typed_dict_contract(class_type)
            if is_typeddict(class_type)
            else _constructor_contract(class_type)
        )
    ####
    enum_members: dict[str, tuple[tuple[str, str], ...]] = {}
    model_schema_sha256: dict[str, str] = {}
    model_json_schemas: dict[str, dict[str, Any]] = {}
    for name, value in {**root_values, **model_values}.items():
        if isinstance(value, type) and issubclass(value, Enum):
            enum_members[name] = tuple(
                (member_name, repr(member.value))
                for member_name, member in value.__members__.items()
            )
        ####
    ####
    for name, value in model_values.items():
        if (
            isinstance(value, type)
            and is_dataclass(value)
            and issubclass(value, models.FrozenModel)
        ):
            model_schema = value.model_json_schema()
            schema_json = json.dumps(
                model_schema,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            model_schema_sha256[name] = sha256(schema_json.encode("utf-8")).hexdigest()
            model_json_schemas[name] = model_schema
        ####
    ####

    return ContractSnapshot(
        package_file=str(package_file),
        root_exports=tuple(package.__all__),
        model_exports=tuple(models.__all__),
        function_signatures=functions,
        class_signatures=classes,
        enum_members=enum_members,
        model_schema_sha256=model_schema_sha256,
        model_json_schemas=model_json_schemas,
    )
####


def _extract_tag(tag: str, destination: Path) -> None:
    """Extract only regular source files from a local Git tag."""

    archived = subprocess.run(
        ["git", "archive", "--format=tar", tag, "src/attitude_convention_inference"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    with tarfile.open(fileobj=BytesIO(archived.stdout), mode="r:") as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if (
                relative.is_absolute()
                or not relative.parts
                or relative.parts[0] != "src"
                or ".." in relative.parts
            ):
                raise ValueError(f"Unsafe archive path: {member.name}")
            ####
            if member.isdir():
                continue
            ####
            if not member.isfile():
                raise ValueError(f"Unsupported archive member: {member.name}")
            ####
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Unable to read archive member: {member.name}")
            ####
            target = destination.joinpath(*relative.parts)
            if target.exists():
                raise ValueError(f"Duplicate archive member: {member.name}")
            ####
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())
        ####
    ####
    if sys.version_info < (3, 12):
        _normalize_legacy_pep695_syntax(destination / "src")
    ####
####


def _isolated_snapshot(source_root: Path) -> ContractSnapshot:
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--probe-root", str(source_root)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return ContractSnapshot.model_validate_json(completed.stdout)
####


def _compare(alpha: ContractSnapshot, current: ContractSnapshot) -> list[str]:
    issues: list[str] = []
    for label, old, new in (
        ("package exports", alpha.root_exports, current.root_exports),
        ("model exports", alpha.model_exports, current.model_exports),
    ):
        removed = sorted(set(old) - set(new))
        if removed:
            issues.append(f"{label} removed: {', '.join(removed)}")
        ####
    ####

    for label, old, new in (
        ("function signature", alpha.function_signatures, current.function_signatures),
        ("class signature", alpha.class_signatures, current.class_signatures),
        ("enum members", alpha.enum_members, current.enum_members),
    ):
        for name, expected in old.items():
            actual = new.get(name)
            if actual == expected:
                continue
            ####
            if (
                label == "function signature"
                and isinstance(expected, str)
                and isinstance(actual, str)
                and (
                    _is_approved_return_type_refinement(name, expected, actual)
                    or _is_approved_input_model_extension(name, expected, actual)
                )
            ):
                continue
            ####
            if (
                label == "class signature"
                and isinstance(expected, str)
                and isinstance(actual, str)
                and _is_approved_trailing_inference_report_extension(name, expected, actual)
            ):
                continue
            else:
                issues.append(f"{label} changed: {name}")
            ####
        ####
    ####

    for name, expected in alpha.model_schema_sha256.items():
        actual = current.model_schema_sha256.get(name)
        if actual == expected:
            continue
        ####
        old_schema = alpha.model_json_schemas.get(name)
        new_schema = current.model_json_schemas.get(name)
        if (
            old_schema is not None
            and new_schema is not None
            and (
                _is_approved_confidence_schema_refinement(name, old_schema, new_schema)
                or _is_approved_sparse_report_schema_extension(name, old_schema, new_schema)
            )
        ):
            continue
        ####
        issues.append(f"model JSON schema changed: {name}")
    ####
    return issues
####


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_TAG, help="local compatibility baseline tag")
    parser.add_argument("--probe-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.probe_root is not None:
        print(_snapshot(args.probe_root).model_dump_json())
        return 0
    ####

    try:
        with TemporaryDirectory(prefix="aci-contract-", dir=ROOT) as temporary:
            alpha_root = Path(temporary)
            _extract_tag(str(args.tag), alpha_root)
            alpha = _isolated_snapshot(alpha_root / "src")
            current = _isolated_snapshot(ROOT / "src")
        ####
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"Contract compatibility check could not run: {error}", file=sys.stderr)
        return 2
    ####

    issues = _compare(alpha, current)
    if issues:
        for issue in issues:
            print(issue, file=sys.stderr)
        ####
        return 1
    ####
    print(
        f"Contract compatibility with {args.tag} passed: "
        f"{len(current.root_exports)} package exports, "
        f"{len(current.function_signatures)} function signatures, "
        f"{len(current.class_signatures)} class signatures, "
        f"{len(current.enum_members)} enums, "
        f"{len(current.model_schema_sha256)} model schemas."
    )
    return 0
####


if __name__ == "__main__":
    raise SystemExit(main())
####
