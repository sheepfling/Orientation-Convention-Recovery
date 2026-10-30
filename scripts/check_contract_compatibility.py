"""Compare public contracts with the local alpha tag without stored schemas."""

from __future__ import annotations

import argparse
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
from typing import cast

from pydantic import BaseModel

__all__ = ["ContractSnapshot", "_compare", "main"]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAG = "0.0.1.a0"


class ContractSnapshot(BaseModel):
    """Named export and schema fingerprints from one isolated package import."""

    package_file: str
    root_exports: tuple[str, ...]
    model_exports: tuple[str, ...]
    function_signatures: dict[str, str]
    class_signatures: dict[str, str]
    enum_members: dict[str, tuple[tuple[str, str], ...]]
    model_schema_sha256: dict[str, str]
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
    classes = {
        name: _constructor_contract(cast("type[object]", value))
        for name, value in root_values.items()
        if isinstance(value, type) and not issubclass(value, Enum)
    }
    enum_members: dict[str, tuple[tuple[str, str], ...]] = {}
    model_schema_sha256: dict[str, str] = {}
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
            schema_json = json.dumps(
                value.model_json_schema(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            model_schema_sha256[name] = sha256(schema_json.encode("utf-8")).hexdigest()
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
        ("model JSON schema", alpha.model_schema_sha256, current.model_schema_sha256),
    ):
        for name, expected in old.items():
            if new.get(name) != expected:
                issues.append(f"{label} changed: {name}")
            ####
        ####
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
        f"{len(alpha.root_exports)} package exports, "
        f"{len(alpha.function_signatures)} function signatures, "
        f"{len(alpha.class_signatures)} class signatures, "
        f"{len(alpha.enum_members)} enums, "
        f"{len(alpha.model_schema_sha256)} model schemas."
    )
    return 0
####


if __name__ == "__main__":
    raise SystemExit(main())
####
