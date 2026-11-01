"""Run the repository's complete cleanliness and test pipeline."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPE_PATHS = ("src", "tests", "examples", "scripts")


def _tool(name: str) -> str:
    suffix = ".exe" if os.name == "nt" else ""
    sibling = Path(sys.executable).with_name(f"{name}{suffix}")
    executable = str(sibling) if sibling.is_file() else shutil.which(name)
    if executable is None:
        raise SystemExit(f"Required development tool is not installed or not on PATH: {name}")
    ####
    return executable
####


def _run(command: list[str], label: str) -> None:
    print(f"\n== {label} ==")
    subprocess.run(command, cwd=ROOT, check=True)
####


def _require_python_module(module_name: str, package_name: str) -> None:
    """Fail early with the dev-install command when a Python prerequisite is absent."""

    if find_spec(module_name) is None:
        raise SystemExit(
            f"Required Python package is not installed: {package_name}. "
            'Install development dependencies with `python -m pip install -e ".[dev]"`.'
        )
    ####
####


def _check_ruff_format(ruff: str, scope_markers: str) -> None:
    """Check that Ruff formatting plus scope-marker restoration is idempotent."""

    with tempfile.TemporaryDirectory(dir=ROOT) as temporary_directory:
        scratch = Path(temporary_directory)
        python_files: list[Path] = []
        listed = subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
        )
        for relative_name in listed.decode("utf-8").split("\0"):
            relative_path = Path(relative_name)
            if (
                relative_path.suffix != ".py"
                or not relative_path.parts
                or relative_path.parts[0] not in SCOPE_PATHS
            ):
                continue
            ####
            source = ROOT / relative_path
            if not source.is_file():
                continue
            ####
            target = scratch / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            python_files.append(relative_path)
        ####
        paths = [str(scratch / directory) for directory in SCOPE_PATHS]
        _run(
            [scope_markers, "--strip", "--fix", "--quiet", *paths],
            "Strip markers in temporary copy",
        )
        _run(
            [ruff, "format", "--quiet", "--config", str(ROOT / "pyproject.toml"), *paths],
            "Ruff format in temporary copy",
        )
        _run(
            [scope_markers, "--fix", "--quiet", *paths],
            "Restore markers in temporary copy",
        )
        changed_files = tuple(
            relative_path
            for relative_path in python_files
            if (scratch / relative_path).read_bytes() != (ROOT / relative_path).read_bytes()
        )
        if changed_files:
            changed_names = ", ".join(path.as_posix() for path in changed_files)
            raise SystemExit(
                "Python formatting or scope markers are not clean; run `python scripts/ci.py "
                f"--fix`. Changed files: {changed_names}"
            )
        ####
    ####
####


def run_pipeline(fix: bool) -> None:
    _require_python_module("pytest_cov", "pytest-cov")
    mdrepo = _tool("mdrepo")
    _run([mdrepo, "fix" if fix else "check", "."], "mdrepo")

    rumdl = _tool("rumdl")
    rumdl_args = ["check"]
    if fix:
        rumdl_args.append("--fix")
    ####
    rumdl_args.append(".")
    _run([rumdl, *rumdl_args], "Markdown")

    generator_args = [
        str(ROOT / "scripts" / "generate_schemas.py"),
        "--fix" if fix else "--check",
    ]
    _run([sys.executable, *generator_args], "Generated schemas and examples")

    ruff = _tool("ruff")
    ruff_check_args = ["check"]
    if fix:
        ruff_check_args.append("--fix")
    ####
    ruff_check_args.append(".")
    _run([ruff, *ruff_check_args], "Ruff lint")
    scope_markers = _tool("scope-markers")
    if fix:
        # Ruff formats standalone #### comments as ordinary top-level comments.
        # Remove markers before formatting so repeated --fix runs stay idempotent.
        try:
            _run([scope_markers, "--strip", "--fix", *SCOPE_PATHS], "Strip scope markers")
            _run([ruff, "format", "."], "Ruff format")
            _run([_tool("pyright"), "--pythonpath", sys.executable], "Pyright")
        finally:
            _run([scope_markers, "--fix", *SCOPE_PATHS], "Restore scope markers")
        ####
        _run([scope_markers, *SCOPE_PATHS], "Scope marker check")
    else:
        _check_ruff_format(ruff, scope_markers)
        _run([_tool("pyright"), "--pythonpath", sys.executable], "Pyright")
        _run([scope_markers, *SCOPE_PATHS], "Scope markers")
    ####

    _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--cov=attitude_convention_inference",
            "--cov-branch",
            "--cov-report=term-missing:skip-covered",
            "--cov-fail-under=85",
        ],
        "Tests and branch coverage",
    )
    _run(
        [sys.executable, str(ROOT / "scripts" / "check_contract_compatibility.py")],
        "Alpha public-contract compatibility",
    )
####


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix",
        action="store_true",
        help="apply auto-fixes before checking the repository",
    )
    args = parser.parse_args()
    run_pipeline(args.fix)
####


if __name__ == "__main__":
    main()
####
