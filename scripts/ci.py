"""Run the repository's complete cleanliness and test pipeline."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
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


def run_pipeline(fix: bool) -> None:
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
    if fix:
        _run([ruff, "format", "."], "Ruff format")
    else:
        print("\n== Ruff format ==")
        print("Skipped check: scope markers intentionally run after Ruff formatting.")
    ####

    _run([_tool("pyright"), "--pythonpath", sys.executable], "Pyright")

    scope_markers = _tool("scope-markers")
    scope_args = ["--fix", *SCOPE_PATHS] if fix else list(SCOPE_PATHS)
    _run([scope_markers, *scope_args], "Scope markers")
    if fix:
        _run([scope_markers, *SCOPE_PATHS], "Scope marker check")
    ####

    _run([sys.executable, "-m", "pytest", "-q"], "Tests")
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
