"""Check packaging configuration contracts."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from scripts import ci


def test_ci_explains_how_to_install_missing_coverage_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_module_spec(module_name: str, package: str | None = None) -> None:
        return None
    ####

    monkeypatch.setattr(ci, "find_spec", missing_module_spec)

    with pytest.raises(SystemExit, match=r"pytest-cov.*pip install -e"):
        ci.run_pipeline(fix=False)
    ####
####


def test_publish_extra_includes_build_backend_requirements() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))

    build_requirements = set(pyproject["build-system"]["requires"])
    publish_requirements = set(pyproject["project"]["optional-dependencies"]["publish"])

    assert build_requirements <= publish_requirements
####


def test_distribution_checker_exposes_temp_directory_option() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "check_distribution.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--temp-dir" in result.stdout
####
