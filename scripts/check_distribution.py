"""Build and smoke-test the sdist and wheel from an isolated working directory."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DIRECT_REFERENCE_REQUIREMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:\[[^]]+\])?\s*@")


def _run(command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)
####


def _temporary_environment(
    temporary_directory: Path,
    *,
    base_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if base_environment is None else base_environment)
    environment.update(
        TMP=str(temporary_directory),
        TEMP=str(temporary_directory),
        TMPDIR=str(temporary_directory),
    )
    return environment
####


def _check_metadata(content: bytes, artifact: str) -> None:
    """Reject direct URL dependencies that public package indexes should reject."""

    metadata = Parser().parsestr(content.decode("utf-8"))
    requirements = metadata.get_all("Requires-Dist", [])
    direct_references = [
        requirement
        for requirement in requirements
        if DIRECT_REFERENCE_REQUIREMENT.match(requirement)
    ]
    if direct_references:
        rendered = ", ".join(direct_references)
        raise RuntimeError(f"{artifact} contains direct URL dependencies: {rendered}")
    ####
####


def _check_sdist(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        metadata_members = [
            member
            for member in members
            if member.isfile()
            and PurePosixPath(member.name).name == "PKG-INFO"
            and len(PurePosixPath(member.name).parts) == 2
        ]
        if len(metadata_members) != 1:
            raise RuntimeError("Source distribution must contain exactly one PKG-INFO file.")
        ####
        metadata_file = archive.extractfile(metadata_members[0])
        if metadata_file is None:
            raise RuntimeError("Unable to read source distribution metadata.")
        ####
        _check_metadata(metadata_file.read(), path.name)
    ####
    required = (
        "pyproject.toml",
        "src/attitude_convention_inference/py.typed",
        "src/attitude_convention_inference/data/profiles/neutral-v1.json",
        "src/attitude_convention_inference/data/profiles/common-engineering-v1.json",
        "docs/ARCHITECTURE.md",
        "docs/PERFORMANCE.md",
        "docs/PREFERENCE_RESEARCH.md",
        "docs/ROADMAP.md",
        "docs/VALIDATION.md",
        "scripts/check_distribution.py",
        "scripts/generate_validation_corpus.py",
        "scripts/generate_euler_validation_corpus.py",
        "scripts/prepare_tumvi_corpus.py",
        "scripts/validate_sensor_mismatch.py",
        "scripts/validate_sparse_anchors.py",
        "scripts/validate_sparse_tumvi.py",
        "scripts/tumvi_data.py",
        "scripts/tumvi_sparse_models.py",
        "scripts/validation_noise.py",
        "examples/synthetic_quaternion.csv",
        "tests/test_independent_recovery.py",
        "tests/test_prepare_tumvi_corpus.py",
        "tests/test_sparse_anchor_validation.py",
        "tests/test_sparse_orientation.py",
    )
    missing = [item for item in required if not any(name.endswith("/" + item) for name in names)]
    if missing:
        raise RuntimeError(f"Source distribution is missing: {', '.join(missing)}")
    ####
####


def check_distribution(
    *,
    no_isolation: bool,
    skip_twine: bool,
    outdir: Path | None = None,
    temp_dir: Path | None = None,
) -> None:
    temporary_parent = None
    if temp_dir is not None:
        temporary_parent = temp_dir if temp_dir.is_absolute() else ROOT / temp_dir
        temporary_parent.mkdir(parents=True, exist_ok=True)
    ####
    with tempfile.TemporaryDirectory(dir=temporary_parent) as temporary_directory:
        scratch = Path(temporary_directory)
        child_environment = _temporary_environment(scratch)
        output = scratch / "dist"
        build_command = [sys.executable, "-m", "build", "--sdist", "--outdir", str(output)]
        if no_isolation:
            build_command.append("--no-isolation")
        ####
        _run(build_command, env=child_environment)

        sdists = tuple(output.glob("*.tar.gz"))
        if len(sdists) != 1:
            raise RuntimeError("Expected exactly one source distribution.")
        ####
        sdist = sdists[0]
        _check_sdist(sdist)
        pip_cache = scratch
        wheel_command = [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(sdist),
            "--no-deps",
            "--cache-dir",
            str(pip_cache),
            "--wheel-dir",
            str(output),
        ]
        if no_isolation:
            wheel_command.append("--no-build-isolation")
        ####
        _run(wheel_command, cwd=scratch, env=child_environment)
        wheels = tuple(output.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("Expected exactly one wheel built from the sdist.")
        ####
        wheel = wheels[0]
        with zipfile.ZipFile(wheel) as archive:
            metadata_names = [
                name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                raise RuntimeError("Wheel must contain exactly one .dist-info/METADATA file.")
            ####
            _check_metadata(archive.read(metadata_names[0]), wheel.name)
        ####
        if not skip_twine:
            _run(
                [sys.executable, "-m", "twine", "check", str(wheel), str(sdist)],
                env=child_environment,
            )
        ####

        installed = scratch / "installed-wheel"
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--target",
                str(installed),
                str(wheel),
            ],
            env=child_environment,
        )

        smoke = "\n".join(
            (
                "import os",
                "import numpy as np",
                "from importlib import resources",
                "from importlib.metadata import distribution",
                "from pathlib import Path",
                "import attitude_convention_inference as package",
                "report_schema = package.InferenceReport.model_json_schema()",
                "diagnostics_schema = report_schema['$defs']['OverallDiagnostics']",
                "confidence_schema = diagnostics_schema['properties']['confidence']",
                "assert confidence_schema['minimum'] == 0.0",
                "assert confidence_schema['maximum'] == 1.0",
                "assert 'Uncalibrated composite heuristic score' in "
                "confidence_schema['description']",
                "confidence_label_schema = diagnostics_schema['properties']['confidence_label']",
                "assert 'not a calibrated probability' in confidence_label_schema['description']",
                "assert {'interval', 'dt_s', 'error_angle_rad'} <= package."
                "IntervalDiagnosticRow.__required_keys__",
                "assert package.IntervalDiagnosticRow.__optional_keys__ == "
                "{'time_start_s', 'time_end_s'}",
                "decision = package.InferenceDecision.from_evidence(",
                "    evidence_sufficient=True, evidence_contender_class_ids=('class-a',)",
                ")",
                "assert decision.status is package.InferenceDecisionStatus.RECOMMENDATION",
                "assert decision.recommended_class_id == 'class-a'",
                "assert Path(package.__file__).resolve().is_relative_to(",
                "    Path(os.environ['ACI_WHEEL_ROOT']).resolve()",
                ")",
                "dist = distribution('attitude-convention-inference')",
                "assert package.__version__ == dist.version",
                "sparse_data = package.SparseOrientationRateData(",
                "    representation=package.AttitudeRepresentation.QUATERNION,",
                "    orientation_anchors=np.tile([0.0, 0.0, 0.0, 1.0], (3, 1)),",
                "    orientation_anchor_times_s=np.array([0.0, 0.5, 1.0]),",
                "    rates=np.zeros((5, 3)),",
                "    rate_times_s=np.linspace(0.0, 1.0, 5),",
                ")",
                "sparse_report = package.infer_sparse_orientation_conventions(sparse_data)",
                "assert sparse_report.observation_model == 'sparse_orientation_anchors'",
                "assert sparse_report.sample_count == 3",
                "assert sparse_report.rate_sample_count == 5",
                "assert package.infer_conventions(sparse_data) == sparse_report",
                "entry_point_names = {entry.name for entry in dist.entry_points}",
                "expected_entry_points = {",
                "    'attitude-conventions',",
                "    'attitude-convention-inference',",
                "    'attitude-cohort-fusion',",
                "}",
                "assert expected_entry_points <= entry_point_names",
                "classifiers = set(dist.metadata.get_all('Classifier', []))",
                "assert {f'Programming Language :: Python :: {version}' for version in",
                "        ('3.11', '3.12', '3.13', '3.14')} <= classifiers",
                "root = resources.files('attitude_convention_inference')",
                "assert root.joinpath('py.typed').is_file()",
                "assert root.joinpath('data/profiles/neutral-v1.json').is_file()",
                "assert root.joinpath('data/profiles/common-engineering-v1.json').is_file()",
                "print(package.__version__)",
            )
        )
        environment = dict(
            child_environment,
            PYTHONPATH=str(installed),
            ACI_WHEEL_ROOT=str(installed),
        )
        _run([sys.executable, "-c", smoke], cwd=scratch, env=environment)
        _run(
            [sys.executable, "-m", "attitude_convention_inference", "--version"],
            cwd=scratch,
            env=environment,
        )
        if outdir is not None:
            outdir.mkdir(parents=True, exist_ok=True)
            if any(outdir.glob("*.whl")) or any(outdir.glob("*.tar.gz")):
                raise FileExistsError(
                    f"{outdir} already contains distribution files; choose an empty directory"
                )
            ####
            shutil.copy2(wheel, outdir / wheel.name)
            shutil.copy2(sdist, outdir / sdist.name)
        ####
        print("Distribution smoke test passed.")
    ####
####


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-isolation",
        action="store_true",
        help="reuse installed build dependencies when offline",
    )
    parser.add_argument(
        "--skip-twine",
        action="store_true",
        help="skip metadata validation when Twine is not installed",
    )
    parser.add_argument(
        "--outdir", type=Path, help="copy validated release files into an empty directory"
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        help="parent directory for temporary build files (use a short local path on Windows)",
    )
    args = parser.parse_args()
    check_distribution(
        no_isolation=args.no_isolation,
        skip_twine=args.skip_twine,
        outdir=args.outdir,
        temp_dir=args.temp_dir,
    )
####


if __name__ == "__main__":
    main()
####
