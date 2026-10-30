"""Build and smoke-test the sdist and wheel from an isolated working directory."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)
####


def _check_sdist(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
    ####
    required = (
        "pyproject.toml",
        "src/attitude_convention_inference/py.typed",
        "src/attitude_convention_inference/data/profiles/neutral-v1.json",
        "src/attitude_convention_inference/data/profiles/common-engineering-v1.json",
        "docs/VALIDATION.md",
        "scripts/check_distribution.py",
        "scripts/generate_validation_corpus.py",
        "examples/synthetic_quaternion.csv",
    )
    missing = [item for item in required if not any(name.endswith("/" + item) for name in names)]
    if missing:
        raise RuntimeError(f"Source distribution is missing: {', '.join(missing)}")
    ####
####


def check_distribution(*, no_isolation: bool, skip_twine: bool, outdir: Path | None = None) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        scratch = Path(temporary_directory)
        output = scratch / "dist"
        build_command = [sys.executable, "-m", "build", "--sdist", "--outdir", str(output)]
        if no_isolation:
            build_command.append("--no-isolation")
        ####
        _run(build_command)

        sdists = tuple(output.glob("*.tar.gz"))
        if len(sdists) != 1:
            raise RuntimeError("Expected exactly one source distribution.")
        ####
        sdist = sdists[0]
        _check_sdist(sdist)
        wheel_command = [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(sdist),
            "--no-deps",
            "--no-cache-dir",
            "--wheel-dir",
            str(output),
        ]
        if no_isolation:
            wheel_command.append("--no-build-isolation")
        ####
        _run(wheel_command)
        wheels = tuple(output.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("Expected exactly one wheel built from the sdist.")
        ####
        wheel = wheels[0]
        if not skip_twine:
            _run([sys.executable, "-m", "twine", "check", str(wheel), str(sdist)])
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
            ]
        )

        smoke = "\n".join(
            (
                "import os",
                "from importlib import resources",
                "from importlib.metadata import distribution",
                "from pathlib import Path",
                "import attitude_convention_inference as package",
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
                "assert any(e.name == 'attitude-conventions' for e in dist.entry_points)",
                "root = resources.files('attitude_convention_inference')",
                "assert root.joinpath('py.typed').is_file()",
                "assert root.joinpath('data/profiles/neutral-v1.json').is_file()",
                "assert root.joinpath('data/profiles/common-engineering-v1.json').is_file()",
                "print(package.__version__)",
            )
        )
        environment = dict(os.environ, PYTHONPATH=str(installed), ACI_WHEEL_ROOT=str(installed))
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
    args = parser.parse_args()
    check_distribution(
        no_isolation=args.no_isolation,
        skip_twine=args.skip_twine,
        outdir=args.outdir,
    )
####


if __name__ == "__main__":
    main()
####
