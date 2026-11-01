# Publishing

The distribution name is `attitude-convention-inference`. The canonical command is
`attitude-conventions`; the full distribution-name command remains as a 0.0.x
compatibility alias.

## Validate locally

```bash
python -m pip install -e '.[dev,publish]'
python scripts/ci.py
python scripts/check_distribution.py
```

The distribution check builds an sdist, builds a wheel from that sdist, validates
metadata with Twine, and imports the installed wheel from outside the checkout.
It also rejects direct-URL dependencies in either artifact's metadata, which
public package indexes should not accept. The check uses a temporary directory
and does not leave release files in `dist/`. The `publish` extra includes the
build backend requirements so the offline check can reuse them without isolation.
After installing `.[dev,publish]`, run
`python scripts/check_distribution.py --no-isolation`. If Twine is not installed,
`--skip-twine` skips only the metadata check; the wheel and sdist smoke checks
still run. On restricted Windows environments, pass a short workspace-local
scratch parent such as `--temp-dir .t`; the checker propagates it to build and
install subprocesses as well as using it for the outer temporary directory.

`scripts/ci.py` also compares public exports, function signatures, class
constructor shapes, enum members, and generated model schemas against the local
`0.0.1.a0` tag. It imports the tag
in a temporary directory, so no generated schemas are tracked. A clone running
the complete checks needs that tag; the GitHub checkout fetches tags and history.
An intentional contract change needs explicit compatibility review before
changing this gate.

To save the validated files for a release, use a fresh output directory:

```bash
python scripts/check_distribution.py --outdir dist
```

The command refuses to mix new release files with older wheels or sdists already
in `dist/`.

To repair auto-fixable issues before checking, run `python scripts/ci.py --fix`.

To regenerate the Pydantic schemas and deterministic example artifacts directly:

```bash
python scripts/generate_schemas.py --fix
python scripts/generate_schemas.py --check
```

Test the wheel as a one-off application:

```bash
uvx --from ./dist/*.whl \
  attitude-conventions --help
```

Test a persistent isolated installation:

```bash
pipx install --force ./dist/*.whl
attitude-conventions --version
pipx uninstall attitude-convention-inference
```

## TestPyPI

```bash
python -m twine upload --repository testpypi dist/*
```

Resolve NumPy and SciPy from the normal package index during the TestPyPI check:

```bash
uvx \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  attitude-convention-inference --version
```

## PyPI

After TestPyPI validation and confirmation that the project name is available:

```bash
python -m twine upload dist/*
```

Prefer trusted publishing in CI over storing a long-lived PyPI token.
