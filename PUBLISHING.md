# Publishing

The distribution name is `attitude-convention-inference`. The canonical command is `attitude-conventions`; the full distribution-name command remains as a 0.4.x compatibility alias.

## Validate locally

```bash
python -m pip install -e '.[dev,publish]'
pytest -q
ruff check .
pyright
python -m build
python -m twine check dist/*
```

Test the wheel as a one-off application:

```bash
uvx --from ./dist/attitude_convention_inference-0.4.0-py3-none-any.whl \
  attitude-conventions --help
```

Test a persistent isolated installation:

```bash
pipx install --force ./dist/attitude_convention_inference-0.4.0-py3-none-any.whl
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
