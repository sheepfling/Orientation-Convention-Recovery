from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """Provide test directories inside the repository instead of global temp."""
    repo_tmp = Path(__file__).resolve().parents[1] / ".pytest-tmp"
    repo_tmp.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(prefix="pytest-", dir=repo_tmp) as directory:
        yield Path(directory)
    ####
####


