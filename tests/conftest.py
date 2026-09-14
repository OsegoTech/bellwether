"""Suite-wide fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test in its own empty directory.

    Config loads ``.env`` from the working directory, so a developer's real
    ``.env`` in the repo root must never leak into a test run.
    """
    monkeypatch.chdir(tmp_path)
