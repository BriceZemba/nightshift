"""Shared test setup.

The suite must never read the developer's real ``.env``. Without this, whatever is sitting
in a local config file silently becomes test input - a real GitHub token, a real
allowlist, ``NIGHTSHIFT_DRY_RUN`` flipped the wrong way - and tests start passing or
failing for reasons that have nothing to do with the code under test.
"""

from __future__ import annotations

import os

import pytest

from nightshift.config import Settings, get_settings

#: Environment variables that would leak a developer's local configuration into tests.
_LEAKY_PREFIXES = ("NIGHTSHIFT_", "GITHUB_", "GOOGLE_", "NVD_")


@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test against a clean, file-free configuration."""
    # Disable .env loading for the duration of the test.
    monkeypatch.setitem(Settings.model_config, "env_file", None)

    for key in [k for k in os.environ if k.startswith(_LEAKY_PREFIXES)]:
        monkeypatch.delenv(key, raising=False)

    # Settings are cached process-wide; a stale entry would defeat the isolation above.
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
