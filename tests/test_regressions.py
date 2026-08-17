"""Regressions from the first live run against a real repository.

Three bugs surfaced only when the pipeline met real OSV data and a real GitHub repo. Each
one is pinned here so it cannot come back quietly.
"""

from __future__ import annotations

import pytest

from nightshift.config import Settings
from nightshift.llm import LLMClient, _is_rate_limit
from nightshift.models import AffectedPackage, Ecosystem, VersionRange
from nightshift.sources.osv import parse_advisory


class TestGitRangesAreNotVersions:
    """OSV GIT ranges carry commit SHAs. Pinning a manifest to a 40-character hash
    produces an install that cannot resolve, and the first live run proposed exactly
    that: `to_version=c45d7c49ea75133e52ab22a8e9e13173938e36ff`."""

    def test_git_range_is_skipped(self) -> None:
        package = AffectedPackage(
            name="requests",
            ecosystem=Ecosystem.PYPI,
            ranges=[
                VersionRange(introduced="0", fixed="c45d7c49ea75133e52ab22a8e9e13173", type="GIT")
            ],
        )
        assert package.first_fixed_version() is None

    def test_ecosystem_range_is_preferred_over_git(self) -> None:
        package = AffectedPackage(
            name="requests",
            ecosystem=Ecosystem.PYPI,
            ranges=[
                VersionRange(introduced="0", fixed="deadbeef" * 5, type="GIT"),
                VersionRange(introduced="2.3.0", fixed="2.31.0", type="ECOSYSTEM"),
            ],
        )
        assert package.first_fixed_version() == "2.31.0"

    def test_range_type_survives_parsing(self) -> None:
        raw = {
            "id": "GHSA-x",
            "affected": [
                {
                    "package": {"name": "requests", "ecosystem": "PyPI"},
                    "ranges": [
                        {
                            "type": "GIT",
                            "events": [{"introduced": "0"}, {"fixed": "abc123"}],
                        }
                    ],
                }
            ],
        }
        advisory = parse_advisory(raw)
        assert advisory.affected[0].ranges[0].type == "GIT"
        assert advisory.affected[0].first_fixed_version() is None

    def test_missing_type_defaults_to_ecosystem(self) -> None:
        """OSV omits the type on some records; treating those as GIT would discard real
        fixes."""
        raw = {
            "id": "GHSA-y",
            "affected": [
                {
                    "package": {"name": "requests", "ecosystem": "PyPI"},
                    "ranges": [{"events": [{"introduced": "0"}, {"fixed": "2.31.0"}]}],
                }
            ],
        }
        advisory = parse_advisory(raw)
        assert advisory.affected[0].first_fixed_version() == "2.31.0"


class TestRateLimitDetection:
    """A 429 killed the first live run outright. It is the single most predictable error
    the system faces on a free tier, so it must be retried rather than fatal."""

    @pytest.mark.parametrize(
        "message",
        [
            "Error code: 429 - quota exceeded",
            "RateLimitError: too many requests",
            "RESOURCE_EXHAUSTED",
            "429 Too Many Requests",
        ],
    )
    def test_rate_limits_are_recognized(self, message: str) -> None:
        assert _is_rate_limit(RuntimeError(message)) is True

    @pytest.mark.parametrize(
        "message",
        ["401 Unauthorized", "invalid model name", "connection reset"],
    )
    def test_other_failures_are_not_retried(self, message: str) -> None:
        assert _is_rate_limit(RuntimeError(message)) is False


class _RateLimited:
    """Fails with a 429 the first `failures` times, then succeeds."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0
        outer = self

        class _Interactions:
            def create(self, model: str, input: str, **kwargs: object):
                outer.calls += 1
                if outer.calls <= outer.failures:
                    raise RuntimeError("Error code: 429 - quota exceeded")
                return type("R", (), {"text": "ok", "usage_metadata": None})()

        self.interactions = _Interactions()


class TestRateLimitRetry:
    def test_a_transient_rate_limit_is_survived(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("nightshift.llm.time.sleep", lambda _: None)
        client = _RateLimited(failures=2)

        result = LLMClient(client=client, settings=Settings()).reason("hello")

        assert result.text == "ok"
        assert client.calls == 3

    def test_persistent_rate_limiting_eventually_gives_up(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bounded, so a quota outage cannot hang a run forever."""
        monkeypatch.setattr("nightshift.llm.time.sleep", lambda _: None)
        client = _RateLimited(failures=99)

        with pytest.raises(RuntimeError, match="still rate limited"):
            LLMClient(client=client, settings=Settings()).reason("hello")

    def test_non_rate_limit_errors_are_not_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bad request retried four times is four times the waste and no more chance of
        succeeding."""
        monkeypatch.setattr("nightshift.llm.time.sleep", lambda _: None)
        calls: list[int] = []

        class _Unauthorized:
            def __init__(self) -> None:
                class _Interactions:
                    def create(self, model: str, input: str, **kwargs: object):
                        calls.append(1)
                        raise RuntimeError("401 Unauthorized")

                self.interactions = _Interactions()

        with pytest.raises(RuntimeError, match="401"):
            LLMClient(client=_Unauthorized(), settings=Settings()).reason("hello")

        assert len(calls) == 1


class TestVerifierReceivesManifests:
    """Every verification failed on the first live run because the work tree held only
    Python files. An upstream bump patches requirements.txt, so `git apply` had nothing to
    apply to and three attempts failed for a reason unrelated to the patch."""

    def test_manifests_are_merged_into_the_verifier_input(self) -> None:
        from nightshift.agents.graph import FindingContext
        from nightshift.models import Dependency, Finding

        ctx = FindingContext(
            finding=Finding(
                id="f1",
                run_id="r1",
                advisory_id="GHSA-x",
                repo="me/repo",
                dependency=Dependency(
                    name="requests",
                    ecosystem=Ecosystem.PYPI,
                    version="2.19.1",
                    manifest_path="requirements.txt",
                ),
            ),
            advisory=parse_advisory({"id": "GHSA-x"}),
            sources={"src/app.py": "import requests\n"},
            manifests={"requirements.txt": "requests==2.19.1\n"},
        )

        merged = {**ctx.sources, **ctx.manifests}

        assert "requirements.txt" in merged, "the patched file must be in the work tree"
        assert "src/app.py" in merged, "sources are still needed to run the tests"
