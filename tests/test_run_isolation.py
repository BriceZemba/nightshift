"""One bad finding must not take down the night's work.

A run died whole because a single model call got an invalid API key. Sixty findings, one
unlucky, nothing produced. A nightly job that behaves that way is not tolerant of anything,
and the failure it hit is among the most ordinary a deployment can have.
"""

from __future__ import annotations

import pytest
from test_integration import (
    AlwaysPassVerifier,
    FakeGenAI,
    FakeGitHub,
    FakeOSV,
    FakeStore,
)
from test_integration import _advisory_with_fix as _advisory

from nightshift.config import Settings
from nightshift.models import FindingStatus
from nightshift.run import NightshiftRun


def _settings(**overrides: object) -> Settings:
    base = {
        "NIGHTSHIFT_REPO_ALLOWLIST": "me/myrepo",
        "NIGHTSHIFT_DRY_RUN": False,
        "NIGHTSHIFT_MAX_PRS_PER_RUN": 5,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


SOURCES = {
    "src/api.py": "import requests\n\ndef f(u):\n    return requests.get(u)\n",
    "tests/t.py": "def test_ok():\n    assert True\n",
}


class _ExplodingTriager:
    """Fails on the first finding it sees, succeeds afterwards."""

    def __init__(self, llm) -> None:
        self.calls = 0

    def triage(self, advisory, dependency, sources, *, sibling_repos=None):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("API key not valid. Please pass a valid API key.")
        from nightshift.models import CallSite, Reachability, TriageVerdict

        return TriageVerdict(
            reachability=Reachability.REACHABLE,
            call_path=[CallSite(file_path="src/api.py", line=4, symbol="requests.get")],
            rationale="reachable",
        )


def _build(settings: Settings, hits: dict, advisories: list):
    github = FakeGitHub(
        {
            "me/myrepo": {
                "manifests": {"requirements.txt": "requests==2.19.1\nflask==0.12.2\n"},
                "sources": SOURCES,
            }
        }
    )
    from nightshift.llm import LLMClient

    # Injected at construction, not assigned afterwards. The constructor builds
    # Triager(self.llm) and Patcher(self.llm) immediately, so reassigning run.llm later
    # leaves those agents holding the original client -- which, with no API key, means a
    # real network call. That passed locally and failed in CI, which is the only place the
    # absence of a key was honest.
    run = NightshiftRun(
        github=github,
        osv=FakeOSV(hits, advisories),
        store=FakeStore(),
        llm=LLMClient(client=FakeGenAI(), settings=settings),
        settings=settings,
        verifier=AlwaysPassVerifier(),
    )
    return run, github


class TestFindingFailureIsolation:
    async def test_one_failing_finding_does_not_abort_the_run(self) -> None:
        """The behaviour that was missing: the run completes and reports."""
        settings = _settings()
        advisory_a = _advisory()
        advisory_b = _advisory()
        advisory_b.id = "GHSA-second"

        run, _ = _build(
            settings,
            {"requests@2.19.1": ["GHSA-fixable"], "flask@0.12.2": ["GHSA-second"]},
            [advisory_a, advisory_b],
        )
        run.triager = _ExplodingTriager(run.llm)

        record = await run.execute()

        # execute() returning at all is the assertion: before this fix it raised.
        assert record.failed >= 1, "the failing finding should be recorded as failed"
        assert record.advisories_ingested == 2, "discovery still completed"

    async def test_the_failure_is_recorded_not_swallowed(self) -> None:
        """Isolating a failure must not mean hiding it."""
        settings = _settings()
        run, _ = _build(settings, {"requests@2.19.1": ["GHSA-fixable"]}, [_advisory()])
        run.triager = _ExplodingTriager(run.llm)

        record = await run.execute()

        assert record.failed == 1
        stored = list(run.store.findings.values())
        assert any(f.status is FindingStatus.FAILED for f in stored)

    async def test_a_clean_run_records_no_failures(self) -> None:
        """The isolation must not turn ordinary outcomes into failures."""
        settings = _settings()
        run, _ = _build(settings, {"requests@2.19.1": ["GHSA-fixable"]}, [_advisory()])

        record = await run.execute()

        assert record.failed == 0


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("API key not valid"),
        ValueError("malformed advisory"),
        ConnectionError("network unreachable"),
    ],
)
async def test_any_ordinary_failure_is_contained(error: Exception) -> None:
    """Not just the one that happened. Anything a single unit of work can raise."""
    settings = _settings()

    class _Raises:
        def __init__(self, llm) -> None:
            pass

        def triage(self, *a: object, **k: object):
            raise error

    run, github = _build(settings, {"requests@2.19.1": ["GHSA-fixable"]}, [_advisory()])
    run.triager = _Raises(run.llm)

    record = await run.execute()

    assert record.failed == 1
    assert github.prs_opened == []
