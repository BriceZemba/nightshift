"""Staying under the model's rate limit rather than bouncing off it.

Backoff alone was not enough on the free tier: it reacts to a 429 already spent, and each
retry burns more of the same quota. A run died after four attempts. The fix is two-sided,
and both sides are tested here: make fewer calls, and pace the ones that remain.
"""

from __future__ import annotations

import pytest

from nightshift.agents.triager import Triager
from nightshift.config import Settings
from nightshift.llm import LLMClient, RateLimiter
from nightshift.models import (
    Advisory,
    AffectedPackage,
    CallSite,
    Dependency,
    Ecosystem,
    Reachability,
    TriageVerdict,
    VersionRange,
)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> dict:
    """A fake clock that advances when the limiter sleeps.

    Stubbing sleep to a no-op while leaving monotonic real makes the limiter spin: it
    waits for a clock that never moves. Time has to advance for the wait to mean anything.
    """
    state = {"now": 1000.0, "slept": []}

    def fake_sleep(seconds: float) -> None:
        state["slept"].append(seconds)
        state["now"] += seconds

    monkeypatch.setattr("nightshift.llm.time.monotonic", lambda: state["now"])
    monkeypatch.setattr("nightshift.llm.time.sleep", fake_sleep)
    return state


class TestRateLimiter:
    def test_allows_calls_under_the_limit(self, clock: dict) -> None:
        limiter = RateLimiter(requests_per_minute=5)
        for _ in range(5):
            limiter.acquire()

        assert clock["slept"] == [], "nothing should wait while under the ceiling"

    def test_waits_once_the_limit_is_reached(self, clock: dict) -> None:
        limiter = RateLimiter(requests_per_minute=3)
        for _ in range(4):
            limiter.acquire()

        assert len(clock["slept"]) == 1
        assert 0 < clock["slept"][0] <= 61

    def test_the_wait_is_only_as_long_as_needed(self, clock: dict) -> None:
        """Sleeping a full minute when the oldest call is 50 seconds old wastes ten
        seconds of every window."""
        limiter = RateLimiter(requests_per_minute=1)
        limiter.acquire()
        clock["now"] += 50

        limiter.acquire()

        assert clock["slept"][0] == pytest.approx(10.1, abs=0.2)

    def test_window_slides(self, clock: dict) -> None:
        """A fixed window lets a burst at the boundary spend two windows' budget at once,
        which is exactly the shape that trips the real quota."""
        limiter = RateLimiter(requests_per_minute=2)
        limiter.acquire()
        limiter.acquire()

        clock["now"] += 61  # the first two calls age out
        limiter.acquire()

        assert clock["slept"] == []

    def test_a_zero_limit_is_clamped(self) -> None:
        """Misconfiguration should throttle hard, not divide by zero or spin."""
        assert RateLimiter(requests_per_minute=0).limit == 1


class _CountingLLM:
    """Counts model calls without making any."""

    def __init__(self) -> None:
        self.calls = 0
        self.settings = Settings()
        outer = self

        class _Interactions:
            def create(self, model: str, input: str, **kwargs: object):
                outer.calls += 1
                return type("R", (), {"text": "explained", "usage_metadata": None})()

        self.interactions = _Interactions()


def _advisory(advisory_id: str) -> Advisory:
    return Advisory(
        id=advisory_id,
        summary="Header leak",
        details="Upgrade.",
        screened=True,
        affected=[
            AffectedPackage(
                name="requests",
                ecosystem=Ecosystem.PYPI,
                ranges=[VersionRange(introduced="2.3.0", fixed="2.31.0")],
            )
        ],
    )


def _dependency() -> Dependency:
    return Dependency(
        name="requests",
        ecosystem=Ecosystem.PYPI,
        version="2.19.1",
        manifest_path="requirements.txt",
    )


SOURCES = {"src/fetcher.py": "import requests\n\ndef f(u):\n    return requests.get(u)\n"}


class TestRationaleReuse:
    """The live run made fourteen model calls for five packages. The explanation describes
    how *this repository* uses a package, which does not change between advisories."""

    def _triager(self) -> tuple[Triager, _CountingLLM]:
        fake = _CountingLLM()
        return Triager(LLMClient(client=fake, settings=Settings())), fake

    def test_many_advisories_on_one_package_cost_one_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("nightshift.llm.time.sleep", lambda _: None)
        triager, fake = self._triager()

        for i in range(10):
            triager.triage(_advisory(f"GHSA-{i}"), _dependency(), SOURCES)

        assert fake.calls == 1, "ten advisories, one explanation"

    def test_every_finding_still_gets_a_rationale(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reuse must not mean later findings come back empty."""
        monkeypatch.setattr("nightshift.llm.time.sleep", lambda _: None)
        triager, _ = self._triager()

        first = triager.triage(_advisory("GHSA-1"), _dependency(), SOURCES)
        second = triager.triage(_advisory("GHSA-2"), _dependency(), SOURCES)

        assert first.rationale == "explained"
        assert second.rationale == "explained"

    def test_different_call_paths_are_explained_separately(self) -> None:
        """A repository using a package in two places deserves both explained."""
        verdict_a = TriageVerdict(
            reachability=Reachability.REACHABLE,
            call_path=[CallSite(file_path="a.py", line=1, symbol="requests.get")],
        )
        verdict_b = TriageVerdict(
            reachability=Reachability.REACHABLE,
            call_path=[CallSite(file_path="b.py", line=9, symbol="requests.post")],
        )

        assert Triager._rationale_key("requests", verdict_a) != Triager._rationale_key(
            "requests", verdict_b
        )

    def test_unreachable_findings_cost_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("nightshift.llm.time.sleep", lambda _: None)
        triager, fake = self._triager()

        triager.triage(_advisory("GHSA-1"), _dependency(), {"a.py": "import os\n"})

        assert fake.calls == 0


class TestConcurrencyIsConfigurable:
    def test_default_leaves_quota_headroom(self) -> None:
        """More parallelism buys nothing once the limiter is the bottleneck; it just makes
        every worker wait."""
        settings = Settings()
        assert settings.concurrency <= 2
        assert settings.llm_requests_per_minute < 20
