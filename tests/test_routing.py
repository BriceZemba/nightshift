"""Routing decides what costs money and what reaches a human, so it is tested directly
rather than through the graph."""

from __future__ import annotations

import pytest

from nightshift.agents import routing
from nightshift.agents.routing import (
    APPROVE,
    AUTO,
    DISMISS,
    ESCALATE,
    HUMAN,
    PATCH,
    RETRY,
)
from nightshift.models import (
    Dependency,
    Ecosystem,
    Finding,
    PatchAttempt,
    PatchStrategy,
    Reachability,
    TriageVerdict,
)


def _finding(**overrides: object) -> Finding:
    base = {
        "id": "f1",
        "run_id": "r1",
        "advisory_id": "GHSA-xxxx",
        "repo": "me/myrepo",
        "dependency": Dependency(
            name="requests",
            ecosystem=Ecosystem.PYPI,
            version="2.19.1",
            manifest_path="requirements.txt",
        ),
    }
    base.update(overrides)
    return Finding(**base)  # type: ignore[arg-type]


def _attempt(n: int, *, passed: bool) -> PatchAttempt:
    return PatchAttempt(attempt=n, strategy=PatchStrategy.UPSTREAM_BUMP, tests_passed=passed)


class TestTriageRoute:
    def test_unreachable_is_dismissed(self) -> None:
        """The 180 -> 6 reduction happens here."""
        finding = _finding(verdict=TriageVerdict(reachability=Reachability.NOT_REACHABLE))
        assert routing.triage_route(finding) == DISMISS

    def test_reachable_is_patched(self) -> None:
        finding = _finding(verdict=TriageVerdict(reachability=Reachability.REACHABLE))
        assert routing.triage_route(finding) == PATCH

    def test_unknown_reachability_is_patched_not_dismissed(self) -> None:
        """Failing safe: an unnecessary patch is recoverable, a missed vulnerability is not.
        The approval gate downstream still routes it to a human."""
        finding = _finding(verdict=TriageVerdict(reachability=Reachability.UNKNOWN))
        assert routing.triage_route(finding) == PATCH

    def test_missing_verdict_is_patched(self) -> None:
        assert routing.triage_route(_finding()) == PATCH


class TestVerifyRoute:
    def test_passing_tests_approve(self) -> None:
        finding = _finding(attempts=[_attempt(1, passed=True)])
        assert routing.verify_route(finding, max_attempts=3) == APPROVE

    def test_failure_below_limit_retries(self) -> None:
        finding = _finding(attempts=[_attempt(1, passed=False)])
        assert routing.verify_route(finding, max_attempts=3) == RETRY

    def test_failure_at_limit_escalates(self) -> None:
        """The critique loop must terminate. Without a bound, a model that cannot solve the
        problem is asked forever while the run's cost grows."""
        finding = _finding(
            attempts=[_attempt(1, passed=False), _attempt(2, passed=False), _attempt(3, passed=False)]
        )
        assert routing.verify_route(finding, max_attempts=3) == ESCALATE

    def test_failure_past_limit_still_escalates(self) -> None:
        finding = _finding(attempts=[_attempt(n, passed=False) for n in range(1, 6)])
        assert routing.verify_route(finding, max_attempts=3) == ESCALATE

    def test_no_attempts_escalates(self) -> None:
        """Nothing was produced, so there is nothing to approve."""
        assert routing.verify_route(_finding(), max_attempts=3) == ESCALATE

    def test_only_latest_attempt_decides(self) -> None:
        """A pass after earlier failures is still a pass - that is the loop working."""
        finding = _finding(attempts=[_attempt(1, passed=False), _attempt(2, passed=True)])
        assert routing.verify_route(finding, max_attempts=3) == APPROVE

    @pytest.mark.parametrize("max_attempts", [1, 2, 5])
    def test_respects_configured_bound(self, max_attempts: int) -> None:
        finding = _finding(attempts=[_attempt(n, passed=False) for n in range(1, max_attempts + 1)])
        assert routing.verify_route(finding, max_attempts) == ESCALATE


class TestApprovalRoute:
    def test_no_reason_proceeds_automatically(self) -> None:
        assert routing.approval_route(None) == AUTO

    def test_any_reason_routes_to_human(self) -> None:
        assert routing.approval_route("Major version bump") == HUMAN

    def test_empty_string_is_not_a_reason(self) -> None:
        assert routing.approval_route("") == AUTO
