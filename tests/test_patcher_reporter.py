"""Deterministic patching and pull-request rendering.

Both are pure functions on purpose. The version bump is a text edit rather than a model
call because a regex cannot invent a version that was never published, and the PR body is
the thing a maintainer actually reads, so its exact text is worth testing.
"""

from __future__ import annotations

from nightshift.agents.patcher import bump_manifest, bump_requirement_line
from nightshift.agents.reporter import (
    branch_name,
    pull_request_title,
    render_pull_request_body,
)
from nightshift.llm import Usage
from nightshift.models import (
    Advisory,
    AffectedPackage,
    CallSite,
    Dependency,
    Ecosystem,
    Finding,
    PatchAttempt,
    PatchStrategy,
    Reachability,
    Severity,
    TriageVerdict,
    VersionRange,
)


class TestBumpRequirementLine:
    def test_bumps_a_pinned_line(self) -> None:
        assert bump_requirement_line("requests==2.19.1", "requests", "2.31.0") == "requests==2.31.0"

    def test_returns_none_for_a_different_package(self) -> None:
        """None distinguishes 'not my line' from 'changed'."""
        assert bump_requirement_line("flask==3.0.0", "requests", "2.31.0") is None

    def test_preserves_extras(self) -> None:
        result = bump_requirement_line("celery[redis]==5.3.0", "celery", "5.3.6")
        assert result == "celery[redis]==5.3.6"

    def test_preserves_trailing_comment(self) -> None:
        result = bump_requirement_line("requests==2.19.1  # pinned", "requests", "2.31.0")
        assert result == "requests==2.31.0  # pinned"

    def test_preserves_environment_marker(self) -> None:
        result = bump_requirement_line('tomli==2.0.0 ; python_version < "3.11"', "tomli", "2.0.1")
        assert result is not None
        assert 'python_version < "3.11"' in result

    def test_preserves_indentation(self) -> None:
        assert bump_requirement_line("  requests==2.19.1", "requests", "2.31.0") == "  requests==2.31.0"

    def test_matches_normalized_names(self) -> None:
        result = bump_requirement_line("Flask_SQLAlchemy==3.1.0", "flask-sqlalchemy", "3.1.1")
        assert result == "Flask_SQLAlchemy==3.1.1"

    def test_ignores_unpinned_lines(self) -> None:
        assert bump_requirement_line("requests>=2.0", "requests", "2.31.0") is None


class TestBumpManifest:
    def test_reports_change(self) -> None:
        content = "flask==3.0.0\nrequests==2.19.1\n"
        updated, changed = bump_manifest(content, "requests", "2.31.0")
        assert changed is True
        assert "requests==2.31.0" in updated
        assert "flask==3.0.0" in updated

    def test_reports_no_change_when_absent(self) -> None:
        _, changed = bump_manifest("flask==3.0.0\n", "requests", "2.31.0")
        assert changed is False

    def test_preserves_line_endings(self) -> None:
        updated, _ = bump_manifest("requests==2.19.1\r\n", "requests", "2.31.0")
        assert updated.endswith("\r\n")

    def test_leaves_other_lines_untouched(self) -> None:
        content = "# comment\n\nrequests==2.19.1\n-r other.txt\n"
        updated, _ = bump_manifest(content, "requests", "2.31.0")
        assert "# comment" in updated
        assert "-r other.txt" in updated


def _finding(**overrides: object) -> Finding:
    base = {
        "id": "finding-abc",
        "run_id": "run-123",
        "advisory_id": "GHSA-x84v",
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


def _advisory() -> Advisory:
    return Advisory(
        id="GHSA-x84v",
        summary="Proxy-Authorization header leak",
        details="Details here.",
        severity=[Severity(type="CVSS_V3", score="x", numeric=6.1)],
        affected=[
            AffectedPackage(
                name="requests",
                ecosystem=Ecosystem.PYPI,
                ranges=[VersionRange(introduced="2.3.0", fixed="2.31.0")],
            )
        ],
        screened=True,
    )


class TestBranchAndTitle:
    def test_branch_name_is_deterministic(self) -> None:
        """A re-run must target the same branch, not accumulate near-duplicates."""
        assert branch_name(_finding(run_id="a")) == branch_name(_finding(run_id="b"))

    def test_branch_name_includes_the_package(self) -> None:
        assert "requests" in branch_name(_finding())

    def test_scoped_npm_name_is_made_branch_safe(self) -> None:
        finding = _finding(
            dependency=Dependency(
                name="@babel/core",
                ecosystem=Ecosystem.NPM,
                version="7.0.0",
                manifest_path="package-lock.json",
            )
        )
        name = branch_name(finding)
        assert "@" not in name
        assert name.count("/") == 1  # only the "nightshift/" prefix

    def test_title_names_advisory_and_package(self) -> None:
        title = pull_request_title(_finding(), _advisory())
        assert "GHSA-x84v" in title
        assert "requests" in title


class TestRenderPullRequestBody:
    def test_includes_severity_and_summary(self) -> None:
        body = render_pull_request_body(_finding(), _advisory())
        assert "GHSA-x84v" in body
        assert "CVSS 6.1" in body
        assert "Proxy-Authorization" in body

    def test_renders_call_sites_as_evidence(self) -> None:
        """The call path is the reason a maintainer reads this instead of closing it."""
        finding = _finding(
            verdict=TriageVerdict(
                reachability=Reachability.REACHABLE,
                call_path=[
                    CallSite(
                        file_path="src/api.py",
                        line=40,
                        symbol="requests.get",
                        snippet="requests.get(url)",
                    )
                ],
                rationale="Called directly in the request handler.",
            )
        )
        body = render_pull_request_body(finding, _advisory())
        assert "src/api.py:40" in body
        assert "requests.get" in body
        assert "Called directly in the request handler." in body

    def test_caps_the_call_site_list(self) -> None:
        sites = [
            CallSite(file_path=f"f{i}.py", line=i, symbol="requests.get") for i in range(20)
        ]
        finding = _finding(
            verdict=TriageVerdict(reachability=Reachability.REACHABLE, call_path=sites)
        )
        body = render_pull_request_body(finding, _advisory())
        assert "and 10 more" in body

    def test_backport_carries_a_review_warning(self) -> None:
        """A synthesized code change deserves more scrutiny than a version bump, and the
        body must say so."""
        finding = _finding(
            attempts=[
                PatchAttempt(
                    attempt=1,
                    strategy=PatchStrategy.BACKPORT,
                    diff="- old\n+ new",
                    tests_passed=True,
                )
            ]
        )
        body = render_pull_request_body(finding, _advisory())
        assert "No fixed version has been published upstream" in body
        assert "closer review" in body

    def test_shows_passing_verification(self) -> None:
        finding = _finding(
            attempts=[
                PatchAttempt(attempt=1, strategy=PatchStrategy.UPSTREAM_BUMP, tests_passed=True)
            ]
        )
        assert "Test suite passed" in render_pull_request_body(finding, _advisory())

    def test_mentions_retry_count_when_the_loop_ran(self) -> None:
        finding = _finding(
            attempts=[
                PatchAttempt(attempt=1, strategy=PatchStrategy.BACKPORT, tests_passed=False),
                PatchAttempt(attempt=2, strategy=PatchStrategy.BACKPORT, tests_passed=True),
            ]
        )
        assert "2 attempts" in render_pull_request_body(finding, _advisory())

    def test_unknown_reachability_is_surfaced(self) -> None:
        finding = _finding(verdict=TriageVerdict(reachability=Reachability.UNKNOWN))
        body = render_pull_request_body(finding, _advisory())
        assert "could not be determined" in body

    def test_blast_radius_is_listed(self) -> None:
        finding = _finding(
            verdict=TriageVerdict(
                reachability=Reachability.REACHABLE, also_affects=["me/other", "me/third"]
            )
        )
        body = render_pull_request_body(finding, _advisory())
        assert "me/other" in body
        assert "me/third" in body

    def test_includes_audit_identifiers(self) -> None:
        body = render_pull_request_body(_finding(), _advisory())
        assert "run-123" in body
        assert "finding-abc" in body


class TestUsageAccounting:
    def test_gemma_is_free(self) -> None:
        """Routing the high-frequency work to Gemma is the cost decision the tiering exists
        for; it must actually cost nothing."""
        usage = Usage()
        cost = usage.add("gemma-4-26b-a4b-it", 1_000_000, 1_000_000)
        assert cost == 0.0
        assert usage.cost_usd == 0.0

    def test_gemini_cost_is_computed(self) -> None:
        usage = Usage()
        cost = usage.add("gemini-3.6-flash", 1_000_000, 0)
        assert cost == 1.50

    def test_costs_accumulate_across_calls(self) -> None:
        usage = Usage()
        usage.add("gemini-3.6-flash", 1_000_000, 0)
        usage.add("gemini-3.6-flash", 1_000_000, 0)
        assert usage.cost_usd == 3.0
        assert usage.calls_by_model["gemini-3.6-flash"] == 2

    def test_unknown_model_does_not_crash_accounting(self) -> None:
        usage = Usage()
        assert usage.add("some-future-model", 1000, 1000) == 0.0
