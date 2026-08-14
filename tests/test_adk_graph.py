"""The ADK graph itself.

The integration suite exercises the graph indirectly, by running the whole pipeline. These
tests assert the thing directly: that the topology is a real ``google.adk.Workflow``, that
routers emit routes ADK can dispatch on, and that a finding driven through an actual
``Runner`` reaches a terminal node.

That matters beyond tidiness. The hackathon requires a Google agent framework to be in the
execution path, not merely in the dependency list, and "we import ADK somewhere" is not the
same claim as "every finding is processed by an ADK graph".
"""

from __future__ import annotations

import pytest
from google.adk import Event, Runner, Workflow
from google.adk.sessions import InMemorySessionService

from nightshift.agents.graph import (
    FindingContext,
    build_finding_workflow,
    materialize_patch,
    payload_text,
)
from nightshift.agents.guardian import Guardian
from nightshift.config import Settings
from nightshift.models import (
    Advisory,
    AffectedPackage,
    Dependency,
    Ecosystem,
    Finding,
    FindingStatus,
    VersionRange,
)


def _settings(**overrides: object) -> Settings:
    base = {
        "NIGHTSHIFT_REPO_ALLOWLIST": "me/myrepo",
        "NIGHTSHIFT_DRY_RUN": True,
        "NIGHTSHIFT_MAX_PATCH_ATTEMPTS": 3,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _advisory(fixed: str | None = "2.31.0") -> Advisory:
    ranges = [VersionRange(introduced="2.3.0", fixed=fixed)] if fixed else [
        VersionRange(introduced="2.3.0")
    ]
    return Advisory(
        id="GHSA-test",
        summary="Header leak",
        details="Upgrade.",
        affected=[
            AffectedPackage(name="requests", ecosystem=Ecosystem.PYPI, ranges=ranges)
        ],
    )


def _context(**overrides: object) -> FindingContext:
    base: dict = {
        "finding": Finding(
            id="find1",
            run_id="run1",
            advisory_id="GHSA-test",
            repo="me/myrepo",
            dependency=Dependency(
                name="requests",
                ecosystem=Ecosystem.PYPI,
                version="2.19.1",
                manifest_path="requirements.txt",
            ),
        ),
        "advisory": _advisory(),
        "sources": {
            "src/api.py": "import requests\n\ndef f(u):\n    return requests.get(u)\n",
            "tests/t.py": "def test_ok():\n    assert True\n",
        },
        "manifests": {"requirements.txt": "requests==2.19.1\n"},
        "repo_has_tests": True,
        "test_command": "pytest",
    }
    base.update(overrides)
    return FindingContext(**base)


class _Verifier:
    def __init__(self, passes: bool = True) -> None:
        self.passes = passes

    def verify(self, sources, diff, test_command):
        from nightshift.agents.verifier import VerificationResult

        return VerificationResult(passed=self.passes, output="ok" if self.passes else "fail")


class _Triager:
    """Returns a fixed verdict so the graph is tested, not the analysis."""

    def __init__(self, reachability) -> None:
        self.reachability = reachability

    def triage(self, advisory, dependency, sources, *, sibling_repos=None):
        from nightshift.models import CallSite, TriageVerdict

        return TriageVerdict(
            reachability=self.reachability,
            call_path=[CallSite(file_path="src/api.py", line=4, symbol="requests.get")],
            rationale="fixed verdict for graph testing",
        )


class _Patcher:
    def patch(self, finding, advisory, manifest_content):
        from nightshift.agents.patcher import Patcher
        from nightshift.llm import LLMClient

        return Patcher(LLMClient(client=object(), settings=_settings()))._upstream_bump(
            finding, manifest_content, "2.31.0", len(finding.attempts) + 1
        )


class _Reporter:
    def __init__(self) -> None:
        self.reported: list[Finding] = []

    async def report(self, finding, advisory, patched, default_branch):
        finding.status = FindingStatus.PR_OPENED
        finding.pr_url = "https://github.com/me/myrepo/pull/1"
        self.reported.append(finding)
        return finding


def _build(
    reachability,
    *,
    verifier_passes: bool = True,
    on_complete=None,
    settings=None,
    memory=None,
):
    reporter = _Reporter()
    workflow = build_finding_workflow(
        guardian=Guardian(),
        triager=_Triager(reachability),
        patcher=_Patcher(),
        verifier=_Verifier(verifier_passes),
        reporter=reporter,
        settings=settings or _settings(),
        context_lookup={"find1": _context()}.__getitem__,
        memory=memory,
        on_complete=on_complete,
    )
    return workflow, reporter


async def _drive(workflow: Workflow) -> list[Finding]:
    """Run one finding through a real ADK Runner and collect terminal findings."""
    runner = Runner(
        app_name="test",
        agent=workflow,
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )
    await runner.run_debug("find1", quiet=True)
    return []


class TestTopology:
    def test_builds_a_real_adk_workflow(self) -> None:
        from nightshift.models import Reachability

        workflow, _ = _build(Reachability.REACHABLE)
        assert isinstance(workflow, Workflow)
        assert workflow.name == "nightshift_finding"

    def test_declares_the_expected_edges(self) -> None:
        from nightshift.models import Reachability

        workflow, _ = _build(Reachability.REACHABLE)
        # entry chain, triage routes, patch chain, verify routes, approval routes
        assert len(workflow.edges) == 5


class TestExecution:
    async def test_reachable_finding_reaches_the_reporter(self) -> None:
        """A full pass through a real ADK Runner, ending in a pull request."""
        from nightshift.models import Reachability

        completed: list[Finding] = []

        async def on_complete(finding: Finding) -> None:
            completed.append(finding)

        workflow, reporter = _build(Reachability.REACHABLE, on_complete=on_complete)
        await _drive(workflow)

        assert reporter.reported, "the graph never reached the reporter"
        assert completed[-1].status is FindingStatus.PR_OPENED

    async def test_unreachable_finding_is_dismissed_by_the_graph(self) -> None:
        from nightshift.models import Reachability

        completed: list[Finding] = []

        async def on_complete(finding: Finding) -> None:
            completed.append(finding)

        workflow, reporter = _build(Reachability.NOT_REACHABLE, on_complete=on_complete)
        await _drive(workflow)

        assert reporter.reported == []
        assert completed[-1].status is FindingStatus.DISMISSED

    async def test_failing_verification_escalates_through_the_graph(self) -> None:
        """The critique loop runs to its bound inside ADK, then routes to escalate."""
        from nightshift.models import Reachability

        completed: list[Finding] = []

        async def on_complete(finding: Finding) -> None:
            completed.append(finding)

        workflow, reporter = _build(
            Reachability.REACHABLE, verifier_passes=False, on_complete=on_complete
        )
        await _drive(workflow)

        assert reporter.reported == []
        assert completed[-1].status is FindingStatus.ESCALATED
        assert len(completed[-1].attempts) == 3  # max_patch_attempts

    async def test_pr_ceiling_blocks_the_write(self) -> None:
        from nightshift.models import Reachability

        completed: list[Finding] = []

        async def on_complete(finding: Finding) -> None:
            completed.append(finding)

        reporter = _Reporter()
        workflow = build_finding_workflow(
            guardian=Guardian(),
            triager=_Triager(Reachability.REACHABLE),
            patcher=_Patcher(),
            verifier=_Verifier(True),
            reporter=reporter,
            settings=_settings(),
            context_lookup={"find1": _context()}.__getitem__,
            on_complete=on_complete,
            reserve_pr_slot=lambda: False,
        )
        await _drive(workflow)

        assert reporter.reported == []
        assert "limit" in (completed[-1].escalation_reason or "")


class TestMemoryChangesBehaviour:
    """Memory is only worth having if it alters what the fleet does."""

    async def test_a_previously_declined_change_is_not_proposed_again(self) -> None:
        from nightshift.memory import PatchMemory
        from nightshift.models import Reachability

        class _Remembers:
            """Stands in for memory that already holds a refusal for this change."""

            async def remember(self, finding, proposed_version=None) -> None:
                return None

            async def previously_declined(self, repo, package, proposed_version):
                return "This same change was already escalated and has not been approved"

        completed: list[Finding] = []

        async def on_complete(finding: Finding) -> None:
            completed.append(finding)

        workflow, reporter = _build(
            Reachability.REACHABLE, on_complete=on_complete, memory=_Remembers()
        )
        await _drive(workflow)

        assert reporter.reported == [], "a declined change was re-proposed"
        assert completed[-1].status is FindingStatus.ESCALATED
        assert "already escalated" in (completed[-1].escalation_reason or "")
        assert isinstance(PatchMemory, type)

    async def test_without_a_prior_refusal_the_patch_proceeds(self) -> None:
        """The suppression must be specific, or memory silently blocks real fixes."""
        from nightshift.models import Reachability

        class _RemembersNothing:
            async def remember(self, finding, proposed_version=None) -> None:
                return None

            async def previously_declined(self, repo, package, proposed_version):
                return None

        workflow, reporter = _build(Reachability.REACHABLE, memory=_RemembersNothing())
        await _drive(workflow)

        assert reporter.reported, "memory blocked a change that was never declined"


class TestRouterEvents:
    def test_event_route_lands_on_actions(self) -> None:
        """ADK stores the route on EventActions; Event(route=...) is lifted into it.

        The published docs show only the Event spelling, and the type stubs show only the
        EventActions one. Both work, and this pins that down.
        """
        assert Event(author="x", route="KEY").actions.route == "KEY"


class TestPayloadText:
    def test_plain_string(self) -> None:
        assert payload_text("  find1 ") == "find1"

    def test_unwraps_adk_content(self) -> None:
        """ADK wraps the runner argument in UserContent before the first node sees it."""
        from google.genai import types

        content = types.UserContent(parts=[types.Part(text="find1")])
        assert payload_text(content) == "find1"


class TestMaterializePatch:
    def test_returns_only_changed_manifests(self) -> None:
        ctx = _context()
        ctx.proposed_version = "2.31.0"
        ctx.manifests = {
            "requirements.txt": "requests==2.19.1\n",
            "other.txt": "flask==3.0.0\n",
        }
        patched = materialize_patch(ctx)
        assert list(patched) == ["requirements.txt"]
        assert "requests==2.31.0" in patched["requirements.txt"]

    def test_no_proposed_version_yields_nothing(self) -> None:
        ctx = _context()
        ctx.proposed_version = None
        assert materialize_patch(ctx) == {}


@pytest.mark.parametrize("status", [FindingStatus.AWAITING_APPROVAL])
def test_awaiting_approval_is_distinct_from_escalated(status: FindingStatus) -> None:
    """A suspended approval gate is not the same as the fleet giving up."""
    assert status is not FindingStatus.ESCALATED
