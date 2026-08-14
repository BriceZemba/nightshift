"""Per-repository memory.

The behaviour that matters: a change the maintainer already refused is not proposed again
tomorrow night. Everything else here protects that from being wrong in the dangerous
direction, which is suppressing a patch that was never actually declined.
"""

from __future__ import annotations

import pytest

from nightshift.config import Settings
from nightshift.memory import PatchMemory, build_memory
from nightshift.models import Dependency, Ecosystem, Finding, FindingStatus


def _finding(**overrides: object) -> Finding:
    base = {
        "id": "f1",
        "run_id": "r1",
        "advisory_id": "GHSA-x",
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


def _memory() -> PatchMemory:
    from google.adk.memory import InMemoryMemoryService

    return PatchMemory(InMemoryMemoryService())


class TestRemember:
    async def test_escalation_is_recorded_and_recalled(self) -> None:
        memory = _memory()
        finding = _finding(
            status=FindingStatus.ESCALATED,
            escalation_reason="Major version bump may break things",
        )
        await memory.remember(finding, "3.0.0")

        assert await memory.previously_declined("me/myrepo", "requests", "3.0.0")

    async def test_dismissals_are_not_remembered(self) -> None:
        """Reachability is recomputed cheaply every run. Caching it would let a stale
        verdict outlive the code change that invalidated it."""
        memory = _memory()
        await memory.remember(_finding(status=FindingStatus.DISMISSED), "3.0.0")

        assert await memory.previously_declined("me/myrepo", "requests", "3.0.0") is None

    async def test_awaiting_approval_is_remembered(self) -> None:
        memory = _memory()
        await memory.remember(_finding(status=FindingStatus.AWAITING_APPROVAL), "3.0.0")

        assert await memory.previously_declined("me/myrepo", "requests", "3.0.0")

    async def test_an_opened_pull_request_does_not_suppress_future_work(self) -> None:
        """Only refusals suppress. A merged fix must not block a later advisory."""
        memory = _memory()
        await memory.remember(_finding(status=FindingStatus.PR_OPENED), "2.31.0")

        assert await memory.previously_declined("me/myrepo", "requests", "2.31.0") is None


class TestRecallPrecision:
    async def test_memory_is_scoped_per_repository(self) -> None:
        """A refusal in one project says nothing about another."""
        memory = _memory()
        await memory.remember(
            _finding(repo="me/repo-a", status=FindingStatus.ESCALATED), "3.0.0"
        )

        assert await memory.previously_declined("me/repo-b", "requests", "3.0.0") is None

    async def test_a_different_target_version_is_not_suppressed(self) -> None:
        """Declining 3.0.0 says nothing about 2.31.0, and suppressing it would hide a fix
        the maintainer would have accepted."""
        memory = _memory()
        await memory.remember(_finding(status=FindingStatus.ESCALATED), "3.0.0")

        assert await memory.previously_declined("me/myrepo", "requests", "2.31.0") is None

    async def test_a_different_package_is_not_suppressed(self) -> None:
        memory = _memory()
        await memory.remember(_finding(status=FindingStatus.ESCALATED), "3.0.0")

        assert await memory.previously_declined("me/myrepo", "flask", "3.0.0") is None

    async def test_no_proposed_version_recalls_nothing(self) -> None:
        memory = _memory()
        await memory.remember(_finding(status=FindingStatus.ESCALATED), "3.0.0")

        assert await memory.previously_declined("me/myrepo", "requests", None) is None


class TestResilience:
    """Memory is an enhancement. Losing it must never fail a run that did real work."""

    async def test_write_failure_is_swallowed(self) -> None:
        class _Failing:
            async def add_memory(self, **kwargs: object) -> None:
                raise RuntimeError("Memory Bank unavailable")

        await PatchMemory(_Failing()).remember(
            _finding(status=FindingStatus.ESCALATED), "3.0.0"
        )

    async def test_search_failure_returns_none_rather_than_blocking(self) -> None:
        """Failing open here is deliberate and the opposite of Guardian: an unreachable
        memory should not stop a legitimate patch, it should just forget."""

        class _Failing:
            async def search_memory(self, **kwargs: object) -> None:
                raise RuntimeError("Memory Bank unavailable")

        result = await PatchMemory(_Failing()).previously_declined(
            "me/myrepo", "requests", "3.0.0"
        )
        assert result is None


class TestBackendSelection:
    def test_defaults_to_in_memory(self) -> None:
        assert isinstance(build_memory(Settings()), PatchMemory)

    def test_engine_id_without_project_falls_back(self) -> None:
        settings = Settings(MEMORY_BANK_AGENT_ENGINE_ID="123")
        assert isinstance(build_memory(settings), PatchMemory)


@pytest.mark.parametrize(
    "status", [FindingStatus.ESCALATED, FindingStatus.AWAITING_APPROVAL]
)
async def test_every_refusal_status_suppresses(status: FindingStatus) -> None:
    memory = _memory()
    await memory.remember(_finding(status=status), "3.0.0")
    assert await memory.previously_declined("me/myrepo", "requests", "3.0.0")
