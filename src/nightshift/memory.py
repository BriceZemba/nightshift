"""Per-repository memory of past patch decisions.

Without memory the fleet is amnesiac in a way that is actively annoying: it re-proposes
every night the exact bump a maintainer declined last week, and the pull requests start
getting ignored wholesale. Remembering a refusal is the difference between an assistant and
a nuisance.

Backed by ADK's ``BaseMemoryService``, so the same code runs against
``InMemoryMemoryService`` offline and GEAP ``VertexAiMemoryBankService`` when deployed.
Memory is scoped per repository: ADK's ``user_id`` carries the repo name, because what was
declined in one project says nothing about another.
"""

from __future__ import annotations

from typing import Any

import structlog

from nightshift.models import Finding, FindingStatus

log = structlog.get_logger(__name__)

APP_NAME = "nightshift"

#: Outcomes worth remembering. A dismissal is not recorded: "not reachable" is recomputed
#: cheaply from source every run, and caching it would let a stale verdict outlive the code
#: change that invalidated it.
REMEMBERED = frozenset(
    {
        FindingStatus.PR_OPENED,
        FindingStatus.ESCALATED,
        FindingStatus.AWAITING_APPROVAL,
    }
)


#: Statuses that mean "the maintainer has not accepted this".
REFUSALS = frozenset({FindingStatus.ESCALATED, FindingStatus.AWAITING_APPROVAL})


def marker(package: str, proposed_version: str, status: FindingStatus) -> str:
    """A machine-readable tag embedded in the remembered text.

    Recall matches on this rather than on ``custom_metadata``, because metadata does not
    round-trip identically through every ``BaseMemoryService`` implementation while the
    text always does. It also keeps matching exact: a semantic search will happily return
    a *similar* decision, and suppressing a patch on the strength of a near-match is worse
    than not remembering at all.
    """
    return f"[nightshift package={package} to={proposed_version} status={status}]"


def _fact(finding: Finding, proposed_version: str | None) -> str:
    """One line of durable memory, readable by a human and parseable by us."""
    target = f" -> {proposed_version}" if proposed_version else ""
    outcome = {
        FindingStatus.PR_OPENED: "opened a pull request",
        FindingStatus.ESCALATED: "escalated to a human without opening a pull request",
        FindingStatus.AWAITING_APPROVAL: "paused awaiting human approval",
    }.get(finding.status, str(finding.status))

    reason = f" Reason: {finding.escalation_reason}." if finding.escalation_reason else ""
    tag = (
        " " + marker(finding.dependency.name, proposed_version, finding.status)
        if proposed_version
        else ""
    )

    return (
        f"For {finding.repo}, advisory {finding.advisory_id} affecting "
        f"{finding.dependency.name} {finding.dependency.version}{target}: "
        f"{outcome}.{reason}{tag}"
    )


class PatchMemory:
    """Records and recalls what the fleet already decided about a package.

    ``service`` is injected so the pipeline is testable, and runs offline, without a
    Google Cloud project.
    """

    def __init__(self, service: Any) -> None:
        self._service = service

    async def remember(self, finding: Finding, proposed_version: str | None = None) -> None:
        """Record a terminal decision.

        Failures are logged and swallowed here, deliberately and unusually for this
        codebase: memory is an enhancement, and losing a memory write must never fail a run
        that has already done its real work.
        """
        if finding.status not in REMEMBERED:
            return

        from google.adk import Event
        from google.genai import types

        # add_events_to_memory rather than add_memory: InMemoryMemoryService rejects direct
        # memory writes, so add_memory would work against Memory Bank and silently break
        # the offline path. Events are supported by every backend.
        event = Event(
            author="nightshift",
            content=types.Content(
                role="user", parts=[types.Part(text=_fact(finding, proposed_version))]
            ),
        )

        try:
            await self._service.add_events_to_memory(
                app_name=APP_NAME,
                user_id=finding.repo,
                events=[event],
                custom_metadata={
                    "repo": finding.repo,
                    "package": finding.dependency.name,
                    "to_version": proposed_version or "",
                    "advisory": finding.advisory_id,
                    "status": str(finding.status),
                },
            )
        except Exception as exc:  # noqa: BLE001 - see docstring
            log.warning("memory.write_failed", finding_id=finding.id, error=str(exc))
            return

        log.info("memory.recorded", finding_id=finding.id, status=str(finding.status))

    async def previously_declined(
        self, repo: str, package: str, proposed_version: str | None
    ) -> str | None:
        """Return the prior refusal for this exact change, or ``None``.

        Matching is on metadata rather than on the searched text, because a semantic search
        will happily return a *similar* decision and re-proposing a bump on the strength of
        a near-match is worse than not remembering at all.
        """
        if not proposed_version:
            return None

        try:
            response = await self._service.search_memory(
                app_name=APP_NAME, user_id=repo, query=f"{package} {proposed_version}"
            )
        except Exception as exc:  # noqa: BLE001 - memory must not fail the run
            log.warning("memory.search_failed", repo=repo, error=str(exc))
            return None

        wanted = {marker(package, proposed_version, status) for status in REFUSALS}

        for memory in getattr(response, "memories", []) or []:
            text = _text_of(memory)
            if any(tag in text for tag in wanted):
                when = getattr(memory, "timestamp", None)
                suffix = f" on {when}" if when else ""
                return (
                    f"This same change was already escalated{suffix} and has not been "
                    f"approved"
                )

        return None


def _text_of(memory: Any) -> str:
    """Flatten a recalled memory's content back to text."""
    content = getattr(memory, "content", None)
    parts = getattr(content, "parts", None) or []
    return " ".join(str(getattr(part, "text", "") or "") for part in parts)


def build_memory(settings) -> PatchMemory:
    """Build memory from settings.

    Falls back to the in-memory service rather than to nothing, so the recall path is
    always exercised and a missing configuration cannot silently change behaviour.
    """
    if settings.memory_bank_agent_engine_id and settings.google_cloud_project:
        from google.adk.memory import VertexAiMemoryBankService

        log.info("memory.backend", backend="vertex_memory_bank")
        return PatchMemory(
            VertexAiMemoryBankService(
                project=settings.google_cloud_project,
                location=settings.google_cloud_location,
                agent_engine_id=settings.memory_bank_agent_engine_id,
            )
        )

    from google.adk.memory import InMemoryMemoryService

    log.info("memory.backend", backend="in_memory")
    return PatchMemory(InMemoryMemoryService())
