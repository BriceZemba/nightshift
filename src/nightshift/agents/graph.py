"""The finding graph - ADK 2.x Workflow wiring.

Two levels of orchestration, and the split is deliberate:

* The **run orchestrator** (``nightshift.run``) scans allowlisted repos, queries OSV once
  for every dependency, creates one ``Finding`` per advisory-repo pair, and publishes them
  to Pub/Sub. That is the fan-out, and it is dynamic - the count depends on the night.
* This **finding graph** processes exactly one ``Finding``. An ADK graph is a static
  topology, so per-item work belongs inside it and fan-out belongs outside it. Pub/Sub
  gives redelivery and dead-lettering for free; the graph gives routing and interrupts.

Topology::

    START -> guardian_screen -> triager -> triage_router
        triage_router  DISMISS  -> dismiss (terminal)
                       PATCH    -> patcher -> verifier -> verify_router
        verify_router  RETRY    -> patcher (back-edge: the critique loop)
                       ESCALATE -> escalate (terminal)
                       APPROVE  -> approval_router
        approval_router HUMAN   -> ask_human -> apply_human_decision -> reporter
                        AUTO    -> reporter

API shape verified against https://adk.dev/graphs/ on 2026-08-13. See
docs/08-TECH-REFERENCE.md - ADK 2.x is a graph engine and the Python router returns
``Event(route="KEY")`` (singular string; the plural ``Routes`` form is the Go API).

.. warning::
   Do not wrap node bodies in ``except Exception:``. ADK 2.x implements tool retry by
   catching exceptions itself, so swallowing them disables it. Never catch
   ``BaseException`` anywhere in this module: the human-in-the-loop pause is delivered as
   an interrupt, and trapping it breaks the approval gate silently.
"""

from __future__ import annotations

import structlog
from google.adk import Event, Workflow
from google.adk.events import RequestInput

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
from nightshift.config import Settings, get_settings
from nightshift.models import Finding, FindingStatus

log = structlog.get_logger(__name__)


# --- routers ---------------------------------------------------------------
# Routers are thin on purpose: they translate a decision made in `routing` or `policy`
# into an ADK route key, and contain no policy of their own.


def triage_router(finding: Finding) -> Event:
    route = routing.triage_route(finding)
    log.info(
        "route.triage",
        finding_id=finding.id,
        route=route,
        reachability=finding.verdict.reachability if finding.verdict else None,
    )
    return Event(route=route)


def verify_router(finding: Finding, settings: Settings | None = None) -> Event:
    settings = settings or get_settings()
    route = routing.verify_route(finding, settings.max_patch_attempts)
    log.info(
        "route.verify", finding_id=finding.id, route=route, attempts=len(finding.attempts)
    )
    return Event(route=route)


def approval_router(finding: Finding) -> Event:
    route = routing.approval_route(finding.escalation_reason)
    log.info("route.approval", finding_id=finding.id, route=route)
    return Event(route=route)


# --- terminal nodes --------------------------------------------------------


def dismiss(finding: Finding) -> Finding:
    """Not reachable from this repo's entrypoints. This is the common case, and saying so
    explicitly is most of the product's value."""
    finding.status = FindingStatus.DISMISSED
    return finding


def escalate(finding: Finding) -> Finding:
    """Hand to a human without opening a pull request."""
    finding.status = FindingStatus.ESCALATED
    finding.escalation_reason = finding.escalation_reason or (
        f"Could not produce a verified patch after {len(finding.attempts)} attempts"
    )
    log.info("finding.escalated", finding_id=finding.id, reason=finding.escalation_reason)
    return finding


# --- human-in-the-loop -----------------------------------------------------


def ask_human(finding: Finding):
    """Pause the workflow until a person approves the change.

    Yielding ``RequestInput`` suspends the graph; execution resumes at the next node once
    a response arrives. Note the documented limitation: ``response_schema`` is not enforced
    by reformatting, so the caller must supply the response already in shape.
    """
    latest = finding.attempts[-1] if finding.attempts else None
    diff = latest.diff if latest else "(no diff generated)"

    yield RequestInput(
        message=(
            f"Approval required for {finding.repo} - {finding.advisory_id}\n"
            f"Reason: {finding.escalation_reason}\n\n"
            f"{diff}\n\n"
            f"Reply 'approve' to open the pull request, anything else to skip."
        ),
        payload=finding.model_dump(),
    )


def apply_human_decision(node_input: object, finding: Finding) -> Finding:
    """Interpret the human's answer.

    Fails closed: anything that is not an explicit approval skips the pull request. A
    garbled or empty response must never be read as consent to write to a repository.
    """
    approved = isinstance(node_input, str) and node_input.strip().lower() in {
        "approve",
        "approved",
        "yes",
        "y",
    }
    if not approved:
        finding.status = FindingStatus.ESCALATED
        log.info("human.declined", finding_id=finding.id)
    return finding


def build_workflow(
    *,
    guardian_screen,
    triager,
    patcher,
    verifier,
    reporter,
) -> Workflow:
    """Assemble the finding graph.

    Agents are injected rather than imported so the topology can be tested against fakes,
    and so a language-specialist patcher can later be swapped in as a ``RemoteA2aAgent``
    without touching the graph.
    """
    return Workflow(
        name="nightshift_finding",
        edges=[
            # Untrusted advisory text is screened before it reaches any model that can
            # write code. Guardian runs first for that reason, not as a formality.
            ("START", guardian_screen, triager, triage_router),
            (
                triage_router,
                {
                    DISMISS: dismiss,
                    PATCH: patcher,
                },
            ),
            (patcher, verifier, verify_router),
            (
                verify_router,
                {
                    RETRY: patcher,  # back-edge: the critique loop
                    ESCALATE: escalate,
                    APPROVE: approval_router,
                },
            ),
            (
                approval_router,
                {
                    HUMAN: ask_human,
                    AUTO: reporter,
                },
            ),
            (ask_human, apply_human_decision, reporter),
        ],
    )
