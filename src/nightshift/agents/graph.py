"""The finding graph - the ADK 2.x Workflow that processes one advisory.

Two levels of orchestration, and the split is deliberate. The run orchestrator
(``nightshift.run``) scans repositories, queries OSV once, and produces one
:class:`FindingContext` per advisory-repo pair; that is the fan-out, and it is dynamic
because the count depends on the night. This graph is a *static* topology handling exactly
one finding. ADK graphs are declared up front, so per-item work belongs inside and fan-out
belongs outside.

Topology::

    START -> guardian_screen -> triage -> triage_router
        triage_router  DISMISS  -> dismiss   (terminal)
                       ESCALATE -> escalate  (terminal)
                       PATCH    -> patch -> verify -> verify_router
        verify_router  RETRY    -> patch     (back-edge: the critique loop)
                       ESCALATE -> escalate  (terminal)
                       APPROVE  -> approval_router
        approval_router HUMAN   -> ask_human (suspends the graph)
                        AUTO    -> report

Three things verified against the installed google-adk 2.6.3 rather than the published
docs, because the two disagree:

* **Nodes bind parameters by name from workflow state.** The parameter carrying the
  previous node's output must be called ``node_input``; any other name raises
  "Missing value for parameter ... It was not found in state". Every node here therefore
  takes ``node_input`` and aliases it for readability.
* ``route`` is a field on ``EventActions``, not ``Event``. ``Event(route="KEY")`` works
  because a validator lifts it into ``actions.route``, but only the ``EventActions``
  spelling appears in the type stubs.
* A router must return ``Event(route=..., output=ctx)``. Returning a bare route passes
  ``None`` downstream and the next node loses its context.

**On the human gate.** Yielding ``RequestInput`` genuinely suspends the workflow: nodes
after it do not execute and the run ends with the finding parked. That is right for an
unattended 03:00 run, because nobody is awake to answer. The finding is marked
``AWAITING_APPROVAL`` before the yield, so the state is already correct if nothing ever
resumes it.

.. warning::
   Never wrap a node body in ``except Exception:``. ADK 2.x implements retry by catching
   exceptions itself, so swallowing them disables it. Never catch ``BaseException``
   anywhere in this module either: the human-in-the-loop pause is delivered as an
   interrupt, and trapping it breaks the approval gate with no error.
"""

from __future__ import annotations

import structlog
from google.adk import Event, Workflow
from google.adk.events import RequestInput
from pydantic import BaseModel, Field

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
from nightshift.config import Settings
from nightshift.models import Advisory, Finding, FindingStatus

log = structlog.get_logger(__name__)


class FindingContext(BaseModel):
    """Everything one finding needs, carried node to node.

    Passed as the graph's payload rather than held in module state, so findings processed
    concurrently cannot interfere with one another.
    """

    finding: Finding
    advisory: Advisory
    sources: dict[str, str] = Field(default_factory=dict)
    manifests: dict[str, str] = Field(default_factory=dict)
    default_branch: str = "main"
    repo_has_tests: bool = False
    test_command: str | None = None
    sibling_packages: dict[str, set[str]] = Field(default_factory=dict)
    guardian_flagged: bool = False
    proposed_version: str | None = None


def payload_text(node_input) -> str:
    """Extract the entry payload as plain text.

    ADK wraps a runner's argument in ``UserContent(parts=[Part(text=...)])`` before it
    reaches the first node, so the entry node sees a Content object rather than the string
    that was passed. Both shapes are accepted, because the graph is also invoked with a
    plain string from tests and from the dev UI.
    """
    if isinstance(node_input, str):
        return node_input.strip()

    parts = getattr(node_input, "parts", None)
    if parts:
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                return str(text).strip()

    return str(node_input).strip()


def build_finding_workflow(
    *,
    guardian,
    triager,
    patcher,
    verifier,
    reporter,
    settings: Settings,
    context_lookup,
    on_decision=None,
    on_complete=None,
    reserve_pr_slot=None,
) -> Workflow:
    """Assemble the finding graph with its agents bound.

    Agents are injected rather than imported so the topology can be exercised against
    fakes, and so a language-specialist patcher can later be swapped in as a
    ``RemoteA2aAgent`` without the graph changing shape.

    ``context_lookup`` maps a finding id to its :class:`FindingContext`. The graph is
    invoked with the id rather than the context itself because ADK's entry points accept
    only text. Passing an id keeps the payload small and leaves the bulky repository
    snapshot, which can be hundreds of source files, out of serialization entirely.

    ``on_decision`` is an optional async callback invoked at each hop; the orchestrator
    uses it to append to the audit trail the pull request body renders.

    ``on_complete`` is called by every terminal node with the finished finding. This is how
    the caller learns the outcome, and it is a callback rather than a return value on
    purpose: **ADK validates the context afresh at each node**, so every node receives a
    copy. Mutations never reach the object the caller passed in, and the workflow's final
    event carries a serialized dict rather than the model. Reading the result out of event
    outputs would mean depending on that serialization behaviour; an explicit callback does
    not.
    """

    async def _record(ctx: FindingContext, agent: str, action: str) -> None:
        if on_decision is not None:
            await on_decision(ctx.finding, agent, action)

    async def _complete(ctx: FindingContext) -> FindingContext:
        """Hand the finished finding back to the caller. Every terminal node ends here."""
        if on_complete is not None:
            await on_complete(ctx.finding)
        return ctx

    # --- nodes --------------------------------------------------------------

    async def guardian_screen(node_input) -> FindingContext:
        """Resolve the context, then screen attacker-controlled advisory text.

        First in the graph for that reason, not as a formality: everything downstream feeds
        this text to a model that can write code and open pull requests.
        """
        ctx = context_lookup(payload_text(node_input))

        result = guardian.screen_advisory(ctx.advisory)
        ctx.guardian_flagged = not result.safe
        if ctx.guardian_flagged:
            ctx.finding.escalation_reason = (
                "Guardian flagged untrusted content in this advisory"
            )
            await _record(ctx, "guardian", f"flagged: {result.reason[:120]}")
        return ctx

    async def triage(node_input: FindingContext) -> FindingContext:
        """The 180-to-6 step. Static analysis decides; the model only explains."""
        ctx = node_input
        if ctx.guardian_flagged:
            return ctx

        verdict = triager.triage(
            ctx.advisory,
            ctx.finding.dependency,
            ctx.sources,
            sibling_repos=ctx.sibling_packages,
        )
        ctx.finding.verdict = verdict
        ctx.finding.status = FindingStatus.TRIAGED

        affected = ctx.advisory.affects(
            ctx.finding.dependency.name, ctx.finding.dependency.ecosystem
        )
        ctx.proposed_version = affected.first_fixed_version() if affected else None

        await _record(
            ctx, "triager", f"{verdict.reachability} ({len(verdict.call_path)} call sites)"
        )
        return ctx

    def triage_router(node_input: FindingContext) -> Event:
        ctx = node_input
        route = ESCALATE if ctx.guardian_flagged else routing.triage_route(ctx.finding)
        log.info("route.triage", finding_id=ctx.finding.id, route=route)
        return Event(author="triage_router", route=route, output=ctx)

    async def patch(node_input: FindingContext) -> FindingContext:
        ctx = node_input
        manifest = ctx.manifests.get(ctx.finding.dependency.manifest_path, "")
        ctx.finding.attempts.append(patcher.patch(ctx.finding, ctx.advisory, manifest))
        return ctx

    async def verify(node_input: FindingContext) -> FindingContext:
        ctx = node_input
        attempt = ctx.finding.attempts[-1]
        result = verifier.verify(ctx.sources, attempt.diff, ctx.test_command)
        attempt.tests_passed = result.passed
        attempt.test_output = result.output
        if result.skipped_reason:
            attempt.error = result.skipped_reason

        await _record(
            ctx,
            "patcher/verifier",
            f"attempt {attempt.attempt} ({attempt.strategy}) "
            f"{'passed' if attempt.tests_passed else 'failed'}",
        )
        return ctx

    def verify_router(node_input: FindingContext) -> Event:
        ctx = node_input
        route = routing.verify_route(ctx.finding, settings.max_patch_attempts)
        log.info(
            "route.verify",
            finding_id=ctx.finding.id,
            route=route,
            attempts=len(ctx.finding.attempts),
        )
        return Event(author="verify_router", route=route, output=ctx)

    def approval_router(node_input: FindingContext) -> Event:
        """Consult the policy layer, which sits outside the model's reach."""
        from nightshift.policy import requires_human_approval

        ctx = node_input
        reason = requires_human_approval(
            ctx.finding,
            proposed_version=ctx.proposed_version,
            repo_has_tests=ctx.repo_has_tests,
            guardian_flagged=ctx.guardian_flagged,
        )
        ctx.finding.escalation_reason = reason
        route = routing.approval_route(reason)
        log.info("route.approval", finding_id=ctx.finding.id, route=route, reason=reason)
        return Event(author="approval_router", route=route, output=ctx)

    # --- terminals ----------------------------------------------------------

    async def dismiss(node_input: FindingContext) -> FindingContext:
        """Not reachable here. The common case, and the whole product's value."""
        ctx = node_input
        ctx.finding.status = FindingStatus.DISMISSED
        await _record(ctx, "triager", "dismissed: not reachable")
        return await _complete(ctx)

    async def escalate(node_input: FindingContext) -> FindingContext:
        ctx = node_input
        ctx.finding.status = FindingStatus.ESCALATED
        ctx.finding.escalation_reason = ctx.finding.escalation_reason or (
            f"No verified patch after {len(ctx.finding.attempts)} attempt(s)"
        )
        await _record(ctx, "policy", f"escalated: {ctx.finding.escalation_reason}")
        return await _complete(ctx)

    def ask_human(node_input: FindingContext):
        """Suspend the graph until a person approves.

        The status is set *before* yielding, because an unattended run will never resume:
        the finding must already read as awaiting approval when the run ends.
        """
        ctx = node_input
        ctx.finding.status = FindingStatus.AWAITING_APPROVAL
        latest = ctx.finding.attempts[-1] if ctx.finding.attempts else None
        diff = latest.diff if latest else "(no diff generated)"

        log.info(
            "hitl.suspended",
            finding_id=ctx.finding.id,
            reason=ctx.finding.escalation_reason,
        )

        yield RequestInput(
            message=(
                f"Approval required for {ctx.finding.repo} - {ctx.finding.advisory_id}\n"
                f"Reason: {ctx.finding.escalation_reason}\n\n"
                f"{diff}\n\n"
                f"Reply 'approve' to open the pull request, anything else to skip."
            ),
            payload=ctx.finding.model_dump(mode="json"),
        )

    async def report(node_input: FindingContext) -> FindingContext:
        ctx = node_input

        if reserve_pr_slot is not None and not reserve_pr_slot():
            # A hard ceiling, so a bug cannot flood a repository however confident the
            # fleet is. This *reserves* a slot rather than reading a counter: findings are
            # processed concurrently, and a check that only reads would let every one of
            # them pass the gate before any of them had opened anything.
            ctx.finding.status = FindingStatus.ESCALATED
            ctx.finding.escalation_reason = (
                f"Per-run pull request limit ({settings.max_prs_per_run}) reached"
            )
            await _record(ctx, "policy", "per-run pull request limit reached")
            return await _complete(ctx)

        patched = materialize_patch(ctx)
        if not patched:
            ctx.finding.status = FindingStatus.ESCALATED
            ctx.finding.escalation_reason = (
                "Patch produced no file change that could be committed"
            )
            await _record(ctx, "reporter", "no committable change; escalated")
            return await _complete(ctx)

        ctx.finding.status = FindingStatus.VERIFIED
        ctx.finding = await reporter.report(
            ctx.finding, ctx.advisory, patched, ctx.default_branch
        )
        return await _complete(ctx)

    return Workflow(
        name="nightshift_finding",
        edges=[
            ("START", guardian_screen, triage, triage_router),
            (triage_router, {DISMISS: dismiss, ESCALATE: escalate, PATCH: patch}),
            (patch, verify, verify_router),
            (
                verify_router,
                {
                    RETRY: patch,  # back-edge: the critique loop
                    ESCALATE: escalate,
                    APPROVE: approval_router,
                },
            ),
            (approval_router, {HUMAN: ask_human, AUTO: report}),
        ],
    )


def materialize_patch(ctx: FindingContext) -> dict[str, str]:
    """Produce the files to commit, containing the actual change.

    Only manifests whose content genuinely differs are returned. An empty result means the
    patch could not be turned into a committable file, which routes to a human rather than
    opening a pull request with an empty diff.
    """
    if not ctx.proposed_version:
        return {}

    from nightshift.agents.patcher import bump_manifest

    patched: dict[str, str] = {}
    for path, content in ctx.manifests.items():
        updated, changed = bump_manifest(
            content, ctx.finding.dependency.name, ctx.proposed_version
        )
        if changed:
            patched[path] = updated
    return patched
