"""Routing decisions for the finding graph.

These are deliberately pure functions, separate from the ADK wiring in ``graph.py``. The
routing logic is where the fleet decides whether to spend money on a model, whether to
retry, and whether a human must look - so it needs to be testable without a graph engine,
a network, or a Google Cloud project.

Every route key here appears verbatim in ``graph.py``'s edge map. Keeping them as module
constants means a typo is an ImportError rather than a node that silently never runs.
"""

from __future__ import annotations

from nightshift.models import Finding, Reachability

# Route keys - shared with graph.py's edge dictionaries.
DISMISS = "DISMISS"
PATCH = "PATCH"
RETRY = "RETRY"
ESCALATE = "ESCALATE"
APPROVE = "APPROVE"
HUMAN = "HUMAN"
AUTO = "AUTO"


def triage_route(finding: Finding) -> str:
    """Decide whether an advisory is worth patching at all.

    This is the single highest-leverage decision in the system: it is what turns 180
    advisories into 6. Everything downstream costs model tokens and, eventually, a human's
    attention, so anything dismissed here is saved twice over.

    ``UNKNOWN`` routes to PATCH rather than DISMISS. When the analysis could not reach a
    conclusion, the safe failure is a patch nobody needed, not a vulnerability nobody
    noticed - and the approval gate downstream will hand it to a human anyway.
    """
    if finding.verdict is None:
        return PATCH
    if finding.verdict.reachability is Reachability.NOT_REACHABLE:
        return DISMISS
    return PATCH


def verify_route(finding: Finding, max_attempts: int) -> str:
    """Decide what happens after the Verifier runs the test suite.

    Implements the critique loop's exit conditions. The bound matters: without it a model
    that cannot solve a problem will keep being asked to solve it, and the run's cost grows
    without limit while nothing improves.
    """
    if not finding.attempts:
        return ESCALATE

    latest = finding.attempts[-1]
    if latest.tests_passed:
        return APPROVE
    if len(finding.attempts) >= max_attempts:
        return ESCALATE
    return RETRY


def approval_route(escalation_reason: str | None) -> str:
    """Route to a human when the action is irreversible or unverifiable.

    ``escalation_reason`` comes from ``policy.requires_human_approval``. The decision lives
    in the policy module rather than here so that it cannot be reasoned around by a model.
    """
    return HUMAN if escalation_reason else AUTO
