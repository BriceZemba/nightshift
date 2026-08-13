"""Reporter - writes the pull request, or hands the finding to a human.

The body is the product. A pull request that says "upgrade this" gets closed unread; one
that says "you call this function on line 40 of your request handler, here is the fix, the
tests pass, it cost three cents to work out" gets merged. Everything the fleet learned is
rendered here so the maintainer can check the reasoning instead of redoing it.

Rendering is a pure function, deliberately, so the exact text a maintainer will see is
testable without a network or a repository.
"""

from __future__ import annotations

import structlog

from nightshift.models import (
    Advisory,
    Decision,
    Finding,
    FindingStatus,
    PatchStrategy,
    Reachability,
)

log = structlog.get_logger(__name__)

_STRATEGY_LABEL = {
    PatchStrategy.UPSTREAM_BUMP: "Upstream version bump",
    PatchStrategy.BACKPORT: "Synthesized backport (no upstream fix exists)",
    PatchStrategy.NO_FIX_AVAILABLE: "No fix available",
}


def branch_name(finding: Finding) -> str:
    """Deterministic branch name.

    Derived from the finding's idempotency key, so a re-run targets the same branch rather
    than accumulating near-duplicates in the repository.
    """
    package = finding.dependency.name.replace("/", "-").replace("@", "")
    return f"nightshift/{package}-{finding.idempotency_key()[:12]}"


def pull_request_title(finding: Finding, advisory: Advisory) -> str:
    return (
        f"Fix {advisory.id} in {finding.dependency.name} "
        f"{finding.dependency.version}"
    )


def render_pull_request_body(
    finding: Finding,
    advisory: Advisory,
    decisions: list[Decision] | None = None,
) -> str:
    """Render the full evidence trail.

    Ordered by what a maintainer needs first: why this one matters, what changed, whether
    it is proven, and only then the audit detail.
    """
    verdict = finding.verdict
    latest = finding.attempts[-1] if finding.attempts else None
    lines: list[str] = []

    cvss = advisory.cvss_score()
    severity = f" · CVSS {cvss}" if cvss is not None else ""
    lines.append(f"## {advisory.id}{severity}")
    lines.append("")
    if advisory.summary:
        lines.append(f"> {advisory.summary}")
        lines.append("")

    lines.append(
        f"**Package:** `{finding.dependency.name}` "
        f"`{finding.dependency.version}` "
        f"(`{finding.dependency.manifest_path}`)"
    )
    lines.append("")

    # --- why this one, out of all of them ---
    lines.append("### Why this advisory applies here")
    lines.append("")
    if verdict:
        lines.append(verdict.rationale or "(no rationale recorded)")
        lines.append("")

        if verdict.reachability is Reachability.REACHABLE and verdict.call_path:
            lines.append("**Reached from:**")
            lines.append("")
            for site in verdict.call_path[:10]:
                snippet = f" - `{site.snippet}`" if site.snippet else ""
                lines.append(f"- `{site.file_path}:{site.line}` → `{site.symbol}`{snippet}")
            if len(verdict.call_path) > 10:
                lines.append(f"- …and {len(verdict.call_path) - 10} more")
            lines.append("")
        elif verdict.reachability is Reachability.UNKNOWN:
            lines.append(
                "⚠️ Reachability could not be determined conclusively, so this was "
                "escalated for review rather than applied automatically."
            )
            lines.append("")

        if verdict.also_affects:
            lines.append(
                f"**Also present in:** {', '.join(f'`{r}`' for r in verdict.also_affects)}"
            )
            lines.append("")

    # --- what changed ---
    lines.append("### The change")
    lines.append("")
    if latest:
        lines.append(f"**Strategy:** {_STRATEGY_LABEL.get(latest.strategy, latest.strategy)}")
        lines.append("")
        if latest.strategy is PatchStrategy.BACKPORT:
            lines.append(
                "> No fixed version has been published upstream. This patch changes this "
                "repository's own code to remove the exposure, and therefore warrants a "
                "closer review than a version bump."
            )
            lines.append("")
        if latest.diff:
            lines.append("```diff")
            lines.append(latest.diff.strip())
            lines.append("```")
            lines.append("")

    # --- is it proven ---
    lines.append("### Verification")
    lines.append("")
    if latest and latest.tests_passed:
        lines.append(f"✅ Test suite passed on attempt {latest.attempt}.")
    elif latest and latest.error:
        lines.append(f"❌ {latest.error}")
    else:
        lines.append("⚠️ Not verified - see escalation reason below.")
    lines.append("")

    if len(finding.attempts) > 1:
        lines.append(
            f"Reached after {len(finding.attempts)} attempts; earlier attempts failed "
            f"verification and were revised."
        )
        lines.append("")

    if finding.escalation_reason:
        lines.append(f"**Escalated:** {finding.escalation_reason}")
        lines.append("")

    # --- audit ---
    lines.append("### Audit trail")
    lines.append("")
    lines.append(f"- Run: `{finding.run_id}`")
    lines.append(f"- Finding: `{finding.id}`")
    lines.append(f"- Cost: ${finding.cost_usd:.4f}")
    if decisions:
        lines.append("")
        for decision in decisions:
            model = f" ({decision.model})" if decision.model else ""
            lines.append(f"  - `{decision.agent}`{model}: {decision.action}")

    lines.append("")
    lines.append("---")
    lines.append(
        "🌙 Opened by [Nightshift](https://github.com/) while nobody was watching. "
        "Every decision above is reproducible from the audit trail."
    )

    return "\n".join(lines)


class Reporter:
    """Opens the pull request, or records the escalation.

    ``github`` and ``store`` are injected so the rendering and idempotency behaviour can
    be tested without touching a real repository.
    """

    def __init__(self, github, store, settings) -> None:
        self._github = github
        self._store = store
        self._settings = settings

    async def report(
        self,
        finding: Finding,
        advisory: Advisory,
        patched_files: dict[str, str],
        default_branch: str,
    ) -> Finding:
        """Open a pull request for a verified finding.

        Claims the idempotency key first. If the claim fails, another run already opened
        this pull request and this one stops - that is what keeps an at-least-once
        delivery from becoming a duplicate PR.
        """
        if finding.status is FindingStatus.ESCALATED:
            log.info("report.escalated_no_pr", finding_id=finding.id)
            return finding

        key = finding.idempotency_key()
        if not await self._store.try_claim_pr(key, finding.id):
            log.info("report.already_reported", finding_id=finding.id)
            return finding

        try:
            branch = branch_name(finding)
            await self._github.create_branch(finding.repo, branch, default_branch)

            for path, content in patched_files.items():
                await self._github.update_file(
                    finding.repo,
                    path,
                    content,
                    message=f"Fix {advisory.id} in {finding.dependency.name}",
                    branch=branch,
                )

            decisions = await self._store.decisions_for_finding(finding.id)

            url = await self._github.open_pull_request(
                finding.repo,
                title=pull_request_title(finding, advisory),
                body=render_pull_request_body(finding, advisory, decisions),
                head=branch,
                base=default_branch,
            )

            finding.pr_url = url
            finding.status = FindingStatus.PR_OPENED
            log.info("report.pr_opened", finding_id=finding.id, url=url)

        except Exception:
            # The claim is released only because the pull request definitively was not
            # created. Releasing it for an action that did happen would reintroduce the
            # duplicate it exists to prevent.
            await self._store.release_pr_claim(key)
            finding.status = FindingStatus.FAILED
            raise

        return finding
