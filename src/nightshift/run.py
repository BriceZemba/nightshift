"""The run orchestrator - the Watcher, and the fan-out around the finding graph.

An ADK graph is a static topology, so the per-item work lives inside the graph and the
fan-out lives here. In production this stage publishes findings to Pub/Sub, which supplies
redelivery and dead-lettering; the in-process path below is the same pipeline with an
``asyncio.Semaphore`` standing in for the queue, which is what makes a full run
demonstrable on a laptop with no cloud project attached.

Ordering is chosen to spend the least money possible before the cheapest filter has run:

1. Read manifests and sources once per repository.
2. One batched OSV query for every dependency across every repository.
3. Static reachability analysis - free, deterministic, and where 180 becomes 6.
4. Only what survives reaches a model.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog

from nightshift.agents.guardian import Guardian
from nightshift.agents.patcher import Patcher
from nightshift.agents.reporter import Reporter
from nightshift.agents.routing import APPROVE, ESCALATE, RETRY, verify_route
from nightshift.agents.triager import Triager
from nightshift.agents.verifier import Verifier
from nightshift.config import Settings, get_settings
from nightshift.llm import LLMClient
from nightshift.models import (
    Advisory,
    Decision,
    Dependency,
    Finding,
    FindingStatus,
    Reachability,
    Repo,
    RunRecord,
)
from nightshift.policy import PolicyViolation, assert_repo_allowed, requires_human_approval
from nightshift.sources.manifests import scan_files

log = structlog.get_logger(__name__)


@dataclass
class RepoSnapshot:
    """Everything read from one repository, fetched once and reused."""

    repo: Repo
    dependencies: list[Dependency] = field(default_factory=list)
    unresolved_count: int = 0
    manifests: dict[str, str] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)

    @property
    def package_names(self) -> set[str]:
        return {d.name for d in self.dependencies}


class NightshiftRun:
    """One nightly run across every allowlisted repository."""

    def __init__(
        self,
        *,
        github,
        osv,
        store,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
        verifier: Verifier | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.github = github
        self.osv = osv
        self.store = store
        self.llm = llm or LLMClient(settings=self.settings)

        self.guardian = Guardian()
        self.triager = Triager(self.llm)
        self.patcher = Patcher(self.llm)
        self.verifier = verifier or Verifier()
        self.reporter = Reporter(github, store, self.settings)

        self.record = RunRecord(
            id=f"run-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}",
            dry_run=self.settings.dry_run,
        )

    async def execute(self) -> RunRecord:
        """Run the full pipeline and return the summary record."""
        self.record.started_at = datetime.now(UTC)
        await self.store.start_run(self.record)

        try:
            snapshots = await self._scan_repositories()
            self.record.repos_scanned = len(snapshots)

            advisories, findings = await self._discover(snapshots)
            self.record.advisories_ingested = len(advisories)

            await self._process(snapshots, advisories, findings)

        finally:
            self.record.cost_usd = self.llm.usage.cost_usd
            await self.store.finish_run(self.record)

        return self.record

    # --- stage 1: read every repository once --------------------------------

    async def _scan_repositories(self) -> dict[str, RepoSnapshot]:
        if not self.settings.repo_allowlist:
            raise PolicyViolation(
                "No repositories are allowlisted. Set NIGHTSHIFT_REPO_ALLOWLIST to "
                "repositories you own or have forked."
            )

        snapshots: dict[str, RepoSnapshot] = {}

        for full_name in self.settings.repo_allowlist:
            assert_repo_allowed(full_name, self.settings)

            repo = await self.github.get_repo(full_name)
            manifests = await self.github.fetch_manifests(full_name, repo.default_branch)
            scan = scan_files(manifests)
            sources = await self.github.fetch_sources(full_name, repo.default_branch)

            repo.has_tests = _detect_tests(sources)
            repo.test_command = _detect_test_command(manifests, repo.has_tests)

            snapshots[full_name] = RepoSnapshot(
                repo=repo,
                dependencies=scan.dependencies,
                unresolved_count=len(scan.unresolved),
                manifests=manifests,
                sources=sources,
            )

            log.info(
                "run.repo_scanned",
                repo=full_name,
                dependencies=len(scan.dependencies),
                unresolved=len(scan.unresolved),
                sources=len(sources),
                has_tests=repo.has_tests,
            )

        return snapshots

    # --- stage 2: one batched advisory query --------------------------------

    async def _discover(
        self, snapshots: dict[str, RepoSnapshot]
    ) -> tuple[dict[str, Advisory], list[Finding]]:
        """Query OSV once for every dependency, then build the work list."""
        all_dependencies: list[Dependency] = []
        owner_of: dict[int, str] = {}

        for full_name, snapshot in snapshots.items():
            for dependency in snapshot.dependencies:
                owner_of[len(all_dependencies)] = full_name
                all_dependencies.append(dependency)

        hits = await self.osv.query_dependencies(all_dependencies)
        if not hits:
            log.info("run.no_advisories")
            return {}, []

        advisory_ids = sorted({a for ids in hits.values() for a in ids})
        advisories = {a.id: a for a in await self.osv.get_advisories(advisory_ids)}

        # Screen every advisory before any of it reaches a model. Guardian runs here, at
        # the boundary where untrusted third-party text enters the system.
        flagged: set[str] = set()
        for advisory in advisories.values():
            result = self.guardian.screen_advisory(advisory)
            if not result.safe:
                flagged.add(advisory.id)
            await self.store.upsert_advisory(advisory)

        self._flagged_advisories = flagged

        findings: list[Finding] = []
        for index, dependency in enumerate(all_dependencies):
            key = f"{dependency.name}@{dependency.version}"
            if key not in hits:
                continue
            repo_name = owner_of[index]
            for advisory_id in hits[key]:
                if advisory_id not in advisories:
                    continue
                findings.append(
                    Finding(
                        id=Finding.make_id(
                            self.record.id, advisory_id, repo_name, dependency.name
                        ),
                        run_id=self.record.id,
                        advisory_id=advisory_id,
                        repo=repo_name,
                        dependency=dependency,
                    )
                )

        log.info("run.discovered", advisories=len(advisories), findings=len(findings))
        return advisories, findings

    # --- stage 3: triage, patch, verify, report -----------------------------

    async def _process(
        self,
        snapshots: dict[str, RepoSnapshot],
        advisories: dict[str, Advisory],
        findings: list[Finding],
    ) -> None:
        sibling_packages = {name: s.package_names for name, s in snapshots.items()}
        semaphore = asyncio.Semaphore(4)

        async def handle(finding: Finding) -> None:
            async with semaphore:
                try:
                    await self._process_one(
                        finding, snapshots[finding.repo], advisories[finding.advisory_id],
                        {k: v for k, v in sibling_packages.items() if k != finding.repo},
                    )
                except PolicyViolation as exc:
                    # A boundary crossing aborts this finding. It is never retried into
                    # success, and it never fails the whole run.
                    finding.status = FindingStatus.FAILED
                    log.warning("run.policy_violation", finding_id=finding.id, error=str(exc))
                    self.record.failed += 1
                    await self.store.upsert_finding(finding)

        await asyncio.gather(*(handle(f) for f in findings))

    async def _process_one(
        self,
        finding: Finding,
        snapshot: RepoSnapshot,
        advisory: Advisory,
        siblings: dict[str, set[str]],
    ) -> None:
        await self._log_decision(finding, "watcher", f"matched {advisory.id}")

        if advisory.id in getattr(self, "_flagged_advisories", set()):
            finding.status = FindingStatus.ESCALATED
            finding.escalation_reason = "Guardian flagged untrusted content in this advisory"
            self.record.escalated += 1
            await self._log_decision(finding, "guardian", "flagged; escalated without patching")
            await self.store.upsert_finding(finding)
            return

        # --- triage: the 180 -> 6 step ---
        verdict = self.triager.triage(
            advisory, finding.dependency, snapshot.sources, sibling_repos=siblings
        )
        finding.verdict = verdict
        finding.status = FindingStatus.TRIAGED
        await self._log_decision(
            finding, "triager", f"{verdict.reachability} ({len(verdict.call_path)} sites)"
        )

        if verdict.reachability is Reachability.NOT_REACHABLE:
            finding.status = FindingStatus.DISMISSED
            self.record.dismissed += 1
            await self.store.upsert_finding(finding)
            return

        self.record.findings_reachable += 1

        # --- critique loop: patch until verified, or give up and escalate ---
        manifest = snapshot.manifests.get(finding.dependency.manifest_path, "")

        while True:
            attempt = self.patcher.patch(finding, advisory, manifest)

            verification = self.verifier.verify(
                snapshot.sources, attempt.diff, snapshot.repo.test_command
            )
            attempt.tests_passed = verification.passed
            attempt.test_output = verification.output
            if verification.skipped_reason:
                attempt.error = verification.skipped_reason

            finding.attempts.append(attempt)
            await self._log_decision(
                finding,
                "patcher/verifier",
                f"attempt {attempt.attempt} ({attempt.strategy}) "
                f"{'passed' if attempt.tests_passed else 'failed'}",
            )

            route = verify_route(finding, self.settings.max_patch_attempts)
            if route != RETRY:
                break

        if route == ESCALATE:
            finding.status = FindingStatus.ESCALATED
            finding.escalation_reason = (
                finding.escalation_reason
                or f"No verified patch after {len(finding.attempts)} attempt(s)"
            )
            self.record.escalated += 1
            await self.store.upsert_finding(finding)
            return

        # --- approval gate ---
        latest = finding.attempts[-1]
        affected = advisory.affects(finding.dependency.name, finding.dependency.ecosystem)
        proposed = affected.first_fixed_version() if affected else None

        reason = requires_human_approval(
            finding,
            proposed_version=proposed,
            repo_has_tests=snapshot.repo.has_tests,
            guardian_flagged=False,
        )

        if reason:
            finding.status = FindingStatus.ESCALATED
            finding.escalation_reason = reason
            self.record.escalated += 1
            await self._log_decision(finding, "policy", f"human approval required: {reason}")
            await self.store.upsert_finding(finding)
            return

        if self.record.prs_opened >= self.settings.max_prs_per_run:
            # A hard ceiling so a bug cannot flood a repository, however confident the
            # fleet is.
            finding.status = FindingStatus.ESCALATED
            finding.escalation_reason = (
                f"Per-run pull request limit ({self.settings.max_prs_per_run}) reached"
            )
            self.record.escalated += 1
            await self.store.upsert_finding(finding)
            return

        # --- report ---
        # Only the files that actually changed are committed. Passing the whole manifest
        # set through unmodified would produce a pull request with an empty diff.
        patched = _apply_upstream_bump(snapshot, finding, proposed)
        if not patched:
            finding.status = FindingStatus.ESCALATED
            finding.escalation_reason = (
                "Patch produced no file change that could be committed"
            )
            self.record.escalated += 1
            await self.store.upsert_finding(finding)
            return

        if route == APPROVE and latest.diff:
            finding.status = FindingStatus.VERIFIED

        finding = await self.reporter.report(
            finding, advisory, patched, snapshot.repo.default_branch
        )
        if finding.status is FindingStatus.PR_OPENED:
            self.record.prs_opened += 1

        await self.store.upsert_finding(finding)

    async def _log_decision(self, finding: Finding, agent: str, action: str) -> None:
        await self.store.record_decision(
            Decision(
                run_id=self.record.id,
                finding_id=finding.id,
                agent=agent,
                action=action,
            )
        )


def _apply_upstream_bump(
    snapshot: RepoSnapshot, finding: Finding, proposed_version: str | None
) -> dict[str, str]:
    """Produce the files to commit, containing the actual change.

    Returns only manifests whose content genuinely differs. An empty result means the
    patch could not be materialized into a committable file - which routes to a human
    rather than opening a pull request with an empty diff.

    Backports are not handled here: they modify source files rather than manifests, and
    every backport escalates for human approval before reaching this point.
    """
    if not proposed_version:
        return {}

    from nightshift.agents.patcher import bump_manifest

    patched: dict[str, str] = {}
    for path, content in snapshot.manifests.items():
        updated, changed = bump_manifest(content, finding.dependency.name, proposed_version)
        if changed:
            patched[path] = updated

    return patched


def _detect_tests(sources: dict[str, str]) -> bool:
    """Whether the repository appears to have a runnable test suite.

    Conservative: a repository wrongly marked as having tests would let an unverified
    patch through the approval gate.
    """
    return any(
        "test" in path.rsplit("/", 1)[-1].lower() or path.startswith("tests/")
        for path in sources
    )


def _detect_test_command(manifests: dict[str, str], has_tests: bool) -> str | None:
    """Infer a test command, or ``None`` when nothing can be inferred.

    Returning ``None`` is a meaningful answer: it routes the finding to a human rather
    than pretending an unverified patch was verified.
    """
    if not has_tests:
        return None
    for path in manifests:
        if path.rsplit("/", 1)[-1] in {"pyproject.toml", "requirements.txt", "poetry.lock"}:
            return "python -m pytest -q"
    return None
