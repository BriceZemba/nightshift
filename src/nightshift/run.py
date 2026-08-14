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
from google.adk import Runner, Workflow
from google.adk.sessions import InMemorySessionService

from nightshift.agents.graph import FindingContext, build_finding_workflow
from nightshift.agents.guardian import Guardian
from nightshift.agents.patcher import Patcher
from nightshift.agents.reporter import Reporter
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
from nightshift.policy import PolicyViolation, assert_repo_allowed
from nightshift.security import build_screener
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

        # Model Armor screens attacker-authored advisory text before it reaches a model
        # that can write code. When it is not configured the screener is None, and Guardian
        # falls back to deterministic pattern screening while recording that Model Armor
        # was not consulted.
        self.guardian = Guardian(model_armor=build_screener(self.settings))
        self.triager = Triager(self.llm)
        self.patcher = Patcher(self.llm)
        self.verifier = verifier or Verifier()
        self.reporter = Reporter(github, store, self.settings)

        #: Pull request slots consumed this run. Separate from ``record.prs_opened``,
        #: which is a tally of outcomes; this one gates the writes as they happen.
        self._pr_slots_used = 0

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
        """Run every finding through the ADK graph.

        The topology is built once and reused; only the payload differs per finding. That
        is the point of a static graph, and it keeps the fan-out here rather than smearing
        it across the agent definitions.
        """
        sibling_packages = {name: s.package_names for name, s in snapshots.items()}

        # The graph is invoked with a finding id, not the context itself, because ADK's
        # entry points accept only text. This registry is what the id is exchanged for.
        contexts: dict[str, FindingContext] = {}

        # ADK validates the context afresh at every node, so each node works on a copy and
        # mutations never reach the object passed in. Terminal nodes therefore report the
        # finished finding through this callback rather than by returning it.
        results: dict[str, Finding] = {}

        async def on_complete(finding: Finding) -> None:
            results[finding.id] = finding

        workflow = build_finding_workflow(
            guardian=self.guardian,
            triager=self.triager,
            patcher=self.patcher,
            verifier=self.verifier,
            reporter=self.reporter,
            settings=self.settings,
            context_lookup=contexts.__getitem__,
            on_decision=self._log_decision,
            on_complete=on_complete,
            reserve_pr_slot=self._reserve_pr_slot,
        )

        semaphore = asyncio.Semaphore(4)

        async def handle(finding: Finding) -> None:
            async with semaphore:
                snapshot = snapshots[finding.repo]
                ctx = FindingContext(
                    finding=finding,
                    advisory=advisories[finding.advisory_id],
                    sources=snapshot.sources,
                    manifests=snapshot.manifests,
                    default_branch=snapshot.repo.default_branch,
                    repo_has_tests=snapshot.repo.has_tests,
                    test_command=snapshot.repo.test_command,
                    sibling_packages={
                        name: packages
                        for name, packages in sibling_packages.items()
                        if name != finding.repo
                    },
                )
                contexts[finding.id] = ctx

                await self._log_decision(finding, "watcher", f"matched {finding.advisory_id}")

                try:
                    await self._run_graph(workflow, ctx)
                except PolicyViolation as exc:
                    # A boundary crossing aborts this finding. It is never retried into
                    # success, and it never fails the whole run.
                    finding.status = FindingStatus.FAILED
                    log.warning("run.policy_violation", finding_id=finding.id, error=str(exc))
                    self.record.failed += 1
                    await self.store.upsert_finding(finding)
                    return

                final = results.get(finding.id)
                if final is None:
                    # No terminal node ran, which means the graph suspended on the
                    # human-approval gate. ``ask_human`` is a plain generator and cannot
                    # await the completion callback, so the outcome is inferred here.
                    final = finding
                    final.status = FindingStatus.AWAITING_APPROVAL

                self._tally(final)
                await self.store.upsert_finding(final)

        await asyncio.gather(*(handle(f) for f in findings))

    async def _run_graph(self, workflow: Workflow, ctx: FindingContext) -> None:
        """Execute one finding through the graph.

        A fresh Runner per finding keeps sessions isolated, which matters because findings
        are processed concurrently. Construction is cheap.

        The outcome arrives through the graph's ``on_complete`` callback rather than a
        return value here, because node outputs are serialized and the workflow's final
        event carries a plain dict.
        """
        runner = Runner(
            app_name="nightshift",
            agent=workflow,
            session_service=InMemorySessionService(),
            auto_create_session=True,
        )

        # The payload is the finding id: ADK wraps the argument in a text Content part, so
        # it cannot carry an object. The entry node exchanges the id for the full context.
        await runner.run_debug(ctx.finding.id, quiet=True)

    def _reserve_pr_slot(self) -> bool:
        """Claim one of the run's allowed pull requests, or refuse.

        Deliberately synchronous. Findings are processed concurrently, so a gate that
        merely *read* a counter would let every pending finding through before any of them
        had opened anything; the check and the increment have to happen without an await
        between them.

        Conservative by design: a reserved slot is not returned if the pull request later
        fails, so the run can under-open by a slot rather than over-open. For a flood guard
        that is the right direction to be wrong in.
        """
        if self._pr_slots_used >= self.settings.max_prs_per_run:
            return False
        self._pr_slots_used += 1
        return True

    def _tally(self, finding: Finding) -> None:
        """Fold one finished finding into the run summary."""
        if finding.verdict and finding.verdict.reachability is not Reachability.NOT_REACHABLE:
            self.record.findings_reachable += 1

        if finding.status is FindingStatus.PR_OPENED:
            self.record.prs_opened += 1
        elif finding.status is FindingStatus.DISMISSED:
            self.record.dismissed += 1
        elif finding.status in (FindingStatus.ESCALATED, FindingStatus.AWAITING_APPROVAL):
            self.record.escalated += 1
        elif finding.status is FindingStatus.FAILED:
            self.record.failed += 1

    async def _log_decision(self, finding: Finding, agent: str, action: str) -> None:
        await self.store.record_decision(
            Decision(
                run_id=self.record.id,
                finding_id=finding.id,
                agent=agent,
                action=action,
            )
        )


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
