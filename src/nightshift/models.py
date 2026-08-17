"""Domain types for Nightshift.

Every type here is designed to be persisted to Firestore and replayed. The pipeline is
resumable: a run that dies halfway can be restarted, and each stage recomputes only what
is missing. That means these models carry *evidence*, not just verdicts - the call path
that justified a reachability decision is as important as the decision itself, because it
is what ends up in the pull request a human has to trust.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Ecosystem(StrEnum):
    """Package ecosystems, spelled the way OSV.dev spells them."""

    PYPI = "PyPI"
    NPM = "npm"
    GO = "Go"
    CRATES_IO = "crates.io"
    MAVEN = "Maven"
    NUGET = "NuGet"
    RUBYGEMS = "RubyGems"


class Reachability(StrEnum):
    """Whether the vulnerable symbol can actually be reached from this repo's entrypoints.

    This is the core judgement Nightshift makes and the reason it exists: most advisories
    that match a manifest are not reachable in the consuming code, and the existing tools
    cannot tell the difference.
    """

    REACHABLE = "reachable"
    NOT_REACHABLE = "not_reachable"
    #: Analysis was inconclusive. Treated as reachable for safety, but flagged for the human.
    UNKNOWN = "unknown"


class PatchStrategy(StrEnum):
    UPSTREAM_BUMP = "upstream_bump"
    #: No fixed version exists upstream. This is the ~40% of advisories that version
    #: bumping tools have nothing to say about, and the case Nightshift is built for.
    BACKPORT = "backport"
    NO_FIX_AVAILABLE = "no_fix_available"


class FindingStatus(StrEnum):
    DISCOVERED = "discovered"
    TRIAGED = "triaged"
    PATCHED = "patched"
    VERIFIED = "verified"
    PR_OPENED = "pr_opened"
    #: The graph is suspended on a human-approval gate. Nothing was written; the run ended
    #: with this finding waiting for a person. Distinct from ESCALATED, which means the
    #: fleet gave up on its own.
    AWAITING_APPROVAL = "awaiting_approval"
    #: Handed to a human: major version bump, no test coverage, or Guardian flagged it.
    ESCALATED = "escalated"
    DISMISSED = "dismissed"
    FAILED = "failed"


class Severity(BaseModel):
    type: str
    score: str
    numeric: float | None = None


class VersionRange(BaseModel):
    introduced: str | None = None
    fixed: str | None = None
    last_affected: str | None = None
    #: OSV range type: ECOSYSTEM, SEMVER or GIT. It matters because a GIT range's "fixed"
    #: value is a commit SHA, not a release, and pinning a manifest to a 40-character hash
    #: produces an install that cannot resolve.
    type: str = "ECOSYSTEM"


class AffectedPackage(BaseModel):
    name: str
    ecosystem: Ecosystem
    ranges: list[VersionRange] = Field(default_factory=list)
    versions: list[str] = Field(default_factory=list)

    def first_fixed_version(self) -> str | None:
        """The earliest *released* version that resolves the advisory, if one exists.

        GIT ranges are skipped. Their ``fixed`` value is a commit SHA, which is a real
        answer to "where was this fixed" and a useless answer to "what do I pin to".
        """
        for r in self.ranges:
            if r.fixed and r.type.upper() != "GIT":
                return r.fixed
        return None


class Advisory(BaseModel):
    """A normalized security advisory, merged from OSV / GHSA / NVD.

    ``summary`` and ``details`` are attacker-controlled text: they are written by whoever
    filed the advisory. They must pass through Model Armor before reaching any model that
    can write code or open a pull request. See ``guardian``.
    """

    id: str
    aliases: list[str] = Field(default_factory=list)
    summary: str = ""
    details: str = ""
    published: datetime | None = None
    modified: datetime | None = None
    severity: list[Severity] = Field(default_factory=list)
    affected: list[AffectedPackage] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    #: Set once Model Armor has cleared the free-text fields. Untrusted until then.
    screened: bool = False

    def cvss_score(self) -> float | None:
        for s in self.severity:
            if s.numeric is not None:
                return s.numeric
        return None

    def affects(self, package: str, ecosystem: Ecosystem) -> AffectedPackage | None:
        for a in self.affected:
            if a.name == package and a.ecosystem == ecosystem:
                return a
        return None


class Dependency(BaseModel):
    name: str
    ecosystem: Ecosystem
    version: str
    #: Path to the manifest that declares it, e.g. "requirements.txt", "pyproject.toml".
    manifest_path: str
    #: Direct dependencies are patchable here; transitive ones may need an upstream fix
    #: or a constraint pin, which changes the patch strategy.
    is_direct: bool = True


class Repo(BaseModel):
    """A target repository. Must be on the allowlist - see ``policy``."""

    owner: str
    name: str
    default_branch: str = "main"
    #: Whether a usable test suite exists. Without one, the Verifier cannot prove a patch
    #: is safe, so the finding is escalated to a human instead of auto-opened.
    has_tests: bool = False
    test_command: str | None = None

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"


class CallSite(BaseModel):
    """One step in the path from a repo entrypoint to the vulnerable symbol.

    This is the evidence a maintainer actually needs. A PR that says "upgrade this" is
    ignored; a PR that says "you call this function on line 40 of your request handler"
    gets read.
    """

    file_path: str
    line: int
    symbol: str
    snippet: str = ""


class TriageVerdict(BaseModel):
    reachability: Reachability
    #: Ordered entrypoint -> vulnerable symbol. Empty when not reachable.
    call_path: list[CallSite] = Field(default_factory=list)
    rationale: str = ""
    #: Other allowlisted repos sharing this dependency - the cross-repo blast radius.
    also_affects: list[str] = Field(default_factory=list)
    model_used: str = ""


class PatchAttempt(BaseModel):
    attempt: int
    strategy: PatchStrategy
    diff: str = ""
    tests_passed: bool = False
    test_output: str = ""
    error: str = ""


class Finding(BaseModel):
    """One advisory as it applies to one repository - the unit of work in the pipeline."""

    id: str
    run_id: str
    advisory_id: str
    repo: str
    dependency: Dependency
    status: FindingStatus = FindingStatus.DISCOVERED
    verdict: TriageVerdict | None = None
    attempts: list[PatchAttempt] = Field(default_factory=list)
    pr_url: str | None = None
    escalation_reason: str | None = None
    cost_usd: float = 0.0
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @staticmethod
    def make_id(run_id: str, advisory_id: str, repo: str, package: str) -> str:
        """Deterministic id, so re-processing the same work updates rather than duplicates."""
        raw = f"{run_id}|{advisory_id}|{repo}|{package}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def idempotency_key(self) -> str:
        """Stable across runs - this is what stops a retry opening a second pull request.

        Deliberately excludes ``run_id``: the same advisory against the same package in the
        same repo is the same real-world action, whether it is discovered tonight or next week.
        """
        raw = f"{self.advisory_id}|{self.repo}|{self.dependency.name}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]


class Decision(BaseModel):
    """One audit-log entry. Every agent hop appends one.

    The chain of these is what the pull request body renders, and what makes the system
    inspectable rather than a black box.
    """

    run_id: str
    finding_id: str | None
    agent: str
    action: str
    detail: str = ""
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    trace_id: str | None = None
    at: datetime | None = None


class RunRecord(BaseModel):
    """One nightly run. The numbers quoted in the demo come straight from this."""

    id: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    repos_scanned: int = 0
    advisories_ingested: int = 0
    findings_reachable: int = 0
    prs_opened: int = 0
    escalated: int = 0
    dismissed: int = 0
    failed: int = 0
    cost_usd: float = 0.0
    dry_run: bool = True
