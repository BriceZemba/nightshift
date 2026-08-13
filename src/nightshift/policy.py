"""Policy invariants - the boundary Nightshift will not cross.

Nightshift writes code and opens pull requests based partly on text written by strangers
(advisory descriptions, upstream commit messages, third-party diffs). That is a real
prompt-injection path with a code-execution payoff, so the safety properties cannot live
only in a prompt. They are enforced here, deterministically, outside the model's reach.

Three invariants:

1. **Own repos only.** Nightshift opens pull requests solely against repositories on an
   explicit allowlist. Automated PRs against repositories you do not own are spam, and
   they burden maintainers who never asked for them.
2. **Never touch execution surfaces.** CI workflows, deploy manifests and credential files
   are off limits. A poisoned advisory that talks a model into editing
   ``.github/workflows/`` would be handing an attacker the repository.
3. **Humans approve the irreversible.** Major version bumps, repos without tests, and
   anything Guardian flagged stop and wait for a person.
"""

from __future__ import annotations

import fnmatch

from nightshift.config import Settings
from nightshift.models import Finding, PatchStrategy, Reachability

#: Paths an automated patch may never modify.
#:
#: Workflow files are the sharpest edge: a change under .github/workflows/ executes with
#: repository credentials on the next push, so writing there converts a patch suggestion
#: into arbitrary code execution.
#: Note on matching: ``fnmatch`` is not glob. Its ``*`` happily matches ``/``, and ``**``
#: carries no special recursive meaning - it collapses to a single ``*``. Patterns here are
#: therefore written flat, and anything that needs to match at arbitrary depth is handled
#: by FORBIDDEN_DIR_SEGMENTS instead.
FORBIDDEN_PATH_PATTERNS: tuple[str, ...] = (
    ".gitlab-ci.yml",
    "Jenkinsfile",
    "azure-pipelines.y*ml",
    "cloudbuild.y*ml",
    "skaffold.y*ml",
    "*.tf",
    "*.tfvars",
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*credentials*",
    "*service-account*.json",
)

#: Any path with one of these as a *directory* component is off limits, at any depth.
#:
#: ``.github`` is the sharpest of them: a file under ``.github/workflows/`` executes with
#: repository credentials on the next push, so write access there is arbitrary code
#: execution rather than a code suggestion.
FORBIDDEN_DIR_SEGMENTS: frozenset[str] = frozenset(
    {
        ".github",
        ".circleci",
        ".gitlab",
        "k8s",
        "kubernetes",
        "helm",
        "secrets",
        "terraform",
    }
)

#: Changing these is legitimate for some advisories (a vulnerable base image, for example)
#: but the blast radius is large enough that a human should look first.
APPROVAL_PATH_PATTERNS: tuple[str, ...] = (
    "Dockerfile",
    "Dockerfile.*",
    "docker-compose.y*ml",
    "setup.py",
)


class PolicyViolation(Exception):
    """Raised when an action would cross a hard boundary.

    Deliberately not caught anywhere in the agent graph: a policy violation should abort
    the finding, not be retried into success.
    """


def assert_repo_allowed(repo_full_name: str, settings: Settings) -> None:
    """Refuse any repository that is not explicitly allowlisted.

    Enforced at the point of action rather than at configuration time, so a model that
    invents a plausible-looking repository name cannot route around it.
    """
    if not settings.repo_allowlist:
        raise PolicyViolation(
            "No repositories are allowlisted. Set NIGHTSHIFT_REPO_ALLOWLIST to repositories "
            "you own or have forked. Nightshift will not open pull requests against "
            "repositories belonging to other people."
        )
    if repo_full_name not in settings.repo_allowlist:
        raise PolicyViolation(
            f"{repo_full_name!r} is not on the allowlist. Nightshift only opens pull requests "
            f"against repositories you own or have forked."
        )


def normalize_path(path: str) -> str:
    """Reduce a path to one comparable form.

    Backslashes first, then leading ``./`` and ``/`` segments. Order matters: a Windows-style
    ``.github\\workflows\\ci.yml`` has to become POSIX before any prefix work, or the guard
    reads it as a single opaque filename.

    ``removeprefix`` rather than ``lstrip`` - ``lstrip`` strips *characters*, so
    ``".github/x".lstrip("./")`` yields ``"github/x"`` and defeats the ``.github`` check
    entirely.
    """
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized.removeprefix("./")
    return normalized.lstrip("/")


def _matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def is_forbidden_path(path: str) -> bool:
    """True when an automated patch may never touch this path."""
    normalized = normalize_path(path)

    directories = normalized.split("/")[:-1]
    if any(segment in FORBIDDEN_DIR_SEGMENTS for segment in directories):
        return True

    return _matches_any(normalized, FORBIDDEN_PATH_PATTERNS)


def assert_paths_allowed(paths: list[str]) -> None:
    """Reject a patch that touches an execution surface or a credential file."""
    violations = [normalize_path(p) for p in paths if is_forbidden_path(p)]
    if violations:
        raise PolicyViolation(
            "Patch touches protected paths that an automated change may never modify: "
            + ", ".join(sorted(violations))
        )


def is_major_bump(current: str, proposed: str) -> bool:
    """True when the leading version component changes.

    Intentionally lenient about odd version strings: anything unparseable is treated as a
    major bump, because "I could not tell" should route to a human rather than proceed.
    """
    try:
        current_major = current.lstrip("v=<>~^ ").split(".")[0]
        proposed_major = proposed.lstrip("v=<>~^ ").split(".")[0]
    except (AttributeError, IndexError):
        return True
    if not current_major.isdigit() or not proposed_major.isdigit():
        return True
    return current_major != proposed_major


def requires_human_approval(
    finding: Finding,
    proposed_version: str | None,
    repo_has_tests: bool,
    guardian_flagged: bool,
) -> str | None:
    """Return the reason a human must approve, or ``None`` if the fleet may proceed alone.

    Ordered most-severe first so the escalation message names the strongest reason.
    """
    if guardian_flagged:
        return "Guardian flagged untrusted content in this advisory"

    if not repo_has_tests:
        return "Repository has no runnable test suite, so the patch cannot be verified"

    if finding.verdict and finding.verdict.reachability is Reachability.UNKNOWN:
        return "Reachability analysis was inconclusive"

    if proposed_version and is_major_bump(finding.dependency.version, proposed_version):
        return (
            f"Major version bump {finding.dependency.version} -> {proposed_version} "
            f"may contain breaking changes"
        )

    latest = finding.attempts[-1] if finding.attempts else None
    if latest and latest.strategy is PatchStrategy.BACKPORT:
        return "Backported patch has no upstream release to compare against"

    return None
