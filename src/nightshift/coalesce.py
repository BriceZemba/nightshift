"""Collapse many advisories against one package into a single pull request.

A real run against a five-dependency repository produced **five pull requests, all bumping
`requests`, each to a different version.** They conflict with each other, and a maintainer
opening that inbox is looking at exactly the noise this project exists to remove. Producing
it would be self-refuting.

The insight is that the advisories are not independent work. Bumping `requests` to the
highest fixed version among them resolves all of them at once, because fixed versions are
cumulative: a release that fixes the newest advisory also contains the fixes for the older
ones. So the right unit of work is **(repository, package)**, not (repository, advisory).

The advisory kept is the one demanding the highest version. The rest are recorded on
``Finding.also_fixes`` and listed in the pull request body, so nothing is hidden, it is
merely not opened five times.
"""

from __future__ import annotations

import structlog
from packaging.version import InvalidVersion, Version

from nightshift.models import Advisory, Finding

log = structlog.get_logger(__name__)


def _sort_key(version: str) -> tuple[int, Version | str]:
    """Order versions, tolerating strings PyPI would not accept.

    Unparseable versions sort *below* every real one: a bump target we cannot compare is a
    bump target we should not prefer.
    """
    try:
        return (1, Version(version))
    except InvalidVersion:
        return (0, version)


def highest_fixed_version(
    findings: list[Finding], advisories: dict[str, Advisory]
) -> str | None:
    """The greatest fixed version demanded by any advisory in the group."""
    targets: list[str] = []
    for finding in findings:
        advisory = advisories.get(finding.advisory_id)
        if advisory is None:
            continue
        affected = advisory.affects(finding.dependency.name, finding.dependency.ecosystem)
        fixed = affected.first_fixed_version() if affected else None
        if fixed:
            targets.append(fixed)

    return max(targets, key=_sort_key) if targets else None


def _demands(finding: Finding, advisories: dict[str, Advisory], target: str) -> bool:
    """Whether this finding is the one asking for ``target``.

    Module level rather than a closure inside the loop: a nested function capturing the
    loop variable is a late-binding bug waiting for someone to defer the call.
    """
    advisory = advisories.get(finding.advisory_id)
    if advisory is None:
        return False
    affected = advisory.affects(finding.dependency.name, finding.dependency.ecosystem)
    return bool(affected and affected.first_fixed_version() == target)


def coalesce(
    findings: list[Finding], advisories: dict[str, Advisory]
) -> list[Finding]:
    """Reduce findings to one per (repository, package).

    Groups with no fixed version anywhere are left untouched. Those are the backport cases,
    where each advisory needs its own code change and collapsing them would merge genuinely
    separate work.
    """
    groups: dict[tuple[str, str], list[Finding]] = {}
    for finding in findings:
        groups.setdefault((finding.repo, finding.dependency.name), []).append(finding)

    coalesced: list[Finding] = []

    for (repo, package), group in groups.items():
        if len(group) == 1:
            coalesced.append(group[0])
            continue

        target = highest_fixed_version(group, advisories)
        if target is None:
            # Every advisory here needs its own backport. Keep them separate.
            coalesced.extend(group)
            continue

        primary = next(
            (f for f in group if _demands(f, advisories, target)), group[0]
        )
        primary.also_fixes = sorted(
            f.advisory_id for f in group if f.advisory_id != primary.advisory_id
        )

        log.info(
            "coalesce.group",
            repo=repo,
            package=package,
            advisories=len(group),
            keeping=primary.advisory_id,
            target=target,
        )
        coalesced.append(primary)

    if len(coalesced) < len(findings):
        log.info("coalesce.summary", before=len(findings), after=len(coalesced))

    return coalesced
