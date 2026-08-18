"""Collapsing many advisories against one package into a single pull request.

A real run produced five pull requests, all bumping `requests`, each to a different version.
They conflict with each other, and that is precisely the noise this project exists to
remove. A tool that generates it is self-refuting, so these tests are about the product's
central claim rather than an optimisation.
"""

from __future__ import annotations

import pytest

from nightshift.coalesce import coalesce, highest_fixed_version
from nightshift.models import (
    Advisory,
    AffectedPackage,
    Dependency,
    Ecosystem,
    Finding,
    VersionRange,
)


def _advisory(advisory_id: str, fixed: str | None, package: str = "requests") -> Advisory:
    ranges = [VersionRange(introduced="0", fixed=fixed)] if fixed else [
        VersionRange(introduced="0")
    ]
    return Advisory(
        id=advisory_id,
        summary=f"Issue {advisory_id}",
        screened=True,
        affected=[
            AffectedPackage(name=package, ecosystem=Ecosystem.PYPI, ranges=ranges)
        ],
    )


def _finding(advisory_id: str, package: str = "requests", repo: str = "me/repo") -> Finding:
    return Finding(
        id=f"f-{advisory_id}-{package}",
        run_id="r1",
        advisory_id=advisory_id,
        repo=repo,
        dependency=Dependency(
            name=package,
            ecosystem=Ecosystem.PYPI,
            version="2.19.1",
            manifest_path="requirements.txt",
        ),
    )


class TestHighestFixedVersion:
    def test_picks_the_greatest(self) -> None:
        advisories = {
            "A": _advisory("A", "2.20.0"),
            "B": _advisory("B", "2.32.4"),
            "C": _advisory("C", "2.31.0"),
        }
        findings = [_finding(a) for a in advisories]

        assert highest_fixed_version(findings, advisories) == "2.32.4"

    def test_orders_by_version_not_string(self) -> None:
        """String ordering puts 2.9.0 above 2.10.0, which would leave a real advisory
        unfixed while claiming otherwise."""
        advisories = {"A": _advisory("A", "2.9.0"), "B": _advisory("B", "2.10.0")}
        findings = [_finding(a) for a in advisories]

        assert highest_fixed_version(findings, advisories) == "2.10.0"

    def test_unparseable_versions_lose_to_real_ones(self) -> None:
        """A target we cannot compare is a target we should not prefer."""
        advisories = {"A": _advisory("A", "not-a-version"), "B": _advisory("B", "2.31.0")}
        findings = [_finding(a) for a in advisories]

        assert highest_fixed_version(findings, advisories) == "2.31.0"

    def test_no_fixed_version_anywhere(self) -> None:
        advisories = {"A": _advisory("A", None), "B": _advisory("B", None)}
        findings = [_finding(a) for a in advisories]

        assert highest_fixed_version(findings, advisories) is None


class TestCoalesce:
    def test_five_advisories_become_one_pull_request(self) -> None:
        """The exact shape of the real run that motivated this."""
        advisories = {
            "GHSA-1": _advisory("GHSA-1", "2.20.0"),
            "GHSA-2": _advisory("GHSA-2", "2.32.0"),
            "GHSA-3": _advisory("GHSA-3", "2.31.0"),
            "PYSEC-4": _advisory("PYSEC-4", "2.32.4"),
            "PYSEC-5": _advisory("PYSEC-5", "2.33.0"),
        }
        findings = [_finding(a) for a in advisories]

        result = coalesce(findings, advisories)

        assert len(result) == 1
        assert result[0].advisory_id == "PYSEC-5", "keep the one demanding the highest bump"
        assert len(result[0].also_fixes) == 4

    def test_nothing_is_hidden(self) -> None:
        """The other advisories are still reported, just not opened five times."""
        advisories = {"A": _advisory("A", "2.20.0"), "B": _advisory("B", "2.32.4")}
        findings = [_finding(a) for a in advisories]

        result = coalesce(findings, advisories)

        assert result[0].also_fixes == ["A"]

    def test_different_packages_stay_separate(self) -> None:
        """requests and pyyaml are genuinely different work."""
        advisories = {
            "A": _advisory("A", "2.32.4", package="requests"),
            "B": _advisory("B", "5.4", package="pyyaml"),
        }
        findings = [_finding("A", package="requests"), _finding("B", package="pyyaml")]

        assert len(coalesce(findings, advisories)) == 2

    def test_different_repositories_stay_separate(self) -> None:
        advisories = {"A": _advisory("A", "2.32.4"), "B": _advisory("B", "2.32.4")}
        findings = [
            _finding("A", repo="me/repo-a"),
            _finding("B", repo="me/repo-b"),
        ]

        assert len(coalesce(findings, advisories)) == 2

    def test_backports_are_left_alone(self) -> None:
        """With no fixed version anywhere, each advisory needs its own code change.
        Collapsing them would merge genuinely separate work."""
        advisories = {"A": _advisory("A", None), "B": _advisory("B", None)}
        findings = [_finding(a) for a in advisories]

        result = coalesce(findings, advisories)

        assert len(result) == 2
        assert all(not f.also_fixes for f in result)

    def test_a_lone_finding_is_untouched(self) -> None:
        advisories = {"A": _advisory("A", "2.32.4")}
        findings = [_finding("A")]

        result = coalesce(findings, advisories)

        assert len(result) == 1
        assert result[0].also_fixes == []

    def test_empty_input(self) -> None:
        assert coalesce([], {}) == []

    def test_a_missing_advisory_does_not_crash_the_group(self) -> None:
        """Advisory fetches can fail individually; the group must still resolve."""
        advisories = {"A": _advisory("A", "2.32.4")}
        findings = [_finding("A"), _finding("MISSING")]

        result = coalesce(findings, advisories)

        assert len(result) == 1
        assert result[0].advisory_id == "A"


class TestIdempotencyStillHolds:
    def test_the_kept_finding_keeps_a_stable_key(self) -> None:
        """Coalescing must not change the idempotency key, or a re-run opens a second
        pull request for work already done."""
        advisories = {"A": _advisory("A", "2.20.0"), "B": _advisory("B", "2.32.4")}

        first = coalesce([_finding("A"), _finding("B")], advisories)[0]
        second = coalesce([_finding("A"), _finding("B")], advisories)[0]

        assert first.idempotency_key() == second.idempotency_key()

    @pytest.mark.parametrize("order", [("A", "B"), ("B", "A")])
    def test_input_order_does_not_change_the_outcome(self, order: tuple[str, str]) -> None:
        advisories = {"A": _advisory("A", "2.20.0"), "B": _advisory("B", "2.32.4")}
        findings = [_finding(a) for a in order]

        result = coalesce(findings, advisories)

        assert result[0].advisory_id == "B"
        assert result[0].also_fixes == ["A"]
