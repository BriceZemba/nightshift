"""OSV parsing and batching.

The range parser gets the most attention: OSV encodes affected versions as an ordered
event stream rather than as intervals, and getting that wrong means either patching
something that was never vulnerable or missing something that is.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from nightshift.models import Dependency, Ecosystem
from nightshift.sources.osv import (
    OSV_API,
    OSVClient,
    OSVError,
    _parse_ranges,
    parse_advisory,
)

RAW_ADVISORY = {
    "id": "GHSA-x84v-xcm2-53pg",
    "aliases": ["CVE-2023-32681"],
    "summary": "Unintended leak of Proxy-Authorization header",
    "details": "Requests is vulnerable to potential forwarding of headers.",
    "published": "2023-05-26T18:23:22Z",
    "modified": "2024-02-16T08:21:11Z",
    "severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:H/PR:N/UI:R/S:U/C:H/I:N/A:N"}],
    "affected": [
        {
            "package": {"name": "requests", "ecosystem": "PyPI"},
            "ranges": [
                {"type": "ECOSYSTEM", "events": [{"introduced": "2.3.0"}, {"fixed": "2.31.0"}]}
            ],
            "versions": ["2.30.0"],
        }
    ],
    "references": [
        {"type": "ADVISORY", "url": "https://github.com/psf/requests/security/advisories/x"}
    ],
}


class TestParseRanges:
    def test_flattens_introduced_fixed_pair(self) -> None:
        ranges = _parse_ranges(
            [{"events": [{"introduced": "1.0.0"}, {"fixed": "1.2.3"}]}]
        )
        assert len(ranges) == 1
        assert ranges[0].introduced == "1.0.0"
        assert ranges[0].fixed == "1.2.3"

    def test_handles_multiple_intervals_in_one_range(self) -> None:
        """A single OSV range can describe several disjoint affected windows."""
        ranges = _parse_ranges(
            [
                {
                    "events": [
                        {"introduced": "1.0.0"},
                        {"fixed": "1.2.0"},
                        {"introduced": "2.0.0"},
                        {"fixed": "2.1.0"},
                    ]
                }
            ]
        )
        assert len(ranges) == 2
        assert (ranges[0].introduced, ranges[0].fixed) == ("1.0.0", "1.2.0")
        assert (ranges[1].introduced, ranges[1].fixed) == ("2.0.0", "2.1.0")

    def test_unterminated_range_is_still_captured(self) -> None:
        """No 'fixed' event means no upstream fix exists - precisely the case Nightshift
        is built for, so it must not be dropped."""
        ranges = _parse_ranges([{"events": [{"introduced": "1.0.0"}]}])
        assert len(ranges) == 1
        assert ranges[0].introduced == "1.0.0"
        assert ranges[0].fixed is None

    def test_last_affected_closes_a_range(self) -> None:
        ranges = _parse_ranges(
            [{"events": [{"introduced": "1.0.0"}, {"last_affected": "1.9.9"}]}]
        )
        assert len(ranges) == 1
        assert ranges[0].last_affected == "1.9.9"


class TestParseAdvisory:
    def test_parses_core_fields(self) -> None:
        advisory = parse_advisory(RAW_ADVISORY)
        assert advisory.id == "GHSA-x84v-xcm2-53pg"
        assert "CVE-2023-32681" in advisory.aliases
        assert advisory.published is not None
        assert advisory.published.year == 2023

    def test_advisory_starts_unscreened(self) -> None:
        """summary/details are attacker-controlled text. They must not reach a model that
        can write code until Guardian has cleared them."""
        assert parse_advisory(RAW_ADVISORY).screened is False

    def test_finds_first_fixed_version(self) -> None:
        advisory = parse_advisory(RAW_ADVISORY)
        affected = advisory.affects("requests", Ecosystem.PYPI)
        assert affected is not None
        assert affected.first_fixed_version() == "2.31.0"

    def test_unknown_ecosystem_is_skipped_not_fatal(self) -> None:
        """OSV covers ecosystems we do not support; one unfamiliar entry must not discard
        an otherwise usable advisory."""
        raw = {
            **RAW_ADVISORY,
            "affected": [
                {"package": {"name": "thing", "ecosystem": "Hackage"}, "ranges": []},
                *RAW_ADVISORY["affected"],
            ],
        }
        advisory = parse_advisory(raw)
        assert len(advisory.affected) == 1
        assert advisory.affected[0].ecosystem is Ecosystem.PYPI

    def test_tolerates_missing_optional_fields(self) -> None:
        advisory = parse_advisory({"id": "OSV-1"})
        assert advisory.id == "OSV-1"
        assert advisory.summary == ""
        assert advisory.affected == []


def _dep(name: str, version: str) -> Dependency:
    return Dependency(
        name=name,
        ecosystem=Ecosystem.PYPI,
        version=version,
        manifest_path="requirements.txt",
    )


class TestQueryDependencies:
    async def test_returns_empty_without_dependencies(self) -> None:
        async with OSVClient() as osv:
            assert await osv.query_dependencies([]) == {}

    @respx.mock
    async def test_maps_hits_back_to_dependencies(self) -> None:
        respx.post(f"{OSV_API}/querybatch").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {"vulns": [{"id": "GHSA-aaaa"}, {"id": "GHSA-bbbb"}]},
                        {},
                    ]
                },
            )
        )
        async with OSVClient() as osv:
            hits = await osv.query_dependencies([_dep("requests", "2.19.1"), _dep("flask", "3.0.0")])

        assert hits == {"requests@2.19.1": ["GHSA-aaaa", "GHSA-bbbb"]}
        assert "flask@3.0.0" not in hits

    @respx.mock
    async def test_misaligned_result_count_raises(self) -> None:
        """OSV guarantees result order matches query order. If the counts disagree we
        cannot safely zip them, and silently mismatching an advisory to the wrong package
        would be worse than failing."""
        respx.post(f"{OSV_API}/querybatch").mock(
            return_value=httpx.Response(200, json={"results": [{}]})
        )
        async with OSVClient() as osv:
            with pytest.raises(OSVError, match="cannot safely align"):
                await osv.query_dependencies([_dep("a", "1.0"), _dep("b", "2.0")])

    @respx.mock
    async def test_client_error_is_not_retried(self) -> None:
        """A 400 means our request is malformed; retrying it only burns the budget."""
        route = respx.post(f"{OSV_API}/querybatch").mock(
            return_value=httpx.Response(400, text="bad request")
        )
        async with OSVClient(max_retries=3) as osv:
            with pytest.raises(OSVError, match="returned 400"):
                await osv.query_dependencies([_dep("a", "1.0")])

        assert route.call_count == 1


class TestGetAdvisory:
    @respx.mock
    async def test_fetches_and_parses(self) -> None:
        respx.get(f"{OSV_API}/vulns/GHSA-x84v-xcm2-53pg").mock(
            return_value=httpx.Response(200, json=RAW_ADVISORY)
        )
        async with OSVClient() as osv:
            advisory = await osv.get_advisory("GHSA-x84v-xcm2-53pg")
        assert advisory.id == "GHSA-x84v-xcm2-53pg"

    @respx.mock
    async def test_missing_advisory_raises(self) -> None:
        respx.get(f"{OSV_API}/vulns/NOPE").mock(return_value=httpx.Response(404))
        async with OSVClient() as osv:
            with pytest.raises(OSVError, match="not found"):
                await osv.get_advisory("NOPE")

    @respx.mock
    async def test_batch_fetch_skips_individual_failures(self) -> None:
        """One unavailable advisory must not abort a run over four hundred dependencies."""
        respx.get(f"{OSV_API}/vulns/GOOD").mock(
            return_value=httpx.Response(200, json=RAW_ADVISORY)
        )
        respx.get(f"{OSV_API}/vulns/BAD").mock(return_value=httpx.Response(404))

        async with OSVClient() as osv:
            advisories = await osv.get_advisories(["GOOD", "BAD"])

        assert len(advisories) == 1
