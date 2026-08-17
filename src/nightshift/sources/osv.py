"""OSV.dev client - the Watcher's primary advisory feed.

OSV is the right primary source here: no authentication, broad ecosystem coverage, and a
batch endpoint that turns "check 400 dependencies" into a handful of requests.

Two-step by design. ``querybatch`` returns only ids and modification times, which is
enough to diff against what we already stored; full records are then fetched only for
advisories we have not seen. On a nightly run against a stable dependency set that is
almost always zero detail fetches.

Note on error handling: transport failures propagate rather than being swallowed. Callers
inside the agent graph must let them surface - a broad ``except Exception:`` inside an ADK
2.x tool silently defeats the runtime's own retry. See docs/08-TECH-REFERENCE.md.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import httpx
import structlog

from nightshift.models import (
    Advisory,
    AffectedPackage,
    Dependency,
    Ecosystem,
    Severity,
    VersionRange,
)

log = structlog.get_logger(__name__)

OSV_API = "https://api.osv.dev/v1"

#: OSV accepts larger batches, but smaller chunks fail more cheaply and retry faster.
BATCH_SIZE = 100

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class OSVError(RuntimeError):
    """OSV returned something we cannot act on."""


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        log.warning("osv.unparseable_timestamp", value=value)
        return None


def _parse_ranges(raw_ranges: list[dict[str, Any]]) -> list[VersionRange]:
    """Flatten OSV's event-stream range format into explicit intervals.

    OSV encodes ranges as an ordered event list -- ``[{"introduced": "0"},
    {"fixed": "1.2.3"}]`` -- rather than as intervals. Each ``introduced`` opens a range
    and the next ``fixed``/``last_affected`` closes it.
    """
    parsed: list[VersionRange] = []
    for raw in raw_ranges:
        # GIT ranges carry commit SHAs rather than releases; the type has to survive so
        # callers can tell a version from a hash.
        range_type = raw.get("type", "ECOSYSTEM")
        current: VersionRange | None = None
        for event in raw.get("events", []):
            if "introduced" in event:
                if current is not None:
                    parsed.append(current)
                current = VersionRange(introduced=event["introduced"], type=range_type)
            elif "fixed" in event:
                if current is None:
                    current = VersionRange(type=range_type)
                current.fixed = event["fixed"]
                parsed.append(current)
                current = None
            elif "last_affected" in event:
                if current is None:
                    current = VersionRange(type=range_type)
                current.last_affected = event["last_affected"]
                parsed.append(current)
                current = None
        if current is not None:
            parsed.append(current)
    return parsed


def _parse_severity(raw: dict[str, Any]) -> list[Severity]:
    severities: list[Severity] = []
    for entry in raw.get("severity", []):
        severities.append(Severity(type=entry.get("type", ""), score=entry.get("score", "")))

    # OSV keeps a numeric score in the loosely-specified database_specific blob when the
    # source database supplies one. CVSS vector strings are left to NVD enrichment.
    db_specific = raw.get("database_specific") or {}
    numeric = db_specific.get("cvss_score") or db_specific.get("score")
    if isinstance(numeric, (int, float)) and severities:
        severities[0].numeric = float(numeric)

    return severities


def parse_advisory(raw: dict[str, Any]) -> Advisory:
    """Convert an OSV vulnerability record into our normalized form.

    Unknown ecosystems are skipped rather than raising: OSV covers ecosystems we do not
    support, and one unfamiliar entry should not discard an otherwise usable advisory.
    """
    affected: list[AffectedPackage] = []
    for entry in raw.get("affected", []):
        package = entry.get("package") or {}
        try:
            ecosystem = Ecosystem(package.get("ecosystem", ""))
        except ValueError:
            continue
        affected.append(
            AffectedPackage(
                name=package.get("name", ""),
                ecosystem=ecosystem,
                ranges=_parse_ranges(entry.get("ranges", [])),
                versions=entry.get("versions", []),
            )
        )

    return Advisory(
        id=raw.get("id", ""),
        aliases=raw.get("aliases", []),
        summary=raw.get("summary", ""),
        details=raw.get("details", ""),
        published=_parse_dt(raw.get("published")),
        modified=_parse_dt(raw.get("modified")),
        severity=_parse_severity(raw),
        affected=affected,
        references=[r.get("url", "") for r in raw.get("references", []) if r.get("url")],
        screened=False,  # free-text fields are untrusted until Guardian clears them
    )


class OSVClient:
    """Async OSV.dev client with bounded retries.

    Usage::

        async with OSVClient() as osv:
            hits = await osv.query_dependencies(deps)
    """

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> OSVClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={"User-Agent": "nightshift/0.1 (+https://github.com/)"},
            )
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("OSVClient must be used as an async context manager")
        return self._client

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with exponential backoff on transient failures.

        Only transport errors and retryable status codes are retried. A 400 means our
        request is malformed and retrying it just wastes the budget.
        """
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = await self.client.post(f"{OSV_API}{path}", json=payload)
            except httpx.TransportError as exc:
                last_error = exc
            else:
                if response.status_code == 200:
                    return response.json()
                if response.status_code not in RETRYABLE_STATUS:
                    raise OSVError(
                        f"OSV {path} returned {response.status_code}: {response.text[:200]}"
                    )
                last_error = OSVError(f"OSV {path} returned {response.status_code}")

            if attempt < self._max_retries - 1:
                backoff = 2**attempt
                log.warning("osv.retry", path=path, attempt=attempt + 1, backoff=backoff)
                await asyncio.sleep(backoff)

        raise OSVError(f"OSV {path} failed after {self._max_retries} attempts") from last_error

    async def query_dependencies(
        self, dependencies: list[Dependency]
    ) -> dict[str, list[str]]:
        """Map each dependency to the advisory ids affecting its pinned version.

        Returns ``{"package@version": [advisory_id, ...]}``, including only entries with
        at least one hit. OSV guarantees result order matches query order, which is what
        lets us zip them back together.
        """
        if not dependencies:
            return {}

        hits: dict[str, list[str]] = {}

        for start in range(0, len(dependencies), BATCH_SIZE):
            chunk = dependencies[start : start + BATCH_SIZE]
            payload = {
                "queries": [
                    {
                        "package": {"name": dep.name, "ecosystem": dep.ecosystem.value},
                        "version": dep.version,
                    }
                    for dep in chunk
                ]
            }

            data = await self._post("/querybatch", payload)
            results = data.get("results", [])

            if len(results) != len(chunk):
                raise OSVError(
                    f"OSV returned {len(results)} results for {len(chunk)} queries; "
                    "cannot safely align advisories to dependencies"
                )

            for dep, result in zip(chunk, results, strict=True):
                ids = [v["id"] for v in result.get("vulns", []) if "id" in v]
                if ids:
                    hits[f"{dep.name}@{dep.version}"] = ids
                    log.info("osv.hit", package=dep.name, version=dep.version, count=len(ids))

        return hits

    async def get_advisory(self, advisory_id: str) -> Advisory:
        """Fetch one full advisory record."""
        for attempt in range(self._max_retries):
            try:
                response = await self.client.get(f"{OSV_API}/vulns/{advisory_id}")
            except httpx.TransportError as exc:
                if attempt == self._max_retries - 1:
                    raise OSVError(f"OSV vulns/{advisory_id} transport failure") from exc
            else:
                if response.status_code == 200:
                    return parse_advisory(response.json())
                if response.status_code == 404:
                    raise OSVError(f"Advisory {advisory_id} not found")
                if response.status_code not in RETRYABLE_STATUS:
                    raise OSVError(
                        f"OSV vulns/{advisory_id} returned {response.status_code}"
                    )

            if attempt < self._max_retries - 1:
                await asyncio.sleep(2**attempt)

        raise OSVError(f"OSV vulns/{advisory_id} failed after {self._max_retries} attempts")

    async def get_advisories(
        self, advisory_ids: list[str], *, concurrency: int = 8
    ) -> list[Advisory]:
        """Fetch many advisories concurrently, skipping individual failures.

        One unavailable advisory should not abort a nightly run over four hundred
        dependencies, so failures are logged and dropped rather than raised.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def fetch(advisory_id: str) -> Advisory | None:
            async with semaphore:
                try:
                    return await self.get_advisory(advisory_id)
                except OSVError as exc:
                    log.warning("osv.advisory_failed", advisory_id=advisory_id, error=str(exc))
                    return None

        results = await asyncio.gather(*(fetch(a) for a in advisory_ids))
        return [advisory for advisory in results if advisory is not None]
