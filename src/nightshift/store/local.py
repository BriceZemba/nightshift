"""File-backed store - the offline development path.

Implements the same interface as the Firestore store, so the full pipeline runs with no
Google Cloud project, no billing account and no network. That matters for three reasons:
development is not blocked on billing activation, CI can exercise the whole run, and the
demo has a fallback if a cloud dependency misbehaves on the night.

The interesting part is :meth:`LocalStore.try_claim_pr`. Firestore gives atomicity through
a transaction; here it comes from ``O_CREAT | O_EXCL``, which the operating system
guarantees to be atomic - the file is created by exactly one caller, and everyone else
gets ``FileExistsError``. Same contract, same guarantee, no server.

Not intended for production: writes are not concurrent-safe across processes beyond the
claim ledger, and the whole dataset is rewritten per collection on each save.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from nightshift.models import Advisory, Decision, Finding, RunRecord

log = structlog.get_logger(__name__)

DEFAULT_ROOT = Path(".nightshift")


def _now() -> datetime:
    return datetime.now(UTC)


def _serialize(value: Any) -> Any:
    """JSON-encode datetimes, which pydantic leaves as objects in ``model_dump()``."""
    if isinstance(value, datetime):
        return {"__datetime__": value.isoformat()}
    raise TypeError(f"not JSON serializable: {type(value)}")


def _deserialize(obj: dict[str, Any]) -> Any:
    if "__datetime__" in obj:
        return datetime.fromisoformat(obj["__datetime__"])
    return obj


class LocalStore:
    """A drop-in replacement for :class:`nightshift.store.firestore.Store`."""

    def __init__(self, root: Path | str = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "pr_locks").mkdir(exist_ok=True)

    # --- persistence helpers ------------------------------------------------

    def _path(self, collection: str) -> Path:
        return self.root / f"{collection}.json"

    def _load(self, collection: str) -> dict[str, Any]:
        path = self._path(collection)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"), object_hook=_deserialize)
        except json.JSONDecodeError:
            # A truncated file from an interrupted run should not abort the next one.
            log.warning("local_store.corrupt_collection", collection=collection)
            return {}

    def _save(self, collection: str, data: dict[str, Any]) -> None:
        """Write via a temporary file and replace, so an interrupt cannot truncate."""
        path = self._path(collection)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(data, default=_serialize, indent=2), encoding="utf-8"
        )
        tmp.replace(path)

    def _put(self, collection: str, key: str, value: dict[str, Any]) -> None:
        data = self._load(collection)
        data[key] = value
        self._save(collection, data)

    # --- runs ---------------------------------------------------------------

    async def start_run(self, run: RunRecord) -> None:
        run.started_at = run.started_at or _now()
        self._put("runs", run.id, run.model_dump())
        log.info("run.started", run_id=run.id, dry_run=run.dry_run, store="local")

    async def finish_run(self, run: RunRecord) -> None:
        run.finished_at = _now()
        self._put("runs", run.id, run.model_dump())
        log.info(
            "run.finished",
            run_id=run.id,
            advisories=run.advisories_ingested,
            reachable=run.findings_reachable,
            prs=run.prs_opened,
            escalated=run.escalated,
            cost_usd=round(run.cost_usd, 4),
        )

    async def get_run(self, run_id: str) -> RunRecord | None:
        raw = self._load("runs").get(run_id)
        return RunRecord(**raw) if raw else None

    # --- advisories ---------------------------------------------------------

    async def upsert_advisory(self, advisory: Advisory) -> None:
        self._put("advisories", advisory.id, advisory.model_dump())

    async def get_advisory(self, advisory_id: str) -> Advisory | None:
        raw = self._load("advisories").get(advisory_id)
        return Advisory(**raw) if raw else None

    async def known_advisory_ids(self, advisory_ids: list[str]) -> set[str]:
        stored = self._load("advisories")
        return {a for a in advisory_ids if a in stored}

    # --- findings -----------------------------------------------------------

    async def upsert_finding(self, finding: Finding) -> None:
        now = _now()
        finding.created_at = finding.created_at or now
        finding.updated_at = now
        self._put("findings", finding.id, finding.model_dump())

    async def get_finding(self, finding_id: str) -> Finding | None:
        raw = self._load("findings").get(finding_id)
        return Finding(**raw) if raw else None

    async def findings_for_run(self, run_id: str) -> list[Finding]:
        return [
            Finding(**raw)
            for raw in self._load("findings").values()
            if raw.get("run_id") == run_id
        ]

    # --- audit log ----------------------------------------------------------

    async def record_decision(self, decision: Decision) -> None:
        decision.at = decision.at or _now()
        data = self._load("decisions")
        data[str(len(data))] = decision.model_dump()
        self._save("decisions", data)

    async def decisions_for_finding(self, finding_id: str) -> list[Decision]:
        decisions = [
            Decision(**raw)
            for raw in self._load("decisions").values()
            if raw.get("finding_id") == finding_id
        ]
        return sorted(decisions, key=lambda d: d.at or datetime.min.replace(tzinfo=UTC))

    # --- idempotency --------------------------------------------------------

    async def try_claim_pr(self, idempotency_key: str, finding_id: str) -> bool:
        """Claim the right to open one pull request. Returns ``True`` exactly once.

        ``O_CREAT | O_EXCL`` is atomic at the OS level: exactly one caller creates the
        file and every other gets ``FileExistsError``. That is the same guarantee
        Firestore's transaction provides, which is what makes this a faithful stand-in
        rather than an approximation.
        """
        lock = self.root / "pr_locks" / f"{idempotency_key}.json"

        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            log.info("pr.already_claimed", key=idempotency_key, finding_id=finding_id)
            return False

        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                {"finding_id": finding_id, "claimed_at": _now().isoformat()}, handle
            )
        return True

    async def release_pr_claim(self, idempotency_key: str) -> None:
        """Release a claim after a failure, so a later run may retry.

        Only call this when the pull request definitively was **not** created.
        """
        (self.root / "pr_locks" / f"{idempotency_key}.json").unlink(missing_ok=True)
        log.info("pr.claim_released", key=idempotency_key)
