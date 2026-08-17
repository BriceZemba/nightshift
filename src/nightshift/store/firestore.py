"""Firestore persistence - agent state, audit log, and the idempotency ledger.

Firestore rather than Cloud SQL for three reasons: it has a real free tier (Cloud SQL has
none), it supports native KNN vector search (Vertex AI Vector Search bills index-serving
nodes around the clock even when idle), and Cloud Run's filesystem is ephemeral, so a
local SQLite session would silently lose state whenever an instance recycled.

Collections
-----------
``runs/{run_id}``              one nightly run; the demo's numbers come from here
``advisories/{advisory_id}``   normalized advisories, cached across runs
``findings/{finding_id}``      one advisory as it applies to one repo - the unit of work
``decisions/{auto_id}``        append-only audit log; every agent hop writes one
``pr_locks/{idempotency_key}`` the ledger that stops a retry opening a duplicate PR
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from nightshift.models import Advisory, Decision, Finding, RunRecord

log = structlog.get_logger(__name__)

RUNS = "runs"
ADVISORIES = "advisories"
FINDINGS = "findings"
DECISIONS = "decisions"
PR_LOCKS = "pr_locks"


def _now() -> datetime:
    return datetime.now(UTC)


class Store:
    """Async Firestore accessor.

    Every write is idempotent by construction: documents are keyed on deterministic ids
    derived from the work itself, so replaying a partially-completed run overwrites rather
    than duplicates. That is what makes the pipeline resumable after a crash - which is
    the property that matters when the whole system runs unattended at 03:00.
    """

    def __init__(self, client: firestore.AsyncClient | None = None) -> None:
        self._client = client or firestore.AsyncClient()

    @property
    def client(self) -> firestore.AsyncClient:
        return self._client

    # --- runs ---------------------------------------------------------------

    async def start_run(self, run: RunRecord) -> None:
        run.started_at = run.started_at or _now()
        await self._client.collection(RUNS).document(run.id).set(run.model_dump())
        log.info("run.started", run_id=run.id, dry_run=run.dry_run)

    async def finish_run(self, run: RunRecord) -> None:
        run.finished_at = _now()
        await self._client.collection(RUNS).document(run.id).set(run.model_dump())
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
        snapshot = await self._client.collection(RUNS).document(run_id).get()
        return RunRecord(**snapshot.to_dict()) if snapshot.exists else None

    # --- advisories ---------------------------------------------------------

    async def upsert_advisory(self, advisory: Advisory) -> None:
        await (
            self._client.collection(ADVISORIES)
            .document(advisory.id)
            .set(advisory.model_dump())
        )

    async def get_advisory(self, advisory_id: str) -> Advisory | None:
        snapshot = await self._client.collection(ADVISORIES).document(advisory_id).get()
        return Advisory(**snapshot.to_dict()) if snapshot.exists else None

    async def known_advisory_ids(self, advisory_ids: list[str]) -> set[str]:
        """Which of these advisories we have already stored.

        Lets the Watcher fetch full records only for genuinely new advisories. On a stable
        dependency set the nightly delta is usually empty.
        """
        if not advisory_ids:
            return set()

        known: set[str] = set()
        collection = self._client.collection(ADVISORIES)

        # get_all is capped per call, so chunk it.
        for start in range(0, len(advisory_ids), 100):
            chunk = advisory_ids[start : start + 100]
            refs = [collection.document(a) for a in chunk]
            async for snapshot in self._client.get_all(refs):
                if snapshot.exists:
                    known.add(snapshot.id)

        return known

    # --- findings -----------------------------------------------------------

    async def upsert_finding(self, finding: Finding) -> None:
        now = _now()
        finding.created_at = finding.created_at or now
        finding.updated_at = now
        await (
            self._client.collection(FINDINGS)
            .document(finding.id)
            .set(finding.model_dump())
        )

    async def get_finding(self, finding_id: str) -> Finding | None:
        snapshot = await self._client.collection(FINDINGS).document(finding_id).get()
        return Finding(**snapshot.to_dict()) if snapshot.exists else None

    async def findings_for_run(self, run_id: str) -> list[Finding]:
        # FieldFilter rather than positional arguments: the positional form is deprecated
        # and warns on every call.
        query = self._client.collection(FINDINGS).where(
            filter=FieldFilter("run_id", "==", run_id)
        )
        return [Finding(**doc.to_dict()) async for doc in query.stream()]

    # --- audit log ----------------------------------------------------------

    async def record_decision(self, decision: Decision) -> None:
        """Append one audit entry.

        The chain of these renders into the pull request body. It is the difference
        between a patch a maintainer has to re-derive by hand and one they can actually
        check.
        """
        decision.at = decision.at or _now()
        await self._client.collection(DECISIONS).add(decision.model_dump())

    async def decisions_for_finding(self, finding_id: str) -> list[Decision]:
        query = self._client.collection(DECISIONS).where(
            filter=FieldFilter("finding_id", "==", finding_id)
        )
        decisions = [Decision(**doc.to_dict()) async for doc in query.stream()]
        return sorted(decisions, key=lambda d: d.at or datetime.min.replace(tzinfo=UTC))

    # --- idempotency --------------------------------------------------------

    async def try_claim_pr(self, idempotency_key: str, finding_id: str) -> bool:
        """Atomically claim the right to open one pull request.

        Returns ``True`` exactly once per key. Every later caller gets ``False``.

        This is the safety property that makes the whole system re-runnable. A crashed run
        gets restarted, an at-least-once Pub/Sub delivery redelivers, a retry fires twice,
        and the maintainer still sees exactly one pull request. Without it, "resumable"
        would mean "spams the repository".
        """
        ref = self._client.collection(PR_LOCKS).document(idempotency_key)
        transaction = self._client.transaction()

        @firestore.async_transactional
        async def claim(tx: firestore.AsyncTransaction) -> bool:
            snapshot = await ref.get(transaction=tx)
            if snapshot.exists:
                return False
            tx.set(
                ref,
                {
                    "finding_id": finding_id,
                    "claimed_at": _now(),
                },
            )
            return True

        claimed: bool = await claim(transaction)
        if not claimed:
            log.info("pr.already_claimed", key=idempotency_key, finding_id=finding_id)
        return claimed

    async def release_pr_claim(self, idempotency_key: str) -> None:
        """Release a claim after a failure, so a later run may retry the action.

        Only call this when the pull request definitively was **not** created. Releasing a
        claim for an action that did happen reintroduces the duplicate it prevents.
        """
        await self._client.collection(PR_LOCKS).document(idempotency_key).delete()
        log.info("pr.claim_released", key=idempotency_key)
