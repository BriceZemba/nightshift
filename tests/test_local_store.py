"""File-backed store.

The claim ledger gets the most attention. ``try_claim_pr`` is the contract that keeps a
crash recovery, a retry, or an at-least-once redelivery from turning into a second pull
request on someone's repository - so the local backend has to honour it exactly as
Firestore's transaction does, not approximately.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from nightshift.config import Settings
from nightshift.models import (
    Advisory,
    Decision,
    Dependency,
    Ecosystem,
    Finding,
    FindingStatus,
    RunRecord,
)
from nightshift.store import get_store
from nightshift.store.local import LocalStore


@pytest.fixture
def store(tmp_path: Path) -> LocalStore:
    return LocalStore(tmp_path / "state")


def _finding(**overrides: object) -> Finding:
    base = {
        "id": "f1",
        "run_id": "r1",
        "advisory_id": "GHSA-xxxx",
        "repo": "me/myrepo",
        "dependency": Dependency(
            name="requests",
            ecosystem=Ecosystem.PYPI,
            version="2.19.1",
            manifest_path="requirements.txt",
        ),
    }
    base.update(overrides)
    return Finding(**base)  # type: ignore[arg-type]


class TestClaimLedger:
    async def test_first_claim_succeeds(self, store: LocalStore) -> None:
        assert await store.try_claim_pr("key-1", "f1") is True

    async def test_second_claim_fails(self, store: LocalStore) -> None:
        await store.try_claim_pr("key-1", "f1")
        assert await store.try_claim_pr("key-1", "f1") is False

    async def test_concurrent_claims_grant_exactly_one(self, store: LocalStore) -> None:
        """The whole point. O_CREAT|O_EXCL is atomic at the OS level, so exactly one of
        twenty simultaneous callers may open the pull request."""
        results = await asyncio.gather(
            *(store.try_claim_pr("shared-key", f"f{i}") for i in range(20))
        )
        assert sum(results) == 1

    async def test_release_allows_a_retry(self, store: LocalStore) -> None:
        await store.try_claim_pr("key-1", "f1")
        await store.release_pr_claim("key-1")
        assert await store.try_claim_pr("key-1", "f1") is True

    async def test_releasing_an_unclaimed_key_is_harmless(self, store: LocalStore) -> None:
        await store.release_pr_claim("never-claimed")

    async def test_distinct_keys_are_independent(self, store: LocalStore) -> None:
        assert await store.try_claim_pr("a", "f1") is True
        assert await store.try_claim_pr("b", "f2") is True

    async def test_claims_survive_a_new_store_instance(self, tmp_path: Path) -> None:
        """A restarted process must not re-open a pull request it already opened."""
        first = LocalStore(tmp_path / "state")
        await first.try_claim_pr("key-1", "f1")

        second = LocalStore(tmp_path / "state")
        assert await second.try_claim_pr("key-1", "f1") is False


class TestRoundTrips:
    async def test_run_record(self, store: LocalStore) -> None:
        run = RunRecord(id="run-1", advisories_ingested=60)
        await store.start_run(run)
        await store.finish_run(run)

        loaded = await store.get_run("run-1")
        assert loaded is not None
        assert loaded.advisories_ingested == 60
        assert loaded.started_at is not None
        assert loaded.finished_at is not None

    async def test_advisory(self, store: LocalStore) -> None:
        await store.upsert_advisory(Advisory(id="GHSA-1", summary="Leak", screened=True))
        loaded = await store.get_advisory("GHSA-1")
        assert loaded is not None
        assert loaded.screened is True

    async def test_finding_with_nested_models(self, store: LocalStore) -> None:
        finding = _finding(status=FindingStatus.TRIAGED)
        await store.upsert_finding(finding)

        loaded = await store.get_finding("f1")
        assert loaded is not None
        assert loaded.status is FindingStatus.TRIAGED
        assert loaded.dependency.name == "requests"

    async def test_findings_are_filtered_by_run(self, store: LocalStore) -> None:
        await store.upsert_finding(_finding(id="a", run_id="run-1"))
        await store.upsert_finding(_finding(id="b", run_id="run-2"))

        assert len(await store.findings_for_run("run-1")) == 1

    async def test_decisions_are_ordered_by_time(self, store: LocalStore) -> None:
        for agent in ("watcher", "triager", "patcher"):
            await store.record_decision(
                Decision(run_id="r1", finding_id="f1", agent=agent, action="x")
            )

        decisions = await store.decisions_for_finding("f1")
        assert [d.agent for d in decisions] == ["watcher", "triager", "patcher"]

    async def test_decisions_are_filtered_by_finding(self, store: LocalStore) -> None:
        await store.record_decision(Decision(run_id="r1", finding_id="f1", agent="a", action="x"))
        await store.record_decision(Decision(run_id="r1", finding_id="f2", agent="b", action="y"))

        assert len(await store.decisions_for_finding("f1")) == 1

    async def test_known_advisory_ids(self, store: LocalStore) -> None:
        await store.upsert_advisory(Advisory(id="GHSA-1"))
        assert await store.known_advisory_ids(["GHSA-1", "GHSA-2"]) == {"GHSA-1"}

    async def test_missing_documents_return_none(self, store: LocalStore) -> None:
        assert await store.get_run("nope") is None
        assert await store.get_advisory("nope") is None
        assert await store.get_finding("nope") is None


class TestResilience:
    async def test_corrupt_collection_does_not_abort_the_next_run(
        self, tmp_path: Path
    ) -> None:
        """A file truncated by an interrupted run must not stop the following one."""
        store = LocalStore(tmp_path / "state")
        await store.upsert_advisory(Advisory(id="GHSA-1"))

        (tmp_path / "state" / "advisories.json").write_text("{ truncated", encoding="utf-8")

        assert await store.get_advisory("GHSA-1") is None
        await store.upsert_advisory(Advisory(id="GHSA-2"))
        assert await store.get_advisory("GHSA-2") is not None


class TestFactory:
    def test_local_backend_needs_no_google_cloud(self, tmp_path: Path) -> None:
        settings = Settings(
            NIGHTSHIFT_STORE="local", NIGHTSHIFT_STORE_PATH=str(tmp_path / "s")
        )
        assert isinstance(get_store(settings), LocalStore)

    def test_invalid_backend_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="NIGHTSHIFT_STORE"):
            Settings(NIGHTSHIFT_STORE="postgres")
