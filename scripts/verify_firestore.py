"""Exercise the Firestore backend against a real database.

The Firestore store was written early and never run. Its most important property, that
``try_claim_pr`` grants the right to open a pull request exactly once, is the thing keeping
a crash recovery or an at-least-once redelivery from becoming a duplicate pull request on
someone's repository. A transaction that is merely *believed* to be atomic is not a safety
property.

This proves it end to end against whatever Firestore the environment points at, including
the free Spark-plan database that needs no billing account.

    python scripts/verify_firestore.py

Requires GOOGLE_CLOUD_PROJECT and application default credentials, or
GOOGLE_APPLICATION_CREDENTIALS pointing at a service account key.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from google.api_core.exceptions import NotFound, PermissionDenied
from google.auth.exceptions import DefaultCredentialsError

from nightshift.models import (
    Advisory,
    Decision,
    Dependency,
    Ecosystem,
    Finding,
    FindingStatus,
    RunRecord,
)

GREEN, RED, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[90m", "\033[1m", "\033[0m"

CONCURRENT_CLAIMERS = 25

_failures = 0


def check(label: str, passed: bool, detail: str = "") -> None:
    global _failures
    mark = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    if not passed:
        _failures += 1
    suffix = f"  {DIM}{detail}{RESET}" if detail else ""
    print(f"  {mark}  {label}{suffix}")


async def main() -> int:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if not project:
        print(f"{RED}GOOGLE_CLOUD_PROJECT is not set.{RESET}")
        return 2

    print(f"\n{BOLD}Firestore backend, project {project}{RESET}\n")

    from nightshift.store.firestore import Store

    # Credentials are resolved when the client is constructed, not on first use, so this
    # is where a missing key surfaces.
    try:
        store = Store()
    except DefaultCredentialsError:
        print(f"{RED}No credentials found.{RESET}\n")
        print("  Point GOOGLE_APPLICATION_CREDENTIALS at your service account key:")
        print(f"  {DIM}$env:GOOGLE_APPLICATION_CREDENTIALS=\"$HOME\\.secrets\\key.json\"{RESET}")
        print("\n  Get one from the Firebase console:")
        print(f"  {DIM}Project settings -> Service accounts -> Generate new private key{RESET}")
        return 2

    run_id = f"verify-{uuid.uuid4().hex[:8]}"
    repo = "verify/example"

    dependency = Dependency(
        name="requests",
        ecosystem=Ecosystem.PYPI,
        version="2.19.1",
        manifest_path="requirements.txt",
    )
    finding = Finding(
        id=f"{run_id}-f1",
        run_id=run_id,
        advisory_id="GHSA-verify",
        repo=repo,
        dependency=dependency,
        status=FindingStatus.TRIAGED,
    )

    # --- reachability -------------------------------------------------------
    # A first write is the cheapest way to tell "no database" apart from "no permission",
    # and the two have completely different fixes. The raw gRPC traceback buries both.
    record = RunRecord(id=run_id, advisories_ingested=60, findings_reachable=16)
    try:
        await store.start_run(record)
    except NotFound:
        print(f"{RED}No Firestore database exists in project {project}.{RESET}\n")
        print("  Create one in the Firebase console, which needs no billing account:")
        print(f"  {DIM}https://console.firebase.google.com{RESET}")
        print("    Build -> Firestore Database -> Create database")
        print(f"    {BOLD}Native mode{RESET}, not Datastore mode. The client only speaks Native.")
        return 2
    except PermissionDenied:
        print(f"{RED}Authenticated, but not allowed to write to Firestore.{RESET}\n")
        print("  The service account needs roles/datastore.user on the project.")
        print(f"  {DIM}Firebase console -> Project settings -> Service accounts{RESET}")
        return 2

    # --- round trips --------------------------------------------------------
    await store.finish_run(record)
    loaded_run = await store.get_run(run_id)
    check(
        "run record round-trips",
        loaded_run is not None and loaded_run.advisories_ingested == 60,
    )

    await store.upsert_advisory(
        Advisory(id=f"GHSA-{run_id}", summary="verification", screened=True)
    )
    loaded_advisory = await store.get_advisory(f"GHSA-{run_id}")
    check(
        "advisory round-trips with nested fields",
        loaded_advisory is not None and loaded_advisory.screened is True,
    )

    await store.upsert_finding(finding)
    loaded_finding = await store.get_finding(finding.id)
    check(
        "finding round-trips with nested models",
        loaded_finding is not None and loaded_finding.dependency.name == "requests",
    )

    for agent in ("watcher", "triager", "patcher"):
        await store.record_decision(
            Decision(run_id=run_id, finding_id=finding.id, agent=agent, action="verify")
        )
    decisions = await store.decisions_for_finding(finding.id)
    check(
        "audit trail is ordered",
        [d.agent for d in decisions] == ["watcher", "triager", "patcher"],
        f"{len(decisions)} entries",
    )

    found = await store.findings_for_run(run_id)
    check("findings query by run works", len(found) == 1)

    # --- the one that matters ----------------------------------------------
    key = f"verify-claim-{uuid.uuid4().hex[:8]}"
    results = await asyncio.gather(
        *(store.try_claim_pr(key, f"f{i}") for i in range(CONCURRENT_CLAIMERS))
    )
    granted = sum(results)
    check(
        f"exactly one of {CONCURRENT_CLAIMERS} concurrent claims is granted",
        granted == 1,
        f"granted={granted}",
    )

    check("a repeated claim is refused", await store.try_claim_pr(key, "again") is False)

    await store.release_pr_claim(key)
    check(
        "releasing allows a retry",
        await store.try_claim_pr(key, "retry") is True,
    )

    # --- clean up -----------------------------------------------------------
    await store.release_pr_claim(key)
    for collection, doc_id in (
        ("runs", run_id),
        ("advisories", f"GHSA-{run_id}"),
        ("findings", finding.id),
    ):
        await store.client.collection(collection).document(doc_id).delete()
    from google.cloud.firestore_v1.base_query import FieldFilter

    stale = await store.client.collection("decisions").where(
        filter=FieldFilter("run_id", "==", run_id)
    ).get()
    for decision in stale:
        await decision.reference.delete()

    print(f"\n  {DIM}verification documents removed{RESET}")

    if _failures:
        print(f"\n{RED}{BOLD}{_failures} check(s) failed.{RESET}\n")
        return 1

    print(f"\n{GREEN}{BOLD}Firestore backend verified.{RESET}")
    print(f"{DIM}  Open the Firestore console to see it: "
          f"https://console.cloud.google.com/firestore/databases/-default-/data?project={project}{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
