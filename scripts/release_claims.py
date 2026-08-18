"""Release pull-request claims so a repository can be processed again.

The idempotency ledger grants the right to open a pull request exactly once per
(advisory, repo, package), and it deliberately survives the pull request being closed.
Closing a PR means "not this change", not "ask me again tomorrow", so re-proposing it
automatically would be the nagging behaviour the whole project exists to avoid.

That is correct in production and inconvenient in exactly one situation: rehearsing a demo,
where you want the same run to produce the same pull requests several times. This releases
the claims for one repository so the next run starts clean.

    python scripts/release_claims.py BriceZemba/nightshift-demo

Reads NIGHTSHIFT_STORE to decide which backend to clear, so it follows whatever the run
itself would use.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nightshift.config import get_settings

GREEN, RED, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[90m", "\033[1m", "\033[0m"


async def release_local(repo: str, root: Path) -> int:
    """Clear file-backed claims whose finding belongs to this repository."""
    import json

    locks = root / "pr_locks"
    if not locks.exists():
        print(f"{DIM}no local claims at {locks}{RESET}")
        return 0

    findings_file = root / "findings.json"
    findings = (
        json.loads(findings_file.read_text(encoding="utf-8"))
        if findings_file.exists()
        else {}
    )
    ours = {fid for fid, f in findings.items() if f.get("repo") == repo}

    released = 0
    for lock in locks.glob("*.json"):
        try:
            claim = json.loads(lock.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # An unreadable or unattributed claim is cleared too: a lock nobody can explain is
        # a lock that will silently block the next run.
        if not ours or claim.get("finding_id") in ours:
            lock.unlink(missing_ok=True)
            released += 1

    return released


async def release_firestore(repo: str) -> int:
    """Clear Firestore claims whose finding belongs to this repository."""
    from google.cloud.firestore_v1.base_query import FieldFilter

    from nightshift.store.firestore import Store

    store = Store()

    findings = await (
        store.client.collection("findings").where(filter=FieldFilter("repo", "==", repo)).get()
    )
    ours = {doc.id for doc in findings}

    released = 0
    async for lock in store.client.collection("pr_locks").stream():
        claim = lock.to_dict() or {}
        if not ours or claim.get("finding_id") in ours:
            await lock.reference.delete()
            released += 1

    return released


async def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/release_claims.py owner/name", file=sys.stderr)
        return 2

    repo = sys.argv[1]
    settings = get_settings()

    print(f"\n{BOLD}Releasing claims for {repo}{RESET}")
    print(f"{DIM}backend: {settings.store_backend}{RESET}\n")

    if settings.store_backend == "firestore":
        released = await release_firestore(repo)
    else:
        released = await release_local(repo, Path(settings.local_store_path))

    if released:
        print(f"{GREEN}released {released} claim(s){RESET}")
        print(f"{DIM}the next run will open these pull requests again{RESET}\n")
    else:
        print(f"{DIM}nothing to release{RESET}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
