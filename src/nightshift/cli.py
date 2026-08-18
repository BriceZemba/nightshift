"""Command-line entrypoint.

Cloud Scheduler invokes the deployed service; this exists so the same pipeline can be run
and watched locally, which is what makes the run demonstrable and debuggable.

``scan`` is the important one for a first look: it reads repositories, queries OSV against
real advisory data, and runs static reachability - no model calls, no credentials beyond a
GitHub token, no cloud project. It is the 180-to-6 number on its own.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import structlog

from nightshift.analysis.reachability import analyze_repository
from nightshift.config import get_settings
from nightshift.policy import PolicyViolation
from nightshift.sources.github import GitHubClient
from nightshift.sources.manifests import scan_files
from nightshift.sources.osv import OSVClient


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        format="%(message)s", stream=sys.stdout, level=logging.DEBUG if verbose else logging.INFO
    )
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.DEBUG if verbose else logging.INFO
        ),
    )


async def _scan(args: argparse.Namespace) -> int:
    """Read repositories, query OSV, and report reachability - without any model calls."""
    settings = get_settings()
    targets = args.repos or settings.repo_allowlist

    if not targets:
        print(
            "No repositories specified. Pass --repos owner/name or set "
            "NIGHTSHIFT_REPO_ALLOWLIST.",
            file=sys.stderr,
        )
        return 2

    totals = {"deps": 0, "advisories": 0, "reachable": 0, "dismissed": 0, "unknown": 0}

    async with GitHubClient(settings) as github, OSVClient() as osv:
        for full_name in targets:
            print(f"\n\033[1m{full_name}\033[0m")

            repo = await github.get_repo(full_name)
            manifests = await github.fetch_manifests(full_name, repo.default_branch)
            scan = scan_files(manifests)
            sources = await github.fetch_sources(full_name, repo.default_branch)

            print(
                f"  {len(scan.dependencies)} pinned dependencies "
                f"({len(scan.unresolved)} unresolved), {len(sources)} Python files"
            )
            totals["deps"] += len(scan.dependencies)

            if not scan.dependencies:
                continue

            hits = await osv.query_dependencies(scan.dependencies)
            advisory_ids = sorted({a for ids in hits.values() for a in ids})
            print(f"  {len(advisory_ids)} advisories match the dependency versions")
            totals["advisories"] += len(advisory_ids)

            by_key = {f"{d.name}@{d.version}": d for d in scan.dependencies}

            for key, ids in sorted(hits.items()):
                dependency = by_key[key]
                result = analyze_repository(sources, dependency.name)

                if result.reachability.value == "reachable":
                    totals["reachable"] += len(ids)
                    marker, colour = "REACHABLE", "\033[31m"
                elif result.reachability.value == "unknown":
                    totals["unknown"] += len(ids)
                    marker, colour = "UNKNOWN", "\033[33m"
                else:
                    totals["dismissed"] += len(ids)
                    if not args.verbose:
                        continue
                    marker, colour = "not reachable", "\033[90m"

                print(f"    {colour}{marker}\033[0m  {key}  ({', '.join(ids[:3])})")
                if result.call_paths:
                    for site in result.call_paths[:3]:
                        print(f"        {site.file_path}:{site.line} -> {site.symbol}")

    print("\n\033[1mSummary\033[0m")
    print(f"  dependencies scanned : {totals['deps']}")
    print(f"  advisories matched   : {totals['advisories']}")
    print(f"  \033[31mreachable\033[0m            : {totals['reachable']}")
    print(f"  \033[33mneeds review\033[0m         : {totals['unknown']}")
    print(f"  \033[90mdismissed as noise\033[0m   : {totals['dismissed']}")

    if totals["advisories"]:
        kept = totals["reachable"] + totals["unknown"]
        print(
            f"\n  {totals['advisories']} advisories reduced to {kept} worth a human's "
            f"attention."
        )

    return 0


async def _run(args: argparse.Namespace) -> int:
    """Execute the full pipeline: triage, patch, verify, report."""
    from nightshift.llm import LLMClient
    from nightshift.run import NightshiftRun
    from nightshift.store import get_store

    settings = get_settings()

    if args.live:
        # Arming the system is an explicit, deliberate act. Never a default, and never
        # inferred from anything else.
        settings = settings.model_copy(update={"dry_run": False})
        print("\033[31mLIVE MODE - pull requests will be opened.\033[0m")
    else:
        print("\033[33mDRY RUN - no pull requests will be opened. Use --live to arm.\033[0m")

    async with GitHubClient(settings) as github, OSVClient() as osv:
        run = NightshiftRun(
            github=github,
            osv=osv,
            store=get_store(settings),
            llm=LLMClient(settings=settings),
            settings=settings,
        )
        record = await run.execute()

    print("\n\033[1mRun complete\033[0m")
    print(f"  repositories   : {record.repos_scanned}")
    print(f"  advisories     : {record.advisories_ingested}")
    print(f"  reachable      : {record.findings_reachable}")
    print(f"  pull requests  : {record.prs_opened}")
    print(f"  escalated      : {record.escalated}")
    print(f"  dismissed      : {record.dismissed}")
    if record.already_reported:
        print(f"  already open   : {record.already_reported}  (claimed by an earlier run)")
    if record.failed:
        print(f"  [31mfailed[0m         : {record.failed}")
    print(f"  cost           : ${record.cost_usd:.4f}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nightshift",
        description="An agent fleet that works the night shift on your dependency backlog.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser(
        "scan", help="Read repositories and report reachability. No model calls."
    )
    scan.add_argument("--repos", nargs="*", help="owner/name (defaults to the allowlist)")

    run = subparsers.add_parser("run", help="Execute the full pipeline.")
    run.add_argument(
        "--live",
        action="store_true",
        help="Actually open pull requests. Without this, the run is a dry run.",
    )

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    handler = {"scan": _scan, "run": _run}[args.command]

    try:
        return asyncio.run(handler(args))
    except PolicyViolation as exc:
        print(f"\n\033[31mPolicy violation:\033[0m {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
