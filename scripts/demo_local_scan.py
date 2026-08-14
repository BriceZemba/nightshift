"""Zero-credential demo of the noise-reduction step.

Runs the real pipeline stages that need no authentication at all: manifest parsing, a live
query against OSV.dev, and static reachability analysis. No GitHub token, no Google Cloud
project, no model calls, no billing account.

This is the 180-to-6 claim, reproducible by anyone in about ten seconds::

    python scripts/demo_local_scan.py

The fixture below pins genuinely old, genuinely vulnerable releases, so the advisories are
real. Two of the four packages are actually used by the fixture source; two are not. That
asymmetry is the entire product.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nightshift.analysis.reachability import analyze_repository
from nightshift.sources.manifests import scan_files
from nightshift.sources.osv import OSVClient

MANIFEST = """\
requests==2.19.1
flask==0.12.2
pyyaml==5.1
urllib3==1.24.1
jinja2==2.10
"""

# Only requests and pyyaml are actually reached from this code. The rest are transitive
# passengers -- present in the manifest, never called.
SOURCES = {
    "src/api.py": (
        "import requests\n"
        "\n"
        "def fetch(url):\n"
        "    return requests.get(url, timeout=10)\n"
    ),
    "src/config.py": (
        "import yaml\n"
        "\n"
        "def load(path):\n"
        "    with open(path) as handle:\n"
        "        return yaml.load(handle)\n"
    ),
}

BOLD, DIM, RED, YELLOW, GREEN, RESET = (
    "\033[1m",
    "\033[90m",
    "\033[31m",
    "\033[33m",
    "\033[32m",
    "\033[0m",
)


async def main() -> int:
    scan = scan_files({"requirements.txt": MANIFEST})
    print(f"\n{BOLD}Manifest{RESET}")
    print(f"  {len(scan.dependencies)} pinned dependencies, {len(SOURCES)} Python files\n")

    print(f"{BOLD}Querying OSV.dev (live, no authentication){RESET}")
    async with OSVClient() as osv:
        hits = await osv.query_dependencies(scan.dependencies)

    total = sum(len(ids) for ids in hits.values())
    print(f"  {total} advisories match these versions, across {len(hits)} packages\n")

    print(f"{BOLD}Reachability{RESET}")
    reachable = unknown = dismissed = 0

    for key, ids in sorted(hits.items()):
        package = key.split("@")[0]
        result = analyze_repository(SOURCES, package)

        if result.reachability.value == "reachable":
            reachable += len(ids)
            marker = f"{RED}REACHABLE {RESET}"
        elif result.reachability.value == "unknown":
            unknown += len(ids)
            marker = f"{YELLOW}REVIEW    {RESET}"
        else:
            dismissed += len(ids)
            marker = f"{DIM}dismissed {RESET}"

        print(f"  {marker} {key:<20} {len(ids):>3} advisories")
        for site in result.call_paths[:2]:
            print(f"                                  {DIM}{site.file_path}:{site.line}"
                  f" -> {site.symbol}{RESET}")

    kept = reachable + unknown
    print(f"\n{BOLD}Result{RESET}")
    print(f"  {RED}{reachable}{RESET} reachable  ·  {YELLOW}{unknown}{RESET} needs review"
          f"  ·  {DIM}{dismissed} dismissed as noise{RESET}")
    if total:
        percent = 100 * (total - kept) / total
        print(
            f"\n  {BOLD}{total} advisories reduced to {kept}{RESET} "
            f"({percent:.0f}% of the backlog was noise)"
        )
    print(f"\n  {GREEN}No credentials used. No model calls. Cost: $0.00{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
