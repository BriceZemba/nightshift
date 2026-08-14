"""HTTP surface for Cloud Run.

Cloud Run services must serve HTTP on ``$PORT``, so the nightly batch cannot be the
container's command. This module is that surface, and it has two jobs:

* ``GET /`` is a read-only status page. It is the URL a judge, or anyone else, opens to see
  what the fleet did last night without needing credentials.
* ``POST /run`` triggers a run. Cloud Scheduler calls it at 03:00.

**The trigger endpoint is guarded and fails closed.** A run writes pull requests to real
repositories, so an unauthenticated trigger would let anyone on the internet make this
system act on your GitHub account. It requires a bearer token, and when no token is
configured the endpoint returns 503 rather than running: an unset secret disables the
endpoint instead of opening it.

FastAPI and uvicorn arrive with google-adk, so this adds no dependency.
"""

from __future__ import annotations

import asyncio
import os
import secrets
from typing import Any

import structlog
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import HTMLResponse

from nightshift.config import get_settings

log = structlog.get_logger(__name__)

app = FastAPI(title="Nightshift", docs_url=None, redoc_url=None)

#: Guards against two runs overlapping. A nightly job that is still going when the next one
#: fires would double the work and race on the claim ledger.
_run_lock = asyncio.Lock()

#: Summary of the most recent run, rendered by the status page.
_last_run: dict[str, Any] = {}


def _expected_token() -> str:
    return os.environ.get("NIGHTSHIFT_RUN_TOKEN", "")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness probe. Deliberately does no work and touches no credentials."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def status_page() -> str:
    """Public, read-only summary of the last run.

    Shows no repository names, diffs or tokens: this page is reachable by anyone with the
    URL, so it carries counts rather than contents.
    """
    settings = get_settings()

    if _last_run:
        rows = "".join(
            f"<tr><td>{label}</td><td class='n'>{value}</td></tr>"
            for label, value in [
                ("Repositories scanned", _last_run.get("repos_scanned", 0)),
                ("Advisories ingested", _last_run.get("advisories_ingested", 0)),
                ("Reachable", _last_run.get("findings_reachable", 0)),
                ("Dismissed as noise", _last_run.get("dismissed", 0)),
                ("Pull requests opened", _last_run.get("prs_opened", 0)),
                ("Awaiting human approval", _last_run.get("escalated", 0)),
                ("Cost (USD)", f"${_last_run.get('cost_usd', 0):.4f}"),
            ]
        )
        ingested = _last_run.get("advisories_ingested", 0)
        kept = _last_run.get("findings_reachable", 0)
        headline = (
            f"{ingested} advisories reduced to {kept}"
            if ingested
            else "No advisories matched"
        )
        body = f"<p class='headline'>{headline}</p><table>{rows}</table>"
    else:
        body = "<p class='headline'>No run recorded yet.</p>"

    mode = "DRY RUN" if settings.dry_run else "LIVE"

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Nightshift</title>
<style>
 body{{background:#0b0d12;color:#e6e8ee;font:16px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;
      margin:0;display:flex;align-items:center;justify-content:center;min-height:100vh}}
 main{{max-width:34rem;padding:2rem}}
 h1{{font-size:1.4rem;margin:0 0 .2rem}}
 .sub{{color:#8b93a7;margin:0 0 1.6rem}}
 .headline{{font-size:1.15rem;color:#7ee787;margin:0 0 1.2rem}}
 table{{border-collapse:collapse;width:100%}}
 td{{padding:.35rem 0;border-bottom:1px solid #1c2030}}
 td.n{{text-align:right;color:#79c0ff}}
 .mode{{margin-top:1.6rem;font-size:.85rem;color:#8b93a7}}
</style></head>
<body><main>
 <h1>Nightshift</h1>
 <p class="sub">An agent fleet that works the night shift on your dependency backlog.</p>
 {body}
 <p class="mode">Mode: {mode} &middot; Runs nightly at 03:00 via Cloud Scheduler</p>
</main></body></html>"""


@app.post("/run")
async def trigger_run(authorization: str = Header(default="")) -> Response:
    """Execute one run. Called by Cloud Scheduler.

    Returns 503 when no token is configured, so a misconfigured deployment refuses to run
    rather than accepting anonymous triggers.
    """
    expected = _expected_token()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="NIGHTSHIFT_RUN_TOKEN is not configured; the trigger is disabled.",
        )

    presented = authorization.removeprefix("Bearer ").strip()
    # Constant-time comparison: a plain != leaks the token a character at a time.
    if not secrets.compare_digest(presented, expected):
        log.warning("server.unauthorized_trigger")
        raise HTTPException(status_code=401, detail="Unauthorized")

    if _run_lock.locked():
        return Response(status_code=409, content="A run is already in progress.")

    async with _run_lock:
        summary = await _execute_run()

    return Response(status_code=200, content=str(summary), media_type="text/plain")


async def _execute_run() -> dict[str, Any]:
    """Run the pipeline once and record the summary for the status page."""
    from nightshift.llm import LLMClient
    from nightshift.run import NightshiftRun
    from nightshift.sources.github import GitHubClient
    from nightshift.sources.osv import OSVClient
    from nightshift.store import get_store

    settings = get_settings()

    async with GitHubClient(settings) as github, OSVClient() as osv:
        run = NightshiftRun(
            github=github,
            osv=osv,
            store=get_store(settings),
            llm=LLMClient(settings=settings),
            settings=settings,
        )
        record = await run.execute()

    global _last_run
    _last_run = record.model_dump(mode="json")
    return _last_run


def main() -> None:
    """Entrypoint for the container."""
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
