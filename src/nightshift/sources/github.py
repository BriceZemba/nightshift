"""GitHub access - read repository contents, and open pull requests.

Every write path funnels through :meth:`GitHubClient.open_pull_request`, which re-checks
the allowlist immediately before acting. The check exists at configuration time too, but
enforcing it again at the point of action means a model that invents a plausible-looking
repository name cannot route around it.

Reads use the git trees and blobs API rather than cloning: one tree request lists the
repository, and only the files actually needed are fetched.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass

import httpx
import structlog

from nightshift.config import Settings, get_settings
from nightshift.models import Repo
from nightshift.policy import assert_repo_allowed

log = structlog.get_logger(__name__)

GITHUB_API = "https://api.github.com"
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

#: Source extensions worth pulling for reachability analysis. Anything else is noise.
ANALYZABLE_SUFFIXES = (".py",)

#: Directories never worth analyzing - vendored or generated code is not "your code".
SKIP_DIRECTORIES = (
    "node_modules/",
    ".venv/",
    "venv/",
    "site-packages/",
    "vendor/",
    "build/",
    "dist/",
)

#: Cap on files fetched per repository, so one enormous monorepo cannot stall a run.
MAX_SOURCE_FILES = 400


class GitHubError(RuntimeError):
    """GitHub returned something we cannot act on."""


@dataclass
class TreeEntry:
    path: str
    size: int


class GitHubClient:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.settings = settings or get_settings()
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> GitHubClient:
        if self._client is None:
            headers = {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "nightshift/0.1",
            }
            if self.settings.github_token:
                headers["Authorization"] = f"Bearer {self.settings.github_token}"
            self._client = httpx.AsyncClient(timeout=self._timeout, headers=headers)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("GitHubClient must be used as an async context manager")
        return self._client

    async def _request(self, method: str, path: str, **kwargs: object) -> httpx.Response:
        """Issue a request, retrying only what is worth retrying.

        A 404 or 422 is a statement about our request, not a transient fault; retrying
        those burns rate limit for nothing.
        """
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = await self.client.request(method, f"{GITHUB_API}{path}", **kwargs)
            except httpx.TransportError as exc:
                last_error = exc
            else:
                if response.status_code < 400:
                    return response
                if response.status_code not in RETRYABLE_STATUS:
                    raise GitHubError(
                        f"GitHub {method} {path} returned {response.status_code}: "
                        f"{response.text[:300]}"
                    )
                last_error = GitHubError(f"GitHub {method} {path} returned {response.status_code}")

            if attempt < self._max_retries - 1:
                await asyncio.sleep(2**attempt)

        raise GitHubError(
            f"GitHub {method} {path} failed after {self._max_retries} attempts"
        ) from last_error

    # --- reads --------------------------------------------------------------

    async def get_repo(self, full_name: str) -> Repo:
        response = await self._request("GET", f"/repos/{full_name}")
        data = response.json()
        owner, _, name = full_name.partition("/")
        return Repo(
            owner=owner,
            name=name,
            default_branch=data.get("default_branch", "main"),
        )

    async def list_tree(self, full_name: str, ref: str) -> list[TreeEntry]:
        """List every file in the repository with one request."""
        response = await self._request(
            "GET", f"/repos/{full_name}/git/trees/{ref}", params={"recursive": "1"}
        )
        data = response.json()

        if data.get("truncated"):
            # Worth knowing: a truncated tree means the analysis saw only part of the repo,
            # which is exactly the situation where a confident "not reachable" would be wrong.
            log.warning("github.tree_truncated", repo=full_name)

        return [
            TreeEntry(path=item["path"], size=item.get("size", 0))
            for item in data.get("tree", [])
            if item.get("type") == "blob"
        ]

    async def get_file(self, full_name: str, path: str, ref: str) -> str | None:
        """Fetch one file's decoded text, or ``None`` if it is absent or binary."""
        try:
            response = await self._request(
                "GET", f"/repos/{full_name}/contents/{path}", params={"ref": ref}
            )
        except GitHubError as exc:
            if "404" in str(exc):
                return None
            raise

        data = response.json()
        if data.get("encoding") != "base64" or "content" not in data:
            return None

        try:
            return base64.b64decode(data["content"]).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return None  # binary or non-UTF-8; not analyzable either way

    async def fetch_files(
        self, full_name: str, paths: list[str], ref: str, *, concurrency: int = 8
    ) -> dict[str, str]:
        """Fetch many files concurrently, skipping ones that fail."""
        semaphore = asyncio.Semaphore(concurrency)

        async def fetch(path: str) -> tuple[str, str | None]:
            async with semaphore:
                return path, await self.get_file(full_name, path, ref)

        results = await asyncio.gather(*(fetch(p) for p in paths))
        return {path: content for path, content in results if content is not None}

    async def fetch_manifests(self, full_name: str, ref: str) -> dict[str, str]:
        """Fetch every recognized manifest and lockfile in the repository."""
        from nightshift.sources.manifests import manifest_paths_to_fetch

        wanted = set(manifest_paths_to_fetch())
        tree = await self.list_tree(full_name, ref)

        paths = [
            entry.path
            for entry in tree
            if entry.path.rsplit("/", 1)[-1] in wanted
            and not any(skip in entry.path for skip in SKIP_DIRECTORIES)
        ]

        return await self.fetch_files(full_name, paths, ref)

    async def fetch_sources(self, full_name: str, ref: str) -> dict[str, str]:
        """Fetch analyzable source files, excluding vendored and generated trees.

        Vendored code is deliberately skipped: an advisory about a package bundled inside
        ``node_modules`` is not a statement about code this repository wrote.
        """
        tree = await self.list_tree(full_name, ref)

        paths = [
            entry.path
            for entry in tree
            if entry.path.endswith(ANALYZABLE_SUFFIXES)
            and not any(skip in entry.path for skip in SKIP_DIRECTORIES)
        ][:MAX_SOURCE_FILES]

        files = await self.fetch_files(full_name, paths, ref)
        log.info("github.sources_fetched", repo=full_name, count=len(files))
        return files

    # --- writes -------------------------------------------------------------

    async def _get_ref_sha(self, full_name: str, branch: str) -> str:
        response = await self._request("GET", f"/repos/{full_name}/git/ref/heads/{branch}")
        return str(response.json()["object"]["sha"])

    async def create_branch(self, full_name: str, branch: str, from_branch: str) -> None:
        base_sha = await self._get_ref_sha(full_name, from_branch)
        await self._request(
            "POST",
            f"/repos/{full_name}/git/refs",
            json={"ref": f"refs/heads/{branch}", "sha": base_sha},
        )
        log.info("github.branch_created", repo=full_name, branch=branch)

    async def update_file(
        self, full_name: str, path: str, content: str, message: str, branch: str
    ) -> None:
        """Commit a file change. Requires the blob SHA when replacing an existing file."""
        response = await self._request(
            "GET", f"/repos/{full_name}/contents/{path}", params={"ref": branch}
        )
        sha = response.json().get("sha")

        await self._request(
            "PUT",
            f"/repos/{full_name}/contents/{path}",
            json={
                "message": message,
                "content": base64.b64encode(content.encode()).decode(),
                "branch": branch,
                "sha": sha,
            },
        )

    async def open_pull_request(
        self, full_name: str, *, title: str, body: str, head: str, base: str
    ) -> str:
        """Open a pull request and return its URL.

        The allowlist is re-checked here, immediately before the only outward-facing
        action in the system. Everything upstream is analysis; this is the step that
        touches a real repository, so it re-verifies rather than trusting its caller.
        """
        assert_repo_allowed(full_name, self.settings)

        if self.settings.dry_run:
            log.info("github.dry_run_pr", repo=full_name, title=title, head=head)
            return f"(dry-run) would open PR on {full_name}: {title}"

        response = await self._request(
            "POST",
            f"/repos/{full_name}/pulls",
            json={"title": title, "body": body, "head": head, "base": base},
        )
        url = str(response.json()["html_url"])
        log.info("github.pr_opened", repo=full_name, url=url)
        return url
