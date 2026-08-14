"""The Cloud Run HTTP surface.

The trigger endpoint is the one place where an internet-reachable request causes writes to
real GitHub repositories, so most of these tests are about it refusing.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nightshift import server


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(server, "_last_run", {}, raising=False)
    return TestClient(server.app)


class TestHealth:
    def test_healthz_is_open_and_cheap(self, client: TestClient) -> None:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestStatusPage:
    def test_renders_without_a_run(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "No run recorded yet" in response.text

    def test_renders_the_headline_number(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            server,
            "_last_run",
            {
                "repos_scanned": 23,
                "advisories_ingested": 180,
                "findings_reachable": 6,
                "dismissed": 174,
                "prs_opened": 4,
                "escalated": 2,
                "cost_usd": 0.31,
            },
            raising=False,
        )
        text = client.get("/").text
        assert "180 advisories reduced to 6" in text
        assert "$0.3100" in text

    def test_does_not_leak_repository_names_or_secrets(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The page is reachable by anyone with the URL, so it carries counts only."""
        monkeypatch.setenv("NIGHTSHIFT_RUN_TOKEN", "super-secret-token")
        monkeypatch.setattr(
            server, "_last_run", {"advisories_ingested": 1, "findings_reachable": 1}, raising=False
        )
        text = client.get("/").text
        assert "super-secret-token" not in text
        assert "github.com" not in text


class TestTriggerAuth:
    def test_unconfigured_token_disables_the_endpoint(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fails closed. A missing secret must disable the trigger, never open it."""
        monkeypatch.delenv("NIGHTSHIFT_RUN_TOKEN", raising=False)
        response = client.post("/run")
        assert response.status_code == 503
        assert "disabled" in response.json()["detail"]

    def test_missing_credentials_are_rejected(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NIGHTSHIFT_RUN_TOKEN", "correct-token")
        assert client.post("/run").status_code == 401

    def test_wrong_token_is_rejected(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NIGHTSHIFT_RUN_TOKEN", "correct-token")
        response = client.post("/run", headers={"Authorization": "Bearer wrong-token"})
        assert response.status_code == 401

    def test_correct_token_reaches_the_pipeline(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NIGHTSHIFT_RUN_TOKEN", "correct-token")

        called: list[bool] = []

        async def fake_run() -> dict:
            called.append(True)
            return {"advisories_ingested": 0}

        monkeypatch.setattr(server, "_execute_run", fake_run)

        response = client.post("/run", headers={"Authorization": "Bearer correct-token"})
        assert response.status_code == 200
        assert called == [True]

    def test_bare_token_without_bearer_prefix_also_works(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NIGHTSHIFT_RUN_TOKEN", "correct-token")

        async def fake_run() -> dict:
            return {}

        monkeypatch.setattr(server, "_execute_run", fake_run)
        assert client.post("/run", headers={"Authorization": "correct-token"}).status_code == 200


class TestConcurrency:
    def test_overlapping_runs_are_refused(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A nightly run still going when the next fires would double the work and race on
        the claim ledger."""
        import asyncio

        monkeypatch.setenv("NIGHTSHIFT_RUN_TOKEN", "correct-token")
        monkeypatch.setattr(server, "_run_lock", asyncio.Lock(), raising=False)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(server._run_lock.acquire())
            response = client.post("/run", headers={"Authorization": "Bearer correct-token"})
            assert response.status_code == 409
        finally:
            loop.close()
