"""Policy is the boundary that does not depend on a model behaving well, so it gets the
most direct tests in the project."""

from __future__ import annotations

import pytest

from nightshift.config import Settings
from nightshift.models import (
    Dependency,
    Ecosystem,
    Finding,
    PatchAttempt,
    PatchStrategy,
    Reachability,
    TriageVerdict,
)
from nightshift.policy import (
    PolicyViolation,
    assert_paths_allowed,
    assert_repo_allowed,
    is_forbidden_path,
    is_major_bump,
    requires_human_approval,
)


def _settings(allowlist: list[str]) -> Settings:
    return Settings(NIGHTSHIFT_REPO_ALLOWLIST=",".join(allowlist))


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


class TestRepoAllowlist:
    def test_allows_listed_repo(self) -> None:
        assert_repo_allowed("me/myrepo", _settings(["me/myrepo"]))

    def test_rejects_unlisted_repo(self) -> None:
        with pytest.raises(PolicyViolation, match="not on the allowlist"):
            assert_repo_allowed("someone-else/their-repo", _settings(["me/myrepo"]))

    def test_empty_allowlist_rejects_everything(self) -> None:
        """Failing closed matters more than convenience: an empty allowlist must never
        mean 'anything goes'."""
        with pytest.raises(PolicyViolation, match="No repositories are allowlisted"):
            assert_repo_allowed("me/myrepo", _settings([]))

    def test_rejects_malformed_allowlist_entry(self) -> None:
        with pytest.raises(ValueError, match="owner/name"):
            _settings(["not-a-repo-spec"])


class TestForbiddenPaths:
    @pytest.mark.parametrize(
        "path",
        [
            ".github/workflows/ci.yml",
            ".github/dependabot.yml",
            "cloudbuild.yaml",
            "main.tf",
            "k8s/deployment.yaml",
            ".env",
            "certs/server.pem",
            "config/credentials.yml",
            "deploy/service-account-prod.json",
        ],
    )
    def test_rejects_execution_and_credential_surfaces(self, path: str) -> None:
        assert is_forbidden_path(path)

    @pytest.mark.parametrize(
        "path",
        ["requirements.txt", "pyproject.toml", "src/app/handler.py", "package-lock.json"],
    )
    def test_allows_ordinary_source_and_manifests(self, path: str) -> None:
        assert not is_forbidden_path(path)

    def test_leading_dot_slash_is_normalized(self) -> None:
        """A patch that names './.github/workflows/ci.yml' is the same attack."""
        assert is_forbidden_path("./.github/workflows/ci.yml")

    def test_backslash_paths_are_normalized(self) -> None:
        assert is_forbidden_path(".github\\workflows\\ci.yml")

    def test_assert_paths_allowed_names_every_violation(self) -> None:
        with pytest.raises(PolicyViolation) as exc:
            assert_paths_allowed(["requirements.txt", ".github/workflows/ci.yml", "main.tf"])
        message = str(exc.value)
        assert ".github/workflows/ci.yml" in message
        assert "main.tf" in message
        assert "requirements.txt" not in message

    def test_clean_patch_passes(self) -> None:
        assert_paths_allowed(["requirements.txt", "src/app.py"])


class TestMajorBump:
    @pytest.mark.parametrize(
        ("current", "proposed", "expected"),
        [
            ("1.2.3", "2.0.0", True),
            ("1.2.3", "1.3.0", False),
            ("1.2.3", "1.2.4", False),
            ("0.1.0", "1.0.0", True),
            ("v1.2.3", "v2.0.0", True),
        ],
    )
    def test_detects_major_changes(self, current: str, proposed: str, expected: bool) -> None:
        assert is_major_bump(current, proposed) is expected

    @pytest.mark.parametrize(("current", "proposed"), [("weird", "1.0.0"), ("1.0.0", "latest")])
    def test_unparseable_versions_escalate(self, current: str, proposed: str) -> None:
        """'I could not tell' must route to a human, not proceed quietly."""
        assert is_major_bump(current, proposed) is True


class TestHumanApproval:
    def test_clean_minor_bump_needs_no_approval(self) -> None:
        finding = _finding(
            verdict=TriageVerdict(reachability=Reachability.REACHABLE),
            attempts=[PatchAttempt(attempt=1, strategy=PatchStrategy.UPSTREAM_BUMP)],
        )
        assert (
            requires_human_approval(
                finding, proposed_version="2.19.2", repo_has_tests=True, guardian_flagged=False
            )
            is None
        )

    def test_guardian_flag_outranks_everything(self) -> None:
        finding = _finding(verdict=TriageVerdict(reachability=Reachability.REACHABLE))
        reason = requires_human_approval(
            finding, proposed_version="3.0.0", repo_has_tests=False, guardian_flagged=True
        )
        assert reason is not None
        assert "Guardian" in reason

    def test_missing_tests_blocks_auto_merge(self) -> None:
        finding = _finding(verdict=TriageVerdict(reachability=Reachability.REACHABLE))
        reason = requires_human_approval(
            finding, proposed_version="2.19.2", repo_has_tests=False, guardian_flagged=False
        )
        assert reason is not None
        assert "test suite" in reason

    def test_inconclusive_reachability_escalates(self) -> None:
        finding = _finding(verdict=TriageVerdict(reachability=Reachability.UNKNOWN))
        reason = requires_human_approval(
            finding, proposed_version="2.19.2", repo_has_tests=True, guardian_flagged=False
        )
        assert reason is not None
        assert "inconclusive" in reason

    def test_major_bump_escalates(self) -> None:
        finding = _finding(verdict=TriageVerdict(reachability=Reachability.REACHABLE))
        reason = requires_human_approval(
            finding, proposed_version="3.0.0", repo_has_tests=True, guardian_flagged=False
        )
        assert reason is not None
        assert "breaking changes" in reason

    def test_backport_escalates(self) -> None:
        """A synthesized backport has no upstream release to compare against, so it always
        gets a human - this is the highest-value and highest-risk case."""
        finding = _finding(
            verdict=TriageVerdict(reachability=Reachability.REACHABLE),
            attempts=[PatchAttempt(attempt=1, strategy=PatchStrategy.BACKPORT)],
        )
        reason = requires_human_approval(
            finding, proposed_version=None, repo_has_tests=True, guardian_flagged=False
        )
        assert reason is not None
        assert "Backported" in reason


class TestIdempotencyKey:
    def test_key_is_stable_across_runs(self) -> None:
        """The same advisory against the same package in the same repo is the same
        real-world action, whether discovered tonight or next week."""
        a = _finding(run_id="run-1")
        b = _finding(run_id="run-2")
        assert a.idempotency_key() == b.idempotency_key()

    def test_key_differs_per_repo(self) -> None:
        a = _finding(repo="me/repo-a")
        b = _finding(repo="me/repo-b")
        assert a.idempotency_key() != b.idempotency_key()

    def test_key_differs_per_advisory(self) -> None:
        a = _finding(advisory_id="GHSA-aaaa")
        b = _finding(advisory_id="GHSA-bbbb")
        assert a.idempotency_key() != b.idempotency_key()
