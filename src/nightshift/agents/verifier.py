"""Verifier - proves a patch does not break the repository.

Runs the repository's own test suite against the patched tree. Nothing else in the fleet
can distinguish a real fix from a plausible-looking one, so this is what makes the pull
request worth a maintainer's attention rather than another thing to review.

Isolation matters more here than anywhere else in the system: this is the one component
that executes code that a model wrote in response to text a stranger authored. In
production it runs as a Cloud Run job with **no network egress**, so a patch that tries to
reach the internet simply cannot. The local runner below is for development and carries
the weaker guarantee of a scrubbed environment and a hard timeout.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

DEFAULT_TIMEOUT_SECONDS = 600

#: Environment variables scrubbed before running untrusted code. A patch under test must
#: never inherit the credentials the fleet itself runs with.
_SCRUBBED_ENV_PREFIXES = (
    "GITHUB_",
    "GOOGLE_",
    "GCP_",
    "AWS_",
    "AZURE_",
    "NIGHTSHIFT_",
    "NVD_",
    "OPENAI_",
    "ANTHROPIC_",
)


@dataclass
class VerificationResult:
    passed: bool
    output: str
    exit_code: int = 0
    timed_out: bool = False
    skipped_reason: str | None = None


def _clean_environment() -> dict[str, str]:
    """Build a minimal environment with no inherited credentials."""
    clean = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(_SCRUBBED_ENV_PREFIXES)
    }
    # Keep the test run deterministic and offline-ish.
    clean["PYTHONDONTWRITEBYTECODE"] = "1"
    clean["PIP_NO_INPUT"] = "1"
    clean["CI"] = "1"
    return clean


def apply_diff(work_dir: Path, diff: str) -> tuple[bool, str]:
    """Apply a unified diff to a working tree using ``git apply``.

    ``git apply`` is used rather than hand-rolled patching because it validates context
    lines. A diff that does not apply cleanly is a failed attempt, and finding that out
    here is far better than committing something malformed.
    """
    if not diff.strip():
        return False, "empty diff"

    patch_file = work_dir / ".nightshift.patch"
    patch_file.write_text(diff, encoding="utf-8")

    try:
        result = subprocess.run(
            ["git", "apply", "--verbose", str(patch_file)],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=60,
            env=_clean_environment(),
        )
    except FileNotFoundError:
        return False, "git is not available to apply the patch"
    except subprocess.TimeoutExpired:
        return False, "git apply timed out"
    finally:
        patch_file.unlink(missing_ok=True)

    if result.returncode != 0:
        return False, f"git apply failed: {result.stderr[:1000]}"

    return True, "patch applied"


class Verifier:
    """Runs a repository's test suite against a patched tree.

    ``runner`` is injectable so the orchestrator can swap the local subprocess runner for
    a Cloud Run job without the graph changing shape.
    """

    def __init__(self, *, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._timeout = timeout

    def verify(
        self,
        source_files: dict[str, str],
        diff: str,
        test_command: str | None,
    ) -> VerificationResult:
        """Materialize the repository, apply the patch, and run its tests.

        A repository with no test command is **not** a pass. It returns ``passed=False``
        with a skip reason, which routes the finding to a human - an unverifiable patch
        must never be presented as a verified one.
        """
        if not test_command:
            return VerificationResult(
                passed=False,
                output="",
                skipped_reason="repository has no configured test command",
            )

        work_dir = Path(tempfile.mkdtemp(prefix="nightshift-verify-"))

        try:
            for relative_path, content in source_files.items():
                target = work_dir / relative_path
                # Reject path traversal in filenames before writing anything.
                if not target.resolve().is_relative_to(work_dir.resolve()):
                    return VerificationResult(
                        passed=False,
                        output=f"refusing to write outside the work directory: {relative_path}",
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")

            subprocess.run(
                ["git", "init", "-q"],
                cwd=work_dir,
                capture_output=True,
                timeout=60,
                env=_clean_environment(),
            )

            applied, message = apply_diff(work_dir, diff)
            if not applied:
                return VerificationResult(passed=False, output=message)

            return self._run_tests(work_dir, test_command)

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _run_tests(self, work_dir: Path, test_command: str) -> VerificationResult:
        # shell=True is required because test commands are arbitrary strings like
        # "pytest -q && ruff check". The command comes from repository configuration the
        # operator controls, never from a model or an advisory.
        try:
            result = subprocess.run(
                test_command,
                cwd=work_dir,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                env=_clean_environment(),
            )
        except subprocess.TimeoutExpired:
            log.warning("verify.timeout", timeout=self._timeout)
            return VerificationResult(
                passed=False,
                output=f"test command exceeded {self._timeout}s and was killed",
                timed_out=True,
            )

        output = (result.stdout + result.stderr)[-8000:]
        passed = result.returncode == 0

        log.info("verify.complete", passed=passed, exit_code=result.returncode)

        return VerificationResult(
            passed=passed,
            output=output,
            exit_code=result.returncode,
        )
