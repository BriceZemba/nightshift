"""Configuration, loaded from environment or .env.

Two settings here are safety rails rather than preferences: ``dry_run`` and
``max_prs_per_run``. Nightshift writes to real repositories, so the default posture is
"change nothing" and the blast radius is capped even when it is armed.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Google Cloud -------------------------------------------------------
    google_cloud_project: str = Field(default="", alias="GOOGLE_CLOUD_PROJECT")
    google_cloud_location: str = Field(default="us-central1", alias="GOOGLE_CLOUD_LOCATION")

    # --- Models -------------------------------------------------------------
    # gemini-3.5-pro does not exist; gemini-3.6-flash is the current GA default.
    # See docs/08-TECH-REFERENCE.md.
    model_reasoning: str = Field(
        default="gemini-3.6-flash", alias="NIGHTSHIFT_MODEL_REASONING"
    )
    #: Gemma is $0/token on the Gemini API. It runs the high-frequency relevance filter so
    #: the reasoning model only ever sees candidates that survived it - a cost decision
    #: first, and a tiering story second.
    model_filter: str = Field(default="gemma-4-26b-a4b-it", alias="NIGHTSHIFT_MODEL_FILTER")

    # --- GitHub -------------------------------------------------------------
    github_token: str = Field(default="", alias="GITHUB_TOKEN")
    github_actor: str = Field(default="", alias="GITHUB_ACTOR")

    #: Own repos or own forks only. Enforced in policy.py, not merely documented.
    #:
    #: ``NoDecode`` is required, not cosmetic. pydantic-settings treats any ``list`` field
    #: as "complex" and runs ``json.loads()`` on the raw environment value *before* field
    #: validators run - so a perfectly ordinary ``NIGHTSHIFT_REPO_ALLOWLIST=me/repo`` (or
    #: an empty one) raises a JSON decode error rather than reaching the splitter below.
    repo_allowlist: Annotated[list[str], NoDecode] = Field(
        default_factory=list, alias="NIGHTSHIFT_REPO_ALLOWLIST"
    )

    # --- Advisory sources ---------------------------------------------------
    nvd_api_key: str = Field(default="", alias="NVD_API_KEY")

    # --- Persistence --------------------------------------------------------
    #: "local" is a file-backed store needing no Google Cloud project, so the full
    #: pipeline runs offline. "firestore" is the deployed backend.
    store_backend: str = Field(default="local", alias="NIGHTSHIFT_STORE")
    local_store_path: str = Field(default=".nightshift", alias="NIGHTSHIFT_STORE_PATH")

    # --- Safety rails -------------------------------------------------------
    dry_run: bool = Field(default=True, alias="NIGHTSHIFT_DRY_RUN")
    max_prs_per_run: int = Field(default=5, alias="NIGHTSHIFT_MAX_PRS_PER_RUN")
    max_patch_attempts: int = Field(default=3, alias="NIGHTSHIFT_MAX_PATCH_ATTEMPTS")

    @field_validator("store_backend")
    @classmethod
    def _validate_store_backend(cls, v: str) -> str:
        allowed = {"local", "firestore"}
        if v not in allowed:
            raise ValueError(f"NIGHTSHIFT_STORE must be one of {sorted(allowed)}, got {v!r}")
        return v

    @field_validator("repo_allowlist", mode="before")
    @classmethod
    def _split_allowlist(cls, v: object) -> object:
        """Accept a comma-separated string from the environment.

        An empty or whitespace-only value yields an empty list, which the policy layer
        treats as "nothing is allowlisted" and therefore refuses everything.
        """
        if v is None:
            return []
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("repo_allowlist")
    @classmethod
    def _validate_allowlist_format(cls, v: list[str]) -> list[str]:
        for entry in v:
            if entry.count("/") != 1 or entry.startswith("/") or entry.endswith("/"):
                raise ValueError(
                    f"repo allowlist entries must look like 'owner/name', got {entry!r}"
                )
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
