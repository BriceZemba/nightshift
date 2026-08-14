"""Model Armor integration - managed prompt-injection screening.

Nightshift reads attacker-authored text. CVE descriptions, advisory bodies, upstream commit
messages and third-party diffs are written by strangers, and they are fed to a model that
writes code and opens pull requests. That is a prompt-injection path with a code-execution
payoff, which is why this is a real control here rather than a checkbox.

Model Armor sits between the deterministic pattern screen in :mod:`nightshift.agents.guardian`
and the structural policy in :mod:`nightshift.policy`. It catches the semantic attacks that
regexes miss; the structural policy is what still holds if it misses them too.

**Fails closed.** If the service errors, times out, or is misconfigured, the content is
reported as unsafe and the finding routes to a human. A security control that quietly
becomes a no-op when the network hiccups is worse than no control, because it is trusted.

Requires ``google-cloud-modelarmor``. The import is lazy so the rest of the system runs
without it, and without a Google Cloud project at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger(__name__)

#: Model Armor is a regional service and rejects the global endpoint.
ENDPOINT_TEMPLATE = "modelarmor.{location}.rep.googleapis.com"

#: Response states that mean "something was found". Compared case-insensitively against the
#: enum's name, because the exact enum surface has moved between client versions.
MATCH_STATES = frozenset({"MATCH_FOUND", "MATCH", "FOUND"})


@dataclass
class ArmorVerdict:
    """Matches the shape :class:`nightshift.agents.guardian.Guardian` expects."""

    blocked: bool
    categories: list[str] = field(default_factory=list)


def template_path(project: str, location: str, template_id: str) -> str:
    """Build the fully-qualified template resource name.

    A bare template id is accepted for convenience in config; anything already fully
    qualified is passed through untouched.
    """
    if template_id.startswith("projects/"):
        return template_id
    return f"projects/{project}/locations/{location}/templates/{template_id}"


class ModelArmorScreener:
    """Screens untrusted text through Model Armor.

    Exposes ``sanitize(text) -> ArmorVerdict``, which is the interface Guardian already
    expects, so wiring this in changes no call sites.
    """

    def __init__(
        self,
        *,
        project: str,
        location: str,
        template_id: str,
        client: Any | None = None,
    ) -> None:
        self.project = project
        self.location = location
        self.template = template_path(project, location, template_id)
        self._client = client

    @property
    def client(self) -> Any:
        """Lazily build the regional client.

        Deferred so that importing this module never requires the dependency or
        credentials, which keeps the offline development path working.
        """
        if self._client is None:
            from google.api_core.client_options import ClientOptions
            from google.cloud import modelarmor_v1

            self._client = modelarmor_v1.ModelArmorClient(
                client_options=ClientOptions(
                    api_endpoint=ENDPOINT_TEMPLATE.format(location=self.location)
                )
            )
        return self._client

    def sanitize(self, text: str) -> ArmorVerdict:
        """Screen one piece of untrusted text.

        Raises rather than returning a verdict on failure. Guardian treats an exception as
        unsafe, which is the fail-closed behaviour we want: an unreachable screener must
        not read as a clean bill of health.
        """
        from google.cloud import modelarmor_v1

        request = modelarmor_v1.SanitizeUserPromptRequest(
            name=self.template,
            user_prompt_data=modelarmor_v1.DataItem(text=text),
        )

        response = self.client.sanitize_user_prompt(request=request)
        return self._interpret(response)

    def _interpret(self, response: Any) -> ArmorVerdict:
        """Turn a Model Armor response into a verdict.

        Deliberately defensive about the response shape. The client library has moved the
        result object between versions, and a parsing failure here must not silently become
        "not blocked" - anything unreadable is treated as blocked.
        """
        result = getattr(response, "sanitization_result", None)
        if result is None:
            log.warning("model_armor.unreadable_response")
            return ArmorVerdict(blocked=True, categories=["unreadable response"])

        overall = _state_name(getattr(result, "filter_match_state", None))
        matched_filters = _matched_filters(result)

        blocked = overall in MATCH_STATES or bool(matched_filters)

        if blocked:
            log.warning("model_armor.blocked", categories=matched_filters, state=overall)

        categories = matched_filters or ([overall] if blocked else [])
        return ArmorVerdict(blocked=blocked, categories=categories)


def _state_name(state: Any) -> str:
    """Read an enum's name whether it arrives as an enum, an int, or a string."""
    if state is None:
        return ""
    name = getattr(state, "name", None)
    return str(name if name is not None else state).upper()


def _matched_filters(result: Any) -> list[str]:
    """Collect the names of filters that fired.

    ``filter_results`` is a map of filter name to a per-filter result, and each result
    nests its own match state under a differently-named field depending on the filter. The
    search is therefore structural rather than keyed on specific attribute names.
    """
    filter_results = getattr(result, "filter_results", None)
    if not filter_results:
        return []

    matched: list[str] = []

    items = filter_results.items() if hasattr(filter_results, "items") else []
    for name, filter_result in items:
        if _contains_match(filter_result):
            matched.append(str(name))

    return sorted(matched)


def _contains_match(node: Any, depth: int = 0) -> bool:
    """Recursively look for a match state on a filter result."""
    if depth > 4 or node is None:
        return False

    state = getattr(node, "match_state", None)
    if state is not None and _state_name(state) in MATCH_STATES:
        return True

    # Each filter nests its result under its own field name, so descend through whatever
    # message-shaped attributes exist rather than guessing at names.
    for attribute in dir(node):
        if attribute.startswith("_") or attribute in {"match_state"}:
            continue
        try:
            child = getattr(node, attribute)
        except (AttributeError, TypeError):
            continue
        if hasattr(child, "match_state") and _contains_match(child, depth + 1):
            return True

    return False


def build_screener(settings) -> ModelArmorScreener | None:
    """Construct a screener from settings, or ``None`` when it is not configured.

    Returning ``None`` leaves Guardian on deterministic screening alone and records that
    Model Armor was not consulted, which is an honest degraded mode rather than a silent
    one.
    """
    if not settings.model_armor_template:
        return None
    if not settings.google_cloud_project:
        log.warning("model_armor.no_project_configured")
        return None

    return ModelArmorScreener(
        project=settings.google_cloud_project,
        location=settings.model_armor_location or settings.google_cloud_location,
        template_id=settings.model_armor_template,
    )
