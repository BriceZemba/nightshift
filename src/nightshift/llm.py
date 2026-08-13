"""Model access - a deliberately thin seam over the Google GenAI SDK.

Two reasons this is one small module rather than scattered SDK calls:

**Tiered routing is a cost decision.** Gemma runs the high-frequency classification work at
zero token cost; Gemini only ever sees candidates that survived it. On a nightly run over
several hundred dependencies that is the difference between cents and dollars, and it is
the kind of engineering choice worth making explicit rather than incidental.

**The SDK surface moved in 2026.** ``client.interactions.create()`` replaced
``generate_content``, and ``temperature``/``top_p``/``top_k`` were deprecated in favour of
``thinking_level``. Confining that to one file means a further change costs one edit
instead of a sweep. See docs/08-TECH-REFERENCE.md.

Nothing here catches broad exceptions. Failures propagate so ADK's own retry can act on
them; swallowing them here would disable it silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from nightshift.config import Settings, get_settings

log = structlog.get_logger(__name__)

#: Published rates per 1M tokens, used for the cost figure quoted in the run summary.
#: Gemma is free on the Gemini API, which is the whole point of routing to it first.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "gemini-3.6-flash": (1.50, 7.50),
    "gemini-3.5-flash": (1.50, 7.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemma-4-26b-a4b-it": (0.0, 0.0),
    "gemma-4-31b-it": (0.0, 0.0),
}


@dataclass
class Usage:
    """Running token and cost totals for one run."""

    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    calls_by_model: dict[str, int] = field(default_factory=dict)

    def add(self, model: str, tokens_in: int, tokens_out: int) -> float:
        rate_in, rate_out = PRICING_PER_MTOK.get(model, (0.0, 0.0))
        cost = (tokens_in / 1_000_000) * rate_in + (tokens_out / 1_000_000) * rate_out
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.cost_usd += cost
        self.calls_by_model[model] = self.calls_by_model.get(model, 0) + 1
        return cost


@dataclass
class Completion:
    text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


class LLMClient:
    """Wraps the GenAI client with model tiering and usage accounting.

    The SDK client is injected rather than constructed here so the agents can be tested
    against a fake without credentials, a network, or a billing account.
    """

    def __init__(self, client: Any | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.usage = Usage()
        self._client = client

    @property
    def client(self) -> Any:
        """Lazily construct the real SDK client.

        Deferred so that importing this module -- which the CLI and tests do -- never
        requires credentials to be present.
        """
        if self._client is None:
            from google import genai

            self._client = genai.Client()
        return self._client

    def _extract_usage(self, response: Any) -> tuple[int, int]:
        """Read token counts from a response, tolerating shape differences.

        The usage block has been spelled several ways across SDK versions. Accounting is
        for a cost figure in a report, so an unreadable count is worth degrading on rather
        than failing the run over.
        """
        meta = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
        if meta is None:
            return 0, 0
        tokens_in = (
            getattr(meta, "prompt_token_count", None)
            or getattr(meta, "input_tokens", None)
            or 0
        )
        tokens_out = (
            getattr(meta, "candidates_token_count", None)
            or getattr(meta, "output_tokens", None)
            or 0
        )
        return int(tokens_in), int(tokens_out)

    def _extract_text(self, response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str):
            return text
        output = getattr(response, "output_text", None)
        if isinstance(output, str):
            return output
        return str(response)

    def _call(self, model: str, prompt: str, **kwargs: Any) -> Completion:
        response = self.client.interactions.create(model=model, input=prompt, **kwargs)

        tokens_in, tokens_out = self._extract_usage(response)
        cost = self.usage.add(model, tokens_in, tokens_out)

        log.info(
            "llm.call",
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=round(cost, 6),
        )

        return Completion(
            text=self._extract_text(response),
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
        )

    def classify(self, prompt: str) -> Completion:
        """Cheap, high-frequency work - routed to Gemma at zero token cost.

        Used for relevance filtering and policy classification, where the question is
        closed and the answer is short. Anything needing real reasoning goes to
        :meth:`reason` instead.
        """
        return self._call(self.settings.model_filter, prompt)

    def reason(self, prompt: str, *, thinking_level: str | None = None) -> Completion:
        """Deep reasoning - routed to Gemini.

        ``thinking_level`` replaced the older ``thinking_budget``; ``temperature``,
        ``top_p`` and ``top_k`` were deprecated in July 2026 and must not be passed.
        """
        kwargs: dict[str, Any] = {}
        if thinking_level:
            kwargs["thinking_level"] = thinking_level
        return self._call(self.settings.model_reasoning, prompt, **kwargs)

    def summary(self) -> dict[str, Any]:
        return {
            "tokens_in": self.usage.tokens_in,
            "tokens_out": self.usage.tokens_out,
            "cost_usd": round(self.usage.cost_usd, 4),
            "calls_by_model": dict(self.usage.calls_by_model),
        }
