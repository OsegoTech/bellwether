"""Provider abstraction and failover — the only door to a model.

``Provider.analyze(finding, context)`` returns the model's raw JSON object.
Providers are interchangeable: each renders the same prompt from the shared
template and asks for the same schema, so ``ProviderChain`` can move from one
to the next without anything upstream knowing which answered.

``ProviderChain.run`` tries the primary up to ``max_retries + 1`` times, then the
fallback the same way. An attempt fails if the call raises (transport, timeout,
refusal) *or* the returned object fails validation — invalid output is retried,
never coerced. If every attempt fails it raises ``AnalysisUnavailable`` and the
pipeline records the finding un-analyzed rather than inventing a proposal.

Per-provider timeouts are enforced by each SDK client (``timeout_seconds``);
the SDKs' own retries are disabled so the chain alone owns retry policy.
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypeVar

from bellwether.models import Finding

logger = logging.getLogger(__name__)

T = TypeVar("T")

_FENCED = re.compile(r"^```(?:json)?\s*\n(.*)\n```$", re.DOTALL)


class ProviderError(Exception):
    """A provider call failed: transport error, timeout, or refusal."""


class InvalidResponse(ProviderError):
    """The provider answered with something that is not a JSON object."""


class AnalysisUnavailable(Exception):
    """Every attempt on every provider failed."""

    def __init__(self, failures: Sequence[str]) -> None:
        self.failures = list(failures)
        super().__init__("all providers failed: " + "; ".join(self.failures))


class Provider(ABC):
    name: str  # "claude" | "openai" — matches config and Proposal.provider

    @abstractmethod
    def analyze(self, finding: Finding, context: dict[str, Any]) -> dict[str, Any]:
        """Return the model's raw JSON object for this finding."""


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse model text as one JSON object (a surrounding code fence is tolerated)."""
    body = text.strip()
    fenced = _FENCED.match(body)
    if fenced:
        body = fenced.group(1)
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise InvalidResponse(f"response is not JSON ({exc.msg} at char {exc.pos})") from None
    if not isinstance(value, dict):
        raise InvalidResponse(f"response JSON is a {type(value).__name__}, not an object")
    return value


@dataclass
class ProviderHealth:
    consecutive_failures: int = 0
    total_failures: int = 0
    last_error: str | None = None
    last_success_at: datetime | None = None


class ProviderChain:
    def __init__(self, providers: Sequence[Provider], *, max_retries: int = 1) -> None:
        if not providers:
            raise ValueError("ProviderChain needs at least one provider")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self._providers = tuple(providers)
        self._max_retries = max_retries
        self.health: dict[str, ProviderHealth] = {p.name: ProviderHealth() for p in providers}

    def run(
        self,
        finding: Finding,
        context: dict[str, Any],
        validate: Callable[[dict[str, Any]], T],
    ) -> tuple[str, T]:
        """Return (provider name, validated result) from the first good answer."""
        failures: list[str] = []
        for index, provider in enumerate(self._providers):
            health = self.health[provider.name]
            for attempt in range(1, self._max_retries + 2):
                started = time.monotonic()
                try:
                    result = validate(provider.analyze(finding, context))
                except Exception as exc:  # transport, garbage and schema failures alike
                    error = _describe(exc)
                    health.consecutive_failures += 1
                    health.total_failures += 1
                    health.last_error = error
                    failures.append(f"{provider.name} attempt {attempt}: {error}")
                    logger.warning(
                        "provider attempt failed",
                        extra={
                            "provider": provider.name,
                            "attempt": attempt,
                            "finding_id": finding.finding_id,
                            "consecutive_failures": health.consecutive_failures,
                            "error": error,
                        },
                    )
                    continue
                health.consecutive_failures = 0
                health.last_success_at = datetime.now(timezone.utc)
                logger.info(
                    "analysis served",
                    extra={
                        "provider": provider.name,
                        "attempt": attempt,
                        "fallback": index > 0,
                        "finding_id": finding.finding_id,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                    },
                )
                return provider.name, result
            if index + 1 < len(self._providers):
                logger.warning(
                    "provider exhausted; failing over",
                    extra={
                        "provider": provider.name,
                        "next_provider": self._providers[index + 1].name,
                        "finding_id": finding.finding_id,
                    },
                )
        raise AnalysisUnavailable(failures)


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]
