"""Claude provider — a thin wrapper over the official ``anthropic`` SDK.

Builds the shared prompt, constrains the answer to the proposal schema with
structured outputs (``output_config.format``), and returns the parsed object.
Nothing Claude-specific leaks past ``analyze``: SDK errors and refusals become
``ProviderError``, truncation and non-JSON become ``InvalidResponse``.
"""

from __future__ import annotations

import logging
from typing import Any

import anthropic

from bellwether.analysis.analyst import SYSTEM_PROMPT, render_prompt
from bellwether.analysis.provider import InvalidResponse, Provider, ProviderError, parse_json_object
from bellwether.analysis.schema import wire_schema
from bellwether.models import Finding

logger = logging.getLogger(__name__)

_MAX_TOKENS = 16_000


class ClaudeProvider(Provider):
    name = "claude"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self._model = model
        # max_retries=0: ProviderChain owns retries; SDK retries would multiply them.
        self._client = client or anthropic.Anthropic(
            api_key=api_key, timeout=timeout_seconds, max_retries=0
        )

    def analyze(self, finding: Finding, context: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": render_prompt(finding, context)}],
                output_config={"format": {"type": "json_schema", "schema": wire_schema()}},
            )
        except anthropic.APIError as exc:
            raise ProviderError(f"claude request failed: {type(exc).__name__}: {exc}") from exc

        logger.info(
            "provider response",
            extra={
                "provider": self.name,
                "model": self._model,
                "stop_reason": response.stop_reason,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )
        if response.stop_reason == "refusal":
            raise ProviderError("claude declined the request (stop_reason=refusal)")
        if response.stop_reason == "max_tokens":
            raise InvalidResponse("claude response truncated at max_tokens")
        text = "".join(block.text for block in response.content if block.type == "text")
        return parse_json_object(text)
