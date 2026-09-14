"""OpenAI provider — a thin wrapper over the official ``openai`` SDK.

Same prompt, same schema as the Claude provider: the shared system prompt and
user prompt, with the answer constrained by ``response_format`` json_schema in
strict mode. SDK errors and refusals become ``ProviderError``; truncation and
non-JSON become ``InvalidResponse``.
"""

from __future__ import annotations

import logging
from typing import Any

import openai

from bellwether.analysis.analyst import SYSTEM_PROMPT, render_prompt
from bellwether.analysis.provider import InvalidResponse, Provider, ProviderError, parse_json_object
from bellwether.analysis.schema import wire_schema
from bellwether.models import Finding

logger = logging.getLogger(__name__)


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        client: openai.OpenAI | None = None,
    ) -> None:
        self._model = model
        # max_retries=0: ProviderChain owns retries; SDK retries would multiply them.
        self._client = client or openai.OpenAI(
            api_key=api_key, timeout=timeout_seconds, max_retries=0
        )

    def analyze(self, finding: Finding, context: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": render_prompt(finding, context)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "bellwether_proposal",
                        "schema": wire_schema(),
                        "strict": True,
                    },
                },
            )
        except openai.APIError as exc:
            raise ProviderError(f"openai request failed: {type(exc).__name__}: {exc}") from exc

        usage = response.usage
        choice = response.choices[0]
        logger.info(
            "provider response",
            extra={
                "provider": self.name,
                "model": self._model,
                "finish_reason": choice.finish_reason,
                "input_tokens": usage.prompt_tokens if usage else None,
                "output_tokens": usage.completion_tokens if usage else None,
            },
        )
        if choice.message.refusal:
            raise ProviderError("openai declined the request (refusal)")
        if choice.finish_reason == "length":
            raise InvalidResponse("openai response truncated at max tokens")
        return parse_json_object(choice.message.content or "")
