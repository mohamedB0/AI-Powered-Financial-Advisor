"""
LLM Service
───────────
Streams tokens from OpenAI's chat completions API.

Config (env vars / config.py):
  OPENAI_API_KEY, LLM_MODEL, LLM_SYSTEM_PROMPT, LLM_MAX_TOKENS,
  LLM_TEMPERATURE, MAX_HISTORY_TURNS
"""

import logging
from typing import AsyncGenerator

import config

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

    async def stream(
        self,
        conversation: list[dict],
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Yield response tokens one-by-one as they arrive from the API.

        conversation: list of {"role": "user"|"assistant", "content": str}
        """
        # Trim history to avoid exceeding context windows
        trimmed = conversation[-config.MAX_HISTORY_TURNS * 2:]

        messages = [
            {"role": "system", "content": system_prompt or config.LLM_SYSTEM_PROMPT},
            *trimmed,
        ]

        logger.debug("LLM request: model=%s turns=%d", config.LLM_MODEL, len(trimmed))

        stream = await self._client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
