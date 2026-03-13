"""
Claude (Anthropic) provider implementation for Project Atlas.

Wraps the Anthropic Python SDK and normalizes responses to the
DecisionProvider interface. Uses prompt caching to reduce costs on
the recurring trading loop system prompt.
"""

import logging
import os
from typing import Dict

import anthropic

from src.engine.providers.base import DecisionProvider
from src.utils.logging_config import setup_file_logger

logger = setup_file_logger(__name__, 'claude_provider')


class ClaudeProvider(DecisionProvider):
    """
    Anthropic Claude provider for trading decisions and reviews.

    Model names are loaded from .env — never hardcoded.
    Uses prompt caching: the system prompt is marked as cacheable so
    repeated calls within a 5-minute window pay only 10% of input cost
    for the cached portion.
    """

    def __init__(self, model_override: str = None):
        """
        Initialize the Claude provider.

        Args:
            model_override: Optional model name override (used by eod_review.py
                            to force the review model)
        """
        self._api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self._api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")

        self._decision_model = model_override or os.getenv('CLAUDE_DECISION_MODEL')
        if not self._decision_model:
            raise ValueError("CLAUDE_DECISION_MODEL not set in .env")

        self._client = anthropic.AsyncAnthropic(api_key=self._api_key)

        # Pricing per million tokens — Sonnet (as of build time)
        self._input_price_per_m = 3.00
        self._output_price_per_m = 15.00
        self._cache_write_price_per_m = 3.75   # 1.25x input
        self._cache_read_price_per_m = 0.30    # 0.1x input

    async def decide(self, system_prompt: str, user_prompt: str) -> Dict:
        """
        Make a trading decision via Claude with prompt caching.

        The system prompt is marked cacheable since it stays the same
        across cycles. Cache hits reduce input cost by 90%.

        Args:
            system_prompt: System instructions with risk rules and output format
            user_prompt: Account state and candidate packages

        Returns:
            Normalized response dict
        """
        return await self._call_api(system_prompt, user_prompt, use_cache=True)

    async def review(self, system_prompt: str, user_prompt: str) -> Dict:
        """
        Perform a review analysis via Claude.

        Args:
            system_prompt: System instructions for the review
            user_prompt: Data to review

        Returns:
            Normalized response dict
        """
        return await self._call_api(system_prompt, user_prompt, use_cache=False)

    async def _call_api(
        self, system_prompt: str, user_prompt: str, use_cache: bool = False
    ) -> Dict:
        """
        Make an API call to Claude and return a normalized response.

        Args:
            system_prompt: System message content
            user_prompt: User message content
            use_cache: Whether to use prompt caching on the system prompt

        Returns:
            Normalized response dict
        """
        try:
            if use_cache:
                system = [
                    {
                        'type': 'text',
                        'text': system_prompt,
                        'cache_control': {'type': 'ephemeral'},
                    }
                ]
            else:
                system = system_prompt

            response = await self._client.messages.create(
                model=self._decision_model,
                max_tokens=4096,
                system=system,
                messages=[
                    {'role': 'user', 'content': user_prompt}
                ],
            )

            content = response.content[0].text
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            cache_creation = getattr(response.usage, 'cache_creation_input_tokens', 0) or 0
            cache_read = getattr(response.usage, 'cache_read_input_tokens', 0) or 0
            cost = self.get_cost_estimate(
                prompt_tokens, completion_tokens, cache_creation, cache_read
            )

            cache_status = ''
            if cache_read > 0:
                cache_status = f' cache_hit={cache_read}'
            elif cache_creation > 0:
                cache_status = f' cache_write={cache_creation}'

            logger.info(
                f"Claude API call: model={self._decision_model} "
                f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} "
                f"cost=${cost:.4f}{cache_status}"
            )

            return self._build_response(
                success=True,
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read,
            )

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return self._build_response(
                success=False,
                content='',
                prompt_tokens=0,
                completion_tokens=0,
                error=str(e),
            )

    def get_cost_estimate(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> float:
        """
        Calculate estimated cost with cache-aware pricing.

        Cache writes cost 1.25x input price. Cache reads cost 0.1x input price.
        Uncached input tokens are charged at the standard rate.

        Args:
            prompt_tokens: Number of uncached input tokens
            completion_tokens: Number of output tokens
            cache_creation_input_tokens: Tokens written to cache
            cache_read_input_tokens: Tokens read from cache

        Returns:
            Estimated cost in USD
        """
        input_cost = (prompt_tokens / 1_000_000) * self._input_price_per_m
        output_cost = (completion_tokens / 1_000_000) * self._output_price_per_m
        cache_write_cost = (cache_creation_input_tokens / 1_000_000) * self._cache_write_price_per_m
        cache_read_cost = (cache_read_input_tokens / 1_000_000) * self._cache_read_price_per_m
        return input_cost + output_cost + cache_write_cost + cache_read_cost

    @property
    def provider_name(self) -> str:
        """Return 'claude'."""
        return 'claude'

    @property
    def model_name(self) -> str:
        """Return the active model name."""
        return self._decision_model
