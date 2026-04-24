"""
Google Gemini provider implementation for Project Atlas.

Wraps the Google GenAI SDK and normalizes responses to the
DecisionProvider interface.
"""

import logging
import os
from typing import Dict

from google import genai

from src.engine.providers.base import DecisionProvider
from src.utils.logging_config import setup_file_logger

logger = setup_file_logger(__name__, 'gemini_provider')


class GeminiProvider(DecisionProvider):
    """
    Google Gemini provider for trading decisions.

    Not used for EOD review — that is always Claude per CLAUDE.md Section 13.4.
    Model name loaded from .env — never hardcoded.
    """

    def __init__(self):
        """Initialize the Gemini provider."""
        self._api_key = os.getenv('GEMINI_API_KEY')
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY not set in .env")

        self._decision_model = os.getenv('GEMINI_DECISION_MODEL')
        if not self._decision_model:
            raise ValueError("GEMINI_DECISION_MODEL not set in .env")

        self._client = genai.Client(api_key=self._api_key)

        # Gemini Flash free tier — $0 for now, but track tokens for limits
        self._input_price_per_m = 0.0
        self._output_price_per_m = 0.0

    async def decide(self, system_prompt: str, user_prompt: str) -> Dict:
        """
        Make a trading decision via Gemini.

        Args:
            system_prompt: System instructions with risk rules and output format
            user_prompt: Account state and candidate packages

        Returns:
            Normalized response dict
        """
        return await self._call_api(system_prompt, user_prompt)

    async def review(self, system_prompt: str, user_prompt: str) -> Dict:
        """
        Perform a review analysis via Gemini.

        Args:
            system_prompt: System instructions for the review
            user_prompt: Data to review

        Returns:
            Normalized response dict
        """
        return await self._call_api(system_prompt, user_prompt)

    async def _call_api(self, system_prompt: str, user_prompt: str) -> Dict:
        """
        Make an API call to Gemini and return a normalized response.

        Args:
            system_prompt: System message content
            user_prompt: User message content

        Returns:
            Normalized response dict
        """
        try:
            response = await self._client.aio.models.generate_content(
                model=self._decision_model,
                contents=user_prompt,
                config={
                    'system_instruction': system_prompt,
                },
            )

            content = response.text
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count
            cost = self.get_cost_estimate(prompt_tokens, completion_tokens)

            logger.info(
                f"Gemini API call: model={self._decision_model} "
                f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} "
                f"cost=${cost:.4f}"
            )

            return self._build_response(
                success=True,
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
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
        Calculate estimated cost based on Gemini pricing.

        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            cache_creation_input_tokens: Unused — included for interface compatibility
            cache_read_input_tokens: Unused — included for interface compatibility

        Returns:
            Estimated cost in USD (free tier = 0.0)
        """
        input_cost = (prompt_tokens / 1_000_000) * self._input_price_per_m
        output_cost = (completion_tokens / 1_000_000) * self._output_price_per_m
        return input_cost + output_cost

    @property
    def provider_name(self) -> str:
        """Return 'gemini'."""
        return 'gemini'

    @property
    def model_name(self) -> str:
        """Return the active model name."""
        return self._decision_model
