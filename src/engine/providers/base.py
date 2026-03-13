"""
Abstract base class for AI decision providers.

All providers must implement this interface. decision_engine.py calls these
methods without knowing which provider is active.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class DecisionProvider(ABC):
    """
    Base interface for AI decision providers.

    Every provider returns the same normalized response shape regardless
    of the underlying SDK.
    """

    @abstractmethod
    async def decide(self, system_prompt: str, user_prompt: str) -> Dict:
        """
        Make a trading decision based on candidate data.

        Args:
            system_prompt: System instructions including risk rules and output format
            user_prompt: Account state, candidate packages, and session context

        Returns:
            Normalized response dict:
                success: bool
                content: str (raw response text)
                prompt_tokens: int
                completion_tokens: int
                cost_estimate: float
                provider: str
                model: str
                error: Optional[str]
                cache_creation_input_tokens: int
                cache_read_input_tokens: int
        """
        raise NotImplementedError

    @abstractmethod
    async def review(self, system_prompt: str, user_prompt: str) -> Dict:
        """
        Perform a review analysis (e.g., EOD review, post-mortem).

        Args:
            system_prompt: System instructions for the review
            user_prompt: Data to review (trades, performance, etc.)

        Returns:
            Normalized response dict (same shape as decide())
        """
        raise NotImplementedError

    @abstractmethod
    def get_cost_estimate(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> float:
        """
        Calculate estimated cost for a given token count.

        Args:
            prompt_tokens: Number of input tokens (uncached)
            completion_tokens: Number of output tokens
            cache_creation_input_tokens: Tokens written to cache (1.25x input price)
            cache_read_input_tokens: Tokens read from cache (0.1x input price)

        Returns:
            Estimated cost in USD
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier (e.g., 'claude', 'gemini')."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier being used."""
        raise NotImplementedError

    def _build_response(
        self,
        success: bool,
        content: str,
        prompt_tokens: int,
        completion_tokens: int,
        error: Optional[str] = None,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> Dict:
        """
        Build the normalized response dict.

        Args:
            success: Whether the API call succeeded
            content: Raw response text
            prompt_tokens: Input token count
            completion_tokens: Output token count
            error: Error message if failed
            cache_creation_input_tokens: Tokens written to cache this call
            cache_read_input_tokens: Tokens read from cache (cache hit)

        Returns:
            Normalized response dict
        """
        cost = self.get_cost_estimate(
            prompt_tokens, completion_tokens,
            cache_creation_input_tokens, cache_read_input_tokens,
        )
        return {
            'success': success,
            'content': content,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'cost_estimate': cost,
            'provider': self.provider_name,
            'model': self.model_name,
            'error': error,
            'cache_creation_input_tokens': cache_creation_input_tokens,
            'cache_read_input_tokens': cache_read_input_tokens,
        }
