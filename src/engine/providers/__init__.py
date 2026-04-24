"""
AI Provider abstraction layer for Project Atlas.

Supports swappable providers (Claude, Gemini) selected at startup via .env.
"""

from src.engine.providers.base import DecisionProvider
from src.engine.providers.claude import ClaudeProvider
from src.engine.providers.gemini import GeminiProvider


def create_provider(provider_name: str) -> DecisionProvider:
    """
    Factory function to create the appropriate AI provider.

    Args:
        provider_name: Provider identifier from .env DECISION_PROVIDER ('claude' or 'gemini')

    Returns:
        Configured DecisionProvider instance
    """
    providers = {
        'claude': ClaudeProvider,
        'gemini': GeminiProvider,
    }

    provider_name = provider_name.lower().strip()
    if provider_name not in providers:
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Valid providers: {', '.join(providers.keys())}"
        )

    return providers[provider_name]()
