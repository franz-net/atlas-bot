"""
Provider Abstraction Tests

Verifies:
- Base interface contract enforcement
- ClaudeProvider normalizes responses correctly
- GeminiProvider normalizes responses correctly
- Factory function creates correct provider
- Provider responses always have the same shape
- Error handling returns normalized error responses
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.engine.providers.base import DecisionProvider


@pytest.fixture
def mock_claude_env(monkeypatch):
    """Set Claude provider env vars."""
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-anthropic-key')
    monkeypatch.setenv('CLAUDE_DECISION_MODEL', 'claude-sonnet-4-20250514')
    monkeypatch.setenv('CLAUDE_REVIEW_MODEL', 'claude-opus-4-20250514')
    monkeypatch.setenv('DECISION_PROVIDER', 'claude')


@pytest.fixture
def mock_gemini_env(monkeypatch):
    """Set Gemini provider env vars."""
    monkeypatch.setenv('GEMINI_API_KEY', 'test-gemini-key')
    monkeypatch.setenv('GEMINI_DECISION_MODEL', 'gemini-2.0-flash')
    monkeypatch.setenv('DECISION_PROVIDER', 'gemini')


REQUIRED_RESPONSE_KEYS = {
    'success', 'content', 'prompt_tokens', 'completion_tokens',
    'cost_estimate', 'provider', 'model', 'error',
    'cache_creation_input_tokens', 'cache_read_input_tokens',
}


class TestBaseInterface:
    """Verify the abstract base class cannot be instantiated directly."""

    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            DecisionProvider()

    def test_base_has_required_methods(self):
        assert hasattr(DecisionProvider, 'decide')
        assert hasattr(DecisionProvider, 'review')
        assert hasattr(DecisionProvider, 'get_cost_estimate')
        assert hasattr(DecisionProvider, 'provider_name')
        assert hasattr(DecisionProvider, 'model_name')


class TestClaudeProvider:
    """Verify Claude provider implements the interface correctly."""

    def test_creation(self, mock_claude_env):
        from src.engine.providers.claude import ClaudeProvider
        provider = ClaudeProvider()
        assert provider.provider_name == 'claude'
        assert provider.model_name == 'claude-sonnet-4-20250514'

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
        monkeypatch.setenv('CLAUDE_DECISION_MODEL', 'claude-sonnet-4-20250514')
        from src.engine.providers.claude import ClaudeProvider
        with pytest.raises(ValueError, match='ANTHROPIC_API_KEY'):
            ClaudeProvider()

    def test_missing_model(self, monkeypatch):
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
        monkeypatch.delenv('CLAUDE_DECISION_MODEL', raising=False)
        from src.engine.providers.claude import ClaudeProvider
        with pytest.raises(ValueError, match='CLAUDE_DECISION_MODEL'):
            ClaudeProvider()

    def test_model_override(self, mock_claude_env):
        from src.engine.providers.claude import ClaudeProvider
        provider = ClaudeProvider(model_override='claude-opus-4-20250514')
        assert provider.model_name == 'claude-opus-4-20250514'

    def test_cost_estimate(self, mock_claude_env):
        from src.engine.providers.claude import ClaudeProvider
        provider = ClaudeProvider()
        cost = provider.get_cost_estimate(1000, 500)
        assert cost > 0
        assert isinstance(cost, float)

    def test_cost_estimate_with_cache_write(self, mock_claude_env):
        """Cache writes cost 1.25x input price."""
        from src.engine.providers.claude import ClaudeProvider
        provider = ClaudeProvider()
        cost_no_cache = provider.get_cost_estimate(1000, 500)
        cost_with_write = provider.get_cost_estimate(0, 500, cache_creation_input_tokens=1000)
        # Cache write is 1.25x input, so should be more expensive
        assert cost_with_write > cost_no_cache

    def test_cost_estimate_with_cache_read(self, mock_claude_env):
        """Cache reads cost 0.1x input price — significant savings."""
        from src.engine.providers.claude import ClaudeProvider
        provider = ClaudeProvider()
        cost_no_cache = provider.get_cost_estimate(1000, 500)
        cost_with_read = provider.get_cost_estimate(0, 500, cache_read_input_tokens=1000)
        # Cache read is 0.1x input, so should be much cheaper
        assert cost_with_read < cost_no_cache

    @pytest.mark.asyncio
    async def test_decide_success(self, mock_claude_env):
        from src.engine.providers.claude import ClaudeProvider
        provider = ClaudeProvider()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"action": "HOLD"}')]
        mock_response.usage.input_tokens = 500
        mock_response.usage.output_tokens = 100
        mock_response.usage.cache_creation_input_tokens = 2000
        mock_response.usage.cache_read_input_tokens = 0

        with patch.object(provider._client.messages, 'create',
                          new_callable=AsyncMock, return_value=mock_response):
            result = await provider.decide('system prompt', 'user prompt')

        assert result['success'] is True
        assert result['content'] == '{"action": "HOLD"}'
        assert result['prompt_tokens'] == 500
        assert result['completion_tokens'] == 100
        assert result['cache_creation_input_tokens'] == 2000
        assert result['cache_read_input_tokens'] == 0
        assert result['provider'] == 'claude'
        assert result['model'] == 'claude-sonnet-4-20250514'
        assert result['error'] is None
        assert set(result.keys()) == REQUIRED_RESPONSE_KEYS

    @pytest.mark.asyncio
    async def test_decide_cache_hit(self, mock_claude_env):
        """On subsequent calls, system prompt should be served from cache."""
        from src.engine.providers.claude import ClaudeProvider
        provider = ClaudeProvider()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"action": "HOLD"}')]
        mock_response.usage.input_tokens = 200
        mock_response.usage.output_tokens = 100
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.usage.cache_read_input_tokens = 1800

        with patch.object(provider._client.messages, 'create',
                          new_callable=AsyncMock, return_value=mock_response):
            result = await provider.decide('system prompt', 'user prompt')

        assert result['cache_read_input_tokens'] == 1800
        assert result['cache_creation_input_tokens'] == 0
        # Cost should be lower than without caching
        assert result['cost_estimate'] < provider.get_cost_estimate(2000, 100)

    @pytest.mark.asyncio
    async def test_decide_api_error(self, mock_claude_env):
        from src.engine.providers.claude import ClaudeProvider
        provider = ClaudeProvider()

        with patch.object(provider._client.messages, 'create',
                          new_callable=AsyncMock,
                          side_effect=Exception('API rate limit exceeded')):
            result = await provider.decide('system prompt', 'user prompt')

        assert result['success'] is False
        assert result['content'] == ''
        assert result['error'] == 'API rate limit exceeded'
        assert result['provider'] == 'claude'
        assert set(result.keys()) == REQUIRED_RESPONSE_KEYS

    @pytest.mark.asyncio
    async def test_review_returns_same_shape(self, mock_claude_env):
        from src.engine.providers.claude import ClaudeProvider
        provider = ClaudeProvider()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"review": "good"}')]
        mock_response.usage.input_tokens = 1000
        mock_response.usage.output_tokens = 200
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.usage.cache_read_input_tokens = 0

        with patch.object(provider._client.messages, 'create',
                          new_callable=AsyncMock, return_value=mock_response):
            result = await provider.review('system prompt', 'user prompt')

        assert set(result.keys()) == REQUIRED_RESPONSE_KEYS
        assert result['success'] is True


class TestGeminiProvider:
    """Verify Gemini provider implements the interface correctly."""

    def test_creation(self, mock_gemini_env):
        with patch('src.engine.providers.gemini.genai'):
            from src.engine.providers.gemini import GeminiProvider
            provider = GeminiProvider()
            assert provider.provider_name == 'gemini'
            assert provider.model_name == 'gemini-2.0-flash'

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv('GEMINI_API_KEY', raising=False)
        monkeypatch.setenv('GEMINI_DECISION_MODEL', 'gemini-2.0-flash')
        from src.engine.providers.gemini import GeminiProvider
        with pytest.raises(ValueError, match='GEMINI_API_KEY'):
            GeminiProvider()

    def test_missing_model(self, monkeypatch):
        monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
        monkeypatch.delenv('GEMINI_DECISION_MODEL', raising=False)
        from src.engine.providers.gemini import GeminiProvider
        with pytest.raises(ValueError, match='GEMINI_DECISION_MODEL'):
            GeminiProvider()

    def test_cost_estimate_free_tier(self, mock_gemini_env):
        with patch('src.engine.providers.gemini.genai'):
            from src.engine.providers.gemini import GeminiProvider
            provider = GeminiProvider()
            cost = provider.get_cost_estimate(10000, 5000)
            assert cost == 0.0

    @pytest.mark.asyncio
    async def test_decide_success(self, mock_gemini_env):
        with patch('src.engine.providers.gemini.genai') as mock_genai:
            from src.engine.providers.gemini import GeminiProvider
            provider = GeminiProvider()

            mock_response = MagicMock()
            mock_response.text = '{"action": "HOLD"}'
            mock_response.usage_metadata.prompt_token_count = 800
            mock_response.usage_metadata.candidates_token_count = 150

            provider._client.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )

            result = await provider.decide('system prompt', 'user prompt')

        assert result['success'] is True
        assert result['content'] == '{"action": "HOLD"}'
        assert result['prompt_tokens'] == 800
        assert result['completion_tokens'] == 150
        assert result['provider'] == 'gemini'
        assert result['model'] == 'gemini-2.0-flash'
        assert result['error'] is None
        assert set(result.keys()) == REQUIRED_RESPONSE_KEYS

    @pytest.mark.asyncio
    async def test_decide_api_error(self, mock_gemini_env):
        with patch('src.engine.providers.gemini.genai') as mock_genai:
            from src.engine.providers.gemini import GeminiProvider
            provider = GeminiProvider()

            provider._client.aio.models.generate_content = AsyncMock(
                side_effect=Exception('Quota exceeded')
            )

            result = await provider.decide('system prompt', 'user prompt')

        assert result['success'] is False
        assert result['error'] == 'Quota exceeded'
        assert result['provider'] == 'gemini'
        assert set(result.keys()) == REQUIRED_RESPONSE_KEYS


class TestProviderFactory:
    """Verify the factory function creates the correct provider."""

    def test_create_claude(self, mock_claude_env):
        from src.engine.providers import create_provider
        provider = create_provider('claude')
        assert provider.provider_name == 'claude'

    def test_create_gemini(self, mock_gemini_env):
        with patch('src.engine.providers.gemini.genai'):
            from src.engine.providers import create_provider
            provider = create_provider('gemini')
            assert provider.provider_name == 'gemini'

    def test_create_unknown_raises(self):
        from src.engine.providers import create_provider
        with pytest.raises(ValueError, match='Unknown provider'):
            create_provider('openai')

    def test_case_insensitive(self, mock_claude_env):
        from src.engine.providers import create_provider
        provider = create_provider('Claude')
        assert provider.provider_name == 'claude'

    def test_whitespace_trimmed(self, mock_claude_env):
        from src.engine.providers import create_provider
        provider = create_provider('  claude  ')
        assert provider.provider_name == 'claude'


class TestResponseNormalization:
    """Verify both providers return identical response shapes."""

    @pytest.mark.asyncio
    async def test_claude_and_gemini_same_keys(self, mock_claude_env, mock_gemini_env):
        from src.engine.providers.claude import ClaudeProvider

        claude = ClaudeProvider()
        mock_claude_resp = MagicMock()
        mock_claude_resp.content = [MagicMock(text='{"action":"HOLD"}')]
        mock_claude_resp.usage.input_tokens = 100
        mock_claude_resp.usage.output_tokens = 50
        mock_claude_resp.usage.cache_creation_input_tokens = 0
        mock_claude_resp.usage.cache_read_input_tokens = 0

        with patch.object(claude._client.messages, 'create',
                          new_callable=AsyncMock, return_value=mock_claude_resp):
            claude_result = await claude.decide('sys', 'usr')

        with patch('src.engine.providers.gemini.genai'):
            from src.engine.providers.gemini import GeminiProvider
            gemini = GeminiProvider()

            mock_gemini_resp = MagicMock()
            mock_gemini_resp.text = '{"action":"HOLD"}'
            mock_gemini_resp.usage_metadata.prompt_token_count = 100
            mock_gemini_resp.usage_metadata.candidates_token_count = 50

            gemini._client.aio.models.generate_content = AsyncMock(
                return_value=mock_gemini_resp
            )

            gemini_result = await gemini.decide('sys', 'usr')

        assert set(claude_result.keys()) == set(gemini_result.keys())
        assert set(claude_result.keys()) == REQUIRED_RESPONSE_KEYS
