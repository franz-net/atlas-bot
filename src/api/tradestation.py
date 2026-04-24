"""
TradeStation API Client for Midas Trading Platform
"""

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd
from dotenv import load_dotenv

from src.config.constants import (
    API_RATE_LIMIT_CALLS,
    API_RATE_LIMIT_PERIOD_SECONDS,
    BRIEF_ASYNC_SLEEP_SECONDS,
    DEFAULT_API_TIMEOUT_SECONDS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TICK_SIZE,
    ORDER_VERIFICATION_TIMEOUT_SECONDS,
    SHORT_ASYNC_SLEEP_SECONDS,
)

load_dotenv()


# ==================== HELPER FUNCTIONS (KISS Principle) ====================

def atomic_file_write(file_path: str, content: str) -> bool:
    """
    Atomically write content to file using temp file + rename.

    Prevents file corruption if process crashes during write.

    Args:
        file_path: Target file path
        content: Content to write

    Returns:
        True if write succeeded
    """
    try:
        file_path_obj = Path(file_path)
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file in same directory (for atomic rename)
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=file_path_obj.parent,
            delete=False,
            suffix='.tmp'
        ) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Atomic rename (POSIX guarantees atomicity)
        Path(tmp_path).replace(file_path)

        # Set restrictive permissions
        os.chmod(file_path, 0o600)

        return True

    except Exception as e:
        logging.getLogger(__name__).error(f"Atomic file write failed for {file_path}: {e}")
        # Clean up temp file if it exists
        try:
            if 'tmp_path' in locals():
                Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            # Explicitly catch Exception to avoid catching system exits
            pass
        return False


def ensure_session_cleanup(client_instance: 'TradeStationClient') -> None:
    """
    Ensure aiohttp session is properly closed.

    Args:
        client_instance: TradeStationClient instance
    """
    if client_instance.session and not client_instance.session.closed:
        # Schedule session close (can't await in synchronous context)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(client_instance.session.close())
            else:
                loop.run_until_complete(client_instance.session.close())
        except Exception as e:
            logging.getLogger(__name__).warning(f"Session cleanup warning: {e}")

# Import symlink helper
from src.utils.logging_config import create_or_update_log_symlink

# Configure logging - FILE ONLY, NO STDOUT (to prevent pipe blocking)
logger = logging.getLogger(__name__)

# Use INFO level by default (DEBUG only when explicitly needed)
# Set to DEBUG via: logger.setLevel(logging.DEBUG) for troubleshooting
log_level = os.getenv('TS_API_LOG_LEVEL', 'INFO').upper()
logger.setLevel(getattr(logging, log_level, logging.INFO))

# Remove any existing handlers to prevent STDOUT output
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add timestamped file handler for rolling logs
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f'logs/tradestation_client_{timestamp}.log'
file_handler = logging.FileHandler(log_file)
# Allow DEBUG messages when logger is set to DEBUG, otherwise INFO
file_handler.setLevel(logging.DEBUG if log_level == 'DEBUG' else logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Create symlink to latest log file
create_or_update_log_symlink(Path(log_file), 'tradestation_client')

logger.info(f"📝 TradeStation client logging to: {log_file}")

# Prevent propagation to root logger (which might have STDOUT handlers)
logger.propagate = False

class RateLimiter:
    """Rate limiter for TradeStation API (500 requests/min)"""
    def __init__(self, max_calls: int = API_RATE_LIMIT_CALLS, period: int = API_RATE_LIMIT_PERIOD_SECONDS):
        self.max_calls = max_calls
        self.period = period
        self.calls = []

    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = time.time()
            # Remove old calls outside the period
            self.calls = [call for call in self.calls if call > now - self.period]

            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                logger.warning(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
                return await wrapper(*args, **kwargs)

            self.calls.append(now)
            return await func(*args, **kwargs)
        return wrapper

class TradeStationClient:
    """
    TradeStation API client with OAuth2 authentication
    """

    def __init__(self):
        self.api_key = os.getenv('TS_API_KEY')
        self.api_secret = os.getenv('TS_API_SECRET')
        self.account_id = os.getenv('TS_ACCOUNT_ID')
        self.base_url = os.getenv('TS_BASE_URL', 'https://api.tradestation.com/v3')
        self.sim_base_url = os.getenv('TS_SIM_URL', 'https://sim-api.tradestation.com/v3')
        self.redirect_uri = os.getenv('TS_REDIRECT_URI', 'http://localhost:8080/callback')
        self.use_sim = os.getenv('USE_SIM_ACCOUNT', 'true').lower() == 'true'

        self.access_token = None
        self.refresh_token = None
        self.token_expires = None
        self.session = None
        self.rate_limiter = RateLimiter()

    async def __aenter__(self) -> 'TradeStationClient':
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        await self.authenticate()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def __del__(self):
        """Destructor to ensure session cleanup"""
        ensure_session_cleanup(self)

    async def authenticate(self) -> bool:
        """
        Authenticate with TradeStation OAuth2 using Authorization Code flow
        """
        try:
            # Check if we have a stored refresh token
            refresh_token = self._load_refresh_token()
            if refresh_token:
                logger.info("Found existing refresh token, attempting to refresh access token...")
                if await self._refresh_access_token(refresh_token):
                    return True
                else:
                    logger.warning("Refresh token invalid, need new authorization")

            # No valid token available
            logger.error("No valid authentication token found")
            logger.error("Please run: python main.py auth")
            return False

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False

    def _build_authorization_url(self) -> str:
        """Build the authorization URL for OAuth2 flow"""
        import urllib.parse

        params = {
            'response_type': 'code',
            'client_id': self.api_key,
            'audience': 'https://api.tradestation.com',
            'redirect_uri': self.redirect_uri,
            'scope': 'openid offline_access MarketData ReadAccount Trade',
            'state': 'midas_auth_state'
        }

        query_string = urllib.parse.urlencode(params)
        return f"https://signin.tradestation.com/authorize?{query_string}"

    async def _exchange_code_for_tokens(self, auth_code: str) -> bool:
        """Exchange authorization code for access and refresh tokens"""
        try:
            token_url = "https://signin.tradestation.com/oauth/token"

            data = {
                'grant_type': 'authorization_code',
                'client_id': self.api_key,
                'client_secret': self.api_secret,
                'code': auth_code,
                'redirect_uri': self.redirect_uri
            }

            async with self.session.post(token_url, data=data) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.access_token = token_data['access_token']
                    self.refresh_token = token_data.get('refresh_token')
                    self.token_expires = datetime.now() + timedelta(
                        seconds=token_data.get('expires_in', 1200)  # 20 minutes default
                    )

                    # Store refresh token for future use
                    if self.refresh_token:
                        self._store_refresh_token(self.refresh_token)

                    # Record initial auth date for token age tracking
                    self._store_auth_date()

                    logger.info("Successfully authenticated with TradeStation")
                    return True
                else:
                    error = await response.text()
                    logger.error(f"Token exchange failed: {error}")
                    return False

        except Exception as e:
            logger.error(f"Token exchange error: {e}")
            return False

    async def _refresh_access_token(self, refresh_token: str) -> bool:
        """Refresh access token using refresh token"""
        try:
            token_url = "https://signin.tradestation.com/oauth/token"

            data = {
                'grant_type': 'refresh_token',
                'client_id': self.api_key,
                'client_secret': self.api_secret,
                'refresh_token': refresh_token
            }

            async with self.session.post(token_url, data=data) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.access_token = token_data['access_token']

                    # Update refresh token if provided
                    if 'refresh_token' in token_data:
                        self.refresh_token = token_data['refresh_token']
                        self._store_refresh_token(self.refresh_token)

                    self.token_expires = datetime.now() + timedelta(
                        seconds=token_data.get('expires_in', 1200)
                    )

                    # Warn if auth session is getting old
                    auth_age = self.get_auth_age_days()
                    if auth_age and auth_age > 25:
                        logger.warning(f"⚠️  Token auth is {auth_age:.0f} days old — consider re-authenticating soon")

                    logger.info("Successfully refreshed access token")
                    return True
                else:
                    error = await response.text()
                    logger.error(f"Token refresh failed: {error}")
                    return False

        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return False

    def _store_auth_date(self) -> None:
        """Record when initial OAuth2 auth was performed (for token age tracking)."""
        try:
            auth_date_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', '.ts_auth_date')
            atomic_file_write(auth_date_file, datetime.now().isoformat())
        except Exception as e:
            logger.warning(f"Could not store auth date: {e}")

    def get_auth_age_days(self) -> Optional[float]:
        """Return days since last initial OAuth2 authentication, or None if unknown."""
        try:
            auth_date_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', '.ts_auth_date')
            if os.path.exists(auth_date_file):
                with open(auth_date_file, 'r') as f:
                    auth_date = datetime.fromisoformat(f.read().strip())
                return (datetime.now() - auth_date).total_seconds() / 86400
        except Exception:
            pass
        return None

    def _store_refresh_token(self, refresh_token: str) -> None:
        """Store refresh token securely with atomic write"""
        try:
            token_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', '.ts_token')

            # Use atomic write to prevent corruption
            if atomic_file_write(token_file, refresh_token):
                logger.debug(f"✅ Refresh token stored securely to {token_file}")
            else:
                logger.warning(f"⚠️  Atomic token write failed")

        except Exception as e:
            logger.warning(f"Could not store refresh token: {e}")

    def _load_refresh_token(self) -> Optional[str]:
        """Load stored refresh token"""
        try:
            import os
            token_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', '.ts_token')

            if os.path.exists(token_file):
                with open(token_file, 'r') as f:
                    return f.read().strip()

        except Exception as e:
            logger.warning(f"Could not load refresh token: {e}")

        return None

    async def _check_token(self) -> None:
        """Check and refresh token if needed, ensuring session is initialized"""
        # Ensure session exists before any token operations
        if not self.session:
            self.session = aiohttp.ClientSession()

        # If no token yet, or token expired/expiring, authenticate
        if not self.access_token or (self.token_expires and datetime.now() >= self.token_expires - timedelta(minutes=5)):
            await self.authenticate()

    async def initialize(self) -> None:
        """Initialize the client session and authenticate"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        await self.authenticate()

    @RateLimiter()
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """
        Make authenticated API request with retry logic and timeouts.
        """
        await self._check_token()

        url = f"{self.sim_base_url if self.use_sim else self.base_url}/{endpoint}"
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }

        # IDEMPOTENCY: Add unique key for POST/PUT to prevent duplicate orders on retry
        if method in ['POST', 'PUT']:
            headers['Idempotency-Key'] = str(uuid.uuid4())

        # Add timeout if not specified
        if 'timeout' not in kwargs:
            kwargs['timeout'] = aiohttp.ClientTimeout(total=DEFAULT_API_TIMEOUT_SECONDS)

        # Log the outgoing request
        request_body = kwargs.get('json', {})
        logger.info(f"🌐 API REQUEST: {method} {endpoint}")
        logger.debug(f"   URL: {url}")
        logger.debug(f"   Body: {request_body}")

        max_retries = DEFAULT_MAX_RETRIES
        for attempt in range(max_retries):
            try:
                async with self.session.request(
                    method, url, headers=headers, **kwargs
                ) as response:
                    # Log response details
                    logger.info(f"📡 API RESPONSE: {method} {endpoint} - Status={response.status}")
                    logger.debug(f"   Response headers: {dict(response.headers)}")

                    if response.status in [200, 201, 202, 204]:
                        # Try to parse JSON response
                        try:
                            result = await response.json()

                            # Validate result is not None
                            if result is None:
                                logger.warning(f"⚠️  API returned None JSON for {method} {endpoint}")
                                return {}

                            logger.info(f"✅ API SUCCESS: {method} {endpoint} ({response.status})")

                            # Only log response summary at DEBUG level (not full JSON to avoid massive logs)
                            if logger.isEnabledFor(logging.DEBUG):
                                result_str = str(result)
                                if len(result_str) > 500:
                                    logger.debug(f"   Response (truncated): {result_str[:500]}... [+{len(result_str)-500} chars]")
                                else:
                                    logger.debug(f"   Response: {result}")

                            return result

                        except Exception as json_err:
                            # Some success responses (like 204) might not have JSON body
                            logger.info(f"✅ API SUCCESS: {method} {endpoint} ({response.status}) - No JSON body")
                            logger.debug(f"   JSON parse error: {json_err}")
                            text_response = await response.text()
                            logger.debug(f"   Raw response text: {text_response}")
                            return {}
                    elif response.status == 401:  # Unauthorized
                        logger.warning(f"⚠️  API UNAUTHORIZED: {method} {endpoint} - Re-authenticating...")
                        await self.authenticate()
                        continue
                    elif response.status == 429:  # Too Many Requests
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"⚠️  API RATE LIMITED: {method} {endpoint} - Retrying after {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    else:
                        error = await response.text()
                        # Log all failures at INFO level for auditing
                        logger.info(f"❌ API FAILED: {method} {endpoint} (status={response.status})")
                        logger.info(f"   Error: {error}")
                        return None

            except Exception as e:
                logger.error(f"❌ API EXCEPTION: {method} {endpoint} (attempt {attempt + 1}/{max_retries})")
                logger.error(f"   Exception: {e}")
                if attempt < max_retries - 1:
                    backoff = 2 ** attempt
                    logger.debug(f"   Retrying in {backoff}s...")
                    await asyncio.sleep(backoff)  # Exponential backoff

        logger.error(f"❌ API REQUEST EXHAUSTED: {method} {endpoint} - All {max_retries} retries failed")
        return None

    async def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Get comprehensive symbol information from TradeStation API.

        Returns symbol details including contract specs, point value, tick size, etc.

        Args:
            symbol: Symbol to query (e.g., '@PL', '@MNQ', 'MSFT')

        Returns:
            Dict with symbol info including:
                - AssetType: 'FUTURE', 'STOCK', etc.
                - Description: Human-readable name
                - PriceFormat: {PointValue, Increment, etc.}
                - QuantityFormat: {MinimumTradeQuantity, etc.}
                - Underlying: Specific contract (for continuous futures)
        """
        try:
            endpoint = f"marketdata/symbols/{symbol}"
            response = await self._make_request("GET", endpoint)

            if response and 'Symbols' in response and len(response['Symbols']) > 0:
                symbol_info = response['Symbols'][0]
                logger.debug(f"✅ Retrieved symbol info for {symbol}: {symbol_info.get('Description', 'N/A')}")
                return symbol_info
            else:
                logger.warning(f"⚠️  No symbol info found for {symbol}")
                return None

        except Exception as e:
            logger.error(f"❌ Error fetching symbol info for {symbol}: {e}")
            return None

    async def resolve_continuous_contract(self, symbol: str) -> Optional[str]:
        """
        Resolve a continuous contract symbol to its underlying specific contract.

        For example: @PL → PLF26 (the actual contract to trade)

        Args:
            symbol: Symbol to resolve (e.g., '@PL', 'PLV25')

        Returns:
            Underlying contract symbol (e.g., 'PLF26') or original symbol if not continuous
        """
        # Only resolve continuous contracts (those starting with @)
        if not symbol.startswith('@'):
            logger.debug(f"Symbol {symbol} is not a continuous contract, using as-is")
            return symbol

        try:
            # Use get_symbol_info which already queries the API
            symbol_info = await self.get_symbol_info(symbol)

            if symbol_info:
                underlying = symbol_info.get('Underlying')

                if underlying:
                    logger.info(f"✅ Resolved continuous contract: {symbol} → {underlying}")
                    return underlying
                else:
                    logger.warning(f"⚠️  No underlying contract found for {symbol}, using as-is")
                    return symbol
            else:
                logger.error(f"❌ Failed to resolve {symbol}: No response from API")
                return symbol

        except Exception as e:
            logger.error(f"❌ Error resolving continuous contract {symbol}: {e}")
            return symbol

    async def get_account(self) -> Dict:
        """Get account information"""
        # For v3 API brokerage endpoints, try to get user accounts
        # Based on swagger pattern: /v2/users/{user_id}/accounts but adapted for v3
        accounts = await self._make_request('GET', 'brokerage/accounts')

        if accounts and isinstance(accounts, dict) and 'Accounts' in accounts:
            # Handle the nested structure: {"Accounts": [...]}
            account_list = accounts['Accounts']
            if account_list and len(account_list) > 0:
                # Find the account that matches our configured account ID, or return the first one
                target_account = None
                for acc in account_list:
                    if acc.get('AccountID', '').lower() == self.account_id.lower():
                        target_account = acc
                        break

                if not target_account:
                    target_account = account_list[0]  # Fallback to first account

                logger.info(f"Found account: {target_account.get('AccountID', 'Unknown')} ({target_account.get('AccountType', 'Unknown')})")
                return target_account
        elif accounts and isinstance(accounts, list) and len(accounts) > 0:
            # Handle direct list structure (fallback)
            account = accounts[0]
            logger.info(f"Found account: {account.get('Key', account.get('AccountID', 'Unknown'))}")
            return account
        else:
            # Log what we actually got from the accounts endpoint
            logger.warning(f"No accounts found. Response was: {accounts}")
            # Fallback to direct account lookup if list doesn't work
            logger.info(f"Trying to get account info for: {self.account_id}")
            return await self._make_request('GET', f'brokerage/accounts/{self.account_id}')

    async def get_balances(self) -> Dict:
        """Get account balances"""
        logger.info(f"📊 GET_BALANCES: Querying account {self.account_id}")
        result = await self._make_request('GET', f'brokerage/accounts/{self.account_id}/balances')
        if result:
            logger.info(f"✅ GET_BALANCES SUCCESS")
            logger.debug(f"   Balances: {result}")
        else:
            logger.warning(f"⚠️  GET_BALANCES returned no data")
        return result

    async def get_positions(self) -> List[Dict]:
        """Get current positions from TradeStation v3 API"""
        logger.info(f"📊 GET_POSITIONS: Querying account {self.account_id}")
        result = await self._make_request('GET', f'brokerage/accounts/{self.account_id}/positions')

        # TradeStation v3 API returns {"Positions": [...], "Errors": [...]}
        positions = []
        if isinstance(result, dict) and 'Positions' in result:
            positions = result['Positions'] if isinstance(result['Positions'], list) else []
        # Fallback for legacy flat list format
        elif isinstance(result, list):
            positions = result

        logger.info(f"✅ GET_POSITIONS SUCCESS: Found {len(positions)} position(s)")
        for pos in positions:
            symbol = pos.get('Symbol', 'unknown')
            qty = pos.get('Quantity', pos.get('NetQuantity', 0))
            logger.debug(f"   Position: {symbol} Qty={qty}")

        return positions

    async def get_orders(self, status: str = 'FILLED') -> List[Dict]:
        """Get orders with specified status. Valid statuses: FILLED, CANCELLED, REJECTED, EXPIRED, etc."""
        params = {'status': status}
        result = await self._make_request('GET', f'brokerage/accounts/{self.account_id}/orders', params=params)
        return result if isinstance(result, list) else []

    async def get_all_orders(self) -> List[Dict]:
        """Get all orders without status filter. Returns orders from today's session."""
        result = await self._make_request('GET', f'brokerage/accounts/{self.account_id}/orders')
        # API may return {"Orders": [...]} or just [...]
        if isinstance(result, dict) and 'Orders' in result:
            return result['Orders']
        return result if isinstance(result, list) else []

    async def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get current market quote for a symbol"""
        try:
            # Try the quotes endpoint first
            result = await self._make_request('GET', f'marketdata/quotes/{symbol}')

            # TradeStation v3 API wraps quotes in {"Quotes": [...]}
            if isinstance(result, dict):
                if 'Quotes' in result and isinstance(result['Quotes'], list) and len(result['Quotes']) > 0:
                    return result['Quotes'][0]
                # Direct dict response (legacy format)
                elif 'Last' in result:
                    return result
            elif isinstance(result, list) and len(result) > 0:
                return result[0]  # Return first quote if multiple

            # Fallback: try stream quotes (not recommended for single quotes)
            result = await self._make_request('GET', f'marketdata/stream/quotes/{symbol}')
            if isinstance(result, dict):
                if 'Quotes' in result and isinstance(result['Quotes'], list) and len(result['Quotes']) > 0:
                    return result['Quotes'][0]
                elif 'Last' in result:
                    return result
            elif isinstance(result, list) and len(result) > 0:
                return result[0]

            return None
        except Exception as e:
            logger.debug(f"Failed to get quote for {symbol}: {e}")
            return None

    def _parse_interval(self, interval: str) -> Tuple[str, str]:
        """
        Parse legacy interval format to new API format

        Args:
            interval: Legacy format like '1min', '5min', '15min', 'daily', etc.

        Returns:
            Tuple of (interval_number, unit) for API
        """
        interval = interval.lower()

        if interval.endswith('min'):
            # Extract number from '1min', '5min', etc.
            number = interval[:-3]
            return (number, 'Minute')
        elif interval in ['daily', 'day']:
            return ('1', 'Daily')
        elif interval in ['weekly', 'week']:
            return ('1', 'Weekly')
        elif interval in ['monthly', 'month']:
            return ('1', 'Monthly')
        else:
            # Default to minute if can't parse
            return ('1', 'Minute')

    async def get_historical_bars(
        self,
        symbol: str,
        interval: str = '1min',
        start_date: str = None,
        end_date: str = None,
        bars_back: int = None
    ) -> pd.DataFrame:
        """
        Get historical bar data

        Args:
            symbol: Trading symbol (e.g., 'MBTU25')
            interval: Bar interval (e.g., '1min', '5min', '15min', 'daily')
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            bars_back: Number of bars to retrieve (alternative to date range)

        Returns:
            DataFrame with OHLCV data
        """
        endpoint = f'marketdata/barcharts/{symbol}'

        # Parse interval format for API
        interval_num, unit = self._parse_interval(interval)

        # Use appropriate session template based on symbol type
        if any(symbol.startswith(prefix) for prefix in ['PL', 'MBT', 'MET', 'MNQ', 'CL', 'NG', 'GC', 'SI']):
            # Futures symbols - try without session template first, then 'Default' if needed
            params = {
                'interval': interval_num,
                'unit': unit
                # No sessiontemplate for futures - let API use default trading hours
            }
        else:
            # Equity symbols - use pre/post market
            params = {
                'interval': interval_num,
                'unit': unit,
                'sessiontemplate': 'USEQPreAndPost'
            }

        if bars_back:
            params['barsback'] = bars_back
        else:
            if start_date:
                # Convert YYYY-MM-DD to ISO 8601 format with UTC timezone
                if len(start_date) == 10:  # YYYY-MM-DD format
                    params['firstdate'] = f"{start_date}T00:00:00Z"
                else:
                    params['firstdate'] = start_date
            if end_date:
                # Convert YYYY-MM-DD to ISO 8601 format with UTC timezone
                if len(end_date) == 10:  # YYYY-MM-DD format
                    # Check if end_date is today, if so use current UTC time instead of 23:59:59
                    today_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d')
                    if end_date == today_utc:
                        # Use current UTC time for today to avoid future date error
                        current_utc = datetime.now(timezone.utc)
                        params['lastdate'] = current_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
                    else:
                        params['lastdate'] = f"{end_date}T23:59:59Z"
                else:
                    params['lastdate'] = end_date

        data = await self._make_request('GET', endpoint, params=params)

        if data and 'Bars' in data:
            df = pd.DataFrame(data['Bars'])
            if not df.empty:
                # Convert timestamp and set as index
                df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
                df.set_index('TimeStamp', inplace=True)

                # Rename columns to standardized names, but only if they exist
                column_mapping = {
                    'Open': 'Open',
                    'High': 'High',
                    'Low': 'Low',
                    'Close': 'Close',
                    'Volume': 'Volume',
                    'TotalVolume': 'Volume',  # Some responses use TotalVolume
                    'OpenInterest': 'OpenInterest'
                }

                # Apply column mapping for existing columns only
                existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
                df = df.rename(columns=existing_mapping)

                # Ensure we have the required columns, fill missing ones with 0
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest']
                for col in required_columns:
                    if col not in df.columns:
                        df[col] = 0

                # Convert to numeric types
                numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

                # Reorder columns to standard format
                df = df[required_columns]
                return df

        return pd.DataFrame()

    async def get_historical_bars_batched(
        self,
        symbol: str,
        interval: str = '1min',
        start_date: str = None,
        end_date: str = None,
        days_back: int = None
    ) -> pd.DataFrame:
        """
        Get historical bar data with automatic batching for large requests

        This method automatically splits large requests into smaller batches
        to respect the 57,600 bar limit and then combines the results.

        Args:
            symbol: Trading symbol (e.g., 'MBTU25')
            interval: Bar interval (e.g., '1min', '5min', '15min', 'daily')
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            days_back: Number of days to retrieve (alternative to date range)

        Returns:
            DataFrame with OHLCV data
        """
        # Calculate date range if days_back is provided
        if days_back:
            # Use UTC time to avoid "future date" API errors
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=days_back)
            start_date = start_dt.strftime('%Y-%m-%d')
            end_date = end_dt.strftime('%Y-%m-%d')

        if not start_date or not end_date:
            # If no date range specified, use single request
            return await self.get_historical_bars(symbol, interval, start_date, end_date)

        # Parse interval to determine batch size
        interval_num, unit = self._parse_interval(interval)

        # Calculate maximum days per batch based on 57,600 bar limit
        # Be very conservative with calculations to avoid API limits
        if unit == 'Minute':
            bars_per_day = int(390 / int(interval_num))  # 390 minutes per trading day
            # Use much smaller safety margin and account for weekends/holidays
            max_days_per_batch = min(50000 // bars_per_day, 30)  # Max 30 days per batch
        else:
            # For daily/weekly/monthly, no batching needed
            max_days_per_batch = 9999

        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        total_days = (end_dt - start_dt).days

        if total_days <= max_days_per_batch:
            # Single request is sufficient
            logger.info(f"Single request for {symbol} {interval}: {total_days} days")
            return await self.get_historical_bars(symbol, interval, start_date, end_date)

        # Need to batch the requests
        logger.info(f"Batching request for {symbol} {interval}: {total_days} days in {max_days_per_batch}-day chunks")

        all_data = []
        current_start = start_dt
        batch_num = 1

        while current_start < end_dt:
            batch_end = min(current_start + timedelta(days=max_days_per_batch), end_dt)

            batch_start_str = current_start.strftime('%Y-%m-%d')
            batch_end_str = batch_end.strftime('%Y-%m-%d')

            logger.info(f"Batch {batch_num}: {batch_start_str} to {batch_end_str}")

            try:
                batch_df = await self.get_historical_bars(
                    symbol=symbol,
                    interval=interval,
                    start_date=batch_start_str,
                    end_date=batch_end_str
                )

                if not batch_df.empty:
                    all_data.append(batch_df)
                    logger.info(f"Batch {batch_num}: Got {len(batch_df)} bars")
                else:
                    logger.warning(f"Batch {batch_num}: No data returned")

                # Small delay between requests to be respectful to API
                await asyncio.sleep(BRIEF_ASYNC_SLEEP_SECONDS)

            except Exception as e:
                logger.error(f"Batch {batch_num} failed: {e}")
                # Continue with next batch rather than failing completely

            current_start = batch_end + timedelta(days=1)
            batch_num += 1

        # Combine all batches
        if all_data:
            combined_df = pd.concat(all_data, axis=0)
            # Remove any duplicate timestamps and sort
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
            combined_df = combined_df.sort_index()
            logger.info(f"Combined {len(all_data)} batches into {len(combined_df)} total bars")
            return combined_df
        else:
            logger.warning(f"No data retrieved for {symbol} {interval}")
            return pd.DataFrame()

    async def stream_quotes(self, symbols: List[str], callback) -> None:
        """
        Stream real-time quotes

        Args:
            symbols: List of symbols to stream
            callback: Async function to handle quote updates
        """
        # WebSocket implementation for real-time data
        ws_url = "wss://stream.tradestation.com/v3/marketdata/stream"

        # Implementation would use websocket-client library
        # This is a placeholder for the WebSocket streaming logic

    async def place_order(self, order_data: Dict) -> Dict:
        """
        Place an order using TradeStation v3 API

        Args:
            order_data: Order dictionary with required fields

        Returns:
            Order confirmation details
        """
        endpoint = f'orderexecution/orders'
        operation_name = "PLACE_ORDER"

        # Log request (DRY)
        self._log_order_request(order_data, operation_name)

        # Validate required fields (DRY)
        required_fields = ['AccountID', 'Symbol', 'Quantity', 'OrderType', 'TradeAction']
        error_msg = self._validate_required_fields(order_data, required_fields, operation_name)
        if error_msg:
            return {'success': False, 'error': error_msg}

        try:
            result = await self._make_request('POST', endpoint, json=order_data)
            logger.info(f"📥 {operation_name} RESPONSE: {result}")

            # Parse response (DRY)
            return self._parse_order_response(result, operation_name)

        except Exception as e:
            logger.error(f"❌ {operation_name} EXCEPTION: {e}")
            logger.debug(f"   Order data: {order_data}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def _round_to_tick(self, price: float, tick_size: float) -> float:
        """
        Round price to valid tick increment

        Args:
            price: Price to round
            tick_size: Tick size for the instrument (e.g., 0.1 for @PL, 0.25 for @MNQ)

        Returns:
            Price rounded to nearest valid tick
        """
        # Ensure inputs are Python floats (not numpy types or strings)
        price = float(price)
        tick_size = float(tick_size)
        return round(price / tick_size) * tick_size

    def _parse_order_response(self, result: Dict, operation_name: str = "ORDER") -> Dict:
        """
        Parse TradeStation API order response (DRY helper).

        Handles both v3 API format (Orders array) and legacy format.

        Args:
            result: API response dictionary
            operation_name: Operation name for logging (e.g., "PLACE_ORDER", "MODIFY_ORDER")

        Returns:
            Dict with 'success', 'order_id', 'message', 'order_status' or 'error'
        """
        if not result or not isinstance(result, dict):
            return {'success': False, 'error': 'API request returned no response'}

        # Check for v3 API format (Orders array)
        if 'Orders' in result and result['Orders']:
            order = result['Orders'][0]
            if 'OrderID' in order:
                order_id = order['OrderID']

                # CRITICAL: Check for Error field - API returns OrderID even on FAILED orders
                # e.g., {"Error": "FAILED", "Message": "Order requires $X purchasing power...", "OrderID": "123"}
                error_field = order.get('Error', '')
                if error_field and error_field.upper() in ('FAILED', 'REJECT', 'REJECTED', 'ERROR'):
                    error_msg = order.get('Message', f'Order failed: {error_field}')
                    logger.error(f"❌ {operation_name} REJECTED (v3 format): OrderID={order_id}")
                    logger.error(f"   Reason: {error_msg}")
                    return {
                        'success': False,
                        'order_id': order_id,
                        'error': error_msg,
                        'order_status': 'Rejected'
                    }

                logger.info(f"✅ {operation_name} SUCCESS (v3 format): OrderID={order_id}")
                logger.debug(f"   Order details: {order}")
                return {
                    'success': True,
                    'order_id': order_id,
                    'message': order.get('Message', 'Operation successful'),
                    'order_status': 'Submitted'
                }

        # Check for legacy format (single order response)
        if 'OrderID' in result:
            order_id = result['OrderID']
            logger.info(f"✅ {operation_name} SUCCESS (legacy format): OrderID={order_id}")
            logger.debug(f"   Order details: {result}")
            return {
                'success': True,
                'order_id': order_id,
                'message': result.get('Message', 'Operation successful'),
                'order_status': result.get('OrderStatus', 'Pending')
            }

        # Check for errors
        if 'Errors' in result and result['Errors']:
            error = result['Errors'][0]
            error_msg = error.get('Message', error.get('Error', 'Operation failed'))
            logger.error(f"❌ {operation_name} FAILED (API error): {error_msg}")
            logger.debug(f"   Full error: {error}")
            return {'success': False, 'error': error_msg}

        # Unexpected format
        error_msg = result.get('Message', result.get('Error', 'Unknown error'))
        logger.error(f"❌ {operation_name} FAILED (unexpected format): {error_msg}")
        logger.debug(f"   Full response: {result}")
        return {'success': False, 'error': error_msg}

    def _validate_required_fields(self, data: Dict, required_fields: List[str], operation_name: str = "ORDER") -> Optional[str]:
        """
        Validate required fields are present in data (DRY helper).

        Args:
            data: Data dictionary to validate
            required_fields: List of required field names
            operation_name: Operation name for logging

        Returns:
            Error message if validation fails, None if successful
        """
        for field in required_fields:
            if field not in data:
                error_msg = f'Missing required field: {field}'
                logger.error(f"❌ {operation_name} VALIDATION FAILED: {error_msg}")
                return error_msg
        return None

    def _log_order_request(self, order_data: Dict, operation_name: str = "ORDER") -> None:
        """
        Log order request details in standard format (DRY helper).

        Args:
            order_data: Order dictionary
            operation_name: Operation name for logging
        """
        logger.info(f"📋 {operation_name} REQUEST:")
        logger.info(f"   Account: {order_data.get('AccountID')}")
        logger.info(f"   Symbol: {order_data.get('Symbol')}")
        logger.info(f"   Quantity: {order_data.get('Quantity')}")
        logger.info(f"   Action: {order_data.get('TradeAction')}")
        logger.info(f"   Type: {order_data.get('OrderType')}")
        logger.debug(f"   Full data: {order_data}")

    async def verify_orders_placed(self, order_ids: List[str], timeout_seconds: int = ORDER_VERIFICATION_TIMEOUT_SECONDS) -> Dict[str, bool]:
        """
        Verify that orders were successfully placed and are active.

        Args:
            order_ids: List of order IDs to verify
            timeout_seconds: Maximum time to wait for orders to appear

        Returns:
            Dict mapping order_id -> bool (True if found and active)
        """
        verification_results = {order_id: False for order_id in order_ids}

        try:
            # Give the broker a moment to process orders
            await asyncio.sleep(SHORT_ASYNC_SLEEP_SECONDS)

            # Query for orders (both open and recent)
            endpoint = f'brokerage/accounts/{self.account_id}/orders'
            result = await self._make_request('GET', endpoint)

            if not result:
                logger.warning(f"Could not retrieve orders for verification")
                return verification_results

            # Handle both list and dict responses
            orders = result if isinstance(result, list) else result.get('Orders', [])

            if not orders:
                logger.warning(f"No orders in verification response")
                return verification_results

            # Check which order IDs are present
            found_order_ids = {order.get('OrderID') for order in orders if order.get('OrderID')}

            for order_id in order_ids:
                if order_id in found_order_ids:
                    verification_results[order_id] = True
                    logger.debug(f"✅ Order {order_id} verified as active")
                else:
                    logger.warning(f"⚠️  Order {order_id} not found in broker orders")

            return verification_results

        except Exception as e:
            logger.error(f"Error verifying orders: {e}")
            return verification_results

    async def place_bracket_order(self, entry_order: Dict, stop_loss_price: float, take_profit_price: float, tick_size: float = DEFAULT_TICK_SIZE) -> Dict:
        """
        Place a bracket order using OSO (Order Sends Order) group.
        Entry order is placed, and when filled, triggers OCO bracket (stop + limit) for exit.

        Note: BRK order type is only for exiting existing positions (closing transactions).
        OSO allows entry + protective bracket as atomic group.

        Args:
            entry_order: Primary entry order dict with AccountID, Symbol, Quantity, TradeAction
            stop_loss_price: Stop loss price level
            take_profit_price: Take profit price level
            tick_size: Tick size for the instrument (default 0.01 for stocks)

        Returns:
            Dict with success status, order IDs for entry, stop, and take profit
        """
        # Round prices to valid ticks
        stop_loss_price = self._round_to_tick(stop_loss_price, tick_size)
        take_profit_price = self._round_to_tick(take_profit_price, tick_size)

        symbol = entry_order.get('Symbol')
        trade_action = entry_order.get('TradeAction', '').upper()
        quantity = entry_order.get('Quantity')

        logger.info(f"📋 PLACE_BRACKET_ORDER REQUEST:")
        logger.info(f"   Entry: {trade_action} {quantity} {symbol} @ Market")
        logger.info(f"   Stop Loss: ${stop_loss_price:.2f} (tick=${tick_size})")
        logger.info(f"   Take Profit: ${take_profit_price:.2f} (tick=${tick_size})")

        # Note: Duplicate prevention is handled at the trading engine level
        # The engine checks active_positions before calling this method
        # This allows flexibility for future features like position scaling/pyramiding

        try:
            # Determine entry and exit actions with explicit position effects
            # TradeStation bracket orders may require explicit OPEN/CLOSE designations
            entry_action_raw = entry_order.get('TradeAction', '').upper()

            # Map to position-specific actions for bracket orders
            if entry_action_raw == 'BUY':
                entry_action = 'BUY'    # Open long position
                exit_action = 'SELL'   # Close long position (TP and SL)
            elif entry_action_raw == 'SELL':
                entry_action = 'SELL'   # Open short position
                exit_action = 'BUY'    # Close short position (TP and SL)
            else:
                # Fallback to simple BUY/SELL
                entry_action = entry_action_raw
                exit_action = 'SELL' if entry_action_raw == 'BUY' else 'BUY'

            # Create entry order with OSO bracket (Order Sends Order)
            # Entry order is a normal single order, OSOs contain the BRK bracket that triggers on fill
            order_type = entry_order.get('OrderType', 'Market')

            entry_with_bracket = {
                # Entry order fields
                'AccountID': entry_order.get('AccountID'),
                'Symbol': entry_order.get('Symbol'),
                'Quantity': str(entry_order.get('Quantity')),
                'OrderType': order_type,
                'TradeAction': entry_action,  # BUY or SELL for entry
                'TimeInForce': entry_order.get('TimeInForce', {'Duration': 'DAY'}),
                'Route': 'Intelligent'
            }

            # Add LimitPrice only for Limit orders
            if order_type == 'Limit' and entry_order.get('LimitPrice'):
                entry_with_bracket['LimitPrice'] = entry_order.get('LimitPrice')

            # Add OSOs bracket
            entry_with_bracket['OSOs'] = [
                    {
                        'Type': 'BRK',  # Bracket order type for exits
                        'Orders': [
                            # Take profit order (Limit)
                            {
                                'AccountID': entry_order.get('AccountID'),
                                'Symbol': entry_order.get('Symbol'),
                                'Quantity': str(entry_order.get('Quantity')),
                                'OrderType': 'Limit',
                                'TradeAction': exit_action,  # SELL for long, BUY for short
                                'LimitPrice': f"{take_profit_price:.2f}",
                                'TimeInForce': {'Duration': 'GTC'},
                                'Route': 'Intelligent'
                            },
                            # Stop loss order (StopMarket)
                            {
                                'AccountID': entry_order.get('AccountID'),
                                'Symbol': entry_order.get('Symbol'),
                                'Quantity': str(entry_order.get('Quantity')),
                                'OrderType': 'StopMarket',
                                'TradeAction': exit_action,  # SELL for long, BUY for short
                                'StopPrice': f"{stop_loss_price:.2f}",
                                'TimeInForce': {'Duration': 'GTC'},
                                'Route': 'Intelligent'
                            }
                        ]
                    }
                ]

            # Submit as a single order (not order group) - use /orders endpoint
            endpoint = 'orderexecution/orders'  # Changed from ordergroups to orders
            logger.info(f"📤 Placing entry order with OSO bracket...")
            logger.debug(f"📤 ORDER WITH BRACKET PAYLOAD: {json.dumps(entry_with_bracket, indent=2)}")

            result = await self._make_request('POST', endpoint, json=entry_with_bracket)

            logger.info(f"📥 BRACKET RESPONSE TYPE: {type(result)}")
            logger.info(f"📥 BRACKET RESPONSE: {json.dumps(result, indent=2) if result else 'None'}")

            # Parse response to extract order IDs
            if result and isinstance(result, dict) and 'Orders' in result:
                orders = result['Orders']

                if len(orders) < 3:
                    error_msg = f"Bracket order failed: Expected 3 orders, got {len(orders)}"
                    logger.error(f"❌ {error_msg}")
                    return {'success': False, 'error': error_msg}

                # Parse orders from API response
                # API returns: [TP (Limit), SL (Stop), Entry (Limit/Market)]
                # We identify by parsing the Message field
                entry_order_obj = None
                tp_order_obj = None
                sl_order_obj = None

                for order in orders:
                    msg = order.get('Message', '')
                    # Stop Market = Stop Loss
                    if 'Stop Market' in msg or 'Stop Limit' in msg:
                        sl_order_obj = order
                        logger.debug(f"   Identified Stop Loss: {msg}")
                    # Otherwise it's a Limit order - need to distinguish TP from Entry
                    elif 'Limit' in msg:
                        # First limit = TP, second limit = Entry
                        if tp_order_obj is None:
                            tp_order_obj = order
                            logger.debug(f"   Identified Take Profit: {msg}")
                        else:
                            entry_order_obj = order
                            logger.debug(f"   Identified Entry: {msg}")
                    # Market order = Entry
                    elif 'Market' in msg:
                        entry_order_obj = order
                        logger.debug(f"   Identified Entry (Market): {msg}")

                # Verify we found all three
                if not entry_order_obj or not tp_order_obj or not sl_order_obj:
                    logger.error(f"❌ Failed to parse all orders from response:")
                    for i, order in enumerate(orders):
                        logger.error(f"   [{i}] {order.get('Message', 'No message')}")
                    return {'success': False, 'error': 'Could not parse order IDs from response'}

                entry_failed = entry_order_obj.get('Error') == 'FAILED'
                tp_failed = tp_order_obj.get('Error') == 'FAILED'
                sl_failed = sl_order_obj.get('Error') == 'FAILED'

                if entry_failed or tp_failed or sl_failed:
                    logger.error(f"❌ BRACKET ORDER FAILED:")
                    if entry_failed:
                        logger.error(f"   Entry: {entry_order_obj.get('Message', 'Unknown error')}")
                    if tp_failed:
                        logger.error(f"   Take Profit: {tp_order_obj.get('Message', 'Unknown error')}")
                    if sl_failed:
                        logger.error(f"   Stop Loss: {sl_order_obj.get('Message', 'Unknown error')}")

                    return {
                        'success': False,
                        'error': f"Bracket order failed: {entry_order_obj.get('Message', '')}"
                    }

                # Extract order IDs
                entry_order_id = entry_order_obj.get('OrderID')
                tp_order_id = tp_order_obj.get('OrderID')
                sl_order_id = sl_order_obj.get('OrderID')

                logger.info(f"✅ BRACKET ORDER SUCCESS:")
                logger.info(f"   Entry OrderID: {entry_order_id}")
                logger.info(f"   Take Profit OrderID: {tp_order_id}")
                logger.info(f"   Stop Loss OrderID: {sl_order_id}")

                # Verify orders are actually placed and active
                logger.info(f"🔍 Verifying orders are active in broker system...")
                order_ids_to_verify = [entry_order_id, tp_order_id, sl_order_id]
                verification_results = await self.verify_orders_placed(order_ids_to_verify)

                all_verified = all(verification_results.values())
                if all_verified:
                    logger.info(f"✅ All orders verified as active in broker system")
                else:
                    # Log which orders weren't verified
                    for order_id, verified in verification_results.items():
                        if not verified:
                            if order_id == entry_order_id:
                                logger.warning(f"⚠️  Entry order {order_id} not verified")
                            elif order_id == tp_order_id:
                                logger.warning(f"⚠️  Take Profit order {order_id} not verified")
                            elif order_id == sl_order_id:
                                logger.warning(f"⚠️  Stop Loss order {order_id} not verified")

                return {
                    'success': True,
                    'entry_order_id': entry_order_id,
                    'stop_order_id': sl_order_id,
                    'tp_order_id': tp_order_id,
                    'message': 'Bracket order placed successfully',
                    'verified': all_verified,
                    'verification_details': verification_results
                }

            # Unexpected response format
            error_msg = f"Bracket order failed: Unexpected response format: {result}"
            logger.error(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}

        except Exception as e:
            logger.error(f"❌ PLACE_BRACKET_ORDER EXCEPTION: {e}")
            logger.debug(f"   Entry order: {entry_order}", exc_info=True)
            return {'success': False, 'error': str(e)}

    async def place_oco_bracket(self, symbol: str, quantity: int, side: str,
                                stop_price: float, limit_price: float, tick_size: float = DEFAULT_TICK_SIZE,
                                account_id: str = None) -> Dict:
        """
        Place OCO bracket (SL + TP) for an EXISTING position without entry order.

        Used during position recovery when protective orders need to be re-placed
        for positions that are already open at the broker.

        Args:
            symbol: Trading symbol
            quantity: Position quantity (absolute value)
            side: Exit side ('BUY' to close short, 'SELL' to close long)
            stop_price: Stop loss price
            limit_price: Take profit price
            tick_size: Tick size for the instrument
            account_id: Account ID (will use default if not provided)

        Returns:
            Dict with success status and order IDs: {'stop_order_id': ..., 'limit_order_id': ...}
        """
        # Round prices to valid ticks
        stop_price = self._round_to_tick(stop_price, tick_size)
        limit_price = self._round_to_tick(limit_price, tick_size)

        account_id = account_id or self.account_id

        logger.info(f"📋 PLACE_OCO_BRACKET (for existing position):")
        logger.info(f"   Symbol: {symbol}, Quantity: {quantity}, Side: {side}")
        logger.info(f"   Stop Loss: ${stop_price:.2f}, Take Profit: ${limit_price:.2f}")
        logger.info(f"   Tick size: ${tick_size}")

        try:
            # Place SL + TP as a linked BRK (bracket) group via ordergroups endpoint.
            # This ensures TradeStation treats them as a single OCO bracket pair,
            # avoiding the margin issue where two separate SELL orders are seen as
            # opening a new short position.
            bracket_group = {
                'Type': 'BRK',
                'Orders': [
                    # Take profit order (Limit)
                    {
                        'AccountID': account_id,
                        'Symbol': symbol,
                        'Quantity': str(quantity),
                        'OrderType': 'Limit',
                        'TradeAction': side,
                        'LimitPrice': f"{limit_price:.2f}",
                        'TimeInForce': {'Duration': 'GTC'},
                        'Route': 'Intelligent'
                    },
                    # Stop loss order (StopMarket)
                    {
                        'AccountID': account_id,
                        'Symbol': symbol,
                        'Quantity': str(quantity),
                        'OrderType': 'StopMarket',
                        'TradeAction': side,
                        'StopPrice': f"{stop_price:.2f}",
                        'TimeInForce': {'Duration': 'GTC'},
                        'Route': 'Intelligent'
                    }
                ]
            }

            endpoint = 'orderexecution/ordergroups'
            logger.info(f"📤 Placing OCO bracket group via {endpoint}...")
            logger.debug(f"📤 OCO BRACKET PAYLOAD: {json.dumps(bracket_group, indent=2)}")

            result = await self._make_request('POST', endpoint, json=bracket_group)

            logger.info(f"📥 OCO BRACKET RESPONSE: {json.dumps(result, indent=2) if result else 'None'}")

            # Parse response to extract order IDs
            if result and isinstance(result, dict) and 'Orders' in result:
                orders = result['Orders']

                if len(orders) < 2:
                    error_msg = f"OCO bracket failed: Expected 2 orders, got {len(orders)}"
                    logger.error(f"❌ {error_msg}")
                    return {'success': False, 'error': error_msg}

                # Identify SL and TP from response by order type in Message
                tp_order_obj = None
                sl_order_obj = None

                for order in orders:
                    msg = order.get('Message', '')
                    if 'Stop Market' in msg or 'Stop Limit' in msg:
                        sl_order_obj = order
                        logger.debug(f"   Identified Stop Loss: {msg}")
                    elif 'Limit' in msg:
                        tp_order_obj = order
                        logger.debug(f"   Identified Take Profit: {msg}")

                if not tp_order_obj or not sl_order_obj:
                    logger.error(f"❌ Failed to parse OCO bracket orders from response:")
                    for i, order in enumerate(orders):
                        logger.error(f"   [{i}] {order.get('Message', 'No message')}")
                    return {'success': False, 'error': 'Could not parse order IDs from bracket response'}

                # Check for failures (Error field in individual order responses)
                tp_error = tp_order_obj.get('Error', '')
                sl_error = sl_order_obj.get('Error', '')
                tp_failed = tp_error and tp_error.upper() in ('FAILED', 'REJECT', 'REJECTED', 'ERROR')
                sl_failed = sl_error and sl_error.upper() in ('FAILED', 'REJECT', 'REJECTED', 'ERROR')

                if tp_failed or sl_failed:
                    logger.error(f"❌ OCO BRACKET ORDER FAILED:")
                    if sl_failed:
                        logger.error(f"   Stop Loss: {sl_order_obj.get('Message', 'Unknown error')}")
                    if tp_failed:
                        logger.error(f"   Take Profit: {tp_order_obj.get('Message', 'Unknown error')}")
                    return {
                        'success': False,
                        'error': f"OCO bracket failed: SL={sl_order_obj.get('Message', '')}, TP={tp_order_obj.get('Message', '')}"
                    }

                stop_order_id = sl_order_obj.get('OrderID')
                limit_order_id = tp_order_obj.get('OrderID')

                logger.info(f"✅ OCO BRACKET PLACED SUCCESSFULLY (linked BRK group):")
                logger.info(f"   Stop Loss OrderID: {stop_order_id}")
                logger.info(f"   Take Profit OrderID: {limit_order_id}")

                return {
                    'success': True,
                    'stop_order_id': stop_order_id,
                    'limit_order_id': limit_order_id
                }

            # Unexpected response format
            error_msg = f"OCO bracket failed: Unexpected response format: {result}"
            logger.error(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}

        except Exception as e:
            logger.error(f"❌ PLACE_OCO_BRACKET EXCEPTION: {e}")
            logger.debug(f"   Symbol: {symbol}, Side: {side}", exc_info=True)
            return {'success': False, 'error': str(e)}

    async def _modify_order_price(self, order_id: str, new_price: float, quantity: int,
                                   price_field: str, order_type: str, tick_size: float = DEFAULT_TICK_SIZE) -> tuple:
        """
        Generic helper to modify order price (DRY - used by modify_stop_loss and modify_take_profit).

        Args:
            order_id: The broker order ID
            new_price: New price level
            quantity: Order quantity (required by API)
            price_field: Price field name ('StopPrice' or 'LimitPrice')
            order_type: Order type for logging ('Stop Loss' or 'Take Profit')
            tick_size: Tick size for the instrument

        Returns:
            Tuple of (success: bool, new_order_id: str or None)
            - new_order_id is the order ID to track (may be different if broker replaced the order)
        """
        new_price = self._round_to_tick(new_price, tick_size)
        logger.info(f"📝 MODIFY_{order_type.upper().replace(' ', '_')}: OrderID={order_id}, NewPrice=${new_price:.2f}, Qty={quantity}")

        try:
            replace_payload = {
                'Quantity': str(quantity),
                price_field: f"{new_price:.2f}"
            }

            endpoint = f'orderexecution/orders/{order_id}'
            result = await self._make_request('PUT', endpoint, json=replace_payload)

            if result:
                # Log the full response for debugging
                logger.info(f"📥 {order_type} modification response: {result}")

                # Check for error messages in the response
                if isinstance(result, dict):
                    # Check for explicit Error field (actual errors)
                    if result.get('Error'):
                        error_msg = result.get('Error')
                        logger.error(f"❌ {order_type} modification rejected: {error_msg}")
                        return (False, None)

                    # Check Message field - "Cancel/Replace order sent." is SUCCESS, not error
                    message = result.get('Message', '')
                    if message:
                        # These are SUCCESS messages from TradeStation
                        if 'Cancel/Replace order sent' in message or 'order sent' in message.lower():
                            logger.info(f"✅ {order_type} modification accepted: {message}")
                        # Check for actual rejection messages
                        elif 'reject' in message.lower() or 'invalid' in message.lower() or 'error' in message.lower():
                            logger.error(f"❌ {order_type} modification rejected: {message}")
                            return (False, None)

                    # Check if order was replaced (new order ID returned)
                    new_order_id = result.get('OrderID') or result.get('OrderId')
                    if new_order_id and str(new_order_id) != str(order_id):
                        logger.warning(f"⚠️  Order was REPLACED: {order_id} → {new_order_id}")
                        logger.info(f"✅ {order_type} modified successfully (new order ID: {new_order_id})")
                        return (True, str(new_order_id))

                logger.info(f"✅ {order_type} modified successfully")
                return (True, order_id)  # Return original ID if not replaced
            else:
                logger.error(f"❌ {order_type} modification failed: No response")
                return (False, None)

        except Exception as e:
            logger.error(f"❌ Error modifying {order_type.lower()}: {e}")
            return (False, None)

    async def modify_stop_loss(self, order_id: str, new_stop_price: float, quantity: int, tick_size: float = DEFAULT_TICK_SIZE) -> tuple:
        """
        Modify an existing stop loss order using PUT request.

        Args:
            order_id: The broker order ID of the stop loss order
            new_stop_price: New stop price level
            quantity: Order quantity (required by API)
            tick_size: Tick size for the instrument (default 0.01 for stocks)

        Returns:
            Tuple of (success: bool, new_order_id: str or None)
            - new_order_id may differ from order_id if broker replaced the order
        """
        return await self._modify_order_price(order_id, new_stop_price, quantity, 'StopPrice', 'Stop Loss', tick_size)

    async def modify_take_profit(self, order_id: str, new_tp_price: float, quantity: int, tick_size: float = DEFAULT_TICK_SIZE) -> tuple:
        """
        Modify an existing take profit order using PUT request.

        Args:
            order_id: The broker order ID of the take profit order
            new_tp_price: New take profit price level
            quantity: Order quantity (required by API)
            tick_size: Tick size for the instrument (default 0.01 for stocks)

        Returns:
            Tuple of (success: bool, new_order_id: str or None)
            - new_order_id may differ from order_id if broker replaced the order
        """
        return await self._modify_order_price(order_id, new_tp_price, quantity, 'LimitPrice', 'Take Profit', tick_size)

    async def update_bracket(self, stop_order_id: str, tp_order_id: str, symbol: str, quantity: int,
                            new_stop_price: float, new_tp_price: float, tick_size: float = DEFAULT_TICK_SIZE,
                            account_id: str = None) -> Dict:
        """
        Update bracket (BRK) orders by modifying SL and TP individually using PUT.

        This method uses the simplified PUT-based modify_stop_loss() and modify_take_profit()
        methods to update bracket orders without requiring additional margin.

        Args:
            stop_order_id: Current stop loss order ID to modify
            tp_order_id: Current take profit order ID to modify
            symbol: Trading symbol
            quantity: Position quantity (absolute value)
            new_stop_price: New stop loss price
            new_tp_price: New take profit price
            tick_size: Tick size for the instrument
            account_id: Account ID (will use default if not provided)

        Returns:
            Dict with success status and order IDs
        """
        logger.info(f"🔄 UPDATE_BRACKET (BRK): {symbol}")
        logger.info(f"   Quantity: {quantity}")
        logger.info(f"   New SL: ${new_stop_price:.2f}, New TP: ${new_tp_price:.2f}")
        logger.info(f"   Modifying SL OrderID: {stop_order_id}, TP OrderID: {tp_order_id}")

        # Round prices to valid ticks
        new_stop_price = self._round_to_tick(new_stop_price, tick_size)
        new_tp_price = self._round_to_tick(new_tp_price, tick_size)

        try:
            # STEP 1: Verify position still exists at broker before updating
            logger.info(f"📤 Step 1: Verifying position exists...")
            positions = await self.get_positions()

            # Parse positions response
            positions_list = []
            if isinstance(positions, list):
                positions_list = positions
            elif positions and 'Positions' in positions:
                positions_list = positions['Positions']

            # Check if position exists
            position_exists = False
            broker_symbols = [pos.get('Symbol') for pos in positions_list]
            logger.debug(f"Broker positions: {broker_symbols}, Looking for: {symbol}")

            for pos in positions_list:
                pos_symbol = pos.get('Symbol', '')
                symbol_root = symbol.lstrip('@')
                if pos_symbol == symbol or pos_symbol.startswith(symbol_root):
                    position_exists = True
                    logger.debug(f"Found position: {pos_symbol} matches {symbol}")
                    break

            if not position_exists:
                logger.warning(f"⚠️  Position {symbol} not found at broker - skipping bracket update")
                logger.warning(f"   Broker has positions for: {broker_symbols}")
                return {
                    'success': False,
                    'error': 'Position already closed',
                    'reason': 'position_not_found'
                }

            # STEP 2: Modify stop loss using simplified method
            logger.info(f"📤 Step 2: Modifying stop loss order {stop_order_id}...")
            sl_success = await self.modify_stop_loss(
                order_id=stop_order_id,
                new_stop_price=new_stop_price,
                quantity=abs(quantity),
                tick_size=tick_size
            )

            # STEP 3: Modify take profit using simplified method
            logger.info(f"📤 Step 3: Modifying take profit order {tp_order_id}...")
            tp_success = await self.modify_take_profit(
                order_id=tp_order_id,
                new_tp_price=new_tp_price,
                quantity=abs(quantity),
                tick_size=tick_size
            )

            # Return results
            if sl_success and tp_success:
                logger.info(f"✅ BRACKET UPDATED SUCCESSFULLY")
                return {
                    'success': True,
                    'stop_order_id': stop_order_id,
                    'tp_order_id': tp_order_id,
                    'new_stop_price': new_stop_price,
                    'new_tp_price': new_tp_price
                }
            else:
                error_msg = f"Partial bracket update: SL={'success' if sl_success else 'failed'}, TP={'success' if tp_success else 'failed'}"
                logger.error(f"⚠️  {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'sl_success': sl_success,
                    'tp_success': tp_success
                }

        except Exception as e:
            logger.error(f"❌ UPDATE_BRACKET EXCEPTION: {e}")
            logger.debug(f"   Symbol: {symbol}, Quantity: {quantity}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    async def cancel_and_replace_oco_bracket(self, old_stop_order_id: str, old_tp_order_id: str,
                                             symbol: str, quantity: int, side: str,
                                             new_stop_price: float, new_tp_price: float,
                                             tick_size: float = DEFAULT_TICK_SIZE,
                                             account_id: str = None) -> Dict:
        """
        Cancel existing OCO bracket and replace with fresh orders.

        Used when protective orders are in non-working states (DON, DOA, SUS, etc.)
        and need to be replaced with actively protecting orders.

        Args:
            old_stop_order_id: Current stop loss order ID to cancel
            old_tp_order_id: Current take profit order ID to cancel
            symbol: Trading symbol
            quantity: Position quantity (absolute value)
            side: Exit side ('BUY' to close short, 'SELL' to close long)
            new_stop_price: New stop loss price
            new_tp_price: New take profit price
            tick_size: Tick size for the instrument
            account_id: Account ID (will use default if not provided)

        Returns:
            Dict with success status and new order IDs or error
        """
        logger.warning(f"🔄 CANCEL & REPLACE OCO BRACKET: {symbol}")
        logger.warning(f"   Reason: Protective orders not actively working")
        logger.warning(f"   Old orders: SL={old_stop_order_id}, TP={old_tp_order_id}")

        account_id = account_id or self.account_id

        try:
            # STEP 1: Cancel old orders (even if they're in DON/DOA/etc. status)
            logger.info(f"📤 Step 1: Canceling old protective orders...")

            cancel_results = []
            for order_id, order_type in [(old_stop_order_id, 'Stop Loss'), (old_tp_order_id, 'Take Profit')]:
                if order_id:
                    logger.info(f"   Canceling {order_type} order {order_id}...")
                    cancelled = await self.cancel_order(order_id)
                    cancel_results.append((order_type, cancelled))

                    if not cancelled:
                        logger.warning(f"⚠️  Failed to cancel {order_type} order {order_id} (may already be inactive)")
                else:
                    logger.warning(f"⚠️  No {order_type} order ID provided - skipping cancel")

            # STEP 2: Place fresh OCO bracket
            logger.info(f"📤 Step 2: Placing fresh OCO bracket...")
            result = await self.place_oco_bracket(
                symbol=symbol,
                quantity=abs(quantity),
                side=side,
                stop_price=new_stop_price,
                limit_price=new_tp_price,
                tick_size=tick_size,
                account_id=account_id
            )

            if not result or not result.get('success'):
                error_msg = result.get('error', 'Unknown error') if result else 'No response'
                logger.error(f"❌ Failed to place fresh OCO bracket: {error_msg}")
                return {
                    'success': False,
                    'error': f'Failed to place fresh OCO bracket: {error_msg}',
                    'old_orders_cancelled': cancel_results
                }

            new_stop_order_id = result.get('stop_order_id')
            new_tp_order_id = result.get('limit_order_id')

            logger.info(f"✅ OCO BRACKET REPLACED SUCCESSFULLY:")
            logger.info(f"   New Stop Loss OrderID: {new_stop_order_id}")
            logger.info(f"   New Take Profit OrderID: {new_tp_order_id}")

            return {
                'success': True,
                'stop_order_id': new_stop_order_id,
                'tp_order_id': new_tp_order_id,
                'new_stop_price': new_stop_price,
                'new_tp_price': new_tp_price,
                'old_orders_cancelled': cancel_results
            }

        except Exception as e:
            logger.error(f"❌ CANCEL_AND_REPLACE_OCO_BRACKET EXCEPTION: {e}")
            logger.debug(f"   Symbol: {symbol}, Side: {side}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    async def is_order_open(self, order_id: str) -> bool:
        """
        Check if an order is still open (not filled/cancelled/rejected/expired).

        This includes ALL "Open" category statuses from TradeStation docs.

        Args:
            order_id: The order ID to check

        Returns:
            True if order is in Open category, False if Filled/Cancelled/Rejected
        """
        try:
            status = await self.get_order_status(order_id)
            if not status:
                logger.debug(f"Order {order_id} not found - assuming closed")
                return False

            order_status = status.get('Status', status.get('OrderStatus', '')).upper()

            # Open statuses (from TradeStation API docs)
            open_statuses = [
                'ACK', 'ASS', 'BRC', 'BRF', 'BRO', 'CHG', 'CND', 'COR', 'CSN',
                'DIS', 'DOA', 'DON', 'ECN', 'EXE', 'FPR', 'LAT', 'OPN', 'OSO',
                'OTHER', 'PLA', 'REC', 'RJC', 'RPD', 'RSN', 'STP', 'STT', 'SUS', 'UCN'
            ]

            # Closed/terminal statuses
            # Canceled: CAN, EXP, OUT, RJR, SCN, TSC, UCH
            # Rejected: REJ
            # Filled: FLL, FLP

            is_open = order_status in open_statuses
            logger.debug(f"Order {order_id} status: {order_status}, is_open: {is_open}")
            return is_open

        except Exception as e:
            logger.warning(f"Error checking order status for {order_id}: {e}")
            # If we can't check, assume it might be open to be safe
            return True

    async def is_order_actively_protecting(self, order_id: str) -> bool:
        """
        Check if a protective order (stop loss/take profit) is ACTIVELY working.

        This is stricter than is_order_open() - only returns True for orders
        that are live on the exchange and will actually execute.

        Statuses considered "actively protecting":
        - ACK: Acknowledged/Received - order accepted by broker
        - OPN: Sent to exchange - actively working
        - DIS: Dispatched - sent to market
        - STP: Stop hit (for stop orders) - triggered and executing
        - FPR: Partial fill (still alive) - partially filled, rest still working

        Statuses NOT considered "actively protecting":
        - DON: Queued - not yet sent to exchange
        - DOA: Dead - not active
        - SUS: Suspended - not active
        - BRC/BRF/BRO: Bracket issues - may not execute
        - All others: pending, changing, or problematic states

        Args:
            order_id: The order ID to check

        Returns:
            True if order is actively protecting, False otherwise
        """
        try:
            status = await self.get_order_status(order_id)
            if not status:
                logger.warning(f"Order {order_id} not found - NOT protecting")
                return False

            order_status = status.get('Status', status.get('OrderStatus', '')).upper()

            # Only these statuses mean the order is truly active/working
            actively_protecting_statuses = ['ACK', 'OPN', 'DIS', 'STP', 'FPR']

            is_protecting = order_status in actively_protecting_statuses

            if not is_protecting:
                logger.warning(
                    f"⚠️  Order {order_id} status '{order_status}' - NOT actively protecting"
                )
            else:
                logger.debug(f"✅ Order {order_id} status '{order_status}' - actively protecting")

            return is_protecting

        except Exception as e:
            logger.error(f"Error checking protective order status for {order_id}: {e}")
            # For protective orders, if we can't verify, assume NOT protecting (fail-safe)
            return False

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order using TradeStation v3 API with smart state checking

        Args:
            order_id: The order ID to cancel

        Returns:
            True if cancelled successfully or already closed, False on error
        """
        logger.info(f"🚫 CANCEL_ORDER: OrderID={order_id}")
        try:
            # First check if order is still open
            is_open = await self.is_order_open(order_id)
            if not is_open:
                logger.info(f"✅ CANCEL_ORDER: OrderID={order_id} already closed - skipping")
                return True

            # Use orderexecution endpoint for DELETE
            endpoint = f'orderexecution/orders/{order_id}'
            result = await self._make_request('DELETE', endpoint)

            # FIX: "Cancel request sent" is actually a SUCCESS response from TradeStation
            if result is not None:
                message = result.get('Message', '').lower()
                # Accept both confirmation messages and "request sent" as success
                success = any(keyword in message for keyword in ['cancel', 'sent'])

                if success:
                    logger.info(f"✅ CANCEL_ORDER SUCCESS: OrderID={order_id}")
                else:
                    logger.warning(f"⚠️  CANCEL_ORDER FAILED: OrderID={order_id}, Response={result}")
                return success
            else:
                logger.warning(f"⚠️  CANCEL_ORDER returned no response: OrderID={order_id}")
                return False

        except Exception as e:
            logger.error(f"❌ CANCEL_ORDER EXCEPTION: OrderID={order_id}, Error={e}")
            return False

    async def get_order_status(self, order_id: str) -> Dict:
        """Get order status using TradeStation v3 API

        API returns: {"Orders": [{"OrderID": "...", "Status": "ACK", ...}], "Errors": []}
        """
        logger.info(f"📊 GET_ORDER_STATUS: OrderID={order_id}")
        try:
            # Use brokerage endpoint instead of orderexecution (which returns 405)
            endpoint = f'brokerage/accounts/{self.account_id}/orders/{order_id}'
            result = await self._make_request('GET', endpoint)

            if result:
                # API returns Orders array wrapper - extract first order
                if 'Orders' in result and isinstance(result['Orders'], list) and len(result['Orders']) > 0:
                    order = result['Orders'][0]
                    status = order.get('Status', 'unknown')
                    status_desc = order.get('StatusDescription', '')
                    logger.info(f"✅ GET_ORDER_STATUS SUCCESS: OrderID={order_id}, Status={status} ({status_desc})")
                    return order  # Return the order object, not the wrapper
                else:
                    logger.warning(f"⚠️  GET_ORDER_STATUS: No orders found in response for OrderID={order_id}")
                    logger.debug(f"   Response structure: {result}")
                    return {}
            else:
                logger.warning(f"⚠️  GET_ORDER_STATUS returned no data for OrderID={order_id}")
                return {}
        except Exception as e:
            logger.error(f"❌ GET_ORDER_STATUS EXCEPTION: OrderID={order_id}, Error={e}")
            return {}

    async def get_accounts(self) -> List[Dict]:
        """Get all accounts for the user"""
        try:
            accounts = await self._make_request('GET', 'brokerage/accounts')

            if accounts and isinstance(accounts, dict) and 'Accounts' in accounts:
                return accounts['Accounts']
            elif accounts and isinstance(accounts, list):
                return accounts
            else:
                return []
        except Exception as e:
            logger.error(f"Error getting accounts: {e}")
            return []

    async def modify_order(self, order_id: str, updates: Dict) -> Dict:
        """Modify an existing order"""
        try:
            endpoint = f'orderexecution/orders/{order_id}'
            result = await self._make_request('PUT', endpoint, json=updates)
            return result if result else {}
        except Exception as e:
            logger.error(f"Error modifying order {order_id}: {e}")
            return {}


# Utility functions for data management
class DataManager:
    """Manage historical data downloads and updates"""

    def __init__(self, client: TradeStationClient):
        self.client = client
        self.data_dir = './data'
        self.symbols = ['MBTU25', 'METU25', 'MNQ25']
        os.makedirs(f"{self.data_dir}/raw", exist_ok=True)

    async def download_historical_data(self, days_back: int = 730) -> None:
        """
        Download historical data for all symbols

        Args:
            days_back: Number of days of history to download
        """
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)

        # MTA requires 5 timeframes: 1m, 5m, 15m, 1h, 1d
        intervals = {
            '1min': min(180, days_back),  # 6 months max for 1-min
            '5min': min(365, days_back),   # 1 year max for 5-min
            '15min': min(365, days_back),  # 1 year max for 15-min
            '60min': min(730, days_back),  # 2 years max for 60-min (hourly)
            'daily': days_back              # Full period for daily
        }

        for symbol in self.symbols:
            logger.info(f"Downloading data for {symbol}")

            for interval, days in intervals.items():
                try:
                    interval_start = end_date - timedelta(days=days)

                    df = await self.client.get_historical_bars(
                        symbol=symbol,
                        interval=interval,
                        start_date=interval_start.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d')
                    )

                    if not df.empty:
                        filepath = f"{self.data_dir}/raw/{symbol}_{interval}.csv"
                        df.to_csv(filepath)
                        logger.info(f"Saved {len(df)} bars of {symbol} {interval} data")

                except Exception as e:
                    logger.error(f"Error downloading {symbol} {interval}: {e}")

            # Rate limiting pause between symbols
            await asyncio.sleep(SHORT_ASYNC_SLEEP_SECONDS)

    async def update_data(self) -> None:
        """Update data with latest bars (MTA uses 5 timeframes)"""
        for symbol in self.symbols:
            for interval in ['1min', '5min', '15min', '60min', 'daily']:
                try:
                    filepath = f"{self.data_dir}/raw/{symbol}_{interval}.csv"

                    # Load existing data
                    if os.path.exists(filepath):
                        existing_df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                        last_timestamp = existing_df.index[-1]

                        # Get new data since last timestamp
                        new_df = await self.client.get_historical_bars(
                            symbol=symbol,
                            interval=interval,
                            start_date=(last_timestamp + timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S')
                        )

                        if not new_df.empty:
                            # Append new data
                            combined_df = pd.concat([existing_df, new_df])
                            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                            combined_df.to_csv(filepath)
                            logger.info(f"Updated {symbol} {interval} with {len(new_df)} new bars")

                except Exception as e:
                    logger.error(f"Error updating {symbol} {interval}: {e}")


# Main execution for testing
async def main() -> None:
    """Test TradeStation connection and data download"""

    async with TradeStationClient() as client:
        # Test authentication
        account = await client.get_account()
        if account:
            logger.info(f"Connected to account: {account.get('AccountID')}")

            # Get account balances
            balances = await client.get_balances()
            if balances:
                logger.info(f"Account balance: ${balances.get('CashBalance', 0):,.2f}")

            # Initialize data manager
            data_manager = DataManager(client)

            # Download historical data
            await data_manager.download_historical_data()

            logger.info("TradeStation API implementation completed successfully!")
        else:
            logger.error("Failed to connect to TradeStation")


class QuoteStreamManager:
    """
    Manages streaming quotes for multiple symbols via TradeStation API.

    Uses HTTP streaming (SSE-like) to receive real-time quote updates without
    hitting REST API rate limits. One stream connection handles multiple symbols.

    Usage:
        async with QuoteStreamManager(client, ['PLF26', 'MNQZ25']) as manager:
            while True:
                quote = manager.get_quote('PLF26')
                if quote:
                    print(f"PLF26 Last: {quote['Last']}")
                await asyncio.sleep(1)
    """

    def __init__(self, client: TradeStationClient, symbols: List[str]):
        """
        Initialize the quote stream manager.

        Args:
            client: Authenticated TradeStationClient instance
            symbols: List of symbols to stream (use actual contracts like PLF26, not @PL)
        """
        self.client = client
        self.symbols = symbols
        self._quotes: Dict[str, Dict] = {}  # Latest quote per symbol
        self._stream_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()
        self._last_update: Dict[str, datetime] = {}

    async def __aenter__(self) -> 'QuoteStreamManager':
        """Start streaming on context enter"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop streaming on context exit"""
        await self.stop()

    async def start(self) -> None:
        """Start the streaming connection"""
        if self._running:
            return

        self._running = True
        self._stream_task = asyncio.create_task(self._stream_loop())
        logger.info(f"📡 QuoteStreamManager started for: {', '.join(self.symbols)}")

    async def stop(self) -> None:
        """Stop the streaming connection"""
        self._running = False
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        logger.info("📡 QuoteStreamManager stopped")

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get the latest quote for a symbol.

        Args:
            symbol: Symbol to get quote for (e.g., 'PLF26')

        Returns:
            Quote dict with 'Last', 'Bid', 'Ask', 'High', 'Low', etc. or None if not available
        """
        return self._quotes.get(symbol)

    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get just the last price for a symbol.

        Args:
            symbol: Symbol to get price for

        Returns:
            Last price as float or None
        """
        quote = self._quotes.get(symbol)
        if quote and 'Last' in quote:
            return float(quote['Last'])
        return None

    def get_all_quotes(self) -> Dict[str, Dict]:
        """Get all current quotes"""
        return self._quotes.copy()

    def is_stale(self, symbol: str, max_age_seconds: float = 30.0) -> bool:
        """
        Check if a quote is stale (hasn't been updated recently).

        Args:
            symbol: Symbol to check
            max_age_seconds: Maximum age in seconds before considered stale

        Returns:
            True if quote is stale or doesn't exist
        """
        last_update = self._last_update.get(symbol)
        if not last_update:
            return True
        age = (datetime.now(timezone.utc) - last_update).total_seconds()
        return age > max_age_seconds

    async def add_symbol(self, symbol: str) -> None:
        """
        Add a symbol to the stream (requires restart).

        Args:
            symbol: Symbol to add
        """
        if symbol not in self.symbols:
            async with self._lock:
                self.symbols.append(symbol)
                # Restart stream to include new symbol
                await self.stop()
                await self.start()

    async def remove_symbol(self, symbol: str) -> None:
        """
        Remove a symbol from the stream.

        Args:
            symbol: Symbol to remove
        """
        if symbol in self.symbols:
            async with self._lock:
                self.symbols.remove(symbol)
                self._quotes.pop(symbol, None)
                self._last_update.pop(symbol, None)
                # Restart stream without the symbol
                if self.symbols:
                    await self.stop()
                    await self.start()
                else:
                    await self.stop()

    async def _stream_loop(self) -> None:
        """Main streaming loop - handles reconnection on failure"""
        retry_delay = 1.0
        max_retry_delay = 60.0

        while self._running:
            try:
                await self._connect_and_stream()
                # If we exit cleanly, reset retry delay
                retry_delay = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if self._running:
                    logger.error(f"📡 Quote stream error: {e}")
                    logger.info(f"📡 Reconnecting in {retry_delay:.1f}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, max_retry_delay)

    async def _connect_and_stream(self) -> None:
        """Connect to stream endpoint and process quotes"""
        if not self.symbols:
            logger.warning("📡 No symbols to stream")
            return

        # Build URL with comma-separated symbols
        symbols_param = ','.join(self.symbols)
        base_url = self.client.sim_base_url if self.client.use_sim else self.client.base_url
        url = f"{base_url}/marketdata/stream/quotes/{symbols_param}"

        headers = {
            'Authorization': f'Bearer {self.client.access_token}',
            'Accept': 'application/json'
        }

        logger.info(f"📡 Connecting to quote stream: {symbols_param}")

        timeout = aiohttp.ClientTimeout(total=None, sock_read=300)  # Long timeout for streaming

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Stream connection failed: {response.status} - {error_text}")

                logger.info(f"📡 Quote stream connected (status={response.status})")

                # Read streaming response line by line (newline-delimited JSON)
                async for line in response.content:
                    if not self._running:
                        break

                    line = line.decode('utf-8').strip()
                    if not line:
                        continue

                    # Handle heartbeat
                    if line == 'heartbeat' or 'Heartbeat' in line:
                        logger.debug("📡 Stream heartbeat received")
                        continue

                    try:
                        data = json.loads(line)
                        await self._process_quote_data(data)
                    except json.JSONDecodeError:
                        logger.debug(f"📡 Non-JSON stream data: {line[:100]}")

    async def _process_quote_data(self, data: Dict) -> None:
        """Process incoming quote data from stream"""
        # TradeStation streams can send various formats
        # Single quote: {"Symbol": "PLF26", "Last": 1650.0, ...}
        # Array: [{"Symbol": "PLF26", ...}, {"Symbol": "MNQZ25", ...}]
        # Wrapped: {"Quotes": [...]}

        quotes = []

        if isinstance(data, list):
            quotes = data
        elif isinstance(data, dict):
            if 'Quotes' in data:
                quotes = data['Quotes']
            elif 'Symbol' in data:
                quotes = [data]
            elif 'Error' in data or 'Message' in data:
                logger.warning(f"📡 Stream message: {data}")
                return

        for quote in quotes:
            symbol = quote.get('Symbol')
            if symbol and symbol in self.symbols:
                async with self._lock:
                    self._quotes[symbol] = quote
                    self._last_update[symbol] = datetime.now(timezone.utc)

                logger.debug(f"📡 Quote update: {symbol} Last=${quote.get('Last', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())
