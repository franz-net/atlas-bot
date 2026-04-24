#!/usr/bin/env python3
"""
TradeStation authentication script for Midas Trading Platform
Handles the OAuth2 authorization code flow
"""

import asyncio
import logging
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.tradestation import TradeStationClient, atomic_file_write

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Authenticate with TradeStation"""
    logger.info("🔐 TradeStation Authentication Setup")
    logger.info("=" * 50)

    try:
        # Check if we already have a valid token
        client = TradeStationClient()

        # Try to load existing refresh token
        refresh_token = client._load_refresh_token()
        if refresh_token:
            print("Found existing authentication token. Testing validity...")

            async with client:
                # Try to make a simple API call to test the token
                try:
                    account = await client.get_account()
                    if account:
                        print(f"✅ Authentication valid! Connected to account: {account.get('AccountID', 'Unknown')}")
                        return True
                except Exception:
                    print("⚠️  Existing token is invalid. Need to re-authenticate.")

        # Need fresh authentication
        print("\n🔐 TradeStation Authentication Required")
        print("=" * 50)
        print("TradeStation requires user authentication through a web browser.")
        print("Please follow these steps:")
        print()

        # Generate authorization URL
        auth_url = client._build_authorization_url()
        print(f"1. Open this URL in your browser:")
        print(f"   {auth_url}")
        print()
        print("2. Log in with your TradeStation credentials")
        print("3. Authorize the application")
        print("4. Copy the authorization code from the redirect URL")
        print("   (Look for 'code=' parameter in the URL)")
        print()

        # Check if authorization code is already in .env file
        auth_code = os.getenv('TS_AUTH_CODE', '').strip()

        if auth_code:
            print(f"Found authorization code in .env file. Using it for authentication...")
            # Test authentication with the provided code
            return await authenticate_with_code(auth_code)

        # Get authorization code from user
        try:
            auth_code = input("Enter the authorization code: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n⚠️  Interactive input not available or cancelled.")
            print()
            print("To complete authentication manually:")
            print("1. Open the URL above in your browser")
            print("2. Complete the authorization process")
            print("3. Get the authorization code from the redirect URL")
            print("4. Add the code to your .env file:")
            print(f"   TS_AUTH_CODE=your_auth_code_here")
            print("5. Run the auth command again:")
            print(f"   python main.py auth")
            return False

        if not auth_code:
            print("❌ No authorization code provided")
            return False

        # Test authentication with the provided code
        return await authenticate_with_code(auth_code)

    except Exception as e:
        logger.error(f"❌ Authentication failed: {e}")
        return False

async def authenticate_with_code(auth_code=None):
    """Authenticate using provided authorization code"""
    if not auth_code:
        auth_code = os.getenv('TS_AUTH_CODE')
        if not auth_code:
            print("❌ No authorization code provided")
            return False

    client = TradeStationClient()
    async with client:
        # Manual authentication with the code
        success = await client._exchange_code_for_tokens(auth_code)

        if success:
            # Test the connection
            account = await client.get_account()
            if account:
                print(f"✅ Authentication successful! Connected to account: {account.get('AccountID', 'Unknown')}")

                # Get account balances
                balances = await client.get_balances()
                if balances:
                    balance = balances.get('CashBalance', 0)
                    print(f"💰 Account balance: ${balance:,.2f}")

                print()
                print("🎉 Authentication setup completed!")

                # Clear the auth code from .env for security
                _clear_auth_code_from_env()

                print("You can now run: python main.py setup")
                return True
            else:
                print("❌ Authentication failed - could not retrieve account information")
                return False
        else:
            print("❌ Failed to exchange authorization code for tokens")
            return False

def _clear_auth_code_from_env():
    """Clear the authorization code from .env file for security"""
    try:
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')

        if os.path.exists(env_file):
            # Read the file
            with open(env_file, 'r') as f:
                lines = f.readlines()

            # Update the TS_AUTH_CODE line
            updated_lines = []
            for line in lines:
                if line.startswith('TS_AUTH_CODE='):
                    updated_lines.append('TS_AUTH_CODE=\n')
                else:
                    updated_lines.append(line)

            # ATOMICITY: Use atomic file write for .env (critical credentials)
            env_content = ''.join(updated_lines)
            if atomic_file_write(env_file, env_content):
                print("🔒 Cleared authorization code from .env file for security")
            else:
                logger.warning("Failed to update .env file atomically")

    except Exception as e:
        logger.warning(f"Could not clear auth code from .env: {e}")

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)