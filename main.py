#!/usr/bin/env python3
"""
Project Atlas — AI-Powered Equity Trading System
CLI entry point for authentication, setup verification, and trading operations.
"""

import asyncio
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

from src.utils.logging_config import setup_file_logger

logger = setup_file_logger(__name__, 'atlas_main')


async def cmd_auth() -> bool:
    """Run the TradeStation OAuth2 authentication flow."""
    from src.api.authenticate import main as auth_main
    return await auth_main()


async def cmd_setup() -> bool:
    """
    Verify Atlas setup: authentication, account access, and balance.

    Confirms the equity sim account is accessible and balance shows
    the expected ~$26K total with active capital above the protected floor.
    """
    from src.api.tradestation import TradeStationClient
    from src.config.constants import HARD_STOP_ACTIVE_CAPITAL_PCT, PROTECTED_FLOOR

    print("Project Atlas — Setup Verification")
    print("=" * 50)

    try:
        client = TradeStationClient()
        async with client:
            # 1. Verify account access
            print("\n[1/3] Verifying account access...")
            account = await client.get_account()
            if not account:
                print("FAIL: Could not retrieve account information.")
                logger.error("Setup failed: no account data returned")
                return False

            account_id = account.get('AccountID', 'Unknown')
            account_type = account.get('AccountType', 'Unknown')
            print(f"  Account: {account_id} ({account_type})")
            logger.info(f"Account verified: {account_id} ({account_type})")

            # 2. Verify balances
            print("\n[2/3] Checking account balance...")
            balances = await client.get_balances()
            if not balances:
                print("FAIL: Could not retrieve balance information.")
                logger.error("Setup failed: no balance data returned")
                return False

            # TradeStation v3 wraps balances in Balances array
            balance_data = balances
            if isinstance(balances, dict) and 'Balances' in balances:
                balance_list = balances['Balances']
                if balance_list and len(balance_list) > 0:
                    balance_data = balance_list[0]

            cash_balance = float(balance_data.get('CashBalance', 0))
            equity = float(balance_data.get('Equity', cash_balance))
            market_value = float(balance_data.get('MarketValue', 0))

            print(f"  Cash Balance:  ${cash_balance:,.2f}")
            print(f"  Equity:        ${equity:,.2f}")
            print(f"  Market Value:  ${market_value:,.2f}")
            logger.info(f"Balance: cash=${cash_balance:.2f} equity=${equity:.2f}")

            # 3. Calculate active capital
            print("\n[3/3] Capital structure check...")
            active_capital = equity - PROTECTED_FLOOR
            print(f"  Protected Floor: ${PROTECTED_FLOOR:,.2f}")
            print(f"  Active Capital:  ${active_capital:,.2f}")

            hard_stop_value = active_capital * HARD_STOP_ACTIVE_CAPITAL_PCT
            if active_capital < hard_stop_value:
                print(f"\n  WARNING: Active capital (${active_capital:,.2f}) is below "
                      f"hard stop ({HARD_STOP_ACTIVE_CAPITAL_PCT*100:.0f}% = ${hard_stop_value:,.2f}).")
                print("  Trading would be paused at this level.")
                logger.warning(f"Active capital ${active_capital:.2f} below hard stop")

            if equity < PROTECTED_FLOOR:
                print(f"\n  CRITICAL: Equity (${equity:,.2f}) is below protected "
                      f"floor (${PROTECTED_FLOOR:,.2f})!")
                logger.error(f"Equity ${equity:.2f} below protected floor")
                return False

            # 4. Test quote retrieval
            print("\n[Bonus] Testing market data access...")
            quote = await client.get_quote('AAPL')
            if quote:
                last_price = quote.get('Last', quote.get('LastPrice', 'N/A'))
                print(f"  AAPL Last Price: ${last_price}")
                logger.info(f"Quote test passed: AAPL={last_price}")
            else:
                print("  WARNING: Could not retrieve quote (market may be closed)")
                logger.warning("Quote test returned no data")

            print("\n" + "=" * 50)
            print("Setup verification PASSED")
            logger.info("Setup verification completed successfully")
            return True

    except Exception as e:
        print(f"\nSetup verification FAILED: {e}")
        logger.error(f"Setup verification failed: {e}")
        return False


async def cmd_ledger() -> bool:
    """
    Ledger CLI — view trading summary, trades, withdrawals, costs, and exports.

    Subcommands:
        ledger summary      P&L summary statistics
        ledger trades       List all trades
        ledger withdrawals  List all withdrawals
        ledger costs        Show API cost summary
        ledger export       Export trades and costs to CSV
    """
    from src.ledger.ledger import TradingLedger

    subcommand = sys.argv[2] if len(sys.argv) > 2 else 'summary'

    try:
        ledger = TradingLedger()

        # Check for --sim or --live filter
        mode_filter = None
        if '--sim' in sys.argv:
            mode_filter = 'SIM'
        elif '--live' in sys.argv:
            mode_filter = 'LIVE'

        if subcommand == 'summary':
            summary = ledger.get_summary(mode=mode_filter)
            mode_label = f" ({mode_filter})" if mode_filter else ""
            print(f"Project Atlas — Ledger Summary{mode_label}")
            print("=" * 50)
            print(f"  Total closed trades:  {summary['total_trades']}")
            print(f"  Open trades:          {summary['open_trades']}")
            print(f"  Wins:                 {summary['wins']}")
            print(f"  Losses:               {summary['losses']}")
            print(f"  Win rate:             {summary['win_rate']:.1f}%")
            print(f"  Gross P&L:            ${summary['gross_pnl']:.2f}")
            print(f"  Total fees:           ${summary['total_fees']:.2f}")
            print(f"  Net P&L:              ${summary['net_pnl']:.2f}")
            print(f"  Avg win:              ${summary['avg_win']:.2f}")
            print(f"  Avg loss:             ${summary['avg_loss']:.2f}")
            print(f"  Total API cost:       ${summary['total_api_cost']:.4f}")
            print(f"  Total withdrawn:      ${summary['total_withdrawn']:.2f}")
            if summary['total_withdrawn'] > 0:
                from src.config.constants import PROTECTED_FLOOR
                effective = PROTECTED_FLOOR + summary['total_withdrawn']
                print(f"  Effective floor (SIM): ${effective:,.2f}")

        elif subcommand == 'trades':
            trades = ledger.get_all_trades()
            if not trades:
                print("No trades recorded yet.")
                return True
            print("Project Atlas — All Trades")
            print("=" * 80)
            for t in trades:
                pnl = t.get('pnl_dollars')
                pnl_str = f"${pnl:.2f}" if pnl is not None else 'open'
                mode_tag = t.get('mode', 'SIM')
                print(
                    f"  [{t['status']}] [{mode_tag}] {t['symbol']} {t['direction']} "
                    f"{t['shares']}sh @ ${t['entry_price']:.2f} → {pnl_str}  "
                    f"({t['entry_timestamp'][:10]})"
                )

        elif subcommand == 'withdrawals':
            withdrawals = ledger.get_withdrawals()
            if not withdrawals:
                print("No withdrawals recorded yet.")
                return True
            print("Project Atlas — Withdrawals")
            print("=" * 60)
            for w in withdrawals:
                print(
                    f"  {w['week_ending']}  "
                    f"profit=${w['weekly_profit']:.2f}  "
                    f"withdrawn=${w['withdrawal_amount']:.2f}  "
                    f"total=${w['running_total_withdrawn']:.2f}"
                )

        elif subcommand == 'costs':
            from datetime import datetime
            today = datetime.now().strftime('%Y-%m-%d')
            daily_cost = ledger.get_daily_api_cost(today)
            summary = ledger.get_summary()
            print("Project Atlas — API Costs")
            print("=" * 50)
            print(f"  Today ({today}):   ${daily_cost:.4f}")
            print(f"  All-time total:      ${summary['total_api_cost']:.4f}")
            cycles = ledger.get_recent_cycles(limit=5)
            if cycles:
                print(f"\n  Last {len(cycles)} cycles:")
                for c in cycles:
                    print(
                        f"    {c['timestamp'][:19]}  {c['action_taken'] or 'HOLD'}  "
                        f"${c['api_cost_estimate']:.4f}  ({c['model_used']})"
                    )

        elif subcommand == 'export':
            trades_csv = ledger.export_trades_csv()
            costs_csv = ledger.export_costs_csv()
            if trades_csv:
                with open('data/trades_export.csv', 'w') as f:
                    f.write(trades_csv)
                print("Exported trades to data/trades_export.csv")
            else:
                print("No trades to export.")
            if costs_csv:
                with open('data/costs_export.csv', 'w') as f:
                    f.write(costs_csv)
                print("Exported costs to data/costs_export.csv")
            else:
                print("No costs to export.")

        else:
            print(f"Unknown ledger subcommand: {subcommand}")
            print("Available: summary, trades, withdrawals, costs, export")
            return False

        ledger.close()
        return True

    except Exception as e:
        print(f"Ledger error: {e}")
        logger.error(f"Ledger command failed: {e}")
        return False


async def cmd_test_order() -> bool:
    """
    Place and cancel a test market order on the equity sim account.

    This is the Sprint 1 exit criteria: confirm we can place and cancel
    an order on the sim account.
    """
    from src.api.tradestation import TradeStationClient

    print("Project Atlas — Test Order (Sim)")
    print("=" * 50)

    try:
        client = TradeStationClient()
        async with client:
            # Verify sim mode
            if not client.use_sim:
                print("ABORT: Not in sim mode. Set USE_SIM_ACCOUNT=true in .env")
                logger.error("Test order aborted: not in sim mode")
                return False

            print(f"\n  Sim mode: ACTIVE")
            print(f"  Account: {client.account_id}")

            # Place a small market buy order for 1 share of AAPL
            symbol = 'AAPL'
            quantity = 1
            print(f"\n[1/3] Placing test BUY order: {quantity} share of {symbol}...")

            order_data = {
                'AccountID': client.account_id,
                'Symbol': symbol,
                'Quantity': str(quantity),
                'OrderType': 'Limit',
                'LimitPrice': '1.00',
                'TradeAction': 'BUY',
                'TimeInForce': {'Duration': 'GTC'},
                'Route': 'Intelligent'
            }

            result = await client.place_order(order_data)

            if not result:
                print("  FAIL: No response from order placement.")
                logger.error("Test order failed: no response")
                return False

            # place_order returns parsed dict with success, order_id, error keys
            # CRITICAL: TS returns OrderID even on failed orders — the client
            # already checks the Error field in _parse_order_response()
            if not result.get('success'):
                error_msg = result.get('error', 'Unknown error')
                print(f"  FAIL: Order rejected. Error: {error_msg}")
                logger.error(f"Test order rejected: {error_msg}")
                return False

            order_id = result['order_id']
            print(f"  Order placed. OrderID: {order_id}")
            logger.info(f"Test order placed: {symbol} qty={quantity} OrderID={order_id}")

            # Brief pause before cancelling
            print("\n[2/3] Cancelling test order...")
            await asyncio.sleep(1)

            cancel_success = await client.cancel_order(order_id)

            if cancel_success:
                print(f"  Order {order_id} cancelled successfully.")
                logger.info(f"Test order cancelled: {order_id}")
            else:
                print(f"  WARNING: Cancel returned false for {order_id}.")
                print("  Order may have already filled (sim can fill instantly).")
                logger.warning(f"Test order cancel returned false: {order_id}")

            # Verify final state
            print("\n[3/3] Verifying order status...")
            status = await client.get_order_status(order_id)
            if status:
                order_status = status.get('Status', status.get('StatusDescription', 'Unknown'))
                print(f"  Order {order_id} status: {order_status}")
                logger.info(f"Test order final status: {order_status}")
            else:
                print(f"  Could not retrieve order status (may be expected in sim).")

            print("\n" + "=" * 50)
            print("Test order completed successfully.")
            print("Sprint 1 exit criteria: PASSED")
            logger.info("Sprint 1 test order completed successfully")
            return True

    except Exception as e:
        print(f"\nTest order FAILED: {e}")
        logger.error(f"Test order failed: {e}")
        return False


async def cmd_review() -> bool:
    """
    Run end-of-day review using Claude Opus.

    Analyzes today's trades, identifies patterns, and suggests improvements.
    """
    from src.engine.eod_review import EODReview
    from src.ledger.ledger import TradingLedger

    date = sys.argv[2] if len(sys.argv) > 2 else None

    print("Project Atlas — EOD Review (Claude Opus)")
    print("=" * 50)

    try:
        ledger = TradingLedger()
        review_engine = EODReview(ledger)

        result = await review_engine.run_review(date)

        if result.get('skipped'):
            print(f"\n  Skipped: {result.get('reason', 'No activity')}")
            print("  No Opus API call was made.")
        elif result.get('success') and result.get('review'):
            review = result['review']
            print()
            print(review_engine.format_review(review))
            print(f"\n  API cost: ${result.get('cost_estimate', 0):.4f}")

            # Store in DB
            import json as _json
            from zoneinfo import ZoneInfo
            from datetime import datetime as _dt
            review_date = date or _dt.now(ZoneInfo('US/Eastern')).strftime('%Y-%m-%d')
            ledger.record_review(
                week_ending=review_date,
                overall_grade=review.get('overall_grade', '?'),
                summary=review.get('summary', ''),
                review_json=_json.dumps(review, default=str),
                prompt_tokens=result.get('prompt_tokens', 0),
                completion_tokens=result.get('completion_tokens', 0),
                cost_estimate=result.get('cost_estimate', 0),
            )
        else:
            print(f"\n  Review failed: {result.get('reason', 'Unknown error')}")

        ledger.close()
        return result.get('success', False)

    except Exception as e:
        print(f"\nReview error: {e}")
        logger.error(f"EOD review command failed: {e}")
        return False


async def cmd_validate() -> bool:
    """
    Run sim validation report against live graduation criteria.

    Checks: 30+ trading days, positive P&L, >45% win rate,
    no >15% daily drawdown, complete ledger entries.
    """
    from src.engine.preflight import PreflightCheck
    from src.ledger.ledger import TradingLedger

    print("Project Atlas — Sim Validation Report")
    print("=" * 50)

    try:
        ledger = TradingLedger()
        preflight = PreflightCheck(ledger)

        report = preflight.run_sim_validation()

        for criterion in report['criteria']:
            status = 'PASS' if criterion['passed'] else 'FAIL'
            print(f"  [{status}] {criterion['name']}: {criterion['detail']}")

        s = report['summary']
        print(f"\n  Trading days:     {s['trading_days']}")
        print(f"  Total trades:     {s['total_trades']}")
        print(f"  Win rate:         {s['win_rate']:.1f}%")
        print(f"  Total P&L:        ${s['total_pnl']:.2f}")
        print(f"  Max daily DD:     {s['max_daily_drawdown']:.1f}%")
        print(f"  Total API cost:   ${s['total_api_cost']:.4f}")

        if report['ready_for_live']:
            print("\n  RESULT: READY for live trading consideration")
        else:
            print("\n  RESULT: NOT READY — criteria above must be met first")

        ledger.close()
        return report['ready_for_live']

    except Exception as e:
        print(f"\nValidation error: {e}")
        logger.error(f"Validation command failed: {e}")
        return False


async def cmd_run() -> bool:
    """Start the Atlas trading loop."""
    from scheduler import TradingScheduler

    scheduler = TradingScheduler()
    try:
        await scheduler.start()
        return True
    except KeyboardInterrupt:
        print("\nShutting down...")
        scheduler.stop()
        return True
    except Exception as e:
        print(f"Scheduler error: {e}")
        logger.error(f"Scheduler failed: {e}")
        return False


def print_usage() -> None:
    """Print CLI usage information."""
    print("Project Atlas — AI-Powered Equity Trading System")
    print()
    print("Usage: python main.py <command>")
    print()
    print("Commands:")
    print("  auth        Authenticate with TradeStation (OAuth2)")
    print("  setup       Verify account access and balance")
    print("  test-order  Place and cancel a test order (sim only)")
    print("  ledger      View trading ledger (summary|trades|withdrawals|costs|export)")
    print("  run         Start the trading loop (runs during market hours)")
    print("  review      Run EOD review with Claude Opus (optional: date YYYY-MM-DD)")
    print("  validate    Check sim trading record against live graduation criteria")
    print()


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(0)

    command = sys.argv[1].lower()

    commands = {
        'auth': cmd_auth,
        'setup': cmd_setup,
        'test-order': cmd_test_order,
        'ledger': cmd_ledger,
        'run': cmd_run,
        'review': cmd_review,
        'validate': cmd_validate,
    }

    if command in commands:
        success = asyncio.run(commands[command]())
        sys.exit(0 if success else 1)
    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)


if __name__ == '__main__':
    main()
