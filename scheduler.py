"""
Trading loop scheduler for Project Atlas.

Orchestrates the full pipeline every 5 minutes during market hours:
screener → candidate builder → news → decision engine → order manager → position monitor.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

from src.api.tradestation import TradeStationClient
from src.config.constants import (
    DEFAULT_LOOP_INTERVAL_SECONDS,
    HARD_STOP_ACTIVE_CAPITAL_PCT,
    MARKET_CLOSE_HOUR,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MINUTE,
    PROTECTED_FLOOR,
    SYMBOL_COOLDOWN_CYCLES,
)
from src.engine.decision_engine import DecisionEngine
from src.engine.eod_review import EODReview
from src.engine.operator_approval import OperatorApproval
from src.engine.preflight import PreflightCheck
from src.ledger.ledger import TradingLedger
from src.ledger.withdrawal_tracker import WithdrawalTracker
from src.orders.order_manager import OrderManager
from src.orders.position_monitor import PositionMonitor
from src.screener.candidate_builder import CandidateBuilder
from src.screener.news_fetcher import NewsFetcher
from src.screener.screener import StockScreener
from src.utils.logging_config import setup_file_logger

logger = setup_file_logger(__name__, 'scheduler')


class TradingScheduler:
    """
    Main trading loop orchestrator.

    Runs the full pipeline on a configurable interval during market hours.
    Manages the lifecycle of all trading components.
    """

    def __init__(self):
        """Initialize the scheduler with all components."""
        self._interval = int(
            os.getenv('LOOP_INTERVAL_SECONDS', DEFAULT_LOOP_INTERVAL_SECONDS)
        )
        self._running = False
        self._cycle_count = 0

        self._eod_review_done_today = False

        # Components initialized in start()
        self._client: Optional[TradeStationClient] = None
        self._screener: Optional[StockScreener] = None
        self._builder: Optional[CandidateBuilder] = None
        self._news: Optional[NewsFetcher] = None
        self._engine: Optional[DecisionEngine] = None
        self._order_mgr: Optional[OrderManager] = None
        self._monitor: Optional[PositionMonitor] = None
        self._ledger: Optional[TradingLedger] = None
        self._eod_review: Optional[EODReview] = None
        self._withdrawal_tracker: Optional[WithdrawalTracker] = None
        self._approval: Optional[OperatorApproval] = None
        self._mode: str = 'SIM'  # Determined from USE_SIM_ACCOUNT at start

    def is_market_open(self) -> bool:
        """
        Check if current time is within market hours (9:30-16:00 EST).

        Returns:
            True if market is open
        """
        est = ZoneInfo('US/Eastern')
        now = datetime.now(est)

        # Weekday check (Mon=0, Fri=4)
        if now.weekday() > 4:
            return False

        market_open = now.replace(
            hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE,
            second=0, microsecond=0,
        )
        market_close = now.replace(
            hour=MARKET_CLOSE_HOUR, minute=0,
            second=0, microsecond=0,
        )

        return market_open <= now <= market_close

    async def _get_account_state(self) -> Optional[dict]:
        """
        Fetch current account state from TradeStation.

        Returns:
            Account state dict or None on error
        """
        try:
            balances = await self._client.get_balances()
            if not balances:
                logger.error("Failed to get account balances")
                return None

            balance_data = balances
            if isinstance(balances, dict) and 'Balances' in balances:
                balance_list = balances['Balances']
                if balance_list:
                    balance_data = balance_list[0]

            equity = float(balance_data.get('Equity', balance_data.get('CashBalance', 0)))

            # In SIM mode, withdrawals raise the effective floor
            # (simulates money leaving the trading pool)
            effective_floor = PROTECTED_FLOOR
            if self._mode == 'SIM':
                effective_floor += self._ledger.get_total_withdrawn()

            active_capital = equity - effective_floor

            # Determine phase
            withdrawal_threshold = float(os.getenv('WITHDRAWAL_THRESHOLD', '2000'))
            phase = 'WITHDRAWAL' if active_capital >= withdrawal_threshold else 'GROWTH'

            return {
                'balance': equity,
                'active_capital': active_capital,
                'buying_power': float(balance_data.get('BuyingPower', 0)),
                'phase': phase,
            }
        except Exception as e:
            logger.error(f"Failed to get account state: {e}")
            return None

    async def run_cycle(self) -> dict:
        """
        Run a single trading cycle.

        Pipeline: position sync → screener → candidates → news → decision → orders → ledger.

        Returns:
            Cycle summary dict
        """
        self._cycle_count += 1
        cycle_summary = {
            'cycle_number': self._cycle_count,
            'timestamp': datetime.now().isoformat(),
            'candidates': 0,
            'action': 'HOLD',
            'trades_executed': 0,
            'exits_detected': 0,
            'error': None,
        }

        try:
            cn = self._cycle_count
            est = ZoneInfo('US/Eastern')
            cycle_start = datetime.now(est).strftime('%H:%M:%S ET')
            _p = lambda msg: print(f"    [{cn}] [{cycle_start}] {msg}", flush=True)

            # 1. Get account state
            _p("Fetching account state...")
            account_state = await self._get_account_state()
            if not account_state:
                cycle_summary['error'] = 'Could not fetch account state'
                return cycle_summary

            active_capital = account_state['active_capital']
            _p(f"Active capital: ${active_capital:,.2f} ({account_state['phase']})")

            # Record start-of-day snapshot (idempotent — first cycle wins)
            today = datetime.now(est).strftime('%Y-%m-%d')
            self._ledger.record_daily_snapshot(
                today, active_capital, account_state['balance'], self._mode,
            )

            # Hard stop check — percentage-based on start-of-day capital
            sod_capital = self._ledger.get_start_of_day_capital(today)
            hard_stop_value = (sod_capital or active_capital) * HARD_STOP_ACTIVE_CAPITAL_PCT
            if active_capital < hard_stop_value:
                logger.warning(
                    f"HARD STOP: Active capital ${active_capital:.2f} "
                    f"below {HARD_STOP_ACTIVE_CAPITAL_PCT*100:.0f}% of start-of-day "
                    f"(${hard_stop_value:.2f}). Trading paused."
                )
                _p(f"HARD STOP: capital ${active_capital:.2f} < ${hard_stop_value:.2f}")
                cycle_summary['error'] = 'Hard stop — active capital too low'
                return cycle_summary

            # 2. Sync positions — detect any exits since last cycle
            _p("Syncing positions...")
            balance = account_state['balance']
            sync_result = await self._monitor.sync_positions(balance)
            cycle_summary['exits_detected'] = len(sync_result['exits_recorded'])
            open_positions = sync_result['broker_positions']
            _p(f"Open positions: {len(open_positions)} | Exits: {cycle_summary['exits_detected']}")

            # 3. Run screener
            _p("Running screener...")
            logger.info(f"Cycle {self._cycle_count}: Running screener")
            candidates, quotes, bars_data = await self._screener.screen()
            cycle_summary['candidates'] = len(candidates)
            _p(f"Screener: {len(candidates)} candidates from {len(quotes)} quotes")

            if not candidates:
                logger.info(f"Cycle {self._cycle_count}: No candidates — skipping AI call")

            # 4. Build candidate packages
            news_data = {}
            if candidates:
                candidate_symbols = [c['symbol'] for c in candidates]
                _p(f"Fetching news for {len(candidate_symbols)} symbols...")
                news_data = await self._news.fetch_news_batch(candidate_symbols)
                news_count = sum(len(v) for v in news_data.values())
                _p(f"News: {news_count} headlines")

                # Build full packages
                packages = self._builder.build_all_packages(
                    candidates, quotes, bars_data, news_data,
                )
                _p(f"Built {len(packages)} candidate packages")
            else:
                packages = []

            # 5. Build cooldown list: open symbols + recently closed symbols
            cooldown_minutes = SYMBOL_COOLDOWN_CYCLES * (self._interval // 60)
            open_symbols = [t['symbol'] for t in self._ledger.get_open_trades()]
            closed_symbols = self._ledger.get_recently_closed_symbols(
                since_minutes=cooldown_minutes,
            )
            recent_symbols = list(set(open_symbols + closed_symbols))
            if recent_symbols:
                _p(f"Cooldown symbols: {', '.join(recent_symbols)}")
                logger.info(f"Cycle {self._cycle_count}: cooldown symbols: {recent_symbols}")

            # 6. AI decision
            ledger_open = self._ledger.get_open_trades()
            _p(f"Waiting for {self._engine._provider.provider_name} decision...")
            decision = await self._engine.decide(
                packages, account_state, open_positions,
                recent_symbols=recent_symbols,
                ledger_open_trades=ledger_open,
            )
            cycle_summary['action'] = decision.get('action', 'HOLD')
            _p(f"Decision: {decision.get('action', 'HOLD')}")

            # 7. Record cycle to ledger (with mode)
            decision['mode'] = self._mode
            self._ledger.record_cycle(decision)

            # 8. Execute trades if any
            if decision.get('action') == 'ENTER' and decision.get('trades'):
                # Operator approval gate (only active when OPERATOR_APPROVAL=true)
                approved_trades = []
                for trade in decision['trades']:
                    if trade.get('action') != 'ENTER':
                        continue
                    if self._approval.request_approval(trade, active_capital):
                        approved_trades.append(trade)
                    else:
                        logger.info(
                            f"Trade {trade.get('symbol')} rejected by operator"
                        )

                if approved_trades:
                    approved_decision = {**decision, 'trades': approved_trades}
                    exec_results = await self._order_mgr.execute_decisions(
                        decision=approved_decision,
                        active_capital=active_capital,
                        open_positions=open_positions,
                        phase=account_state['phase'],
                    )

                    for exec_result in exec_results:
                        if exec_result['success']:
                            cycle_summary['trades_executed'] += 1
                            trade_decision = next(
                                (t for t in decision['trades']
                                 if t['symbol'] == exec_result['symbol']),
                                {},
                            )
                            self._ledger.record_trade_entry(
                                cycle_id=decision['cycle_id'],
                                symbol=exec_result['symbol'],
                                direction=exec_result['direction'],
                                shares=exec_result['shares'],
                                entry_price=exec_result.get('entry_price', 0),
                                stop_loss_price=trade_decision.get('stop_loss', 0),
                                take_profit_price=trade_decision.get('take_profit', 0),
                                entry_reasoning=trade_decision.get('reasoning', ''),
                                news_catalyst=None,
                                entry_order_id=exec_result['entry_order_id'] or '',
                                stop_order_id=exec_result['stop_order_id'] or '',
                                tp_order_id=exec_result['tp_order_id'] or '',
                                phase=account_state['phase'],
                                active_capital=active_capital,
                                mode=self._mode,
                                entry_fees=exec_result.get('entry_fees', 0.0),
                            )
                        else:
                            logger.warning(
                                f"Trade execution failed: {exec_result['symbol']} "
                                f"— {exec_result['error']}"
                            )

            # 9. Execute exits if any
            if decision.get('action') == 'EXIT' and decision.get('exits'):
                # Operator approval gate for exits too
                approved_exits = []
                for exit_dec in decision['exits']:
                    exit_as_trade = {
                        'symbol': exit_dec.get('symbol'),
                        'action': 'EXIT',
                        'reasoning': exit_dec.get('reasoning', ''),
                    }
                    if self._approval.request_approval(exit_as_trade, active_capital):
                        approved_exits.append(exit_dec)
                    else:
                        logger.info(
                            f"Exit {exit_dec.get('symbol')} rejected by operator"
                        )

                if approved_exits:
                    approved_decision = {**decision, 'exits': approved_exits}
                    exit_results = await self._order_mgr.execute_exit_decisions(
                        decision=approved_decision,
                        ledger_open_trades=ledger_open,
                    )

                    for exit_result in exit_results:
                        if exit_result.get('success'):
                            cycle_summary['trades_executed'] += 1
                            trade = next(
                                (t for t in ledger_open
                                 if t['symbol'] == exit_result['symbol']),
                                {},
                            )
                            # Compute P&L
                            entry_price = trade.get('entry_price', 0)
                            exit_price = exit_result.get('exit_price', 0)
                            shares = exit_result.get('shares', 0)
                            direction = exit_result.get('direction', 'LONG')

                            if direction == 'LONG':
                                pnl_dollars = (exit_price - entry_price) * shares
                            else:
                                pnl_dollars = (entry_price - exit_price) * shares

                            pnl_pct = (
                                (pnl_dollars / (entry_price * shares)) * 100
                                if entry_price > 0 and shares > 0 else 0
                            )

                            self._ledger.record_trade_exit(
                                trade_id=trade.get('id', 0),
                                exit_price=exit_price,
                                pnl_dollars=pnl_dollars,
                                pnl_pct=pnl_pct,
                                exit_reasoning=f"AI EXIT: {exit_result.get('reasoning', '')}",
                                active_capital=active_capital,
                                exit_fees=exit_result.get('exit_fees', 0.0),
                            )

                            est_now = datetime.now(est).strftime('%H:%M:%S ET')
                            _p(
                                f"EXIT: {exit_result['symbol']} "
                                f"P&L=${pnl_dollars:+.2f} ({pnl_pct:+.1f}%)"
                            )
                        elif exit_result.get('already_closed'):
                            logger.info(
                                f"Exit {exit_result['symbol']}: "
                                f"position already closed by bracket"
                            )
                        else:
                            logger.warning(
                                f"Exit failed: {exit_result['symbol']} "
                                f"— {exit_result.get('error', 'unknown')}"
                            )

            # Log daily API cost warning
            daily_cost = self._ledger.get_daily_api_cost()
            if daily_cost > 1.0:
                logger.warning(f"Daily API cost ${daily_cost:.4f} exceeds $1.00 threshold")

            logger.info(
                f"Cycle {self._cycle_count} complete: "
                f"action={cycle_summary['action']} "
                f"candidates={cycle_summary['candidates']} "
                f"trades={cycle_summary['trades_executed']} "
                f"exits={cycle_summary['exits_detected']}"
            )

        except Exception as e:
            cycle_summary['error'] = str(e)
            logger.error(f"Cycle {self._cycle_count} failed: {e}")

        return cycle_summary

    async def start(self) -> None:
        """
        Start the trading loop.

        Initializes all components and runs cycles on the configured interval.
        Only runs during market hours.
        """
        logger.info("Starting Atlas trading scheduler")
        print("Atlas Trading Scheduler")
        print("=" * 50)

        self._client = TradeStationClient()

        async with self._client:
            # Initialize components
            self._screener = StockScreener(self._client)
            self._builder = CandidateBuilder()
            self._news = NewsFetcher(self._client)
            self._engine = DecisionEngine()
            self._ledger = TradingLedger()
            self._order_mgr = OrderManager(self._client)
            self._monitor = PositionMonitor(self._client, self._ledger)
            self._eod_review = EODReview(self._ledger)
            self._withdrawal_tracker = WithdrawalTracker(self._ledger)
            self._approval = OperatorApproval()
            self._mode = 'SIM' if self._client.use_sim else 'LIVE'

            # Run pre-flight checks
            preflight = PreflightCheck(self._ledger)
            pf_result = preflight.run_startup_checks()
            for check in pf_result['checks']:
                status = 'OK' if check['passed'] else 'FAIL'
                print(f"  [{status}] {check['name']}: {check['detail']}")
            for warning in pf_result['warnings']:
                print(f"  WARNING: {warning}")

            if not pf_result['passed']:
                print("\nPre-flight checks FAILED. Fix issues above before starting.")
                logger.error("Pre-flight checks failed — aborting")
                return

            self._running = True
            logger.info(
                f"Scheduler initialized: interval={self._interval}s "
                f"provider={self._engine._provider.provider_name} "
                f"mode={self._mode}"
            )
            print(f"\n  Provider: {self._engine._provider.provider_name}")
            print(f"  Model: {self._engine._provider.model_name}")
            print(f"  Interval: {self._interval}s")
            print(f"  Mode: {self._mode}")
            print(f"  {self._approval.format_summary()}")
            print()

            # Start background position monitor
            monitor_task = asyncio.create_task(self._position_monitor_loop())

            while self._running:
                try:
                    if self.is_market_open():
                        summary = await self.run_cycle()
                        if summary.get('error'):
                            print(f"  Cycle {summary['cycle_number']}: ERROR — {summary['error']}")
                        else:
                            print(
                                f"  Cycle {summary['cycle_number']}: "
                                f"{summary['action']} | "
                                f"{summary['candidates']} candidates | "
                                f"{summary['trades_executed']} trades | "
                                f"{summary['exits_detected']} exits"
                            )
                    else:
                        est = ZoneInfo('US/Eastern')
                        now = datetime.now(est)
                        logger.debug(f"Market closed ({now.strftime('%H:%M EST %A')})")

                        # Run weekly review on Fridays after market close
                        if (now.weekday() == 4
                                and now.hour >= MARKET_CLOSE_HOUR
                                and not self._eod_review_done_today):
                            await self._run_eod_tasks(now)

                        # Reset EOD flag at midnight
                        if now.hour == 0:
                            self._eod_review_done_today = False

                except Exception as e:
                    logger.error(f"Cycle loop error: {e}")
                    print(f"  ERROR in cycle loop: {e}", flush=True)

                await asyncio.sleep(self._interval)

            # Clean up background monitor
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    async def _position_monitor_loop(self) -> None:
        """
        Background task that monitors open positions every 30 seconds.

        Detects exits between trading cycles so we get near-real-time
        awareness of stop loss and take profit fills.
        """
        while self._running:
            try:
                await asyncio.sleep(30)

                if not self.is_market_open():
                    continue

                open_trades = self._ledger.get_open_trades()
                if not open_trades:
                    continue

                # Check broker positions
                balances = await self._client.get_balances()
                if not balances:
                    continue

                balance_data = balances
                if isinstance(balances, dict) and 'Balances' in balances:
                    balance_list = balances['Balances']
                    if balance_list:
                        balance_data = balance_list[0]

                equity = float(balance_data.get('Equity', balance_data.get('CashBalance', 0)))
                exits = await self._monitor.check_for_exits(equity)

                for exit_info in exits:
                    pnl = exit_info.get('pnl_dollars', 0)
                    symbol = exit_info.get('symbol', '?')
                    reason = exit_info.get('exit_reasoning', 'unknown')
                    est = ZoneInfo('US/Eastern')
                    ts = datetime.now(est).strftime('%H:%M:%S ET')
                    print(
                        f"    [MONITOR] [{ts}] Exit: {symbol} "
                        f"P&L=${pnl:+.2f} ({reason})",
                        flush=True,
                    )
                    logger.info(
                        f"Monitor detected exit: {symbol} P&L=${pnl:.2f} ({reason})"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Position monitor error: {e}")

    async def _run_eod_tasks(self, now: datetime) -> None:
        """
        Run end-of-week tasks: weekly review and Friday withdrawal calculation.

        Only runs on Fridays after market close.

        Args:
            now: Current datetime in EST
        """
        self._eod_review_done_today = True
        logger.info("Running end-of-week tasks")

        # Weekly Review (Fridays only)
        week_ending = now.strftime('%Y-%m-%d')
        try:
            print("\n  Running weekly review (Claude Opus)...")
            result = await self._eod_review.run_review()
            if result.get('skipped'):
                print(f"  Weekly review skipped: {result.get('reason')}")
            elif result.get('success') and result.get('review'):
                review = result['review']
                grade = review.get('overall_grade', '?')
                print(f"  Weekly review complete: Grade {grade} | Cost ${result.get('cost_estimate', 0):.4f}")

                # Store review in DB
                import json as _json
                self._ledger.record_review(
                    week_ending=week_ending,
                    overall_grade=grade,
                    summary=review.get('summary', ''),
                    review_json=_json.dumps(review, default=str),
                    prompt_tokens=result.get('prompt_tokens', 0),
                    completion_tokens=result.get('completion_tokens', 0),
                    cost_estimate=result.get('cost_estimate', 0),
                    mode=self._mode,
                )
            else:
                print(f"  Weekly review failed: {result.get('reason', 'Unknown')}")
        except Exception as e:
            logger.error(f"Weekly review failed: {e}")
            print(f"  Weekly review error: {e}")

        # Watchlist rotation (Fridays, after weekly review)
        try:
            from src.screener.watchlist_rotation import WatchlistRotation
            rotation = WatchlistRotation(self._ledger, self._screener)
            print("\n  Running watchlist rotation (Claude Opus)...")
            rot_result = await rotation.run_rotation(week_ending)
            if rot_result.get('skipped'):
                print(f"  Watchlist rotation skipped: {rot_result.get('reason', 'No data')}")
            elif rot_result.get('success'):
                added = len(rot_result.get('added', []))
                removed = len(rot_result.get('removed', []))
                print(
                    f"  Watchlist rotation: +{added} -{removed} symbols "
                    f"| Cost ${rot_result.get('cost_estimate', 0):.4f}"
                )
                if rot_result.get('added'):
                    print(f"    Added: {', '.join(rot_result['added'])}")
                if rot_result.get('removed'):
                    print(f"    Removed: {', '.join(rot_result['removed'])}")
            else:
                print(f"  Watchlist rotation failed: {rot_result.get('reason', 'Unknown')}")
        except Exception as e:
            logger.error(f"Watchlist rotation failed: {e}")
            print(f"  Watchlist rotation error: {e}")

        # Friday withdrawal calculation
        if now.weekday() == 4:
            try:
                account_state = await self._get_account_state()
                if account_state:
                    withdrawal = self._withdrawal_tracker.calculate_weekly_withdrawal(
                        current_balance=account_state['balance'],
                    )
                    if withdrawal:
                        self._withdrawal_tracker.record_withdrawal(withdrawal)
                        print(
                            f"  Withdrawal recorded: ${withdrawal['withdrawal_amount']:.2f} "
                            f"(1% of ${withdrawal['weekly_profit']:.2f} weekly profit)"
                        )
                    else:
                        print("  No withdrawal this week (conditions not met)")
            except Exception as e:
                logger.error(f"Withdrawal calculation failed: {e}")

    def stop(self) -> None:
        """Stop the trading loop."""
        self._running = False
        logger.info("Scheduler stop requested")
        if self._ledger:
            self._ledger.close()
