"""
Trading ledger for Project Atlas.

SQLite-based audit trail for all trade decisions, outcomes, and AI costs.
Schema initializes on import — safe to run on every startup via
CREATE TABLE IF NOT EXISTS.
"""

import csv
import io
import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

from src.utils.logging_config import setup_file_logger

logger = setup_file_logger(__name__, 'ledger')

DEFAULT_DB_PATH = os.path.join('data', 'atlas_ledger.db')

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_timestamp DATETIME,
    exit_timestamp DATETIME,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    shares INTEGER NOT NULL,
    entry_price REAL,
    stop_loss_price REAL,
    take_profit_price REAL,
    exit_price REAL,
    pnl_dollars REAL,
    pnl_pct REAL,
    entry_fees REAL DEFAULT 0,
    exit_fees REAL DEFAULT 0,
    entry_reasoning TEXT,
    exit_reasoning TEXT,
    news_catalyst TEXT,
    entry_order_id TEXT,
    stop_order_id TEXT,
    tp_order_id TEXT,
    phase TEXT,
    active_capital_at_entry REAL,
    active_capital_at_exit REAL,
    cycle_id TEXT,
    status TEXT DEFAULT 'OPEN',
    mode TEXT DEFAULT 'SIM'
);

CREATE TABLE IF NOT EXISTS cycles (
    id TEXT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    candidates_evaluated INTEGER DEFAULT 0,
    news_items_processed INTEGER DEFAULT 0,
    action_taken TEXT,
    model_used TEXT,
    provider_used TEXT,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    cache_creation_input_tokens INTEGER DEFAULT 0,
    cache_read_input_tokens INTEGER DEFAULT 0,
    api_cost_estimate REAL DEFAULT 0,
    full_response TEXT,
    mode TEXT DEFAULT 'SIM'
);

CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    week_ending DATE NOT NULL,
    overall_grade TEXT,
    summary TEXT,
    review_json TEXT NOT NULL,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    cost_estimate REAL DEFAULT 0,
    mode TEXT DEFAULT 'SIM'
);

CREATE TABLE IF NOT EXISTS watchlist_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    week_ending DATE NOT NULL,
    symbols_added TEXT,
    symbols_removed TEXT,
    reasoning TEXT,
    full_response TEXT,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    cost_estimate REAL DEFAULT 0,
    mode TEXT DEFAULT 'SIM'
);

CREATE TABLE IF NOT EXISTS daily_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL UNIQUE,
    active_capital REAL NOT NULL,
    balance REAL NOT NULL,
    timestamp DATETIME NOT NULL,
    mode TEXT DEFAULT 'SIM'
);

CREATE TABLE IF NOT EXISTS system_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS withdrawals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    week_ending DATE NOT NULL,
    active_capital REAL NOT NULL,
    weekly_profit REAL NOT NULL,
    withdrawal_amount REAL NOT NULL,
    running_total_withdrawn REAL DEFAULT 0,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS near_misses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    cycle_id TEXT,
    symbol TEXT NOT NULL,
    gate TEXT NOT NULL,
    gate_detail TEXT,
    entry_price REAL,
    would_be_direction TEXT,
    metrics_json TEXT,
    outcome_1d REAL,
    outcome_3d REAL,
    outcome_backfilled_at DATETIME,
    mode TEXT DEFAULT 'SIM'
);

CREATE INDEX IF NOT EXISTS idx_near_misses_timestamp
    ON near_misses(timestamp);
CREATE INDEX IF NOT EXISTS idx_near_misses_backfill
    ON near_misses(outcome_backfilled_at, timestamp);

CREATE TABLE IF NOT EXISTS gate_backtest (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    cycle_timestamp DATETIME NOT NULL,
    symbol TEXT NOT NULL,
    gate TEXT NOT NULL,
    gate_detail TEXT,
    entry_price REAL,
    would_be_direction TEXT,
    metrics_json TEXT,
    outcome_1d REAL,
    outcome_3d REAL,
    passed_all_gates INTEGER DEFAULT 0,
    universe TEXT DEFAULT 'current_160'
);

CREATE INDEX IF NOT EXISTS idx_gate_backtest_run
    ON gate_backtest(run_id, gate);
"""


class TradingLedger:
    """
    SQLite-based trading ledger for audit trail and performance tracking.

    All multi-step writes use transactions. All reads return typed Python
    objects. No raw SQL outside this module.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """
        Initialize the ledger and create schema if needed.

        Args:
            db_path: Path to SQLite database file. Use ':memory:' for tests.
        """
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info(f"Ledger initialized: {db_path}")

    def _init_schema(self) -> None:
        """Create tables if they don't exist and run migrations."""
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()
        self._migrate_add_mode_column()

    def _migrate_add_mode_column(self) -> None:
        """Add mode column to existing tables if missing."""
        for table in ('trades', 'cycles'):
            columns = [
                row[1] for row in
                self._conn.execute(f'PRAGMA table_info({table})').fetchall()
            ]
            if 'mode' not in columns:
                self._conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN mode TEXT DEFAULT 'SIM'"
                )
                self._conn.commit()
                logger.info(f"Migrated {table}: added mode column")

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()

    # ==================== Cycle Operations ====================

    def record_cycle(self, cycle_result: Dict) -> None:
        """
        Record a decision cycle (every cycle, even HOLD).

        Args:
            cycle_result: Decision engine result dict
        """
        cycle_id = cycle_result.get('cycle_id', '')

        # Idempotent: check for existing cycle_id
        existing = self._conn.execute(
            'SELECT id FROM cycles WHERE id = ?', (cycle_id,)
        ).fetchone()
        if existing:
            logger.warning(f"Cycle {cycle_id} already recorded — skipping")
            return

        # Count news items across candidates
        news_count = 0
        raw = cycle_result.get('raw_response', '')

        try:
            self._conn.execute(
                """INSERT INTO cycles
                   (id, timestamp, candidates_evaluated, news_items_processed,
                    action_taken, model_used, provider_used, prompt_tokens,
                    completion_tokens, cache_creation_input_tokens,
                    cache_read_input_tokens, api_cost_estimate, full_response, mode)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    cycle_id,
                    cycle_result.get('timestamp', datetime.now().isoformat()),
                    cycle_result.get('candidates_evaluated', 0),
                    news_count,
                    cycle_result.get('action', 'HOLD'),
                    cycle_result.get('model', ''),
                    cycle_result.get('provider', ''),
                    cycle_result.get('prompt_tokens', 0),
                    cycle_result.get('completion_tokens', 0),
                    cycle_result.get('cache_creation_input_tokens', 0),
                    cycle_result.get('cache_read_input_tokens', 0),
                    cycle_result.get('cost_estimate', 0),
                    raw,
                    cycle_result.get('mode', 'SIM'),
                ),
            )
            self._conn.commit()
            logger.info(f"Recorded cycle {cycle_id}: {cycle_result.get('action')}")
        except Exception as e:
            logger.error(f"Failed to record cycle {cycle_id}: {e}")

    # ==================== Trade Operations ====================

    def record_trade_entry(
        self,
        cycle_id: str,
        symbol: str,
        direction: str,
        shares: int,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        entry_reasoning: str,
        news_catalyst: Optional[str],
        entry_order_id: str,
        stop_order_id: str,
        tp_order_id: str,
        phase: str,
        active_capital: float,
        mode: str = 'SIM',
        entry_fees: float = 0.0,
    ) -> int:
        """
        Record a trade entry in the ledger.

        Args:
            cycle_id: UUID of the decision cycle
            symbol: Ticker symbol
            direction: LONG or SHORT
            shares: Number of shares
            entry_price: Fill price
            stop_loss_price: Initial stop loss
            take_profit_price: Initial take profit
            entry_reasoning: Claude's reasoning text
            news_catalyst: News headline (nullable)
            entry_order_id: TradeStation order ID
            stop_order_id: Stop loss order ID
            tp_order_id: Take profit order ID
            phase: GROWTH or WITHDRAWAL
            active_capital: Active capital at time of entry
            mode: SIM or LIVE
            entry_fees: Commission + routing fees on entry

        Returns:
            Trade ID (row id)
        """
        try:
            cursor = self._conn.execute(
                """INSERT INTO trades
                   (entry_timestamp, symbol, direction, shares, entry_price,
                    stop_loss_price, take_profit_price, entry_reasoning,
                    news_catalyst, entry_order_id, stop_order_id, tp_order_id,
                    phase, active_capital_at_entry, cycle_id, status, mode,
                    entry_fees)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)""",
                (
                    datetime.now().isoformat(),
                    symbol, direction, shares, entry_price,
                    stop_loss_price, take_profit_price, entry_reasoning,
                    news_catalyst, entry_order_id, stop_order_id, tp_order_id,
                    phase, active_capital, cycle_id, mode, entry_fees,
                ),
            )
            self._conn.commit()
            trade_id = cursor.lastrowid
            logger.info(
                f"Trade entry recorded: id={trade_id} {direction} {shares} "
                f"{symbol} @ {entry_price}"
            )
            return trade_id
        except Exception as e:
            logger.error(f"Failed to record trade entry: {e}")
            return -1

    def record_trade_exit(
        self,
        trade_id: int,
        exit_price: float,
        pnl_dollars: float,
        pnl_pct: float,
        exit_reasoning: str,
        active_capital: float,
        exit_fees: float = 0.0,
    ) -> bool:
        """
        Record a trade exit.

        Args:
            trade_id: ID of the trade to close
            exit_price: Fill price on exit
            pnl_dollars: Realized P&L in dollars (gross, before fees)
            pnl_pct: P&L as percentage of position
            exit_reasoning: Claude's exit reasoning
            active_capital: Active capital at time of exit
            exit_fees: Commission + routing fees on exit

        Returns:
            True if update succeeded
        """
        try:
            cursor = self._conn.execute(
                """UPDATE trades SET
                   exit_timestamp = ?, exit_price = ?, pnl_dollars = ?,
                   pnl_pct = ?, exit_reasoning = ?, active_capital_at_exit = ?,
                   exit_fees = ?, status = 'CLOSED'
                   WHERE id = ? AND status = 'OPEN'""",
                (
                    datetime.now().isoformat(),
                    exit_price, pnl_dollars, pnl_pct,
                    exit_reasoning, active_capital, exit_fees, trade_id,
                ),
            )
            self._conn.commit()
            if cursor.rowcount == 0:
                logger.warning(
                    f"Trade exit skipped: id={trade_id} already closed "
                    f"(race condition with monitor)"
                )
                return False
            logger.info(f"Trade exit recorded: id={trade_id} P&L=${pnl_dollars:.2f}")
            return True
        except Exception as e:
            logger.error(f"Failed to record trade exit: {e}")
            return False

    # ==================== Withdrawal Operations ====================

    def record_withdrawal(
        self,
        week_ending: str,
        active_capital: float,
        weekly_profit: float,
        withdrawal_amount: float,
        notes: str = '',
    ) -> bool:
        """
        Record a weekly withdrawal calculation.

        Args:
            week_ending: Friday date (YYYY-MM-DD)
            active_capital: Active capital at calculation time
            weekly_profit: Realized profit for the week
            withdrawal_amount: 1% of weekly profit
            notes: Optional notes

        Returns:
            True if insert succeeded
        """
        try:
            running_total = self.get_total_withdrawn() + withdrawal_amount
            self._conn.execute(
                """INSERT INTO withdrawals
                   (timestamp, week_ending, active_capital, weekly_profit,
                    withdrawal_amount, running_total_withdrawn, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(),
                    week_ending, active_capital, weekly_profit,
                    withdrawal_amount, running_total, notes,
                ),
            )
            self._conn.commit()
            logger.info(f"Withdrawal recorded: ${withdrawal_amount:.2f} for week ending {week_ending}")
            return True
        except Exception as e:
            logger.error(f"Failed to record withdrawal: {e}")
            return False

    # ==================== Review Operations ====================

    def record_review(
        self,
        week_ending: str,
        overall_grade: str,
        summary: str,
        review_json: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_estimate: float = 0.0,
        mode: str = 'SIM',
    ) -> bool:
        """
        Record a weekly review from Claude Opus.

        Args:
            week_ending: Friday date (YYYY-MM-DD)
            overall_grade: Letter grade (A-F)
            summary: 2-3 sentence overview
            review_json: Full JSON review response
            prompt_tokens: Tokens used in prompt
            completion_tokens: Tokens in completion
            cost_estimate: API cost in USD
            mode: SIM or LIVE

        Returns:
            True if insert succeeded
        """
        try:
            self._conn.execute(
                """INSERT INTO reviews
                   (timestamp, week_ending, overall_grade, summary,
                    review_json, prompt_tokens, completion_tokens,
                    cost_estimate, mode)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(),
                    week_ending, overall_grade, summary,
                    review_json, prompt_tokens, completion_tokens,
                    cost_estimate, mode,
                ),
            )
            self._conn.commit()
            logger.info(f"Review recorded: grade={overall_grade} week={week_ending}")
            return True
        except Exception as e:
            logger.error(f"Failed to record review: {e}")
            return False

    def get_reviews(self, limit: int = 10) -> List[Dict]:
        """
        Return recent weekly reviews.

        Args:
            limit: Max number of reviews to return

        Returns:
            List of review dicts ordered by most recent first
        """
        rows = self._conn.execute(
            'SELECT * FROM reviews ORDER BY timestamp DESC LIMIT ?', (limit,)
        ).fetchall()
        return [dict(row) for row in rows]

    # ==================== Watchlist Change Operations ====================

    def record_watchlist_change(
        self,
        week_ending: str,
        symbols_added: str,
        symbols_removed: str,
        reasoning: str,
        full_response: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_estimate: float = 0.0,
        mode: str = 'SIM',
    ) -> bool:
        """
        Record a weekly watchlist rotation change.

        Args:
            week_ending: Friday date (YYYY-MM-DD)
            symbols_added: JSON string of added symbols
            symbols_removed: JSON string of removed symbols
            reasoning: Combined reasoning text
            full_response: Full Opus JSON response
            prompt_tokens: Tokens used in prompt
            completion_tokens: Tokens in completion
            cost_estimate: API cost in USD
            mode: SIM or LIVE

        Returns:
            True if insert succeeded
        """
        try:
            self._conn.execute(
                """INSERT INTO watchlist_changes
                   (timestamp, week_ending, symbols_added, symbols_removed,
                    reasoning, full_response, prompt_tokens, completion_tokens,
                    cost_estimate, mode)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(),
                    week_ending, symbols_added, symbols_removed,
                    reasoning, full_response, prompt_tokens, completion_tokens,
                    cost_estimate, mode,
                ),
            )
            self._conn.commit()
            logger.info(f"Watchlist change recorded: week={week_ending}")
            return True
        except Exception as e:
            logger.error(f"Failed to record watchlist change: {e}")
            return False

    def get_watchlist_changes(self, limit: int = 10) -> List[Dict]:
        """
        Return recent watchlist rotation changes.

        Args:
            limit: Max number of changes to return

        Returns:
            List of change dicts ordered by most recent first
        """
        rows = self._conn.execute(
            'SELECT * FROM watchlist_changes ORDER BY timestamp DESC LIMIT ?',
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    # ==================== Daily Snapshot Operations ====================

    def record_daily_snapshot(
        self,
        date: str,
        active_capital: float,
        balance: float,
        mode: str = 'SIM',
    ) -> bool:
        """
        Record start-of-day active capital. Idempotent — first write per day wins.

        Args:
            date: Date string (YYYY-MM-DD)
            active_capital: Active capital at start of day
            balance: Total account balance
            mode: SIM or LIVE

        Returns:
            True if inserted (first cycle of day), False if already existed
        """
        try:
            self._conn.execute(
                """INSERT OR IGNORE INTO daily_snapshots
                   (date, active_capital, balance, timestamp, mode)
                   VALUES (?, ?, ?, ?, ?)""",
                (date, active_capital, balance, datetime.now().isoformat(), mode),
            )
            self._conn.commit()
            inserted = self._conn.total_changes > 0
            if inserted:
                logger.info(
                    f"Daily snapshot: {date} active_capital=${active_capital:.2f}"
                )
            return inserted
        except Exception as e:
            logger.error(f"Failed to record daily snapshot: {e}")
            return False

    def get_start_of_day_capital(self, date: str) -> Optional[float]:
        """
        Get the start-of-day active capital for a given date.

        Args:
            date: Date string (YYYY-MM-DD)

        Returns:
            Active capital or None if no snapshot exists
        """
        row = self._conn.execute(
            'SELECT active_capital FROM daily_snapshots WHERE date = ?', (date,)
        ).fetchone()
        return float(row['active_capital']) if row else None

    # ==================== Read Operations ====================

    def get_open_trades(self) -> List[Dict]:
        """Return all currently open trades."""
        rows = self._conn.execute(
            "SELECT * FROM trades WHERE status = 'OPEN' ORDER BY entry_timestamp"
        ).fetchall()
        return [dict(row) for row in rows]

    def get_all_trades(self) -> List[Dict]:
        """Return all trades (open and closed)."""
        rows = self._conn.execute(
            'SELECT * FROM trades ORDER BY entry_timestamp'
        ).fetchall()
        return [dict(row) for row in rows]

    def get_closed_trades(self) -> List[Dict]:
        """Return all closed trades."""
        rows = self._conn.execute(
            "SELECT * FROM trades WHERE status = 'CLOSED' ORDER BY exit_timestamp"
        ).fetchall()
        return [dict(row) for row in rows]

    def get_trades_closed_on_date(self, date: str) -> List[Dict]:
        """
        Return trades that were closed on a specific date.

        Args:
            date: Date string (YYYY-MM-DD)

        Returns:
            List of closed trade dicts
        """
        rows = self._conn.execute(
            "SELECT * FROM trades WHERE status = 'CLOSED' AND date(exit_timestamp) = ? "
            "ORDER BY exit_timestamp",
            (date,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_trade_by_id(self, trade_id: int) -> Optional[Dict]:
        """Return a single trade by ID."""
        row = self._conn.execute(
            'SELECT * FROM trades WHERE id = ?', (trade_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_trade_by_cycle(self, cycle_id: str) -> Optional[Dict]:
        """Return a trade by its cycle ID."""
        row = self._conn.execute(
            'SELECT * FROM trades WHERE cycle_id = ?', (cycle_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_recently_closed_symbols(self, since_minutes: int = 30) -> List[str]:
        """
        Return symbols that were closed within the last N minutes.

        Args:
            since_minutes: Look-back window in minutes

        Returns:
            List of symbol strings
        """
        cutoff = datetime.now()
        # SQLite datetime comparison using ISO format
        from datetime import timedelta
        cutoff_str = (cutoff - timedelta(minutes=since_minutes)).isoformat()
        rows = self._conn.execute(
            """SELECT DISTINCT symbol FROM trades
               WHERE status = 'CLOSED' AND exit_timestamp >= ?""",
            (cutoff_str,),
        ).fetchall()
        return [row['symbol'] for row in rows]

    def get_recent_cycles(self, limit: int = 20) -> List[Dict]:
        """Return recent decision cycles."""
        rows = self._conn.execute(
            'SELECT * FROM cycles ORDER BY timestamp DESC LIMIT ?', (limit,)
        ).fetchall()
        return [dict(row) for row in rows]

    def get_total_withdrawn(self) -> float:
        """Return cumulative withdrawal total."""
        row = self._conn.execute(
            'SELECT COALESCE(MAX(running_total_withdrawn), 0) as total FROM withdrawals'
        ).fetchone()
        return float(row['total']) if row else 0.0

    def get_withdrawals(self) -> List[Dict]:
        """Return all withdrawal records."""
        rows = self._conn.execute(
            'SELECT * FROM withdrawals ORDER BY timestamp'
        ).fetchall()
        return [dict(row) for row in rows]

    def get_trades_by_mode(self, mode: str) -> List[Dict]:
        """Return closed trades filtered by mode (SIM or LIVE)."""
        rows = self._conn.execute(
            "SELECT * FROM trades WHERE status = 'CLOSED' AND mode = ? ORDER BY exit_timestamp",
            (mode,),
        ).fetchall()
        return [dict(row) for row in rows]

    # ==================== Summary / Analytics ====================

    def get_summary(self, mode: str = None) -> Dict:
        """
        Return P&L summary statistics.

        Args:
            mode: Optional filter by mode ('SIM' or 'LIVE'). None = all trades.

        Returns:
            Dict with total_trades, wins, losses, win_rate, total_pnl,
            avg_win, avg_loss, total_api_cost
        """
        if mode:
            closed = self.get_trades_by_mode(mode)
        else:
            closed = self.get_closed_trades()
        wins = [t for t in closed if (t['pnl_dollars'] or 0) > 0]
        losses = [t for t in closed if (t['pnl_dollars'] or 0) <= 0]

        gross_pnl = sum(t['pnl_dollars'] or 0 for t in closed)
        total_fees = sum(
            (t.get('entry_fees') or 0) + (t.get('exit_fees') or 0)
            for t in closed
        )
        net_pnl = gross_pnl - total_fees
        avg_win = (
            sum(t['pnl_dollars'] for t in wins) / len(wins)
            if wins else 0
        )
        avg_loss = (
            sum(t['pnl_dollars'] for t in losses) / len(losses)
            if losses else 0
        )
        win_rate = len(wins) / len(closed) * 100 if closed else 0

        # API costs
        cost_row = self._conn.execute(
            'SELECT COALESCE(SUM(api_cost_estimate), 0) as total FROM cycles'
        ).fetchone()
        total_api_cost = float(cost_row['total']) if cost_row else 0

        return {
            'total_trades': len(closed),
            'open_trades': len(self.get_open_trades()),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'gross_pnl': gross_pnl,
            'total_fees': total_fees,
            'net_pnl': net_pnl,
            'total_pnl': net_pnl,  # backward compat — now means net
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_api_cost': total_api_cost,
            'total_withdrawn': self.get_total_withdrawn(),
        }

    def get_daily_api_cost(self, date: str = None) -> float:
        """
        Return total API cost for a given date.

        Args:
            date: Date string (YYYY-MM-DD). Defaults to today.

        Returns:
            Total cost in USD
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        row = self._conn.execute(
            """SELECT COALESCE(SUM(api_cost_estimate), 0) as total
               FROM cycles WHERE timestamp LIKE ?""",
            (f'{date}%',),
        ).fetchone()
        return float(row['total']) if row else 0

    # ==================== Export ====================

    def export_trades_csv(self) -> str:
        """
        Export all trades to CSV format.

        Returns:
            CSV string
        """
        trades = self.get_all_trades()
        if not trades:
            return ''

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=trades[0].keys())
        writer.writeheader()
        writer.writerows(trades)
        return output.getvalue()

    def export_costs_csv(self) -> str:
        """
        Export cycle costs to CSV format.

        Returns:
            CSV string
        """
        cycles = self.get_recent_cycles(limit=10000)
        if not cycles:
            return ''

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=cycles[0].keys())
        writer.writeheader()
        writer.writerows(cycles)
        return output.getvalue()

    # ==================== Near-Miss Operations ====================

    def record_near_miss(
        self,
        symbol: str,
        gate: str,
        gate_detail: str,
        entry_price: Optional[float] = None,
        would_be_direction: Optional[str] = None,
        metrics: Optional[Dict] = None,
        cycle_id: Optional[str] = None,
        mode: str = 'SIM',
    ) -> int:
        """
        Log a candidate that was rejected by a specific gate.

        Used for shadow data collection: records what would-have-been
        trades look like so their forward returns can be analyzed later
        and inform whether gate thresholds should be relaxed.

        Args:
            symbol: Ticker that was rejected
            gate: One of 'price_spread', 'rsi', 'atr', 'relvol', 'regime',
                  'vwap_distance', 'direction_mismatch', 'rsi_direction',
                  'pydantic', 'cooldown'
            gate_detail: Human-readable detail (e.g. 'rsi=72.1, max=70')
            entry_price: Snapshot price at rejection (for outcome calculation)
            would_be_direction: LONG or SHORT if implied by the gate
            metrics: Snapshot of indicator values at rejection time
            cycle_id: Associated decision cycle, if any
            mode: SIM or LIVE

        Returns:
            Row ID of the inserted near-miss, or -1 on failure.
        """
        try:
            cursor = self._conn.execute(
                """INSERT INTO near_misses
                   (timestamp, cycle_id, symbol, gate, gate_detail,
                    entry_price, would_be_direction, metrics_json, mode)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(),
                    cycle_id,
                    symbol,
                    gate,
                    gate_detail,
                    entry_price,
                    would_be_direction,
                    json.dumps(metrics) if metrics else None,
                    mode,
                ),
            )
            self._conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to record near-miss {symbol}/{gate}: {e}")
            return -1

    def get_near_misses_pending_backfill(
        self, max_rows: int = 500
    ) -> List[Dict]:
        """
        Return near-miss rows that still need forward-return backfill.

        A row is pending if outcome_backfilled_at IS NULL and it's at least
        one day old (needs time for a 1d forward return to exist).

        Args:
            max_rows: Cap on rows returned per call (keeps the backfill job small)

        Returns:
            List of near-miss dicts, oldest first.
        """
        rows = self._conn.execute(
            """SELECT * FROM near_misses
               WHERE outcome_backfilled_at IS NULL
                 AND datetime(timestamp) <= datetime('now', '-1 day')
               ORDER BY timestamp ASC
               LIMIT ?""",
            (max_rows,),
        ).fetchall()
        return [dict(row) for row in rows]

    def update_near_miss_outcome(
        self,
        near_miss_id: int,
        outcome_1d: Optional[float],
        outcome_3d: Optional[float],
    ) -> bool:
        """
        Write forward-return outcomes back to a near-miss row.

        Args:
            near_miss_id: Row ID of the near-miss
            outcome_1d: 1-day forward return as a decimal fraction (0.02 = +2%)
            outcome_3d: 3-day forward return as a decimal fraction

        Returns:
            True if the row was updated.
        """
        try:
            cursor = self._conn.execute(
                """UPDATE near_misses
                   SET outcome_1d = ?, outcome_3d = ?,
                       outcome_backfilled_at = ?
                   WHERE id = ?""",
                (
                    outcome_1d,
                    outcome_3d,
                    datetime.now().isoformat(),
                    near_miss_id,
                ),
            )
            self._conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to update near-miss {near_miss_id}: {e}")
            return False

    def get_near_miss_rollup(
        self, start_date: str, end_date: str
    ) -> List[Dict]:
        """
        Return per-gate rejection counts and outcome stats for a date range.

        For each (gate, would_be_direction) pair, computes total rejections,
        how many had backfilled outcomes, mean 1d/3d returns, and hit rate
        (percentage of rejections where the price moved in the direction
        the trade would have taken).

        Args:
            start_date: Inclusive YYYY-MM-DD
            end_date: Inclusive YYYY-MM-DD

        Returns:
            List of rollup dicts, one per (gate, direction) group.
        """
        rows = self._conn.execute(
            """SELECT gate, would_be_direction,
                      COUNT(*) as total,
                      SUM(CASE WHEN outcome_backfilled_at IS NOT NULL
                               THEN 1 ELSE 0 END) as backfilled,
                      AVG(outcome_1d) as mean_1d,
                      AVG(outcome_3d) as mean_3d,
                      SUM(CASE
                          WHEN would_be_direction = 'LONG' AND outcome_1d > 0 THEN 1
                          WHEN would_be_direction = 'SHORT' AND outcome_1d < 0 THEN 1
                          ELSE 0 END) as directional_wins_1d
               FROM near_misses
               WHERE date(timestamp) BETWEEN ? AND ?
               GROUP BY gate, would_be_direction
               ORDER BY total DESC""",
            (start_date, end_date),
        ).fetchall()
        return [dict(row) for row in rows]

    # ==================== Gate Backtest Operations ====================

    def clear_backtest_run(self, run_id: str) -> int:
        """
        Delete all rows for a backtest run. Use before re-running the same
        run_id to avoid stale data.

        Args:
            run_id: Run identifier

        Returns:
            Number of rows deleted.
        """
        cursor = self._conn.execute(
            'DELETE FROM gate_backtest WHERE run_id = ?', (run_id,)
        )
        self._conn.commit()
        return cursor.rowcount

    def record_backtest_rows(self, rows: List[Dict]) -> int:
        """
        Bulk-insert backtest observations in one transaction.

        Args:
            rows: Dicts with keys matching gate_backtest columns. Missing
                  optional keys default to None/0.

        Returns:
            Number of rows inserted.
        """
        if not rows:
            return 0
        try:
            self._conn.executemany(
                """INSERT INTO gate_backtest
                   (run_id, cycle_timestamp, symbol, gate, gate_detail,
                    entry_price, would_be_direction, metrics_json,
                    outcome_1d, outcome_3d, passed_all_gates, universe)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        r['run_id'],
                        r['cycle_timestamp'],
                        r['symbol'],
                        r['gate'],
                        r.get('gate_detail'),
                        r.get('entry_price'),
                        r.get('would_be_direction'),
                        json.dumps(r['metrics']) if r.get('metrics') else None,
                        r.get('outcome_1d'),
                        r.get('outcome_3d'),
                        int(r.get('passed_all_gates', 0)),
                        r.get('universe', 'current_160'),
                    )
                    for r in rows
                ],
            )
            self._conn.commit()
            return len(rows)
        except Exception as e:
            logger.error(f"Bulk backtest insert failed: {e}")
            return 0

    def get_backtest_rollup(self, run_id: str) -> List[Dict]:
        """
        Per-gate rollup for a backtest run.

        For each gate, returns: rejection count, mean 1d/3d returns, and
        hit rate = fraction of rejections that moved in the direction the
        trade would have taken (only computed for rows with a direction).

        Args:
            run_id: Backtest run identifier

        Returns:
            Rollup dicts ordered by rejection count descending.
        """
        rows = self._conn.execute(
            """SELECT gate,
                      COUNT(*) as rejections,
                      AVG(outcome_1d) as mean_1d,
                      AVG(outcome_3d) as mean_3d,
                      SUM(CASE
                          WHEN would_be_direction = 'LONG' AND outcome_1d > 0 THEN 1
                          WHEN would_be_direction = 'SHORT' AND outcome_1d < 0 THEN 1
                          ELSE 0 END) as directional_wins_1d,
                      SUM(CASE
                          WHEN outcome_1d IS NOT NULL
                               AND would_be_direction IS NOT NULL THEN 1
                          ELSE 0 END) as with_direction
               FROM gate_backtest
               WHERE run_id = ? AND passed_all_gates = 0
               GROUP BY gate
               ORDER BY rejections DESC""",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_backtest_passes(self, run_id: str) -> List[Dict]:
        """Return rows that cleared all gates (would-have-been-trades)."""
        rows = self._conn.execute(
            """SELECT * FROM gate_backtest
               WHERE run_id = ? AND passed_all_gates = 1
               ORDER BY cycle_timestamp""",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    # ==================== System State ====================

    def get_state(self, key: str) -> Optional[str]:
        """
        Get a persisted system state value.

        Args:
            key: State key

        Returns:
            Value string or None if not set
        """
        row = self._conn.execute(
            'SELECT value FROM system_state WHERE key = ?', (key,)
        ).fetchone()
        return row['value'] if row else None

    def set_state(self, key: str, value: str) -> None:
        """
        Set a persisted system state value (upsert).

        Args:
            key: State key
            value: Value to store
        """
        self._conn.execute(
            """INSERT INTO system_state (key, value, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?""",
            (key, value, datetime.now().isoformat(),
             value, datetime.now().isoformat()),
        )
        self._conn.commit()
