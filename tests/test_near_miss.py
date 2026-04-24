"""
Tests for near-miss shadow logging.

Covers the ledger table + methods, screener instrumentation, decision
engine instrumentation, and the backfill job. All tests use in-memory
SQLite and mocked API clients — no live calls.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src.engine.decision_engine import DecisionEngine
from src.jobs.backfill_near_miss_outcomes import (
    _compute_forward_return,
    backfill_once,
)
from src.ledger.ledger import TradingLedger
from src.screener.screener import StockScreener


@pytest.fixture
def ledger():
    l = TradingLedger(db_path=':memory:')
    yield l
    l.close()


# ==================== Ledger table + methods ====================

class TestLedgerNearMiss:
    def test_record_and_read_back(self, ledger):
        row_id = ledger.record_near_miss(
            symbol='NVDA',
            gate='rsi',
            gate_detail='rsi=72.1, max=70',
            entry_price=500.0,
            would_be_direction='SHORT',
            metrics={'rsi': 72.1, 'atr_pct': 2.1},
        )
        assert row_id > 0

        rows = ledger._conn.execute(
            'SELECT * FROM near_misses WHERE id = ?', (row_id,)
        ).fetchone()
        assert rows['symbol'] == 'NVDA'
        assert rows['gate'] == 'rsi'
        assert rows['entry_price'] == 500.0
        assert rows['would_be_direction'] == 'SHORT'
        assert rows['outcome_backfilled_at'] is None
        parsed = json.loads(rows['metrics_json'])
        assert parsed['rsi'] == 72.1

    def test_pending_backfill_excludes_recent_rows(self, ledger):
        # Fresh row (today) — excluded because 1d return doesn't exist yet
        ledger.record_near_miss('AAA', 'rsi', 'x', entry_price=10.0)
        # Older row — include
        ledger._conn.execute(
            "INSERT INTO near_misses (timestamp, symbol, gate, entry_price) "
            "VALUES (?, 'BBB', 'rsi', 10.0)",
            ((datetime.now() - timedelta(days=2)).isoformat(),),
        )
        ledger._conn.commit()

        pending = ledger.get_near_misses_pending_backfill()
        symbols = {r['symbol'] for r in pending}
        assert 'BBB' in symbols
        assert 'AAA' not in symbols

    def test_pending_backfill_excludes_already_backfilled(self, ledger):
        ledger._conn.execute(
            "INSERT INTO near_misses "
            "(timestamp, symbol, gate, entry_price, outcome_backfilled_at) "
            "VALUES (?, 'DONE', 'rsi', 10.0, ?)",
            (
                (datetime.now() - timedelta(days=2)).isoformat(),
                datetime.now().isoformat(),
            ),
        )
        ledger._conn.commit()
        pending = ledger.get_near_misses_pending_backfill()
        assert all(r['symbol'] != 'DONE' for r in pending)

    def test_update_outcome(self, ledger):
        row_id = ledger.record_near_miss('NVDA', 'rsi', 'x', entry_price=10.0)
        assert ledger.update_near_miss_outcome(row_id, 0.03, 0.05)

        row = ledger._conn.execute(
            'SELECT * FROM near_misses WHERE id = ?', (row_id,)
        ).fetchone()
        assert row['outcome_1d'] == 0.03
        assert row['outcome_3d'] == 0.05
        assert row['outcome_backfilled_at'] is not None

    def test_rollup_counts_and_hit_rate(self, ledger):
        today = datetime.now().date().isoformat()

        def insert(symbol, direction, outcome_1d, backfilled):
            ledger._conn.execute(
                """INSERT INTO near_misses
                   (timestamp, symbol, gate, entry_price,
                    would_be_direction, outcome_1d, outcome_backfilled_at)
                   VALUES (?, ?, 'rsi', 10.0, ?, ?, ?)""",
                (
                    datetime.now().isoformat(),
                    symbol, direction, outcome_1d,
                    datetime.now().isoformat() if backfilled else None,
                ),
            )

        insert('A', 'LONG', 0.02, True)    # long win
        insert('B', 'LONG', -0.01, True)   # long loss
        insert('C', 'SHORT', -0.03, True)  # short win
        insert('D', 'LONG', None, False)   # not yet backfilled
        ledger._conn.commit()

        rollup = ledger.get_near_miss_rollup(today, today)
        assert rollup

        by_dir = {r['would_be_direction']: r for r in rollup}
        assert by_dir['LONG']['total'] == 3
        assert by_dir['LONG']['backfilled'] == 2
        assert by_dir['LONG']['directional_wins_1d'] == 1
        assert by_dir['SHORT']['total'] == 1
        assert by_dir['SHORT']['directional_wins_1d'] == 1


# ==================== Screener instrumentation ====================

class TestScreenerInstrumentation:
    def test_spread_near_miss_logged_when_in_price_band(self, ledger):
        client = MagicMock()
        screener = StockScreener(client, ledger=ledger)

        # Price in band, spread just above the 0.15% max but under 2x (0.30%).
        # bid=99.9, ask=100.1 → spread = 0.2/100.0 ≈ 0.2% → near miss.
        quotes = {
            'NVDA': {'Last': 100.0, 'Bid': 99.9, 'Ask': 100.1, 'Volume': 1e6},
        }
        survivors, _ = screener.apply_price_spread_filter(quotes)
        assert 'NVDA' not in survivors

        rows = ledger._conn.execute(
            "SELECT * FROM near_misses WHERE gate = 'spread'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]['symbol'] == 'NVDA'

    def test_price_band_fail_does_not_log(self, ledger):
        client = MagicMock()
        screener = StockScreener(client, ledger=ledger)
        quotes = {
            'PENNY': {'Last': 0.5, 'Bid': 0.49, 'Ask': 0.51, 'Volume': 1e6},
        }
        screener.apply_price_spread_filter(quotes)
        rows = ledger._conn.execute(
            "SELECT COUNT(*) FROM near_misses"
        ).fetchone()
        assert rows[0] == 0

    def test_screener_works_without_ledger(self):
        """Backward compat: ledger is optional."""
        client = MagicMock()
        screener = StockScreener(client)  # no ledger kwarg
        quotes = {
            'NVDA': {'Last': 100.0, 'Bid': 99.7, 'Ask': 100.3, 'Volume': 1e6},
        }
        # Should not raise
        survivors, _ = screener.apply_price_spread_filter(quotes)
        assert 'NVDA' not in survivors


# ==================== Decision engine instrumentation ====================

class TestDecisionEngineInstrumentation:
    def _make_engine(self, ledger):
        # Provider is stubbed — we're testing the rejection logging, not AI calls.
        engine = DecisionEngine(provider=MagicMock(), ledger=ledger)
        return engine

    def test_direction_mismatch_logged(self, ledger):
        engine = self._make_engine(ledger)
        # Candidate says LONG_ONLY but AI proposes SHORT → rejected + logged.
        candidates = [{
            'symbol': 'NVDA',
            'allowed_direction': 'LONG_ONLY',
            'regime': 'TREND_LONG',
            'rsi_14': 60,
        }]
        response = {
            'success': True,
            'content': json.dumps({
                'action': 'ENTER',
                'trades': [{
                    'action': 'ENTER',
                    'symbol': 'NVDA',
                    'direction': 'SHORT',
                    'expected_entry_price': 500.0,
                    'stop_loss': 510.0,
                    'take_profit': 475.0,
                    'conviction': 8,
                    'reasoning': 'A' * 100,
                }],
            }),
            'provider': 'mock', 'model': 'm',
            'prompt_tokens': 0, 'completion_tokens': 0, 'cost_estimate': 0,
        }

        result = engine._build_cycle_result(
            cycle_id='c1', candidates=candidates,
            response=response, active_capital=10000,
        )
        assert result['action'] == 'HOLD'  # rejected
        rows = ledger._conn.execute(
            "SELECT * FROM near_misses WHERE gate = 'direction_mismatch'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]['symbol'] == 'NVDA'
        assert rows[0]['cycle_id'] == 'c1'

    def test_pydantic_conviction_logged(self, ledger):
        engine = self._make_engine(ledger)
        response = {
            'success': True,
            'content': json.dumps({
                'action': 'ENTER',
                'trades': [{
                    'action': 'ENTER',
                    'symbol': 'NVDA',
                    'direction': 'LONG',
                    'expected_entry_price': 100.0,
                    'stop_loss': 98.0,
                    'take_profit': 106.0,
                    'conviction': 3,  # below MIN_CONVICTION_SCORE
                    'reasoning': 'A' * 100,
                }],
            }),
            'provider': 'mock', 'model': 'm',
            'prompt_tokens': 0, 'completion_tokens': 0, 'cost_estimate': 0,
        }
        result = engine._build_cycle_result(
            cycle_id='c2', candidates=[],
            response=response, active_capital=10000,
        )
        assert result['parse_error'] is not None
        rows = ledger._conn.execute(
            "SELECT * FROM near_misses WHERE gate = 'pydantic'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]['symbol'] == 'NVDA'


# ==================== Backfill job ====================

class TestBackfillJob:
    def test_compute_forward_return_basic(self):
        reject_date = datetime(2026, 4, 13, 14, 30)
        bars = pd.DataFrame(
            {'Close': [100.0, 102.0, 101.0, 105.0, 110.0]},
            index=pd.to_datetime([
                '2026-04-13', '2026-04-14', '2026-04-15',
                '2026-04-16', '2026-04-17',
            ]),
        )
        # entry 100, +1d close = 102 → +2%
        r1d = _compute_forward_return(100.0, reject_date, bars, 1)
        assert r1d == pytest.approx(0.02)
        # +3d close = 105 → +5%
        r3d = _compute_forward_return(100.0, reject_date, bars, 3)
        assert r3d == pytest.approx(0.05)

    def test_compute_forward_return_insufficient_bars(self):
        reject_date = datetime(2026, 4, 17)
        bars = pd.DataFrame(
            {'Close': [100.0]},
            index=pd.to_datetime(['2026-04-17']),
        )
        # No trading days strictly after — can't compute
        assert _compute_forward_return(100.0, reject_date, bars, 1) is None

    def test_compute_forward_return_missing_price(self):
        reject_date = datetime(2026, 4, 13)
        bars = pd.DataFrame(
            {'Close': [100.0, 102.0]},
            index=pd.to_datetime(['2026-04-13', '2026-04-14']),
        )
        assert _compute_forward_return(0.0, reject_date, bars, 1) is None
        assert _compute_forward_return(None, reject_date, bars, 1) is None

    @pytest.mark.asyncio
    async def test_backfill_updates_pending_rows(self, ledger):
        # Seed a pending row 2 days old
        old_ts = (datetime.now() - timedelta(days=2)).isoformat()
        ledger._conn.execute(
            "INSERT INTO near_misses "
            "(timestamp, symbol, gate, entry_price, would_be_direction) "
            "VALUES (?, 'NVDA', 'rsi', 100.0, 'LONG')",
            (old_ts,),
        )
        ledger._conn.commit()

        # Mock client returns bars where the day AFTER the rejection has close=103
        reject_day = datetime.fromisoformat(old_ts).date()
        bars = pd.DataFrame(
            {'Close': [100.0, 103.0, 104.0, 106.0]},
            index=pd.to_datetime([
                reject_day,
                reject_day + timedelta(days=1),
                reject_day + timedelta(days=2),
                reject_day + timedelta(days=3),
            ]),
        )
        client = MagicMock()
        client.get_historical_bars = AsyncMock(return_value=bars)

        result = await backfill_once(client, ledger)
        assert result['updated'] == 1

        row = ledger._conn.execute(
            "SELECT * FROM near_misses WHERE symbol = 'NVDA'"
        ).fetchone()
        assert row['outcome_1d'] == pytest.approx(0.03)
        assert row['outcome_3d'] == pytest.approx(0.06)
        assert row['outcome_backfilled_at'] is not None
