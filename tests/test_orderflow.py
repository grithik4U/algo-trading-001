from datetime import datetime, timezone

from trading_engine.orderflow.engine import aggregate_l2, aggregate_trades
from trading_engine.orderflow.schema import BookLevel, TradeTick


def test_trade_aggregation_calculates_true_delta():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trades = [
        TradeTick(ts, 100.0, 5.0, "buy"),
        TradeTick(ts, 100.1, 2.0, "sell"),
        TradeTick(ts, 100.2, 3.0, "buy"),
    ]

    result = aggregate_trades(trades)

    assert result.buy_volume == 8.0
    assert result.sell_volume == 2.0
    assert result.delta == 6.0
    assert result.cumulative_delta == 6.0


def test_l2_aggregation_keeps_trade_delta_and_adds_book_liquidity():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trades = [TradeTick(ts, 100.0, 5.0, "buy")]
    flow = aggregate_trades(trades)
    levels = [
        BookLevel(ts, 99.9, 100.0, "bid", 1),
        BookLevel(ts, 100.1, 200.0, "ask", 1),
    ]

    result = aggregate_l2(levels, ts, flow)

    assert result.delta == 5.0
    assert result.bid_volume == 100.0
    assert result.ask_volume == 200.0
    assert result.imbalance_ratio == 2.0
