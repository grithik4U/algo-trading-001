from datetime import datetime, timezone

from trading_engine.orderflow.footprint import build_footprint
from trading_engine.orderflow.schema import TradeTick


def test_footprint_aggregates_buy_and_sell_volume_at_price():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trades = [
        TradeTick(ts, 100.01, 5, "buy"),
        TradeTick(ts, 100.02, 3, "buy"),
        TradeTick(ts, 100.01, 2, "sell"),
    ]

    rows = build_footprint(trades, price_tick=0.01)

    row = next(r for r in rows if r.price == 100.01)
    assert row.buy_volume == 5
    assert row.sell_volume == 2
    assert row.delta == 3
    assert row.dominant_side == "buy"
