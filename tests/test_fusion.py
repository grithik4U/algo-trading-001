from datetime import datetime, timezone

from trading_engine.orderflow.footprint import FootprintRow
from trading_engine.orderflow.fusion import fuse_flow_and_l2
from trading_engine.orderflow.l2_events import L2Event


def test_buy_absorption_requires_flow_price_and_l2_replenishment():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    row = FootprintRow(ts, 100.0, 100, 10, 90, 110, 10.0, "buy")
    l2 = [L2Event(ts, 100.0, "bid", 50, 100, 50, "replenishment")]

    event = fuse_flow_and_l2(row, 100.02, l2, delta_threshold=50, response_threshold=0.05)

    assert event.classification == "buy_absorption_candidate"
    assert event.l2_support


def test_buy_consumption_requires_price_follow_through_and_liquidity_pull():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    row = FootprintRow(ts, 100.0, 100, 10, 90, 110, 10.0, "buy")
    l2 = [L2Event(ts, 100.0, "ask", 50, 10, -40, "pull")]

    event = fuse_flow_and_l2(row, 100.20, l2, delta_threshold=50, response_threshold=0.05)

    assert event.classification == "buy_liquidity_consumption"
    assert event.l2_opposition
