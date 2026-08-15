from datetime import datetime, timezone

from trading_engine.orderflow.footprint import FootprintRow
from trading_engine.orderflow.response import classify_flow_response


def test_large_buy_delta_with_little_upside_is_absorption_candidate():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    row = FootprintRow(ts, 100.0, 100, 10, 90, 110, 10.0, "buy")

    event = classify_flow_response(
        row,
        next_price=100.02,
        delta_threshold=50,
        response_threshold=0.05,
    )

    assert event.kind == "buy_absorption_candidate"


def test_large_buy_delta_with_strong_upside_is_continuation():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    row = FootprintRow(ts, 100.0, 100, 10, 90, 110, 10.0, "buy")

    event = classify_flow_response(
        row,
        next_price=100.20,
        delta_threshold=50,
        response_threshold=0.05,
    )

    assert event.kind == "buy_continuation"
