from datetime import datetime, timezone

from trading_engine.orderflow.footprint import FootprintRow
from trading_engine.orderflow.imbalance import detect_stacked_imbalances


def test_detects_three_level_buy_imbalance_stack():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        FootprintRow(ts, 100.00, 30, 5, 25, 35, 6.0, "buy"),
        FootprintRow(ts, 100.01, 25, 4, 21, 29, 6.25, "buy"),
        FootprintRow(ts, 100.02, 40, 8, 32, 48, 5.0, "buy"),
    ]

    events = detect_stacked_imbalances(rows, ratio_threshold=3.0, min_stack=3)

    assert len(events) == 3
    assert all(event.side == "buy" for event in events)
