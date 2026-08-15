from datetime import datetime, timezone

from trading_engine.orderflow.l2_events import compare_snapshots
from trading_engine.orderflow.schema import BookLevel


def test_l2_detects_pull_and_replenishment():
    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    previous = [
        BookLevel(t1, 100.0, 100.0, "bid", 1),
        BookLevel(t1, 101.0, 50.0, "ask", 1),
    ]
    current = [
        BookLevel(t2, 100.0, 20.0, "bid", 1),
        BookLevel(t2, 101.0, 100.0, "ask", 1),
    ]

    events = compare_snapshots(previous, current, replenishment_threshold=40)

    by_side = {event.side: event for event in events}
    assert by_side["bid"].event_type == "pull"
    assert by_side["ask"].event_type == "replenishment"
