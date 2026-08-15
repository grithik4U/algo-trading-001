from datetime import datetime, timezone

from trading_engine.liquidity.classifier import classify_liquidity


def test_buy_side_swing_at_range_high_is_external():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pool = classify_liquidity(
        timestamp=ts,
        price=105.0,
        side="buy_side",
        swing_price=105.0,
        range_high=105.0,
        range_low=95.0,
        source="confirmed_swing_high",
    )
    assert pool.kind == "external"


def test_sell_side_swing_inside_range_is_internal():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pool = classify_liquidity(
        timestamp=ts,
        price=101.0,
        side="sell_side",
        swing_price=101.0,
        range_high=105.0,
        range_low=95.0,
        source="internal_swing_low",
    )
    assert pool.kind == "internal"
