import pandas as pd

from trading_engine.liquidity import LiquidityEngine
from trading_engine.structure import detect_swings


def sample_ohlc() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100, 101, 103, 102, 100, 99, 101, 104, 103, 101, 98],
            "high": [101, 104, 105, 103, 101, 102, 105, 106, 104, 102, 99],
            "low": [99, 100, 101, 99, 97, 98, 100, 102, 101, 97, 95],
            "close": [100, 103, 102, 100, 98, 101, 104, 103, 102, 98, 96],
            "volume": [100] * 11,
        },
        index=pd.date_range("2026-01-01", periods=11, freq="min"),
    )


def test_swings_require_confirmation_bars():
    df = sample_ohlc()
    result = detect_swings(df, left_bars=1, right_bars=1)
    assert result["swing_high"].notna().any()
    assert result["swing_low"].notna().any()
    assert result["swing_high_confirmed_at"].notna().any()


def test_liquidity_engine_returns_structural_levels():
    result = LiquidityEngine(
        left_bars=1,
        right_bars=1,
        external_lookback=3,
    ).levels(sample_ohlc())
    assert not result.empty
    assert set(result["side"]).issubset({"BSL", "SSL"})
    assert set(result["scope"]).issubset({"internal", "external"})
    assert (result["confirmed_at"] >= result["timestamp"]).all()
