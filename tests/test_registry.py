import pandas as pd

from trading_engine.liquidity.registry import build_liquidity_registry


def test_registry_combines_structure_and_session_levels():
    structural = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 10:00"]),
            "price": [100.0],
            "side": ["high"],
            "timeframe": ["5min"],
            "liquidity_class": ["internal"],
            "structural_significance": [0.30],
        }
    )
    sessions = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 09:00"]),
            "price": [101.0],
            "side": ["BSL"],
            "timeframe": ["session"],
            "session": ["London"],
        }
    )

    registry = build_liquidity_registry(structural, sessions)

    assert len(registry) == 2
    assert set(registry["source"]) == {"structure", "session"}
    session_row = registry[registry["source"] == "session"].iloc[0]
    assert session_row["liquidity_class"] == "external"
