import pandas as pd

from trading_engine.liquidity.confluence import build_confluence_zones


def test_independent_sources_create_confluence_zone():
    registry = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01 09:00", "2026-01-01 10:00", "2026-01-01 11:00"]
            ),
            "price": [100.00, 100.02, 100.01],
            "side": ["BSL", "BSL", "BSL"],
            "source": ["session", "structure", "structure"],
        }
    )

    zones = build_confluence_zones(registry, tolerance=0.0005, min_sources=2)

    assert len(zones) == 1
    assert zones.iloc[0]["levels"] == 3
    assert set(zones.iloc[0]["sources"]) == {"session", "structure"}
    assert zones.iloc[0]["score"] > 2.0


def test_single_source_does_not_meet_default_confluence_requirement():
    registry = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 09:00", "2026-01-01 10:00"]),
            "price": [100.00, 100.01],
            "side": ["SSL", "SSL"],
            "source": ["structure", "structure"],
        }
    )

    zones = build_confluence_zones(registry, tolerance=0.001, min_sources=2)

    assert zones.empty
