import pandas as pd

from trading_engine.liquidity.clustering import cluster_levels


def test_clusters_nearby_buy_side_levels():
    levels = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01 10:00", "2026-01-01 11:00", "2026-01-01 12:00"]
            ),
            "price": [100.00, 100.02, 102.00],
            "side": ["BSL", "BSL", "BSL"],
        }
    )

    pools = cluster_levels(levels, tolerance=0.0005, min_members=2)

    assert len(pools) == 1
    assert pools.iloc[0]["members"] == 2
    assert abs(pools.iloc[0]["price"] - 100.01) < 1e-9


def test_does_not_cluster_opposite_sides():
    levels = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 10:00", "2026-01-01 11:00"]),
            "price": [100.00, 100.01],
            "side": ["BSL", "SSL"],
        }
    )

    pools = cluster_levels(levels, tolerance=0.001, min_members=2)

    assert pools.empty
