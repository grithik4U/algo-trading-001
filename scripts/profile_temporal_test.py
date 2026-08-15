"""Test structural volume-profile persistence across rolling time windows."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trading_engine.data.binance import BinanceConfig, BinancePublicData
from trading_engine.data.dataset import load_aligned_binance_dataset
from trading_engine.volume_profile.trade_profile import build_trade_volume_profile


WINDOW_SIZE = timedelta(hours=1)
STEP = timedelta(minutes=15)
NODE_MATCH_TOLERANCE = 1.25
MIN_ZONE_COVERAGE = 0.30


def _cluster_nodes(nodes, tolerance: float = NODE_MATCH_TOLERANCE):
    """Cluster nearby nodes without chaining distant nodes through a long bridge."""
    clusters = []
    for node in sorted(nodes, key=lambda n: n.center):
        best = None
        best_distance = float("inf")
        for cluster in clusters:
            distance = abs(node.center - cluster["center"])
            if distance <= tolerance and distance < best_distance:
                best = cluster
                best_distance = distance
        if best is None:
            best = {"center": node.center, "nodes": []}
            clusters.append(best)
        best["nodes"].append(node)
        # Robust center: median of observed node centers.
        values = sorted(n.center for n in best["nodes"])
        mid = len(values) // 2
        best["center"] = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
    return clusters


def _zone_coverage(center: float, profiles, node_type: str, tolerance: float = NODE_MATCH_TOLERANCE):
    """Return the number of distinct windows containing a matching node."""
    matched = []
    for i, (_, _, profile) in enumerate(profiles):
        nodes = profile.hvn_nodes if node_type == "HVN" else profile.lvn_nodes
        if any(abs(node.center - center) <= tolerance for node in nodes):
            matched.append(i)
    return matched


def _regime_label(window_index: int, total_windows: int, coverage: float, latest_match: int | None) -> str:
    """Classify a zone by persistence and whether it remains relevant near the latest window."""
    recent_cutoff = max(0, total_windows - 4)
    recent = latest_match is not None and latest_match >= recent_cutoff
    if coverage >= 0.75 and recent:
        return "HIGH_ACTIVE"
    if coverage >= 0.50 and recent:
        return "MEDIUM_ACTIVE"
    if coverage >= 0.50:
        return "HISTORICAL"
    if coverage >= MIN_ZONE_COVERAGE and recent:
        return "DEVELOPING"
    return "LOW"


def main() -> None:
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(hours=4)
    provider = BinancePublicData(BinanceConfig(symbol="BTCUSDT"))
    dataset = load_aligned_binance_dataset(
        provider, interval="1m", start=start, end=end, bar_limit=1000
    )

    windows = []
    cursor = dataset.start
    while cursor + WINDOW_SIZE <= dataset.end:
        window_end = cursor + WINDOW_SIZE
        trades = dataset.trades.loc[
            (dataset.trades.index >= cursor) & (dataset.trades.index < window_end)
        ]
        if len(trades) >= 100:
            windows.append((cursor, window_end, trades))
        cursor += STEP

    profiles = []
    for window_start, window_end, trades in windows:
        profile = build_trade_volume_profile(
            trades,
            tick_size=0.01,
            profile_bin_ticks=25,
            node_smoothing_bins=3,
            node_prominence=0.25,
            node_min_separation_bins=3,
            node_min_relative_volume=1.0,
        )
        profiles.append((window_start, window_end, profile))

    print("=== BTCUSDT TEMPORAL PROFILE TEST ===")
    print(f"dataset={dataset.start} -> {dataset.end}")
    print(f"windows={len(profiles)}")
    print("\nwindow_end            POC       VAH       VAL       HVNs  LVNs")
    print("--------------------  --------  --------  --------  ----  ----")
    for _, window_end, profile in profiles:
        print(
            f"{window_end.isoformat():<20}  {profile.poc:>8.2f}  {profile.vah:>8.2f}  {profile.val:>8.2f}  "
            f"{len(profile.hvn_nodes):>4}  {len(profile.lvn_nodes):>4}"
        )

    latest_window = profiles[-1][1] if profiles else None
    total_windows = len(profiles)

    for node_type in ("HVN", "LVN"):
        all_nodes = []
        for _, _, profile in profiles:
            nodes = profile.hvn_nodes if node_type == "HVN" else profile.lvn_nodes
            all_nodes.extend(nodes)
        clusters = _cluster_nodes(all_nodes)

        zones = []
        for cluster in clusters:
            matches = _zone_coverage(cluster["center"], profiles, node_type)
            coverage = len(matches) / max(1, total_windows)
            if coverage < MIN_ZONE_COVERAGE:
                continue
            latest_match = matches[-1] if matches else None
            label = _regime_label(len(matches) - 1 if matches else 0, total_windows, coverage, latest_match)
            # Current relevance is explicitly higher when the zone appears in recent windows.
            recent_matches = sum(i >= max(0, total_windows - 4) for i in matches)
            relevance = (coverage * 0.6) + ((recent_matches / min(4, total_windows)) * 0.4)
            zones.append((relevance, coverage, cluster["center"], len(matches), label))

        print(f"\n=== TEMPORAL {node_type} ZONES ===")
        print("center≈price | windows | coverage | relevance | status")
        for relevance, coverage, center, count, label in sorted(zones, reverse=True):
            print(
                f"center≈{center:.2f} | windows={count}/{total_windows} | "
                f"coverage={coverage:.0%} | relevance={relevance:.2f} | {label}"
            )

    if latest_window is not None:
        print(f"\nlatest_window_end={latest_window}")
        print("Status definitions: HIGH_ACTIVE >=75% + recent; MEDIUM_ACTIVE >=50% + recent; "
              "HISTORICAL >=50% but not recent; DEVELOPING >=30% + recent; LOW otherwise.")


if __name__ == "__main__":
    main()
