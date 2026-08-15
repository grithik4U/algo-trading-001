"""Test structural volume-profile persistence across rolling time windows."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trading_engine.data.binance import BinanceConfig, BinancePublicData
from trading_engine.data.dataset import load_aligned_binance_dataset
from trading_engine.volume_profile.trade_profile import build_trade_volume_profile


WINDOW_SIZE = timedelta(hours=1)
STEP = timedelta(minutes=15)
NODE_MATCH_TOLERANCE = 1.25
ZONE_GAP_TOLERANCE = 3.5
MAX_ZONE_WIDTH = 10.0
MIN_ZONE_COVERAGE = 0.30


def _cluster_nodes(nodes, tolerance: float = NODE_MATCH_TOLERANCE):
    """Cluster close node observations without chaining through distant nodes."""
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
        values = sorted(n.center for n in best["nodes"])
        mid = len(values) // 2
        best["center"] = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
    return clusters


def _consolidate_zones(clusters, gap_tolerance: float = ZONE_GAP_TOLERANCE):
    """Merge nearby node clusters into structural regions while preserving a width cap."""
    if not clusters:
        return []

    ordered = sorted(clusters, key=lambda c: c["center"])
    zones = []
    current = {"clusters": [ordered[0]]}

    for cluster in ordered[1:]:
        current_low = min(c["center"] for c in current["clusters"])
        current_high = max(c["center"] for c in current["clusters"])
        proposed_high = max(current_high, cluster["center"])
        gap = cluster["center"] - current_high

        if gap <= gap_tolerance and proposed_high - current_low <= MAX_ZONE_WIDTH:
            current["clusters"].append(cluster)
        else:
            zones.append(current)
            current = {"clusters": [cluster]}
    zones.append(current)

    result = []
    for zone in zones:
        centers = [c["center"] for c in zone["clusters"]]
        result.append(
            {
                "low": min(centers),
                "high": max(centers),
                "center": sum(centers) / len(centers),
                "clusters": zone["clusters"],
            }
        )
    return result


def _zone_coverage(zone, profiles, node_type: str, tolerance: float = NODE_MATCH_TOLERANCE):
    """Return windows containing at least one node inside the structural zone."""
    matched = []
    for i, (_, _, profile) in enumerate(profiles):
        nodes = profile.hvn_nodes if node_type == "HVN" else profile.lvn_nodes
        if any(zone["low"] - tolerance <= node.center <= zone["high"] + tolerance for node in nodes):
            matched.append(i)
    return matched


def _regime_label(total_windows: int, coverage: float, latest_match: int | None) -> str:
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
        zones = _consolidate_zones(clusters)
        scored = []

        for zone in zones:
            matches = _zone_coverage(zone, profiles, node_type)
            coverage = len(matches) / max(1, total_windows)
            if coverage < MIN_ZONE_COVERAGE:
                continue

            latest_match = matches[-1] if matches else None
            label = _regime_label(total_windows, coverage, latest_match)
            recent_matches = sum(i >= max(0, total_windows - 4) for i in matches)
            recent_ratio = recent_matches / min(4, total_windows)
            relevance = (coverage * 0.6) + (recent_ratio * 0.4)
            scored.append(
                (
                    relevance,
                    coverage,
                    zone["low"],
                    zone["high"],
                    zone["center"],
                    len(matches),
                    label,
                    sum(len(c["nodes"]) for c in zone["clusters"]),
                )
            )

        print(f"\n=== TEMPORAL {node_type} STRUCTURAL ZONES ===")
        print("zone | center | windows | coverage | relevance | nodes | status")
        for relevance, coverage, low, high, center, count, label, node_count in sorted(scored, reverse=True):
            print(
                f"{low:.2f} -> {high:.2f} | center≈{center:.2f} | "
                f"windows={count}/{total_windows} | coverage={coverage:.0%} | "
                f"relevance={relevance:.2f} | nodes={node_count} | {label}"
            )

    if latest_window is not None:
        print(f"\nlatest_window_end={latest_window}")
        print("Zone rules: nodes within $1.25 are matched; nearby structural clusters within $3.50 are "
              "consolidated, with a maximum zone width of $10.00.")
        print("Status definitions: HIGH_ACTIVE >=75% + recent; MEDIUM_ACTIVE >=50% + recent; "
              "HISTORICAL >=50% but not recent; DEVELOPING >=30% + recent; LOW otherwise.")


if __name__ == "__main__":
    main()
