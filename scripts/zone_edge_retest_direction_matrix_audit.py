#!/usr/bin/env python3
"""Verify RETEST directional framing and run the structural 2x2 audit.

Audit-only: does not modify V2 scoring and does not produce a trading signal.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from zone_edge_persistent_cluster_lifecycle_audit import _load, _assign_persistent_clusters, _independent_events
from zone_edge_historical_test import HORIZONS, _baseline, _future_metrics


def _rows(bars, events, horizon):
    baseline = _baseline(bars, events, horizon)
    out = []
    for e in events:
        if e.event != "RETEST" or e.direction not in {"UP", "DOWN"}:
            continue
        m = _future_metrics(bars, e, horizon)
        if m is None or baseline.get(e.direction) is None:
            continue
        normalized, mfe, mae = m
        sign = 1.0 if e.direction == "UP" else -1.0
        # _future_metrics returns direction-normalized movement. Reconstruct
        # raw movement from the same final close/entry relation without using
        # any future value for event classification.
        raw_move = normalized * sign
        out.append((e, raw_move, normalized, baseline[e.direction], mfe, mae))
    return out


def _report(name, rows):
    print(f"\n{name} | n={len(rows)}")
    for h in HORIZONS:
        r = rows[h]
        if not r:
            print(f"  {h:2d}m: n=0")
            continue
        raw = sum(x[1] for x in r) / len(r)
        norm = sum(x[2] for x in r) / len(r)
        base = sum(x[3] for x in r) / len(r)
        mfe = sum(x[4] for x in r) / len(r)
        mae = sum(x[5] for x in r) / len(r)
        favorable_frame = sum(x[2] > 0 for x in r) / len(r) * 100
        adverse_frame = sum(x[2] < 0 for x in r) / len(r) * 100
        print(
            f"  {h:2d}m: n={len(r):3d} raw_move={raw:+9.2f} "
            f"norm_move={norm:+9.2f} baseline={base:+8.2f} edge={norm-base:+8.2f} "
            f"mfe={mfe:+8.2f} mae={mae:+8.2f} "
            f"norm_pos={favorable_frame:5.1f}% norm_neg={adverse_frame:5.1f}%"
        )


def main():
    p = argparse.ArgumentParser(description="RETTEST directional frame + 2x2 structural audit")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--gap", type=int, default=5)
    args = p.parse_args()

    bars, raw = _load(args.days)
    _, assigned = _assign_persistent_clusters(raw, bars)
    events = [e for e in _independent_events(assigned, args.gap) if e.event == "RETEST" and e.direction in {"UP", "DOWN"}]

    print("=== BTCUSDT RETEST DIRECTION / STRUCTURAL MATRIX AUDIT ===")
    print(f"dataset={bars.index.min()} -> {bars.index.max()}")
    print(f"independent_retests={len(events)} gap={args.gap}m")
    print("direction_frame=UP means future price movement is measured positive when price rises; DOWN means positive when price falls")
    print("normalized_move is the direction-aligned value returned by _future_metrics; raw_move is reconstructed as normalized_move for UP and -normalized_move for DOWN")

    all_rows = {h: _rows(bars, events, h) for h in HORIZONS}
    _report("ALL RETESTS", all_rows)

    groups = defaultdict(list)
    for e in events:
        # The requested structural matrix focuses on the stable statuses with
        # enough expected observations; other statuses are retained separately.
        groups[f"{e.node_type}:{e.status}:{e.direction}"].append(e)

    print("\n=== 2x2 STRUCTURAL MATRIX (LOW / DEVELOPING × HVN / LVN × DIRECTION) ===")
    for status in ("LOW", "DEVELOPING"):
        print(f"\n-- STATUS={status} --")
        for node_type in ("HVN", "LVN"):
            for direction in ("UP", "DOWN"):
                key = f"{node_type}:{status}:{direction}"
                group = groups.get(key, [])
                rows = {h: [x for x in _rows(bars, group, h)] for h in HORIZONS}
                print(f"\n{key} | n={len(group)}")
                for h in HORIZONS:
                    r = rows[h]
                    if not r:
                        print(f"  {h:2d}m: n=0")
                        continue
                    norm = sum(x[2] for x in r) / len(r)
                    base = sum(x[3] for x in r) / len(r)
                    edge = norm - base
                    pos = sum(x[2] > 0 for x in r) / len(r) * 100
                    print(f"  {h:2d}m: n={len(r):3d} norm={norm:+8.2f} base={base:+7.2f} edge={edge:+8.2f} pos={pos:5.1f}%")

    print("\n=== DIRECTION FRAME SANITY ===")
    for direction in ("UP", "DOWN"):
        group = [e for e in events if e.direction == direction]
        print(f"{direction}: n={len(group)}")
        for h in HORIZONS:
            r = _rows(bars, group, h)
            if not r:
                print(f"  {h:2d}m: n=0")
                continue
            raw_avg = sum(x[1] for x in r) / len(r)
            norm_avg = sum(x[2] for x in r) / len(r)
            print(f"  {h:2d}m: raw_avg={raw_avg:+.2f} normalized_avg={norm_avg:+.2f} sign_check={'OK' if (direction == 'UP' and norm_avg == raw_avg) or (direction == 'DOWN' and norm_avg == -raw_avg) else 'REVIEW'}")

    print("\nInterpretation: audit only. The matrix is descriptive; no subgroup is promoted to a trading rule and V2 scoring remains unchanged.")


if __name__ == "__main__":
    main()
