#!/usr/bin/env python3
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import timedelta
from zone_edge_persistent_cluster_audit import _load_bars, _build_profile_snapshots, _find_events

THRESHOLDS = (2, 5, 10, 15, 30, 60)

def get(e, *names):
    for n in names:
        if hasattr(e, n): return getattr(e, n)
        if isinstance(e, dict) and n in e: return e[n]
    return None

def audit(events, minutes):
    groups=defaultdict(list)
    for e in events:
        c=get(e,'persistent_cluster_id','persistent_id','cluster_id')
        t=get(e,'timestamp','event_timestamp','time','ts')
        if c is not None and t is not None: groups[c].append((t,e))
    independent=suppressed=0
    lifecycle_sizes=[]
    for c, items in groups.items():
        items.sort(key=lambda x:x[0]); life=0; last=None
        for t,e in items:
            if last is None or t-last >= timedelta(minutes=minutes):
                independent+=1; life+=1
            else: suppressed+=1
            last=t
        lifecycle_sizes.append(life)
    total=independent+suppressed
    return independent,suppressed,(100*independent/total if total else 0),max(lifecycle_sizes,default=0)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--days',type=int,default=7); a=p.parse_args()
    bars=_load_bars(a.days); snaps=_build_profile_snapshots(bars); events,_,_=_find_events(bars,snaps)
    tscol='timestamp' if 'timestamp' in bars.columns else bars.columns[0]
    print('=== BTCUSDT LIFECYCLE SENSITIVITY AUDIT ===')
    print(f'dataset={bars.iloc[0][tscol]} -> {bars.iloc[-1][tscol]}')
    print(f'raw_events={len(events)}')
    print('threshold_min | independent | suppressed | retained_ratio | max_lifecycles_cluster')
    for m in THRESHOLDS:
        i,s,r,mx=audit(events,m); print(f'{m:14d} | {i:11d} | {s:10d} | {r:13.2f}% | {mx:23d}')
    print('Interpretation: sensitivity only. No v2 edge scoring is changed.')
if __name__=='__main__': main()
