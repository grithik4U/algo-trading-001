#!/usr/bin/env python3
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import timedelta
from zone_edge_persistent_cluster_lifecycle_audit import _load, _assign_persistent_clusters

THRESHOLDS=(2,5,10,15,30,60)

def audit(assigned, minutes):
    independent=suppressed=0; max_life=0
    for items in assigned.values():
        items=sorted(items,key=lambda e:e.timestamp); last=None; life=0
        for e in items:
            if last is None or e.timestamp-last >= timedelta(minutes=minutes):
                independent+=1; life+=1
            else: suppressed+=1
            last=e.timestamp
        max_life=max(max_life,life)
    total=independent+suppressed
    return independent,suppressed,100*independent/total if total else 0,max_life

def main():
    p=argparse.ArgumentParser(); p.add_argument('--days',type=int,default=7); a=p.parse_args()
    bars,events=_load(a.days); _,assigned=_assign_persistent_clusters(events,bars)
    print('=== BTCUSDT LIFECYCLE SENSITIVITY AUDIT ===')
    print(f'dataset={bars.index.min()} -> {bars.index.max()}')
    print(f'raw_events={len(events)}')
    print('threshold_min | independent | suppressed | retained_ratio | max_lifecycles_cluster')
    for m in THRESHOLDS:
        i,s,r,mx=audit(assigned,m); print(f'{m:14d} | {i:11d} | {s:10d} | {r:13.2f}% | {mx:23d}')
    print('Interpretation: sensitivity only. No v2 edge scoring is changed.')
if __name__=='__main__': main()
