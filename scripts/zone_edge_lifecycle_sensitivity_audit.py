#!/usr/bin/env python3
from __future__ import annotations
import argparse
from zone_edge_persistent_cluster_lifecycle_audit import _load,_assign_persistent_clusters,_audit_lifecycles
THRESHOLDS=(2,5,10,15,30,60)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--days',type=int,default=7); a=p.parse_args(); bars,events=_load(a.days); _,assigned=_assign_persistent_clusters(events,bars)
 print('=== BTCUSDT LIFECYCLE SENSITIVITY AUDIT ==='); print(f'dataset={bars.index.min()} -> {bars.index.max()}'); print(f'raw_events={len(events)}'); print('threshold_min | independent | suppressed | retained_ratio | max_lifecycles_cluster')
 for m in THRESHOLDS:
  i,s,c,_,_=_audit_lifecycles(assigned,m); total=i+s; mx=max(c.values(),default=0); print(f'{m:14d} | {i:11d} | {s:10d} | {100*i/total if total else 0:13.2f}% | {mx:23d}')
 print('Interpretation: sensitivity uses the canonical lifecycle engine. No v2 edge scoring is changed.')
if __name__=='__main__': main()
