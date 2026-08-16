#!/usr/bin/env python3
"""Independent-event edge validation. Audit only; V2 scoring is unchanged."""
from __future__ import annotations
import argparse
from collections import Counter
from zone_edge_persistent_cluster_lifecycle_audit import _load,_assign_persistent_clusters,_independent_events
from zone_edge_historical_test import HORIZONS, _baseline, _future_metrics, _outcome

KINDS=("BREAKOUT","RETEST")

def _rows(bars, events, horizon):
    baseline=_baseline(bars,events,horizon)
    out=[]
    for e in events:
        if e.event not in KINDS or e.direction is None: continue
        m=_future_metrics(bars,e,horizon)
        if m is None or baseline.get(e.direction) is None: continue
        move,mfe,mae=m
        out.append((e,move,baseline[e.direction],mfe,mae,_outcome(mfe,mae)))
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--days',type=int,default=7); p.add_argument('--gap',type=int,default=5); a=p.parse_args()
    bars,raw=_load(a.days); _,assigned=_assign_persistent_clusters(raw,bars); events=_independent_events(assigned,a.gap)
    counts=Counter(e.event for e in events)
    print('=== BTCUSDT INDEPENDENT-EVENT EDGE AUDIT ===')
    print(f'dataset={bars.index.min()} -> {bars.index.max()}')
    print(f'raw_events={len(raw)} independent_events={len(events)} retained_ratio={100*len(events)/len(raw):.2f}%')
    print('independent_event_counts='+' '.join(f'{k.lower()}={counts.get(k,0)}' for k in KINDS))
    for h in HORIZONS:
        rows=_rows(bars,events,h)
        print(f'\n=== {h}M INDEPENDENT EDGE ===')
        print('event | n | favorable | adverse | avg dir | avg baseline | edge')
        print('------|---|-----------|---------|----------|---------------|-----')
        for kind in KINDS:
            r=[x for x in rows if x[0].event==kind]
            if not r:
                print(f'{kind:8s}| {0:3d} |      0.0% |     0.0% |     0.00 |          0.00 | +0.00'); continue
            fav=sum(x[5]=='FAVORABLE' for x in r)/len(r)*100
            adv=sum(x[5]=='ADVERSE' for x in r)/len(r)*100
            avg=sum(x[1] for x in r)/len(r); base=sum(x[2] for x in r)/len(r); edge=avg-base
            print(f'{kind:8s}| {len(r):3d} | {fav:9.1f}% | {adv:8.1f}% | {avg:9.2f} | {base:14.2f} | {edge:+.2f}')
        print(f'baseline_anchors: UP={_baseline(bars,events,h).get("UP")} DOWN={_baseline(bars,events,h).get("DOWN")}')
    print('\nInterpretation: only canonical independent lifecycle events are scored. Immediate breakout/retest duplication is removed before edge calculation. No look-ahead is introduced and V2 scoring remains unchanged.')
if __name__=='__main__': main()
