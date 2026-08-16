#!/usr/bin/env python3
"""Independent RETEST subgroup decomposition. Audit only; V2 scoring unchanged."""
from __future__ import annotations
import argparse
from collections import defaultdict
from zone_edge_persistent_cluster_lifecycle_audit import _load, _assign_persistent_clusters, _independent_events
from zone_edge_historical_test import HORIZONS, _baseline, _future_metrics, _outcome


def main():
    p=argparse.ArgumentParser(); p.add_argument('--days',type=int,default=7); p.add_argument('--gap',type=int,default=5); a=p.parse_args()
    bars,raw=_load(a.days); _,assigned=_assign_persistent_clusters(raw,bars); events=[e for e in _independent_events(assigned,a.gap) if e.event=='RETEST' and e.direction]
    print('=== BTCUSDT INDEPENDENT RETEST DECOMPOSITION AUDIT ===')
    print(f'dataset={bars.index.min()} -> {bars.index.max()}')
    print(f'independent_retests={len(events)} gap={a.gap}m')
    groups=defaultdict(list)
    for e in events:
        groups['ALL'].append(e); groups[f'{e.node_type}:{e.status}'].append(e); groups[e.node_type].append(e); groups[e.status].append(e); groups[e.direction].append(e)
    def report(name, group):
        print(f'\n{name} | n={len(group)}')
        for h in HORIZONS:
            b=_baseline(bars,events,h); rows=[]
            for e in group:
                m=_future_metrics(bars,e,h)
                if m is not None and b.get(e.direction) is not None: rows.append((m,b[e.direction]))
            if not rows: print(f'  {h}m: n=0'); continue
            avg=sum(x[0][0] for x in rows)/len(rows); base=sum(x[1] for x in rows)/len(rows); fav=sum(_outcome(x[0][1],x[0][2])=='FAVORABLE' for x in rows)/len(rows)*100; adv=sum(_outcome(x[0][1],x[0][2])=='ADVERSE' for x in rows)/len(rows)*100
            print(f'  {h:2d}m: n={len(rows):3d} fav={fav:5.1f}% adv={adv:5.1f}% avg={avg:+8.2f} base={base:+8.2f} edge={avg-base:+8.2f}')
    report('ALL RETESTS',groups['ALL'])
    for key in sorted(groups):
        if key!='ALL' and ':' in key: report(key,groups[key])
    for key in ('UP','DOWN'):
        if key in groups: report(f'DIRECTION={key}',groups[key])
    print('\nInterpretation: subgroup audit only. No threshold optimization, no V2 scoring changes, and no trading signal is produced.')
if __name__=='__main__': main()
