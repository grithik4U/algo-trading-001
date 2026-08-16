#!/usr/bin/env python3
"""Canonical persistent-cluster lifecycle engine/audit. Audit only; v2 scoring unchanged."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import numpy as np
from trading_engine.data.binance import BinanceConfig, BinancePublicData
from zone_edge_historical_test import _load_klines, _build_profiles_from_bars
from zone_edge_walkforward_v2 import _build_snapshots, _find_events
from zone_edge_persistent_cluster_audit import _atr, _event_range, _match_distance

PERSISTENT_MAX_GAP_MINUTES=120
RETEST_MAX_MINUTES=120

def _load(days:int):
    end=datetime.now(timezone.utc).replace(second=0,microsecond=0); start=end-timedelta(days=days)
    bars=_load_klines(BinancePublicData(BinanceConfig(symbol='BTCUSDT',page_limit=1000)),start,end)
    profiles=_build_profiles_from_bars(bars); snapshots=_build_snapshots(profiles,bars)
    events,_,_=_find_events(bars,snapshots)
    return bars,sorted(events,key=lambda e:e.timestamp)

def _assign_persistent_clusters(events,bars):
    atr=_atr(bars); clusters=[]; assigned=defaultdict(list); next_id=0
    for event in events:
        lo,hi=_event_range(event); center=float(event.center)
        av=atr.loc[event.timestamp] if event.timestamp in atr.index else np.nan
        md=_match_distance(av); candidates=[]
        for c in clusters:
            age=(event.timestamp-c['last_seen']).total_seconds()/60
            if age<=PERSISTENT_MAX_GAP_MINUTES and abs(c['center']-center)<=md:
                candidates.append((abs(c['center']-center),c))
        if candidates:
            _,c=min(candidates,key=lambda x:x[0]); c['center']=(c['center']+center)/2; c['low']=min(c['low'],lo); c['high']=max(c['high'],hi)
        else:
            c={'id':next_id,'center':center,'low':lo,'high':hi,'last_seen':event.timestamp}; clusters.append(c); next_id+=1
        c['last_seen']=event.timestamp; assigned[c['id']].append(event)
    return clusters,assigned

def _audit_lifecycles(assigned,interaction_gap_minutes=5):
    independent=suppressed=0; counts=Counter(); reasons=Counter(); examples=[]
    for cid,items in assigned.items():
        items=sorted(items,key=lambda e:e.timestamp); lifecycle=0; last_ts=None; state='IDLE'; last_breakout=None; seen_retest=False
        for e in items:
            typ=str(e.event).upper(); ts=e.timestamp
            if state=='IDLE':
                lifecycle+=1; independent+=1; state='BROKEN' if typ=='BREAKOUT' else 'IDLE'; last_ts=ts; last_breakout=ts if typ=='BREAKOUT' else None; seen_retest=False; continue
            if typ=='RETEST' and last_breakout is not None and not seen_retest:
                gap=(ts-last_breakout).total_seconds()/60
                if 0<=gap<=RETEST_MAX_MINUTES:
                    suppressed+=1; reasons['breakout_retest_same_lifecycle']+=1; seen_retest=True; state='IDLE'; last_ts=ts; continue
            if typ=='BREAKOUT' and last_ts is not None and (ts-last_ts).total_seconds()/60>=interaction_gap_minutes:
                lifecycle+=1; independent+=1; state='BROKEN'; last_ts=ts; last_breakout=ts; seen_retest=False; continue
            suppressed+=1; reasons['repeated_within_lifecycle']+=1; last_ts=ts
        counts[cid]=lifecycle
        if len(items)>=5: examples.append((cid,len(items),lifecycle))
    return independent,suppressed,counts,reasons,examples

def main():
    p=argparse.ArgumentParser(); p.add_argument('--days',type=int,default=7); p.add_argument('--gap',type=int,default=5); a=p.parse_args()
    bars,events=_load(a.days); clusters,assigned=_assign_persistent_clusters(events,bars); i,s,counts,reasons,examples=_audit_lifecycles(assigned,a.gap)
    raw=Counter(str(e.event).upper() for e in events); total=i+s
    print('=== BTCUSDT PERSISTENT CLUSTER LIFECYCLE AUDIT ==='); print(f'dataset={bars.index.min()} -> {bars.index.max()}'); print(f'raw_events={len(events)}'); print('raw_event_counts='+' '.join(f'{k.lower()}={raw[k]}' for k in ('BREAKOUT','RETEST','SWEEP','REJECTION'))); print(f'persistent_clusters={len(clusters)}'); print(f'interaction_gap_minutes={a.gap}'); print(f'independent_lifecycles={i}'); print(f'suppressed_within_lifecycle={s}'); print(f'retained_ratio={100*i/total:.2f}%' if total else 'retained_ratio=0.00%'); print(f'max_lifecycles_single_cluster={max(counts.values(),default=0)}'); print('suppression_reasons='+' '.join(f'{k}={v}' for k,v in reasons.most_common())); print('lifecycle_distribution='+' '.join(f'{k}:{v}' for k,v in Counter(counts.values()).most_common(10))); print('sample_high_density_clusters:');
    for cid,raw_n,life in sorted(examples,key=lambda x:(-x[1],x[0]))[:10]: print(f'  P{cid} | raw_events={raw_n} lifecycles={life}')
    print('Interpretation: canonical lifecycle engine. Breakout+immediate retest is one lifecycle; later breakout is independent only after the configured interaction gap. Audit only; v2 edge scoring unchanged.')
if __name__=='__main__': main()
