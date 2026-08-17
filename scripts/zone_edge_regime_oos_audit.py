#!/usr/bin/env python3
from __future__ import annotations
import argparse
from datetime import timedelta
from collections import Counter
import numpy as np
import pandas as pd
from zone_edge_persistent_cluster_lifecycle_audit import _load,_assign_persistent_clusters,_independent_events
from zone_edge_historical_test import _future_metrics,_outcome,HORIZONS,_baseline

TEST_DAYS=7
LOW_STATUS='LOW'


def _atr(bars, period=14):
    h=bars['high'].astype(float); l=bars['low'].astype(float); c=bars['close'].astype(float); p=c.shift(1)
    tr=pd.concat([h-l,(h-p).abs(),(l-p).abs()],axis=1).max(axis=1)
    return tr.rolling(period,min_periods=period).mean()


def _regime_series(bars):
    atr=_atr(bars)
    ret=bars['close'].astype(float).pct_change()
    vol=bars['volume'].astype(float)
    atr_pct=atr / bars['close'].astype(float) * 100.0
    vol_ma=vol.rolling(60,min_periods=60).mean()
    # Trend magnitude is a rolling absolute net return, descriptive only.
    trend=ret.rolling(60,min_periods=60).sum().abs()*100.0
    return pd.DataFrame({'atr_pct':atr_pct,'volume_ratio':vol/vol_ma,'trend_pct':trend})


def _terciles(frame):
    vals={}
    for col in frame.columns:
        s=frame[col].dropna()
        vals[col]=(float(s.quantile(1/3)),float(s.quantile(2/3)))
    return vals


def _classify(x, bounds):
    lo,hi=bounds
    if x<=lo:return 'LOW'
    if x<=hi:return 'MEDIUM'
    return 'HIGH'


def main():
    p=argparse.ArgumentParser(); p.add_argument('--days',type=int,default=30); p.add_argument('--test-days',type=int,default=7); a=p.parse_args()
    bars,raw=_load(a.days); _,assigned=_assign_persistent_clusters(raw,bars); events=_independent_events(assigned,5)
    regimes=_regime_series(bars)
    cutoff=bars.index.max()-timedelta(days=a.test_days)
    development=regimes.loc[regimes.index<cutoff].dropna()
    bounds=_terciles(development)
    print('=== BTCUSDT FROZEN REGIME OOS AUDIT ===')
    print(f'dataset={bars.index.min()} -> {bars.index.max()}')
    print(f'development_end={cutoff}')
    print(f'test_start={cutoff}')
    print(f'test_days={a.test_days}')
    print('threshold_source=DEVELOPMENT_ONLY')
    print('atr_pct_terciles=' + ','.join(f'{x:.6f}' for x in bounds['atr_pct']))
    print('volume_ratio_terciles=' + ','.join(f'{x:.6f}' for x in bounds['volume_ratio']))
    print('trend_pct_terciles=' + ','.join(f'{x:.6f}' for x in bounds['trend_pct']))
    test_reg=regimes.loc[regimes.index>=cutoff]
    test_events=[e for e in events if e.timestamp>=cutoff and e.event=='RETEST']
    rows=[]
    for e in test_events:
        if e.status!=LOW_STATUS: continue
        if e.timestamp not in test_reg.index: continue
        r=test_reg.loc[e.timestamp]
        if r.isna().any(): continue
        rows.append((e,_classify(float(r['atr_pct']),bounds['atr_pct']),_classify(float(r['volume_ratio']),bounds['volume_ratio']),_classify(float(r['trend_pct']),bounds['trend_pct'])))
    print(f'raw_events={len(raw)} independent_events={len(events)} test_low_retests={len(rows)}')
    print('condition | n | ' + ' | '.join(f'{h}m_edge' for h in HORIZONS) + ' | ' + ' | '.join(f'{h}m_fav' for h in HORIZONS))
    candidates=[('ALL',lambda v:True),('VOL_HIGH',lambda v:v[1]=='HIGH'),('VOLUME_HIGH',lambda v:v[2]=='HIGH'),('TREND_MID',lambda v:v[3]=='MEDIUM'),('VOL_HIGH_VOLUME_HIGH',lambda v:v[1]=='HIGH' and v[2]=='HIGH')]
    baselines={h:_baseline(bars,[e for e,_,_,_ in rows],h) for h in HORIZONS}
    for name,fn in candidates:
        group=[e for e,*reg in rows if fn((e,*reg))]
        vals={}
        for h in HORIZONS:
            b=baselines[h]
            rr=[]
            for e in group:
                m=_future_metrics(bars,e,h)
                if m is not None and b.get(e.direction) is not None: rr.append((m,b[e.direction]))
            if rr:
                edge=np.mean([x[0][0]-x[1] for x in rr]); fav=np.mean([_outcome(x[0][1],x[0][2])=='FAVORABLE' for x in rr])*100
                vals[h]=(edge,fav)
            else: vals[h]=(float('nan'),float('nan'))
        print(f'{name} | {len(group)} | ' + ' | '.join(f'{vals[h][0]:+.2f}' for h in HORIZONS) + ' | ' + ' | '.join(f'{vals[h][1]:.1f}%' if np.isfinite(vals[h][1]) else 'nan' for h in HORIZONS))
    print('Interpretation: regime thresholds are learned only from development data; candidate rules are evaluated only on the final unseen test window. No V2 scoring changes.')
if __name__=='__main__': main()
