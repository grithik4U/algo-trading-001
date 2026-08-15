# algo-trading-001

Professional quantitative trading research and execution framework.

## Project objective

Build a modular, testable trading research system that converts market-structure, liquidity, volume-profile and order-flow hypotheses into measurable features and statistically validated signals.

This project is research-first. No live execution logic should be enabled until the relevant components have been backtested, stress-tested, validated out-of-sample and evaluated with realistic transaction costs.

## Initial research scope

### Phase 1 — Market Structure & Liquidity
- Swing highs and lows
- Higher-timeframe structural levels
- Session highs and lows
- Internal vs external liquidity classification
- Buy-side and sell-side liquidity pools
- Equal highs and lows
- Liquidity sweeps
- Reclaim vs accepted breakout
- Level strength and ranking

### Planned phases

2. Displacement and structural shifts
3. Fair value / price-overlap features
4. Volume Profile: POC, VAH, VAL, HVN, LVN
5. Order-flow and market-depth features
6. Signal research and feature interaction
7. Backtesting and transaction-cost modelling
8. Walk-forward validation and paper trading
9. Production execution

## Design principles

- Definitions must be explicit and machine-testable.
- Practitioner terminology is treated as a hypothesis, not as a proven market law.
- Avoid look-ahead bias and survivorship bias.
- Separate feature generation, signal generation, risk and execution.
- Prefer out-of-sample and walk-forward validation over in-sample optimisation.
- Record assumptions and research results so experiments are reproducible.

## Repository structure

```text
src/trading_engine/
├── structure/
├── liquidity/
├── indicators/
├── volume_profile/
├── orderflow/
├── signals/
├── risk/
└── execution/

tests/
notebooks/
backtests/
docs/
```

## Status

Phase 1 foundation.
