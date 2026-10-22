"""MGA vs HBA comparison harness.

Runs the SAME corrected HBA config (top_n=2, rebalance 5, inverse_vol,
binary regime @0.5, target_vol 0.15, full_day, 5bps, -8% dd guard) on
any universe directory, so HBA (33 US large caps) and MGA (diversified
cross-sector/asset ETFs) are compared apples to apples. Also reports the
corrected-cost walk-forward (v6 engine) per universe.

Usage: python3 -I mga_compare.py LABEL=DATA_DIR [LABEL=DATA_DIR ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine')); sys.path.insert(0, str(Path(__file__).resolve().parent))
import hba_v8_snapshot_engine as eng

FIXED = eng.Config(2, 5, "inverse_vol", 0.0, True, "binary", 0.5, 0.15, 1.0, 5.0, "full_day")
DD_THRESHOLD, DD_FLOOR = 0.08, 0.0


def apply_dd_guard(net, threshold, floor):
    eq = (1 + net.fillna(0.0)).cumprod()
    dd = eq / eq.cummax() - 1
    scale = np.where(dd.shift(1).fillna(0.0) <= -abs(threshold), floor, 1.0)
    return net * scale


def load(data_dir):
    eng.DATA_DIR = Path(data_dir)
    uni = eng.load_universe()
    prices, loaded, missing = eng.load_prices(uni)
    import os
    end = os.environ.get("COMPARE_END", "2026-08-31")  # common window for fairness
    prices = prices.loc[(prices.index >= "2010-01-01") & (prices.index <= end)]
    opens = eng.load_open_prices(loaded).reindex(index=prices.index)[prices.columns]
    feats = eng.make_features(prices, overnight_ret=opens / prices.shift(1) - 1)
    return prices, feats, loaded, missing


def fmt(m):
    return (f"Sharpe {m['sharpe']:.3f} | CAGR {m['cagr']*100:5.1f}% | "
            f"vol {m['vol']*100:4.1f}% | maxDD {m['max_dd']*100:6.1f}%")


def run(label, data_dir):
    prices, feats, loaded, missing = load(data_dir)
    print(f"\n=== {label}: {len(loaded)} tickers ({'missing: ' + ','.join(missing) if missing else 'none missing'}), "
          f"{prices.index.min().date()} -> {prices.index.max().date()} ===")
    net, W, _ = eng.simulate_full(prices, feats, FIXED)
    guarded = apply_dd_guard(net, DD_THRESHOLD, DD_FLOOR)
    last = prices.index[-1] - pd.Timedelta(days=365)
    last5 = prices.index[-1] - pd.Timedelta(days=365 * 5)
    print("Fixed config, full history, dd-guarded :", fmt(eng.metrics(guarded)))
    print("Fixed config, last 5y                  :", fmt(eng.metrics(guarded[guarded.index >= last5])))
    print("Fixed config, last 12mo                :", fmt(eng.metrics(guarded[guarded.index >= last])))
    print("Fixed config, unguarded (reference)    :", fmt(eng.metrics(net)))
    held = (W.abs().sum(axis=1) > 0).mean()
    top = (W.gt(0).sum().sort_values(ascending=False) / len(W)).head(6)
    print(f"Days invested: {held*100:.0f}% | most-held names (share of days): " +
          ", ".join(f"{k} {v*100:.0f}%" for k, v in top.items()))

    stitched, folds, hold, final_cfg = eng.walk_forward(prices, feats)
    s, h = eng.metrics(stitched), eng.metrics(hold)
    print(f"Walk-forward (72-config honest-cost grid): stitched OOS {fmt(s)}")
    print(f"   median fold OOS Sharpe {folds['oos_sharpe'].median():.3f} | "
          f"folds positive {(folds['oos_sharpe']>0).mean()*100:.0f}% | "
          f"holdout {fmt(h)}")
    print(f"   exec chosen: {folds['execution'].value_counts().to_dict()}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        run(label, path)
