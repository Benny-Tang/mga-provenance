"""Pre-registered second-signal test on HBA's original 33-name universe.

Variants (fixed weights, no tuning after seeing results):
  base        : HBA momentum score as-is
  +reversal   : base + 0.30 * centered rank of (-5-day return)
  +lowvol     : base + 0.30 * centered rank of (-63-day vol)
  +overnight  : base + 0.30 * centered rank of trailing 21-day sum of
                close-to-open (overnight) returns
  +all3       : base + 0.20 each of the three
Each variant gets the identical honest-cost walk-forward (full_day only -
overnight execution already shown to lose in every fold; 36-config grid).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))
import hba_v8_snapshot_engine as eng

eng.DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "hba33"


def grid_full_day(cost_bps):
    out = []
    for top_n in [2, 3, 4]:
        for weighting in ["equal", "inverse_vol", "score_weighted"]:
            for regime_mode in ["off", "binary"]:
                for target in [None, 0.15]:
                    out.append(eng.Config(top_n, 5, weighting, 0.0, True, regime_mode, 0.50,
                                          target, 1.0, cost_bps, "full_day"))
    return out


eng.parameter_grid = grid_full_day  # walk_forward looks this up at call time


def build():
    universe = eng.load_universe()
    prices, loaded, _ = eng.load_prices(universe)
    prices = prices.loc[(prices.index >= "2010-01-01") & (prices.index <= "2026-08-31")]
    opens = eng.load_open_prices(loaded).reindex(index=prices.index)[prices.columns]
    overnight = opens / prices.shift(1) - 1
    feats = eng.make_features(prices, overnight_ret=overnight)
    r = eng.rank01
    comp = {
        "reversal": r(-(prices / prices.shift(5) - 1)) - 0.5,
        "lowvol": r(-feats["vol63"]) - 0.5,
        "overnight": r(overnight.rolling(21, min_periods=15).sum()) - 0.5,
    }
    return prices, feats, comp


def variant(feats, comp, weights):
    f = dict(feats)
    f["score"] = feats["score"] + sum(w * comp[k] for k, w in weights.items())
    return f


VARIANTS = {
    "base": {},
    "+reversal": {"reversal": 0.30},
    "+lowvol": {"lowvol": 0.30},
    "+overnight": {"overnight": 0.30},
    "+all3": {"reversal": 0.20, "lowvol": 0.20, "overnight": 0.20},
}

if __name__ == "__main__":
    prices, feats, comp = build()
    streams = {}
    rows = []
    for name, wts in VARIANTS.items():
        f = variant(feats, comp, wts)
        stitched, folds, hold, cfg = eng.walk_forward(prices, f)
        s, h = eng.metrics(stitched), eng.metrics(hold)
        streams[name] = stitched
        rows.append((name, s["sharpe"], s["cagr"], s["max_dd"], folds["oos_sharpe"].median(),
                     (folds["oos_sharpe"] > 0).mean(), h["sharpe"]))
        print(f"done {name}", flush=True)
    print(f"\n{'variant':<12}{'stitched':>9}{'CAGR':>8}{'maxDD':>8}{'medFold':>9}{'%pos':>6}{'holdout':>9}")
    for n, sh, cg, dd, med, pos, ho in rows:
        print(f"{n:<12}{sh:>9.3f}{cg*100:>7.1f}%{dd*100:>7.1f}%{med:>9.3f}{pos*100:>5.0f}%{ho:>9.3f}")
    df = pd.concat(streams, axis=1).dropna()
    print("\nCorrelation of each variant's OOS daily returns with base:")
    print(df.corr()["base"].round(3).to_string())
