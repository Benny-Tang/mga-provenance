"""HBA + MGA blend test: are the two return streams different enough that
a mix beats HBA alone? Same fixed config as before, plus walk-forward
stitched OOS streams (more honest - no config chosen on the full sample).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine')); sys.path.insert(0, str(Path(__file__).resolve().parent))
import hba_v8_snapshot_engine as eng
import mga_compare_v8 as mc

ROOT = Path(__file__).resolve().parents[1]
HBA_DIR = str(ROOT / 'data' / 'hba33')
MGA_DIR = str(ROOT / 'data' / 'mga30')


def sharpe(r):
    r = r.dropna()
    return r.mean() / r.std(ddof=1) * np.sqrt(252)


def maxdd(r):
    eq = (1 + r.dropna()).cumprod()
    return (eq / eq.cummax() - 1).min()


def report(title, a, b):
    df = pd.concat({"HBA": a, "MGA": b}, axis=1).dropna()
    df = df.loc[(df != 0).any(axis=1)]
    print(f"\n--- {title} ({df.index.min().date()} -> {df.index.max().date()}, {len(df)} days) ---")
    print(f"Correlation of daily returns: {df['HBA'].corr(df['MGA']):.3f}")
    print(f"{'HBA weight':>10} {'Sharpe':>8} {'CAGR':>7} {'maxDD':>8}")
    for w in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0]:
        r = w * df["HBA"] + (1 - w) * df["MGA"]
        cagr = (1 + r).prod() ** (252 / len(r)) - 1
        print(f"{w:>10.1f} {sharpe(r):>8.3f} {cagr*100:>6.1f}% {maxdd(r)*100:>7.1f}%")


def streams(data_dir):
    prices, feats, loaded, _ = mc.load(data_dir)
    fixed_net, _, _ = eng.simulate_full(prices, feats, mc.FIXED)
    stitched, folds, hold, cfg = eng.walk_forward(prices, feats)
    return fixed_net, stitched


if __name__ == "__main__":
    h_fixed, h_wf = streams(HBA_DIR)
    m_fixed, m_wf = streams(MGA_DIR)
    report("Fixed config, unguarded, full history", h_fixed, m_fixed)
    report("Walk-forward stitched out-of-sample", h_wf, m_wf)
