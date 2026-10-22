"""T001 - drawdown guard threshold (hypothesis written in TEST_LOG.md before running).

Fixed HBA config, 33-stock universe, selection window only (before LOCK_DATE).
Compares guard off against thresholds 8/12/15/20% (floor 0 = go to cash).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from common import LOCK_DATE, ROOT
import hba_v8_snapshot_engine as eng
import mga_compare_v8 as mc

mc.eng.DATA_DIR = ROOT / "data" / "hba33"
prices, feats, loaded, missing = mc.load(str(ROOT / "data" / "hba33"))
prices = prices.loc[prices.index < LOCK_DATE]
feats = {k: (v.loc[v.index < LOCK_DATE] if hasattr(v, "loc") else v) for k, v in feats.items()}
net, W, _ = eng.simulate_full(prices, feats, mc.FIXED)

rows = []
for thr in [None, 0.08, 0.12, 0.15, 0.20]:
    r = net if thr is None else mc.apply_dd_guard(net, thr, 0.0)
    m = eng.metrics(r)
    last5 = eng.metrics(r[r.index >= r.index[-1] - pd.Timedelta(days=365 * 5)])
    cash = float((r == 0).mean())
    rows.append(("off" if thr is None else f"{thr:.0%}", m["sharpe"], m["cagr"], m["max_dd"], last5["sharpe"], cash))

print(f"{'guard':<6}{'Sharpe':>8}{'CAGR':>8}{'maxDD':>8}{'5y Sharpe':>10}{'days flat':>10}")
for g, s, c, d, s5, cash in rows:
    print(f"{g:<6}{s:>8.3f}{c*100:>7.1f}%{d*100:>7.1f}%{s5:>10.3f}{cash*100:>9.0f}%")
