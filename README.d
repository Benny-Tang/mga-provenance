# mga-provenance - research sandbox

MGA (30 diversified ETFs) failed as a strategy: walk-forward Sharpe 0.48 vs HBA 1.36.
This repo is now a sandbox for ideas to improve HBA, run on the CORRECTED engine.
Not connected to any Alpaca account. Paper trading lives in hba-provenance only.

## Layout
- engine/hba_v8_snapshot_engine.py  reference engine (fixes the stale-weight carry bug)
- tools/   MGA vs HBA compare, blend test, 336-config sweep, cost reality check
- tests/   one file per numbered test (t001_...), run from repo root: python3 tests/t001_guard_threshold.py
- data/    put CSVs here: data/hba33 (33 stocks), data/mga30 (30 ETFs), data/ext53 (53 stocks)
           hba33 comes from your existing hba_v4_data; run data/mga_download_data.py and data/hba_ext_download.py in a Codespace for the others
- TEST_LOG.md  every test, win or lose

## Rules
1. Log every test in TEST_LOG.md BEFORE running it: idea, expected effect, config. Fill the result after, even if it loses.
2. No unplanned parameter sweeps. A sweep counts as one test per config tried.
3. Data from LOCK_DATE (2025-08-29) on is a locked holdout. Look at it once per idea, after the config is fixed.
4. Use this engine. If your engine builds weights with replace(0,nan).ffill() it carries dropped names forever and inflates Sharpe (HBA fixed config 1.29 -> 1.10 when fixed).
5. Costs: 5 bps, full-day execution only (overnight-only needs a daily round trip and loses).
6. Report stitched walk-forward Sharpe, median fold, % positive folds, max drawdown. Headline = walk-forward, not full-history.
7. Survivorship caveat: the 33 and 86 stock lists were picked with hindsight; treat every Sharpe as an upper bound.
