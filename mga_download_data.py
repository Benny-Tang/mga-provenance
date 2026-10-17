"""Mongoose Alpha (MGA) data downloader.

Run this on YOUR computer (the Claude workspace cannot reach Yahoo).
Needs:  pip install yfinance pandas

Writes ./mga_data/<TICKER>.csv (Date,Open,High,Low,Close,Volume - same
format as the HBA 33-ticker data, split/dividend adjusted) plus
./mga_data/universe.csv and ./mga_data/download_manifest.csv.
Then zip the mga_data folder and upload it.
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

# Cross-sector / cross-asset universe, all with history back to 2010 or earlier.
UNIVERSE = [
    # US sector ETFs (SPDR Select Sector)
    "XLE", "XLU", "XLP", "XLV", "XLI", "XLB", "XLY", "XLF", "XLK",
    # US broad / size / style
    "SPY", "QQQ", "IWM", "MDY",
    # International equity
    "EFA", "EEM", "EWJ", "EWZ", "EWG", "FXI", "EWY",
    # Bonds / credit
    "TLT", "IEF", "HYG", "LQD",
    # Real assets / commodities
    "GLD", "SLV", "USO", "DBC", "VNQ", "DBA",
]

START = "2010-01-01"
OUT = Path("mga_data")
OUT.mkdir(exist_ok=True)

manifest = []
for t in UNIVERSE:
    try:
        d = yf.download(t, start=START, auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
        d.index.name = "Date"
        d.to_csv(OUT / f"{t}.csv")
        manifest.append((t, "OK", len(d), d.index.min().date(), d.index.max().date()))
        print(f"{t}: {len(d)} rows, {d.index.min().date()} -> {d.index.max().date()}")
    except Exception as e:  # noqa: BLE001
        manifest.append((t, f"FAIL {e}", 0, "", ""))
        print(f"{t}: FAILED {e}")

ok = [m[0] for m in manifest if m[1] == "OK"]
pd.DataFrame({"Ticker": ok}).to_csv(OUT / "universe.csv", index=False)
pd.DataFrame(manifest, columns=["Ticker", "Status", "Rows", "Start", "End"]).to_csv(
    OUT / "download_manifest.csv", index=False)
print(f"\nDone: {len(ok)}/{len(UNIVERSE)} tickers OK. Zip the 'mga_data' folder and upload it.")
