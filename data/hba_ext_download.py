"""HBA universe extension downloader (single stocks from under-represented sectors).

Run this in your Codespace (the Claude workspace cannot reach Yahoo).
Needs:  pip install yfinance pandas

Downloads ~53 NEW large-cap stocks (none overlap the existing 33 HBA
tickers) into ./hba_ext_data/<TICKER>.csv - same Date,Open,High,Low,Close,
Volume format, split/dividend adjusted - then zips the folder itself to
hba_ext_data.zip.
"""
import shutil
from pathlib import Path

import pandas as pd
import yfinance as yf

UNIVERSE = [
    # Energy
    "COP", "SLB", "EOG", "OXY",
    # Utilities
    "NEE", "DUK", "SO", "D",
    # Consumer staples
    "KO", "PEP", "MDLZ", "CL", "PM", "MO",
    # Materials
    "LIN", "APD", "SHW", "FCX", "NEM",
    # Real estate
    "PLD", "AMT", "SPG",
    # Communication / media
    "DIS", "VZ", "T", "CMCSA",
    # Consumer discretionary
    "HD", "MCD", "NKE", "LOW", "SBUX", "TJX",
    # Health care
    "PFE", "TMO", "ABT", "AMGN", "ISRG",
    # Financials
    "WFC", "C", "BLK", "AXP", "SCHW",
    # Industrials / defense / transport
    "RTX", "LMT", "BA", "UNP", "MMM",
    # Technology
    "CRM", "QCOM", "TXN", "CSCO", "IBM", "AMD",
]

START = "2010-01-01"
OUT = Path("hba_ext_data")
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
shutil.make_archive("hba_ext_data", "zip", ".", "hba_ext_data")
print(f"\nDone: {len(ok)}/{len(UNIVERSE)} tickers OK. Zip written: hba_ext_data.zip")
