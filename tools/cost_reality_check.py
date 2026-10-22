"""Honest-cost check for the live FINAL_CONFIG (execution='overnight').

The deployed hba_live_signal.py's simulate() charges transaction cost only
on rebalance days (every 5 trading days), because it implicitly assumes
overnight-only exposure can be achieved for free. Actually achieving it
live requires flattening the position every single day (sell at open) and
reloading every evening (buy at/near close) -- a full round-trip EVERY
day the position is held, not just every 5 days.

This script runs the SAME signal/config two ways:
  (a) "backtest-as-coded" cost model: turnover charged only on rebalance days.
  (b) "live-realistic" cost model: turnover charged as a full daily
      round-trip (enter evening + exit next morning) on every day a
      nonzero position is held.
Everything else (features, ranking, weighting, regime gate, dd guard) is
identical and copied verbatim from hba_live_signal.py.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "hba33"
ANNUALIZATION = 252

UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO", "ORCL", "ADBE",
    "JPM", "BAC", "GS", "MS", "V", "MA",
    "UNH", "JNJ", "LLY", "MRK", "ABBV",
    "CAT", "HON", "GE", "UPS", "DE",
    "COST", "WMT", "PG", "XOM", "CVX",
    "TSLA", "INTC",
]


@dataclass(frozen=True)
class Config:
    top_n: int = 2
    rebalance_every: int = 5
    weighting: str = "score_weighted"
    min_momentum: float = 0.0
    require_positive_mom63: bool = True
    regime_mode: str = "binary"
    min_breadth: float = 0.6
    target_vol: float | None = None
    max_exposure: float = 1.0
    cost_bps: float = 5.0
    execution: str = "overnight"


FINAL_CONFIG = Config()


def load_prices():
    closes, opens = {}, {}
    for t in UNIVERSE:
        df = pd.read_csv(DATA_DIR / f"{t}.csv", parse_dates=["Date"]).set_index("Date").sort_index()
        closes[t] = df["Close"]
        opens[t] = df["Open"]
    prices = pd.concat(closes, axis=1).sort_index()
    open_prices = pd.concat(opens, axis=1).sort_index().reindex(index=prices.index)[prices.columns]
    return prices, open_prices


def rank01(x):
    return x.rank(axis=1, pct=True)


def make_features(prices, overnight_ret):
    ret = prices.pct_change()
    mom21 = prices / prices.shift(21) - 1
    mom63 = prices / prices.shift(63) - 1
    mom126 = prices / prices.shift(126) - 1
    vol63 = ret.rolling(63, min_periods=40).std() * np.sqrt(ANNUALIZATION)
    risk_mom = mom63 / vol63.replace(0, np.nan)
    trend100 = prices > prices.rolling(100, min_periods=60).mean()
    trend200 = prices > prices.rolling(200, min_periods=120).mean()
    score = (
        0.35 * (rank01(mom126) - 0.5)
        + 0.25 * (rank01(mom63) - 0.5)
        + 0.15 * (rank01(mom21) - 0.5)
        + 0.15 * (rank01(risk_mom) - 0.5)
        + 0.10 * (rank01(trend100.astype(float)) - 0.5)
    )
    breadth = trend200.mean(axis=1, skipna=True)
    return {
        "ret": ret, "mom21": mom21, "mom63": mom63, "mom126": mom126,
        "vol63": vol63, "score": score, "trend200": trend200, "breadth": breadth,
        "overnight_ret": overnight_ret.reindex(index=prices.index, columns=prices.columns),
    }


def apply_dd_guard(net, threshold, floor):
    if threshold is None or net.empty:
        return net
    eq = (1 + net.fillna(0.0)).cumprod()
    dd = eq / eq.cummax() - 1
    dd_known_yesterday = dd.shift(1).fillna(0.0)
    scale = np.where(dd_known_yesterday <= -abs(threshold), floor, 1.0)
    return net * scale


def build_weights(prices, f, cfg):
    dates = prices.index
    score = f["score"]; mom126 = f["mom126"]; mom63 = f["mom63"]; breadth = f["breadth"]
    tickers = prices.columns.tolist()
    reb_idx = np.arange(0, len(dates), cfg.rebalance_every, dtype=int)
    W = np.zeros((len(dates), len(tickers)), dtype=float)
    current = np.zeros(len(tickers), dtype=float)
    for i in reb_idx:
        date = dates[i]
        s = score.iloc[i].dropna()
        if len(s) < max(4, cfg.top_n):
            W[i, :] = current
            continue
        candidates = s.sort_values(ascending=False).index[: cfg.top_n]
        ok = mom126.loc[date, candidates] > cfg.min_momentum
        if cfg.require_positive_mom63:
            ok &= mom63.loc[date, candidates] > 0
        candidates = candidates[ok.fillna(False)]
        b = breadth.iloc[i]
        if cfg.regime_mode == "binary":
            if pd.isna(b) or b < cfg.min_breadth:
                candidates = []
        new = np.zeros(len(tickers), dtype=float)
        if len(candidates):
            if cfg.weighting == "score_weighted":
                sc = (s.loc[candidates] + 0.5).clip(lower=0.01)
                w = sc / sc.sum()
                for t, wt in w.items():
                    new[tickers.index(t)] = float(wt)
        current = new
        W[i, :] = current
    W_full = pd.DataFrame(W, index=dates, columns=tickers)
    snapshots = W_full.iloc[reb_idx]
    W = snapshots.reindex(dates).ffill().fillna(0.0)
    return W


def simulate_as_coded(prices, f, cfg):
    """Backtest-as-deployed: turnover (and its cost) charged only on
    rebalance days, as in hba_live_signal.py's simulate()."""
    dates = prices.index
    ret = f["overnight_ret"]
    W = build_weights(prices, f, cfg)
    reb_idx = np.arange(0, len(dates), cfg.rebalance_every, dtype=int)
    prev = np.zeros(W.shape[1])
    turnover = np.zeros(len(dates))
    for i in reb_idx:
        cur = W.iloc[i].values
        turnover[i] = np.abs(cur - prev).sum()
        prev = cur
    gross = (W.shift(1).fillna(0.0) * ret).sum(axis=1)
    costs = pd.Series(turnover, index=dates) * (cfg.cost_bps / 10000.0)
    return gross - costs, W


def simulate_live_realistic(prices, f, cfg):
    """Live-realistic: achieving overnight-only exposure requires a full
    buy+sell round trip EVERY day a nonzero position is held (enter near
    that day's close, exit at next open) -- not just on rebalance days.
    Daily turnover = 2 * that day's held weight (full entry + full exit)."""
    dates = prices.index
    ret = f["overnight_ret"]
    W = build_weights(prices, f, cfg)
    daily_turnover = 2.0 * W.abs().sum(axis=1)
    gross = (W.shift(1).fillna(0.0) * ret).sum(axis=1)
    costs = daily_turnover * (cfg.cost_bps / 10000.0)
    return gross - costs, W


def sharpe(net):
    net = net.dropna()
    if net.std() == 0 or len(net) < 20:
        return float("nan")
    return float(net.mean() / net.std() * np.sqrt(ANNUALIZATION))


def total_return(net):
    return float((1 + net.fillna(0.0)).prod() - 1)


def main():
    prices, open_prices = load_prices()
    overnight_ret = open_prices / prices.shift(1) - 1
    f = make_features(prices, overnight_ret)

    net_coded, W = simulate_as_coded(prices, f, FINAL_CONFIG)
    net_live, _ = simulate_live_realistic(prices, f, FINAL_CONFIG)

    net_coded_g = apply_dd_guard(net_coded, 0.08, 0.0)
    net_live_g = apply_dd_guard(net_live, 0.08, 0.0)

    last_year_cut = prices.index[-1] - pd.Timedelta(days=365)

    print("FULL HISTORY (with dd guard, as validated)")
    print(f"  as-coded (5-day cost):   Sharpe {sharpe(net_coded_g):.3f}  total_ret {total_return(net_coded_g)*100:.1f}%")
    print(f"  live-realistic (daily):  Sharpe {sharpe(net_live_g):.3f}  total_ret {total_return(net_live_g)*100:.1f}%")
    print()
    print("LAST 12 MONTHS (with dd guard)")
    nc1 = net_coded_g[net_coded_g.index >= last_year_cut]
    nl1 = net_live_g[net_live_g.index >= last_year_cut]
    print(f"  as-coded (5-day cost):   Sharpe {sharpe(nc1):.3f}  total_ret {total_return(nc1)*100:.1f}%")
    print(f"  live-realistic (daily):  Sharpe {sharpe(nl1):.3f}  total_ret {total_return(nl1)*100:.1f}%")
    print()
    avg_days_held = W[W.sum(axis=1) > 0].shape[0]
    print(f"Days with nonzero position: {avg_days_held} / {len(W)}")
    print(f"Avg daily cost drag, as-coded:  {(net_coded - net_coded).mean():.6f} (reference)")
    coded_cost_total = float((prices.index.to_series().notna()).sum())  # placeholder, real numbers below
    print()
    # explicit cost totals
    reb_idx = np.arange(0, len(prices.index), FINAL_CONFIG.rebalance_every, dtype=int)
    prev = np.zeros(W.shape[1])
    tcoded = 0.0
    for i in reb_idx:
        cur = W.iloc[i].values
        tcoded += np.abs(cur - prev).sum()
        prev = cur
    tlive = float((2.0 * W.abs().sum(axis=1)).sum())
    print(f"Total turnover units, as-coded:  {tcoded:.1f}  -> total cost {tcoded*FINAL_CONFIG.cost_bps/10000*100:.2f}% of equity over the period")
    print(f"Total turnover units, live-realistic: {tlive:.1f}  -> total cost {tlive*FINAL_CONFIG.cost_bps/10000*100:.2f}% of equity over the period")


if __name__ == "__main__":
    main()
