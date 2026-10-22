"""HBA v7: pushing on the remaining gap to 1.5 Sharpe, honest costs throughout.

Fixed from v6's finding: execution='full_day' always wins once overnight
is costed honestly, so this grid only tests full_day (frees up budget for
other axes). New axes swept that v6 held fixed:
  - rebalance_every: [3, 5, 10, 21]  (cadence)
  - min_breadth: [0.4, 0.5, 0.6]     (regime threshold, when regime != off)
  - regime_mode: [off, binary, continuous]
  - target_vol: [None, 0.10, 0.15]
  - top_n: [2, 3]
  - weighting: [inverse_vol, score_weighted]  (equal dropped - never won)

Universe size is NOT swept here - still the real 33-ticker universe from
before. Widening it further needs fresh data pulled, flagged as a
separate follow-up, not run in this pass.

Same simulate-once-per-config caching approach as v6 for speed.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "hba33"
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "sweep"
ANNUALIZATION = 252


@dataclass(frozen=True)
class Config:
    top_n: int
    rebalance_every: int
    weighting: str
    min_momentum: float
    require_positive_mom63: bool
    regime_mode: str
    min_breadth: float
    target_vol: float | None
    max_exposure: float
    cost_bps: float
    execution: str = "full_day"


def load_universe() -> List[str]:
    u = pd.read_csv(DATA_DIR / "universe.csv")
    return [x for x in u["Ticker"].astype(str).str.upper().tolist() if x]


def load_prices(universe: Iterable[str]):
    frames = {}
    missing = []
    for ticker in universe:
        p = DATA_DIR / f"{ticker}.csv"
        if not p.exists():
            missing.append(ticker)
            continue
        df = pd.read_csv(p)
        dt = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_localize(None)
        close = pd.to_numeric(df["Close"], errors="coerce")
        s = pd.Series(close.values, index=dt, name=ticker).dropna()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        frames[ticker] = s
    prices = pd.concat(frames, axis=1).sort_index()
    return prices, list(frames), missing


def load_open_prices(loaded_tickers: List[str]) -> pd.DataFrame:
    frames = {}
    for ticker in loaded_tickers:
        p = DATA_DIR / f"{ticker}.csv"
        d = pd.read_csv(p)
        idx = pd.to_datetime(d["Date"], utc=True, errors="coerce").dt.tz_localize(None)
        frames[ticker] = pd.Series(pd.to_numeric(d["Open"], errors="coerce").values, index=idx).sort_index()
    return pd.concat(frames, axis=1).sort_index()


def rank01(x: pd.DataFrame) -> pd.DataFrame:
    return x.rank(axis=1, pct=True)


def make_features(prices: pd.DataFrame, overnight_ret: pd.DataFrame) -> Dict:
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


def metrics(returns: pd.Series) -> Dict:
    r = returns.dropna()
    if len(r) < 2:
        return {"sharpe": np.nan, "cagr": np.nan, "vol": np.nan, "max_dd": np.nan, "days": int(len(r))}
    eq = (1 + r).cumprod()
    total = eq.iloc[-1] - 1
    years = len(r) / ANNUALIZATION
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and 1 + total > 0 else np.nan
    vol = r.std(ddof=1) * np.sqrt(ANNUALIZATION)
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(ANNUALIZATION) if r.std(ddof=1) else np.nan
    dd = eq / eq.cummax() - 1
    return {
        "start": r.index.min().date().isoformat(), "end": r.index.max().date().isoformat(),
        "days": int(len(r)), "total_return": float(total), "cagr": float(cagr),
        "vol": float(vol), "sharpe": float(sharpe), "max_dd": float(dd.min()),
    }


def simulate_full(prices: pd.DataFrame, f: Dict, cfg: Config):
    dates = prices.index
    ret = f["overnight_ret"] if (cfg.execution == "overnight" and "overnight_ret" in f) else f["ret"]
    score = f["score"]; mom126 = f["mom126"]; mom63 = f["mom63"]; vol63 = f["vol63"]; breadth = f["breadth"]
    tickers = prices.columns.tolist()
    tidx = {t: i for i, t in enumerate(tickers)}

    reb_idx = np.arange(0, len(dates), cfg.rebalance_every, dtype=int)
    W = np.zeros((len(dates), len(tickers)), dtype=float)
    turnover = np.zeros(len(dates), dtype=float)
    current = np.zeros(len(tickers), dtype=float)

    for i in reb_idx:
        date = dates[i]
        s = score.iloc[i].dropna()
        if len(s) < max(4, cfg.top_n):
            continue
        candidates = s.sort_values(ascending=False).index[:cfg.top_n]
        ok = mom126.loc[date, candidates] > cfg.min_momentum
        if cfg.require_positive_mom63:
            ok &= mom63.loc[date, candidates] > 0
        candidates = candidates[ok.fillna(False)]

        regime_scale = 1.0
        b = breadth.iloc[i]
        if cfg.regime_mode == "binary":
            if pd.isna(b) or b < cfg.min_breadth:
                candidates = []
        elif cfg.regime_mode == "continuous":
            regime_scale = 0.6 if pd.isna(b) else float(np.clip(b / cfg.min_breadth, 0.3, 1.0))

        new = np.zeros(len(tickers), dtype=float)
        if len(candidates):
            idx = [tidx[t] for t in candidates]
            if cfg.weighting == "equal":
                new[idx] = 1.0 / len(idx)
            elif cfg.weighting == "inverse_vol":
                v = vol63.loc[date, candidates].replace(0, np.nan)
                inv = (1.0 / v).replace([np.inf, -np.inf], np.nan).dropna()
                if len(inv):
                    for t, wt in (inv / inv.sum()).items():
                        new[tidx[t]] = float(wt)
            elif cfg.weighting == "score_weighted":
                sc = (s.loc[candidates] + 0.5).clip(lower=0.01)
                w = sc / sc.sum()
                for t, wt in w.items():
                    new[tidx[t]] = float(wt)
            new = new * regime_scale
        turnover[i] = np.abs(new - current).sum()
        current = new
        W[i, :] = current

    W_full = pd.DataFrame(W, index=dates, columns=tickers)
    W = W_full.iloc[reb_idx].reindex(dates).ffill().fillna(0.0)
    gross = (W.shift(1).fillna(0.0) * ret).sum(axis=1)

    if cfg.execution == "overnight":
        costs = (2.0 * W.abs().sum(axis=1)) * (cfg.cost_bps / 10000.0)
    else:
        costs = pd.Series(turnover, index=dates) * (cfg.cost_bps / 10000.0)

    net = gross - costs
    if cfg.target_vol is not None:
        rv = net.rolling(63, min_periods=40).std() * np.sqrt(ANNUALIZATION)
        scale = (cfg.target_vol / rv.replace(0, np.nan)).clip(lower=0, upper=cfg.max_exposure)
        net = net * scale.shift(1).fillna(0)

    return net


def slice_net(net, start=None, end=None):
    r = net
    if start is not None:
        r = r.loc[r.index >= pd.Timestamp(start)]
    if end is not None:
        r = r.loc[r.index <= pd.Timestamp(end)]
    return r.dropna()


def parameter_grid(cost_bps: float) -> List[Config]:
    out = []
    seen = set()
    for top_n in [2, 3]:
        for weighting in ["inverse_vol", "score_weighted"]:
            for rebalance_every in [3, 5, 10, 21]:
                for target in [None, 0.10, 0.15]:
                    for regime_mode in ["off", "binary", "continuous"]:
                        breadths = [0.5] if regime_mode == "off" else [0.4, 0.5, 0.6]
                        for min_breadth in breadths:
                            cfg = Config(top_n, rebalance_every, weighting, 0.0, True,
                                         regime_mode, min_breadth, target, 1.0, cost_bps, "full_day")
                            if cfg in seen:
                                continue
                            seen.add(cfg)
                            out.append(cfg)
    return out


def optimize_is(cache: Dict[Config, pd.Series], start, end) -> Tuple[Config, Dict]:
    rows = []
    for cfg, net_full in cache.items():
        r = slice_net(net_full, start, end)
        m = metrics(r)
        score = m["sharpe"] if np.isfinite(m["sharpe"]) else -999
        if np.isfinite(m["max_dd"]):
            score -= 0.35 * max(0, (-m["max_dd"] - 0.20) / 0.10)
        if m["days"] < 300:
            score -= 1.0
        rows.append((score, cfg, m))
    rows.sort(key=lambda x: x[0], reverse=True)
    best_score = rows[0][0]
    plateau = [x for x in rows if x[0] >= best_score - 0.02 * abs(best_score)]
    chosen = sorted(plateau, key=lambda x: (x[1].top_n, x[1].rebalance_every, x[1].target_vol is not None))[0]
    return chosen[1], {"objective": chosen[0], "metrics": chosen[2]}


def walk_forward(prices, features, cost_bps=5.0, is_days=756, oos_days=126, step_days=126, holdout_days=252):
    dates = prices.index
    configs = parameter_grid(cost_bps)
    print(f"Grid size: {len(configs)} configs", flush=True)

    t0 = time.time()
    cache: Dict[Config, pd.Series] = {}
    for n, cfg in enumerate(configs, 1):
        cache[cfg] = simulate_full(prices, features, cfg)
        if n % 100 == 0:
            print(f"  precomputed {n}/{len(configs)} ({time.time()-t0:.1f}s)", flush=True)
    print(f"Precomputed {len(configs)} full-history simulations in {time.time()-t0:.1f}s", flush=True)

    wfo_end_i = len(dates) - holdout_days
    oos_segments = []
    fold_rows = []
    i = is_days
    max_folds = int(os.environ.get("HBA_MAX_FOLDS", "1000000"))
    tfold = time.time()
    while i + oos_days <= wfo_end_i and len(fold_rows) < max_folds:
        is_start, is_end = dates[i - is_days], dates[i - 1]
        oos_start, oos_end = dates[i], dates[i + oos_days - 1]
        cfg, opt = optimize_is(cache, is_start, is_end)
        oos_ret = slice_net(cache[cfg], oos_start, oos_end)
        is_ret = slice_net(cache[cfg], is_start, is_end)
        im = metrics(is_ret); om = metrics(oos_ret)
        fold = {
            "fold": len(fold_rows) + 1,
            "is_start": is_start.date().isoformat(), "is_end": is_end.date().isoformat(),
            "oos_start": oos_start.date().isoformat(), "oos_end": oos_end.date().isoformat(),
            "is_sharpe": im["sharpe"], "oos_sharpe": om["sharpe"],
            "is_cagr": im["cagr"], "oos_cagr": om["cagr"],
            "oos_max_dd": om["max_dd"], "oos_vol": om["vol"],
            "wfe_sharpe": (om["sharpe"] / im["sharpe"]) if im["sharpe"] and np.isfinite(im["sharpe"]) else np.nan,
            **asdict(cfg),
        }
        fold_rows.append(fold)
        oos_segments.append(oos_ret)
        i += step_days
    print(f"All {len(fold_rows)} folds evaluated in {time.time()-tfold:.1f}s", flush=True)

    stitched = pd.concat(oos_segments).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    holdout_start = dates[wfo_end_i]
    final_cfg = Config(**{k: fold_rows[-1][k] for k in Config.__dataclass_fields__.keys()})
    hold_ret = slice_net(cache[final_cfg], holdout_start, dates[-1])
    return stitched, pd.DataFrame(fold_rows), hold_ret, final_cfg


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    universe = load_universe()
    prices, loaded, missing = load_prices(universe)
    prices = prices.loc[prices.index >= "2010-01-01"]
    open_prices = load_open_prices(loaded).reindex(index=prices.index)[prices.columns]
    overnight_ret = open_prices / prices.shift(1) - 1
    features = make_features(prices, overnight_ret=overnight_ret)
    print(f"Loaded {len(loaded)} tickers, {len(prices)} days", flush=True)

    t0 = time.time()
    stitched, folds, holdout, final_cfg = walk_forward(prices, features)
    print(f"Total time: {time.time()-t0:.1f}s", flush=True)

    overall = metrics(stitched)
    hold = metrics(holdout)
    wfe = np.nanmean(folds["wfe_sharpe"].replace([np.inf, -np.inf], np.nan))
    median_oos = float(folds["oos_sharpe"].median())
    pct_positive = float((folds["oos_sharpe"] > 0).mean())

    result = {
        "objective": "push on the gap to Sharpe 1.5, honest costs, cadence/regime/breadth/target_vol swept",
        "loaded_count": len(loaded), "missing": missing,
        "grid_size": len(parameter_grid(5.0)),
        "stitched_oos": overall,
        "median_fold_oos_sharpe": median_oos,
        "positive_oos_fold_pct": pct_positive * 100,
        "mean_sharpe_wfe": float(wfe),
        "final_holdout": hold,
        "final_selected_config": asdict(final_cfg),
        "pass_full_oos_sharpe_gt_1_5": bool(overall["sharpe"] > 1.5),
    }
    folds.to_csv(OUT_DIR / "walk_forward_folds.csv", index=False)
    with open(OUT_DIR / "v7_summary.json", "w") as fh:
        json.dump(result, fh, indent=2, default=str)

    print(json.dumps(result, indent=2, default=str))
    print("\nWFO folds:")
    print(folds[["fold", "is_sharpe", "oos_sharpe", "oos_max_dd", "top_n", "rebalance_every", "weighting", "regime_mode", "min_breadth", "target_vol"]].to_string(index=False))

    print("\nConfig selection frequency across folds:")
    cols = ["top_n", "rebalance_every", "weighting", "regime_mode", "min_breadth", "target_vol"]
    print(folds[cols].value_counts().to_string())


if __name__ == "__main__":
    main()
