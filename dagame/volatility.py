"""Volatility from public historical prices (CoinGecko / Yahoo Finance / Binance CSV exports) with a block-bootstrap interval.
Price files live in data/ (see data/README.md)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

_DATE = ("date", "timestamp", "snapped_at", "open time", "open_time", "time", "datetime")
_PRICE = ("close", "adj close", "adj_close", "price", "closing price")


def load_close(path: str) -> pd.Series:
    df = pd.read_csv(path)
    low = {c.lower().strip(): c for c in df.columns}
    dc = next((low[c] for c in _DATE if c in low), None)
    pc = next((low[c] for c in _PRICE if c in low), None)
    if dc is None or pc is None:
        raise ValueError(f"cannot find date/price columns in {list(df.columns)}")
    t = df[dc]
    t = pd.to_datetime(t, unit="ms" if float(t.iloc[0]) > 1e11 else "s", utc=True) if pd.api.types.is_numeric_dtype(t) else pd.to_datetime(t, utc=True)
    s = pd.Series(pd.to_numeric(df[pc], errors="coerce").to_numpy(float), index=t).dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if len(s) < 30 or (s <= 0).any():
        raise ValueError("need at least 30 strictly positive prices")
    return s


def daily_volatility(prices: pd.Series):
    """Daily-equivalent volatility sqrt(sum r^2 / sum dt), dt in days (valid for irregular spacing). Returns (sigma, n_returns).
    The block-bootstrap interval is ``daily_volatility_ci``."""
    r = np.diff(np.log(prices.to_numpy(float)))
    dt = np.diff(pd.DatetimeIndex(prices.index).as_unit("ns").asi8) / 1e9 / 86400.0
    return math.sqrt(float(np.sum(r ** 2) / np.sum(dt))), len(r)


def daily_volatility_ci(prices: pd.Series, n_boot: int = 400, level: float = 0.95, seed: int = 0):
    """Daily volatility with a moving-block bootstrap interval (blocks keep volatility clustering). Returns (sigma, low, high, n)."""
    r = np.diff(np.log(prices.to_numpy(float)))
    dt = np.diff(pd.DatetimeIndex(prices.index).as_unit("ns").asi8) / 1e9 / 86400.0
    n, b = len(r), max(1, round(len(r) ** (1 / 3)))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        i = (rng.integers(0, n - b + 1, size=math.ceil(n / b))[:, None] + np.arange(b)[None, :]).ravel()[:n]
        draws.append(math.sqrt(float(np.sum(r[i] ** 2) / np.sum(dt[i]))))
    lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])
    return math.sqrt(float(np.sum(r ** 2) / np.sum(dt))), float(lo), float(hi), n
