"""Fat tails and the square-root-of-time rule, from intraday prices (tested on synthetic prices with known truth and run on the one-minute ETH file in data/).

* ``empirical_fstar``: minimal deterring forfeit under the EMPIRICAL distribution of window returns (fat tails and skew included),
  versus the lognormal benchmark with the same window volatility (``lognormal_fstar``, from the game solver).
* ``variance_ratio``: Lo-MacKinlay variance ratios VR(m) = Var(m-step return) / (m Var(1-step return)). Under independent increments
  VR = 1, which is what the square-root-of-time scaling of daily volatility to 12-second blocks assumes. A variance ratio of one is a restriction on variance scaling and does not by itself establish independence of the increments.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .model import ForfeitRule, Market, Protocol, Routing, Trade
from .solver import minimal_deterring_forfeit


def window_returns(prices: pd.Series, window_seconds: float) -> np.ndarray:
    """Overlapping log returns over windows of exactly ``window_seconds`` (regularly spaced observations required)."""
    ns = pd.DatetimeIndex(prices.index).as_unit("ns").asi8
    dt = np.diff(ns) / 1e9
    step = float(np.median(dt))
    if not np.allclose(dt, step, rtol=1e-6):
        raise ValueError("window_returns needs regularly spaced observations")
    w = window_seconds / step
    if abs(w - round(w)) > 1e-9 or round(w) < 1:
        raise ValueError(f"window ({window_seconds}s) must be a positive multiple of the sampling step ({step}s)")
    w = int(round(w))
    lp = np.log(prices.to_numpy(float))
    if len(lp) <= w:
        raise ValueError("series shorter than one window")
    return lp[w:] - lp[:-w]


def empirical_fstar(R, notional: float, fee: float) -> float:
    """g = N (e^R / E[e^R] - 1) recentred to mean zero (no free drift); Pi(f) = E[max(g - fee, -f)] is monotone in f, so bisect on the sample."""
    R = np.asarray(R, float)
    Y = np.exp(R - R.max())
    g = notional * (Y / Y.mean() - 1.0)
    pi = lambda f: float(np.mean(np.maximum(g - fee, -f)))
    if pi(0.0) <= 0:
        return 0.0
    hi = max(fee + notional + abs(float(g.max())), float(-g.min()) + fee) + 1.0
    if pi(hi) > 1e-12:
        return math.inf
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if pi(mid) > 0 else (lo, mid)
        if hi - lo <= 1e-12 * max(1.0, hi):
            break
    return hi


def bootstrap_empirical_fstar(R, notional, fee, n_boot=300, block=20, level=0.95, seed=0):
    R = np.asarray(R, float)
    n, b = len(R), min(block, len(R))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = (rng.integers(0, n - b + 1, size=math.ceil(n / b))[:, None] + np.arange(b)[None, :]).ravel()[:n]
        vals.append(empirical_fstar(R[i], notional, fee))
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return empirical_fstar(R, notional, fee), float(lo), float(hi)


def lognormal_fstar(window_vol: float, notional: float, fee: float, steps: int = 200) -> float:
    """f* of the lattice game with the same window volatility (a fine lattice approximates the lognormal limit; 200 steps within ~1%)."""
    size = notional / 100.0
    return minimal_deterring_forfeit(Market(100.0, window_vol / math.sqrt(steps)), Trade(size=size),
                                     Protocol(steps + 1, ForfeitRule(), Routing.burn(), fee=fee))


def variance_ratio(prices: pd.Series, multiples=(2, 5, 10, 30, 60)) -> pd.DataFrame:
    """Variance ratios with two asymptotic z-statistics for H0: independent increments (VR = 1).

    ``z`` is the Lo-MacKinlay statistic that assumes homoskedastic increments. ``z_robust`` is their heteroskedasticity-robust version,
    z* = (VR - 1) / sqrt(theta), theta = sum_{j=1}^{m-1} [2 (m - j) / m]^2 delta_j and
    delta_j = sum_t (r_t - mu)^2 (r_{t-j} - mu)^2 / [sum_t (r_t - mu)^2]^2, which is valid when volatility changes over time.
    |z| > 1.96 rejects H0 at 5%."""
    lp = np.log(prices.to_numpy(float))
    r1 = np.diff(lp)
    n, mu = len(r1), float(np.mean(r1))
    var1 = float(np.mean((r1 - mu) ** 2))
    a2 = (r1 - mu) ** 2
    denom = float(a2.sum()) ** 2
    delta = np.array([0.0] + [float(np.dot(a2[j:], a2[:-j])) / denom for j in range(1, max(multiples))])
    rows = []
    for m in multiples:
        rm = lp[m:] - lp[:-m]
        vr = float(np.mean((rm - m * mu) ** 2) / (m * var1))
        se = math.sqrt(2.0 * (2 * m - 1) * (m - 1) / (3.0 * m * n))
        theta = float(sum((2.0 * (m - j) / m) ** 2 * delta[j] for j in range(1, m)))
        rows.append(dict(multiple=m, variance_ratio=vr, z=(vr - 1.0) / se, z_robust=(vr - 1.0) / math.sqrt(theta),
                         implied_exponent=0.5 * math.log(vr * m) / math.log(m) if vr > 0 else float("nan")))
    return pd.DataFrame(rows)
