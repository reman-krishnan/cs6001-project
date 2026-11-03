"""Pricing sanity check against Han, Lin and Yu (AFT 2019): the premium of an at-the-money American call, computed
with the Cox-Ross-Rubinstein lattice, n = 36 steps over T = 1 day, annualised volatility sigma_daily * sqrt(365).
Their abstract reports about 2% to 3% of the asset value for cryptocurrency pairs and about 0.3% for stocks and fiat currencies,
and the body of the paper reports about 1% to 2.3% for cryptocurrency pairs at a one-day horizon. Their per-pair volatilities come from data
not shipped here, so the method and the reported ranges are checked, not individual pairs."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .model import ForfeitRule, Market, Protocol, Routing, Scenario, Trade
from .solver import solve


def crr_american(sigma_block: float, n: int) -> float:
    """Textbook backward loop for an ATM American call (S0 = K = 1, r = 0), independent of the game solver."""
    u, d = math.exp(sigma_block), math.exp(-sigma_block)
    p = (1.0 - d) / (u - d)
    j = np.arange(n + 1)
    V = np.maximum(np.exp(sigma_block * (2 * j - n)) - 1.0, 0.0)
    for t in range(n - 1, -1, -1):
        jj = np.arange(t + 1)
        V = np.maximum(p * V[1:] + (1 - p) * V[:-1], np.exp(sigma_block * (2 * jj - t)) - 1.0)
    return float(V[0])


def han_premium_pct(sigma_daily: float, T_days: float = 1.0, n: int = 36) -> float:
    sigma_block = sigma_daily * math.sqrt(365.0) * math.sqrt((T_days / 365.0) / n)
    return 100.0 * crr_american(sigma_block, n)


def han_table(daily_vols=(0.0075, 0.02, 0.03, 0.04, 0.05, 0.058)) -> pd.DataFrame:
    """Han's method vs the game solver with no forfeit, no fee, no censoring (window = n steps, so k = n + 1)."""
    rows = []
    for sd in daily_vols:
        sc = Scenario(Market(100.0, sd / 6.0), Trade(size=1.0), Protocol(37, ForfeitRule(), Routing.burn()))
        ours = solve(sc).value_user
        h = han_premium_pct(sd)
        rows.append(dict(daily_vol=sd, han_method_pct=h, game_solver_pct=ours, abs_diff=abs(h - ours)))
    return pd.DataFrame(rows)
