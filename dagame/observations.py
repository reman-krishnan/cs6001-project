"""Numerical verification of the proposal's Observations 1-3, on the lattice, against closed forms.

These are *checks*, not proofs. The proofs are in the report; these functions provide the independent numerical
evidence and the counter-example search harness for them.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom

from .model import ForfeitRule, Market, PayoffKind, Protocol, Routing, Scenario, Trade
from .numerics import close, gt
from .solver import solve, solve_all_selections


def _protocol(k, f, routing, fee, c=0.0, q=0.0, fallback=False, sigma_policy="required"):
    return Protocol(k, ForfeitRule(flat=f), routing, fee, c, q, fallback, sigma_policy)


def pi_call_form(market: Market, trade: Trade, k: int, fee: float, f: float) -> float:
    """Observation 2, call form:  -f + E[(g - (phi - f))^+], with g evaluated at block k-1.
    Valid for k >= 2, q = 0, no fallback, forfeit not routed back to the user, martingale prices."""
    n = k - 1
    j = np.arange(n + 1)
    S = market.S0 * np.exp(market.sigma * (2 * j - n))
    g = trade.g(S, market.S0)
    return float(-f + np.sum(binom.pmf(j, n, market.p) * np.maximum(g - (fee - f), 0.0)))


def pi_put_form(market: Market, trade: Trade, k: int, fee: float, f: float) -> float:
    """Observation 2, put form:  E[((phi - f) - g)^+] - phi.  Equals the call form iff E[g] = 0."""
    n = k - 1
    j = np.arange(n + 1)
    S = market.S0 * np.exp(market.sigma * (2 * j - n))
    g = trade.g(S, market.S0)
    return float(np.sum(binom.pmf(j, n, market.p) * np.maximum((fee - f) - g, 0.0)) - fee)


def check_observation_1(market: Market, trade: Trade, k: int, fee: float, f_values) -> list[dict]:
    """Self-routing: V_0 must not depend on f at all (the forfeit is a payment from the user to the user)."""
    rows, base = [], None
    for f in f_values:
        sol = solve(Scenario(market, trade, _protocol(k, f, Routing.to_self(), fee)))
        base = sol.value_user if base is None else base
        rows.append({"f": f, "v0": sol.value_user, "independent_of_f": bool(close(sol.value_user, base))})
    return rows


def check_observation_2(market: Market, trade: Trade, k: int, fee: float, f_values) -> list[dict]:
    """Solver V_0 must equal the call form and (for E[g]=0) the put form; deterrence must coincide with
    'call value <= f'. Requires k >= 2."""
    if k < 2:
        raise ValueError("Observation 2 needs a non-empty authorization window (k >= 2)")
    rows = []
    for f in f_values:
        v0 = solve(Scenario(market, trade, _protocol(k, f, Routing.burn(), fee))).value_user
        call, put = pi_call_form(market, trade, k, fee, f), pi_put_form(market, trade, k, fee, f)
        n = k - 1
        j = np.arange(n + 1)
        g = trade.g(market.S0 * np.exp(market.sigma * (2 * j - n)), market.S0)
        call_value = float(np.sum(binom.pmf(j, n, market.p) * np.maximum(g - (fee - f), 0.0)))
        rows.append({"f": f, "v0_solver": v0, "call_form": call, "put_form": put,
                     "match_call": bool(close(v0, call)), "match_put": bool(close(v0, put)),
                     "deterred_solver": not gt(v0, 0.0),
                     "deterred_formula": not gt(call_value, f)})
    return rows


def check_observation_3(market: Market, trade: Trade, k: int, fee: float, c: float, f_values) -> list[dict]:
    """Producer routing + fallback: the producer publishes iff fee - c > f (indifferent at equality).
    Reports, per f, what the solver's producer does under every tie-breaking selection."""
    rows = []
    for f in f_values:
        sc = Scenario(market, trade, _protocol(k, f, Routing.producer(), fee, c, fallback=True))
        sels = solve_all_selections(sc)
        pubs = {bool(s.publish is not None and s.publish.all()) for _, s in sels}
        predicted = "publish" if gt(fee - c, f) else ("withhold" if gt(f, fee - c) else "tie")
        rows.append({"f": f, "threshold": fee - c, "predicted": predicted,
                     "solver_publishes_in_selections": sorted(pubs),
                     "tie_flagged": any(s.ties.producer for _, s in sels),
                     "consistent": (pubs == {True} if predicted == "publish" else
                                    pubs == {False} if predicted == "withhold" else pubs == {True, False})})
    return rows
