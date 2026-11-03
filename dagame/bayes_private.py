"""Bayesian version with a PRIVATE user type (the proposal's theta_u ~ D, "such as a hidden exposure").

A user of type theta has a hidden exposure that makes executing the trade worth an extra theta * N whatever the price does: executing at
block t is worth g_t + theta N (implemented as a lower strike K = S0 (1 - theta)). theta = 0 is the pure speculator. The designer knows
only the prior D over types and commits to the forfeit f of a plain commit-reveal protocol (burn routing, no fallback) before types are
drawn (ex ante). Each type then best-responds (interim):
    V_BR(theta, f)  = its equilibrium value of committing (optimal stopping, solved by backward induction);
    V_hon(theta, f) = its value if it simply authorizes at the first opportunity at every block (the honest strategy);
    it COMMITS iff V_BR >= 0 (outside option 0);
    X = V_BR - V_hon >= 0 is its option-exploitation benefit: the gain from timing its authorization instead of authorizing at once.
        X is a chosen loss proxy. It mixes the counterparty's trading loss E[g_tau 1{exec}] with changes in the user's own fee,
        forfeit and execution probability, so it is not in general equal to the counterparty's loss or to a welfare loss.
Designer's expected loss (ex ante), all in fractions of the notional:
    L(f) = E_theta[ commit * (lam * X + nu * friction) + (1 - commit) * s(theta) ],
    friction = (theta - fee) - V_hon  (shortfall of honest play from sure execution: censoring and forfeits),
    s(theta) = max(theta - fee, 0)    (genuine surplus lost when a type is deterred from committing).
Weights lam (weight on the exploitation proxy) and nu (cost of friction) are parameters.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .model import ForfeitRule, Market, PayoffKind, Protocol, Routing, Scenario, Trade
from .numerics import lt
from .solver import evaluate, solve

NOTIONAL = 100.0
DEFAULT_PRIOR = {0.0: 0.35, 0.0002: 0.20, 0.0005: 0.15, 0.001: 0.12, 0.002: 0.08, 0.005: 0.06, 0.01: 0.04}


def _validate_prior(prior):
    if abs(sum(prior.values()) - 1.0) > 1e-9 or any(p < 0 for p in prior.values()) or any(not 0 <= t < 1 for t in prior):
        raise ValueError("prior must be a probability distribution over types theta in [0, 1)")


def type_scenario(theta, s_tot, k, f, fee, q):
    if not 0.0 <= theta < 1.0:
        raise ValueError("theta must lie in [0, 1)")
    mk = Market(100.0, s_tot / math.sqrt(k - 1))
    return Scenario(mk, Trade(PayoffKind.CALL, 1.0, NOTIONAL * (1.0 - theta)),
                    Protocol(k, ForfeitRule(flat=f * NOTIONAL), Routing.burn(), fee * NOTIONAL, 0.0, q, False))


def interim(theta, s_tot, k, f, fee, q) -> dict:
    sc = type_scenario(theta, s_tot, k, f, fee, q)
    v_br = solve(sc).value_user / NOTIONAL
    attempt = {t: np.ones(t + 1, dtype=bool) for t in range(1, k)}
    v_hon = evaluate(sc, attempt, None, False)[0] / NOTIONAL
    commit = not lt(v_br, 0.0)
    return dict(theta=theta, f=f, v_br=v_br, v_hon=v_hon, commit=bool(commit), extraction=max(v_br - v_hon, 0.0),
                friction=max((theta - fee) - v_hon, 0.0), surplus=max(theta - fee, 0.0))


def design_table(s_tot, k, fee, q, prior=None, f_grid=None, lam=1.0, nu=1.0):
    """Interim rows for every (type, forfeit) and the designer's ex-ante loss for every forfeit."""
    prior = DEFAULT_PRIOR if prior is None else prior
    _validate_prior(prior)
    f_grid = [0.0] if f_grid is None else list(f_grid)
    rows, agg = [], []
    for f in f_grid:
        loss = ex = fr = lost = part = 0.0
        for th, pr in prior.items():
            r = interim(th, s_tot, k, f, fee, q)
            rows.append({**r, "prob": pr})
            if r["commit"]:
                ex += pr * r["extraction"]
                fr += pr * r["friction"]
                part += pr
            else:
                lost += pr * r["surplus"]
        loss = lam * ex + nu * fr + lost
        agg.append(dict(f=f, exploitation=ex, friction=fr, lost_surplus=lost, loss=loss, participation=part))
    return pd.DataFrame(rows), pd.DataFrame(agg)


def optimal_forfeit(agg: pd.DataFrame) -> dict:
    i = int(agg.loss.idxmin())
    return dict(f_opt=float(agg.f[i]), loss_opt=float(agg.loss[i]), loss_at_zero=float(agg.loss[0]))


def eps_optimal_forfeit(agg: pd.DataFrame, eps: float = 0.01) -> dict:
    """Smallest forfeit on the grid whose loss is within ``eps`` of the best achievable loss reduction.

    The designer's loss is almost flat once moderately exposed types stop extracting value, so its exact minimiser lands wherever
    numerical noise puts it and keeps moving as the lattice is refined. The smallest forfeit that captures (1 - eps) of the
    reduction from f = 0 does not have that problem. ``agg.f`` must start at 0."""
    f = agg.f.to_numpy(float)
    loss = agg.loss.to_numpy(float)
    if f[0] != 0.0:
        raise ValueError("the grid must start at f = 0")
    best, base = float(loss.min()), float(loss[0])
    ok = loss <= best + eps * (base - best) + 1e-18
    i = int(np.argmax(ok))
    return dict(f_opt=float(f[i]), loss_opt=float(loss[i]), loss_at_zero=base, loss_min=best, index=i, eps=eps)


def refine_eps_forfeit(agg: pd.DataFrame, s_tot, k, fee, q, *, prior=None, eps: float = 0.01, lam: float = 1.0, nu: float = 1.0,
                       n_fine: int = 40) -> float:
    """The smallest forfeit within ``eps`` of the best loss reduction, located between the grid point that first qualifies and the one
    before it by a finer search, so the answer is not limited to the coarse grid spacing. The target loss is the one defined on the
    coarse grid (its minimum and its value at f = 0)."""
    o = eps_optimal_forfeit(agg, eps)
    i = o["index"]
    if i == 0:
        return o["f_opt"]
    lo, hi = float(agg.f.iloc[i - 1]), float(agg.f.iloc[i])
    target = o["loss_min"] + eps * (o["loss_at_zero"] - o["loss_min"]) + 1e-18
    fine = [float(x) for x in np.linspace(lo, hi, n_fine + 1)[1:]]
    _, a2 = design_table(s_tot, k, fee, q, prior=prior, f_grid=fine, lam=lam, nu=nu)
    ok = a2.loss.to_numpy() <= target
    return float(a2.f.to_numpy()[int(np.argmax(ok))]) if ok.any() else hi


def participation_threshold(rows: pd.DataFrame, f: float):
    """Smallest type that commits at forfeit f (the interim participation cut-off), or None."""
    d = rows[(rows.f == f) & rows.commit]
    return None if d.empty else float(d.theta.min())
