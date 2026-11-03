"""Rotating producers: endogenous censoring by a strategic ENTITY with stake share alpha.

``strategic.py`` made ONE producer decide on every block (a rollup sequencer). On a chain with rotating block producers each block
is produced by the strategic entity only with probability alpha (its stake share, independent across blocks); otherwise the producer
is HONEST (follows the protocol: includes every authorization and publishes sigma at block k). When it is the producer the entity
chooses to include or censor, and at block k to publish or withhold. It collects the fee when it includes, and the forfeit share
s_P f when the commitment is dropped while it produces block k. Its continuation value therefore shrinks with the number of blocks
left, because honest producers can include the user's retry.

alpha = 1 reproduces ``strategic.solve_strategic``; alpha = 0 is the honest-producer game (exogenous q = 0).
The independent check is ``build_game_rotating`` solved by the generic tree engine. Requires q = 0, no contestation, sigma required.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from . import tree as T
from .model import Scenario
from .numerics import close, gt, lt
from .solver import USER_TIES, PRODUCER_TIES

INCLUDE_TIES = ("include", "censor")


def _check(sc: Scenario, alpha: float):
    pr = sc.protocol
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if pr.q != 0.0 or pr.beta_eff != 0.0 or pr.sigma_policy != "required":
        raise ValueError("rotating-producer model requires q = 0, no contestation and sigma_policy='required'")


@dataclass
class RotatingSolution:
    scenario: Scenario
    alpha: float
    value_user: float
    value_entity: float
    p_reveal: float          # user's authorization included (by an honest producer or by the entity)
    p_fallback: float        # sigma published at block k
    p_dropped: float


def _bw(sc, alpha, user_tie, entity_tie, settle_tie):
    k, p, fee, c = sc.protocol.k, sc.market.p, sc.protocol.fee, sc.protocol.publish_cost
    gk = sc.g_at(k)
    drop_u, drop_e = sc.drop_payoff()
    if sc.protocol.fallback:
        pub_e, wh_e = fee - c, drop_e
        ent_pub = True if gt(pub_e, wh_e) else (False if lt(pub_e, wh_e) else settle_tie == "publish")
        Uk = (1 - alpha) * (gk - fee) + alpha * (gk - fee if ent_pub else drop_u)
        Ek = np.full(k + 1, alpha * (pub_e if ent_pub else wh_e))
    else:
        Uk, Ek = np.full(k + 1, drop_u), np.full(k + 1, alpha * drop_e)
    Vu, Ve = {k: Uk}, {k: Ek}
    att_tab, inc_tab = {}, {}
    for t in range(k - 1, 0, -1):
        Cu = p * Vu[t + 1][1:] + (1 - p) * Vu[t + 1][:-1]
        Ce = p * Ve[t + 1][1:] + (1 - p) * Ve[t + 1][:-1]
        g = sc.g_at(t)
        inc = np.where(close(fee, Ce), entity_tie == "include", gt(fee, Ce))          # entity's inclusion rule when it produces
        Au = (1 - alpha) * (g - fee) + alpha * np.where(inc, g - fee, Cu)
        Ae = alpha * np.where(inc, fee, Ce)
        tie_u = close(Au, Cu)
        att = np.where(tie_u, user_tie == "attempt", gt(Au, Cu))
        Vu[t], Ve[t] = np.where(att, Au, Cu), np.where(att, Ae, Ce)
        att_tab[t], inc_tab[t] = att, inc
    V1u, V1e = Vu[1], Ve[1]
    return (float(p * V1u[1] + (1 - p) * V1u[0]), float(p * V1e[1] + (1 - p) * V1e[0]), att_tab, inc_tab, ent_pub if sc.protocol.fallback else None)


def _forward(sc, alpha, att_tab, inc_tab, ent_pub):
    """Outcome probabilities under the profile (independent of the DP)."""
    k, p = sc.protocol.k, sc.market.p
    mass, reveal = np.array([1.0]), 0.0
    for t in range(1, k + 1):
        nxt = np.zeros(t + 1)
        nxt[1:] += p * mass
        nxt[:-1] += (1 - p) * mass
        mass = nxt
        if t < k:
            hit = (1 - alpha) + alpha * inc_tab[t].astype(float)         # P(an attempt is included)
            done = mass * att_tab[t] * hit
            reveal += float(done.sum())
            mass = mass - done
    fb = 0.0
    if sc.protocol.fallback:
        fb = float(mass.sum()) * ((1 - alpha) + (alpha if ent_pub else 0.0))
    return reveal, fb, float(mass.sum()) - fb


def solve_rotating(sc: Scenario, alpha: float, *, user_tie="attempt", entity_tie="include", settle_tie="publish") -> RotatingSolution:
    _check(sc, alpha)
    if user_tie not in USER_TIES or entity_tie not in INCLUDE_TIES or settle_tie not in PRODUCER_TIES:
        raise ValueError("invalid tie-breaking rule")
    vu, ve, att, inc, ent_pub = _bw(sc, alpha, user_tie, entity_tie, settle_tie)
    rev, fb, dr = _forward(sc, alpha, att, inc, ent_pub)
    return RotatingSolution(sc, alpha, vu, ve, rev, fb, dr)


def solve_all_rotating(sc: Scenario, alpha: float):
    return [(dict(user_tie=u, entity_tie=e, settle_tie=s), solve_rotating(sc, alpha, user_tie=u, entity_tie=e, settle_tie=s))
            for u, e, s in itertools.product(USER_TIES, INCLUDE_TIES, PRODUCER_TIES)]


# ---------------------------------------------------------------- independent check: explicit game tree ---------------------
def build_game_rotating(sc: Scenario, alpha: float) -> T.Game:
    """Explicit extensive form with a chance node for the identity of each block's producer. Player 1 is the entity."""
    _check(sc, alpha)
    k, pr, p = sc.protocol.k, sc.protocol, sc.market.p
    if k > 14:
        raise ValueError("tree engine is exponential in k; refusing k > 14")
    S0, sigma, fee, c, h = sc.market.S0, sc.market.sigma, pr.fee, pr.publish_cost, int(pr.fallback)
    decisions = {}

    def reg(n):
        decisions[n.key] = n
        return n

    def gval(path):
        return float(sc.trade.g(np.array([S0 * math.exp(sigma * (path.count("u") - path.count("d")))]), S0)[0])

    bmemo, cmemo = {}, {}

    def cont(path):
        if path not in cmemo:
            cmemo[path] = T.Chance([(p, block(path + "u")), (1 - p, block(path + "d"))])
        return cmemo[path]

    def block(path):
        if path in bmemo:
            return bmemo[path]
        t, g = len(path), gval(path)
        drop_u, drop_e = sc.drop_payoff()
        if t == k:
            if h:
                ent = reg(T.Decision(1, ("publish", h, path), {"publish": T.Terminal((g - fee, fee - c)),
                                                               "withhold": T.Terminal((drop_u, drop_e))}))
                node = T.Chance([(alpha, ent), (1 - alpha, T.Terminal((g - fee, 0.0)))])
            else:
                node = T.Chance([(alpha, T.Terminal((drop_u, drop_e))), (1 - alpha, T.Terminal((drop_u, 0.0)))])
        else:
            nxt = cont(path)
            ent = reg(T.Decision(1, ("include", h, path), {"include": T.Terminal((g - fee, fee)), "censor": nxt}))
            att = T.Chance([(alpha, ent), (1 - alpha, T.Terminal((g - fee, 0.0)))])
            node = reg(T.Decision(0, ("reveal", h, path), {"attempt": att, "wait": nxt}))
        bmemo[path] = node
        return node

    return T.Game(cont(""), decisions, sc)


def extortion_threshold_scan(sc_builder, alpha: float, f_grid, tol: float = 1e-9):
    """Smallest forfeit on ``f_grid`` at which an honest (constant-value) user suffers an expected loss. ``sc_builder(f)`` returns the
    scenario. Worst case over the tie-breaking selections."""
    for f in f_grid:
        sc = sc_builder(float(f))
        ideal = sc.trade.value - sc.protocol.fee
        loss = max(ideal - s.value_user for _, s in solve_all_rotating(sc, alpha))
        if loss > tol * max(1.0, abs(ideal)):
            return float(f), float(loss)
    return math.inf, 0.0


def extortion_threshold(sc_builder, alpha: float, f_start: float, f_max: float, ratio: float = 1.4, iters: int = 40, tol: float = 1e-9) -> float:
    """Smallest forfeit at which an honest user suffers an expected loss (worst over tie-breaking selections): geometric search then
    bisection (the loss switches on once and stays on as f grows). inf if no loss up to f_max."""
    def loses(f):
        sc = sc_builder(float(f))
        ideal = sc.trade.value - sc.protocol.fee
        return max(ideal - x.value_user for _, x in solve_all_rotating(sc, alpha)) > tol * max(1.0, abs(ideal))
    prev, f = 0.0, f_start
    while f <= f_max:
        if loses(f):
            break
        prev, f = f, f * ratio
    else:
        return math.inf
    lo, hi = prev, f
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if loses(mid) else (mid, hi)
    return hi
