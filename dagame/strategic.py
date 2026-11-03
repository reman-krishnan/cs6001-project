"""Strategic censoring: a single producer (sequencer) decides whether to include each reveal attempt.

This removes the main limitation of the base model (exogenous i.i.d. censoring). At block t the user sees S_t and may
attempt to reveal; the producer sees the attempt and either INCLUDES it (executes at S_t; fee to the producer) or
CENSORS it (the game continues exactly as if the user had waited). At block k it decides publish / withhold as before.

Backward induction on the lattice tracks BOTH players' continuation values (C_u, C_p). The producer includes iff
fee > C_p: censoring is only worth it if the producer expects to earn more later, e.g. by collecting a forfeit
(EXTORTION), which needs its forfeit share times f to exceed the fee it forgoes.

Requires q = 0 and no contestation (the extension is about censoring being a choice, not a coin flip). The
independent check is ``tree.build_game(sc, strategic=True)``.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .model import Scenario
from .numerics import close, gt
from .solver import Ties, _settlement, _validate_ties, PRODUCER_TIES, SIGMA_TIES, USER_TIES

INCLUDE_TIES = ("include", "censor")


@dataclass
class StrategicSolution:
    scenario: Scenario
    gave_sigma: bool
    value_user: float
    value_producer: float
    attempt: dict            # t -> bool array: user's (effective) attempt
    include: dict            # t -> bool array: producer includes an attempt
    publish: np.ndarray | None
    p_reveal: float          # probability the user's reveal is included
    p_fallback: float
    p_dropped: float
    ties: Ties = field(default_factory=Ties)
    include_tie: bool = False


def _bw(sc: Scenario, h: bool, user_tie, producer_tie, include_tie):
    k, p, fee, c = sc.protocol.k, sc.market.p, sc.protocol.fee, sc.protocol.publish_cost
    Uk, publish, ptie = _settlement(sc, h, producer_tie)
    drop_u, drop_p = sc.drop_payoff()
    wh_u, wh_p = sc.withhold_payoff(sc.g_at(k))
    Pk = np.where(publish, fee - c, wh_p) if publish is not None else np.full(k + 1, drop_p)
    Vu, Vp = {k: Uk}, {k: Pk}
    attempt, include, tie_nodes, inc_tie = {}, {}, [], False
    for t in range(k - 1, 0, -1):
        Cu = p * Vu[t + 1][1:] + (1 - p) * Vu[t + 1][:-1]
        Cp = p * Vp[t + 1][1:] + (1 - p) * Vp[t + 1][:-1]
        tie_i = close(fee, Cp)
        inc = np.where(tie_i, include_tie == "include", gt(fee, Cp))
        A = np.where(inc, sc.g_at(t) - fee, Cu)                 # user's value of attempting, anticipating the reply
        tie_u = inc & close(A, Cu)                               # a futile attempt (censored) is not a real choice
        att = np.where(inc, np.where(tie_u, user_tie == "attempt", gt(A, Cu)), False)
        go = att & inc
        Vu[t] = np.where(go, A, Cu)
        Vp[t] = np.where(go, fee, Cp)
        attempt[t], include[t] = att, inc
        tie_nodes += [(t, int(j)) for j in np.flatnonzero(tie_u)]
        inc_tie = inc_tie or bool(np.any(tie_i & att))
    V1u, V1p = Vu[1], Vp[1]
    return (float(p * V1u[1] + (1 - p) * V1u[0]), float(p * V1p[1] + (1 - p) * V1p[0]), attempt, include, publish,
            tie_nodes, ptie, inc_tie)


def _forward(sc, attempt, include, publish, h):
    """Outcome probabilities under the profile (independent of the DP)."""
    k, p = sc.protocol.k, sc.market.p
    mass, reveal = np.array([1.0]), 0.0
    for t in range(1, k + 1):
        nxt = np.zeros(t + 1)
        nxt[1:] += p * mass
        nxt[:-1] += (1 - p) * mass
        mass = nxt
        if t < k:
            done = mass * (attempt[t] & include[t])
            reveal += float(done.sum())
            mass = mass - done
    fb = float((mass * publish).sum()) if (h and publish is not None) else 0.0
    return reveal, fb, float(mass.sum()) - fb


def solve_strategic(sc: Scenario, *, user_tie="attempt", producer_tie="publish", sigma_tie="give",
                    include_tie="include") -> StrategicSolution:
    _validate_ties(user_tie, producer_tie, sigma_tie)
    if include_tie not in INCLUDE_TIES:
        raise ValueError(f"include_tie must be one of {INCLUDE_TIES}")
    pr = sc.protocol
    if pr.q != 0.0 or pr.beta_eff != 0.0:
        raise ValueError("strategic censoring requires q = 0 and no contestation")
    ties = Ties()
    if pr.fallback and pr.sigma_policy == "optional":
        outs = {h: _bw(sc, h, user_tie, producer_tie, include_tie) for h in (True, False)}
        if close(outs[True][0], outs[False][0]):
            ties.sigma, h = True, sigma_tie == "give"
        else:
            h = bool(gt(outs[True][0], outs[False][0]))
    else:
        h = bool(pr.fallback)
    vu, vp, attempt, include, publish, tie_nodes, ptie, itie = _bw(sc, h, user_tie, producer_tie, include_tie)
    ties.user_nodes, ties.producer = tie_nodes, ptie
    rev, fb, dr = _forward(sc, attempt, include, publish, h)
    return StrategicSolution(sc, h, vu, vp, attempt, include, publish, rev, fb, dr, ties, itie)


def solve_all_strategic(sc: Scenario) -> list:
    out = []
    for ut, pt, st, it in itertools.product(USER_TIES, PRODUCER_TIES, SIGMA_TIES, INCLUDE_TIES):
        out.append((dict(user_tie=ut, producer_tie=pt, sigma_tie=st, include_tie=it),
                    solve_strategic(sc, user_tie=ut, producer_tie=pt, sigma_tie=st, include_tie=it)))
    return out


def extortion_threshold(sc: Scenario) -> float:
    """Forfeit above which a producer with routing share s_P profits from censoring every attempt and withholding:
    s_P * f > fee  (it forgoes the fee to collect the forfeit). inf if s_P = 0."""
    s_p = sc.protocol.routing.to_producer
    return float("inf") if s_p == 0 else sc.protocol.fee / s_p
