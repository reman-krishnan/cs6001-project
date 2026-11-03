"""Conversion of the extensive-form game to its STRATEGIC FORM ("conversion to normal form") and the comparison
of Nash equilibria with subgame-perfect equilibria (subgame perfection, non-credible threats).

A pure strategy is a complete plan: an action at EVERY decision node of the player, on or off the path. Every subgame-perfect
equilibrium is a Nash equilibrium of the strategic form, but not conversely: the extra Nash equilibria rest on threats that
would not be carried out if the threatened node were reached (non-credible threats). In this project that is exactly the
producer's "I will withhold" threat that a forfeit routed to the producer creates (Observation 3).
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from .numerics import gt
from .tree import Game, expected_payoffs, verify_spe


def to_normal_form(game: Game, max_profiles: int = 2 ** 17):
    """Returns (keys, strategy lists, U, P) where U[i, j] / P[i, j] are the user's / producer's expected payoffs when
    the user plays plan i and the producer plays plan j. Player 0 = user, 1 = producer."""
    keys = [game.keys(0), game.keys(1)]
    plans = [list(itertools.product(*[list(game.decisions[k].actions) for k in ks])) for ks in keys]
    n = len(plans[0]) * len(plans[1])
    if n > max_profiles:
        raise ValueError(f"{n} strategy profiles exceeds max_profiles={max_profiles}")
    U = np.zeros((len(plans[0]), len(plans[1])))
    P = np.zeros_like(U)
    for i, pi in enumerate(plans[0]):
        for j, pj in enumerate(plans[1]):
            prof = dict(zip(keys[0], pi))
            prof.update(zip(keys[1], pj))
            v = expected_payoffs(game, prof)
            U[i, j], P[i, j] = v[0], v[1]
    return keys, plans, U, P


def pure_nash(U: np.ndarray, P: np.ndarray) -> list[tuple[int, int]]:
    """All pure-strategy Nash equilibria of a bimatrix game (mutual best responses, within the shared tolerance)."""
    out = []
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            if not gt(U[:, j].max(), U[i, j]) and not gt(P[i, :].max(), P[i, j]):
                out.append((i, j))
    return out


def nash_vs_spe(game: Game, max_profiles: int = 2 ** 17) -> dict:
    """Classify every pure Nash equilibrium of the strategic form as subgame perfect or not.
    Returns dict(nash=[(payoffs, is_spe, profile)], n_spe, n_nash, non_credible=[...])."""
    keys, plans, U, P = to_normal_form(game, max_profiles)
    rows = []
    for i, j in pure_nash(U, P):
        prof = dict(zip(keys[0], plans[0][i]))
        prof.update(zip(keys[1], plans[1][j]))
        rows.append(((float(U[i, j]), float(P[i, j])), not verify_spe(game, prof), prof))
    return {"nash": rows, "n_nash": len(rows), "n_spe": sum(r[1] for r in rows),
            "non_credible": [r for r in rows if not r[1]], "shape": U.shape}
