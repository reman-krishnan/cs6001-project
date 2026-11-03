"""An independent check of the PAYOFF MODEL, not only of the equilibrium computation.

``solver.py`` and ``tree.py`` both take payoffs from ``Scenario``. This file re-implements the game from the written rules alone
(authorization pays (g - fee, fee), publication pays (g - fee, fee - c), a drop pays (-(1 - r_U) f, r_P f), each attempt is included
with probability 1 - q and a censored attempt may be retried), by explicit enumeration of price paths and inclusion events, and
using none of the package's payoff code. It then checks
  1. the solver's reported equilibrium payoffs against this evaluator, for the solver's own policy, and
  2. that no policy of the user (all Markov policies, small k) and no policy of the producer beats the solver's value.
"""
import itertools
import math
import random

import numpy as np
import pytest

from dagame import ForfeitRule, Market, Protocol, Routing, Scenario, Trade, solve

ROUTINGS = [Routing.burn(), Routing.to_self(), Routing.producer(), Routing.pool(3)]


def evaluate(S0, sigma, k, size, fee, c, q, f, r_u, r_p, user_attempt, producer_publish, fallback):
    """Expected (user, producer) payoff by enumerating every price path and every inclusion event.
    user_attempt[(t, j)] and producer_publish[j] are the policies (j = number of up moves so far)."""
    u, d = math.exp(sigma), math.exp(-sigma)
    p = (1 - d) / (u - d)
    total = np.zeros(2)

    def g(t, j):
        return size * (S0 * math.exp(sigma * (2 * j - t)) - S0)

    def walk(t, j, prob):                                  # state: block t reached, j up moves, commitment still unauthorized
        nonlocal total
        if t == k:                                          # settlement
            if fallback and producer_publish[j]:
                total += prob * np.array([g(k, j) - fee, fee - c])
            else:
                total += prob * np.array([-(1 - r_u) * f, r_p * f])
            return
        # blocks 1 .. k-1: the user may attempt; an attempt is censored with probability q
        for up, pr in ((1, p), (0, 1 - p)):
            j2, t2 = j + up, t + 1
            if t2 == k:
                walk(t2, j2, prob * pr)
            elif user_attempt[(t2, j2)]:
                total += prob * pr * (1 - q) * np.array([g(t2, j2) - fee, fee])      # included
                walk(t2, j2, prob * pr * q)                                         # censored, treated as having waited
            else:
                walk(t2, j2, prob * pr)

    walk(0, 0, 1.0)
    return total


def nodes(k):
    return [(t, j) for t in range(1, k) for j in range(t + 1)]


def scenario_params(rng):
    k = rng.randint(2, 5)
    r = rng.choice(ROUTINGS)
    fee = rng.choice([0.0, 0.5, 1.0, 2.0])
    c = fee * rng.choice([0.0, 0.5, 0.9])
    return dict(k=k, sigma=rng.choice([0.02, 0.05, 0.1]), size=rng.choice([1.0, 2.0]), fee=fee, c=c, q=rng.choice([0.0, 0.2, 0.5]),
                f=rng.choice([0.0, 0.5, 3.0, 12.0]), routing=r, fallback=rng.random() < 0.6)


def build(prm):
    pr = Protocol(prm["k"], ForfeitRule(flat=prm["f"]), prm["routing"], prm["fee"], prm["c"], prm["q"], prm["fallback"])
    return Scenario(Market(100.0, prm["sigma"]), Trade(size=prm["size"]), pr)


def run_eval(prm, ua, pp):
    r = prm["routing"]
    return evaluate(100.0, prm["sigma"], prm["k"], prm["size"], prm["fee"], prm["c"], prm["q"], prm["f"], r.to_user, r.to_producer, ua, pp,
                    prm["fallback"])


def solver_policies(sol, prm):
    k = prm["k"]
    ua = {(t, j): bool(sol.attempt[t][j]) for t in range(1, k) for j in range(t + 1)}
    pp = [bool(sol.publish[j]) for j in range(k + 1)] if (prm["fallback"] and sol.publish is not None) else [False] * (k + 1)
    return ua, pp


def test_solver_payoffs_equal_an_independent_evaluation_of_its_own_policy():
    rng = random.Random(2026)
    for _ in range(120):
        prm = scenario_params(rng)
        sol = solve(build(prm))
        ua, pp = solver_policies(sol, prm)
        got = run_eval(prm, ua, pp)
        assert got[0] == pytest.approx(sol.value_user, abs=1e-8, rel=1e-9), prm
        assert got[1] == pytest.approx(sol.value_producer, abs=1e-8, rel=1e-9), prm


def test_no_user_policy_and_no_producer_policy_beats_the_solver_value():
    rng = random.Random(77)
    checked = 0
    while checked < 25:
        prm = scenario_params(rng)
        if prm["k"] > 3:                                    # k = 3 has 5 user nodes (32 policies), k = 2 has 2
            continue
        sol = solve(build(prm))
        ua, pp = solver_policies(sol, prm)
        ns = nodes(prm["k"])
        best_user = max(run_eval(prm, dict(zip(ns, bits)), pp)[0] for bits in itertools.product([False, True], repeat=len(ns)))
        assert best_user <= sol.value_user + 1e-8, prm
        if prm["fallback"]:
            best_prod = max(run_eval(prm, ua, list(bits))[1] for bits in itertools.product([False, True], repeat=prm["k"] + 1))
            assert best_prod <= sol.value_producer + 1e-8, prm
        checked += 1
