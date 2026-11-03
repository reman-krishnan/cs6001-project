import math
import random
import time

import numpy as np
import pytest

from conftest import make, random_scenario
from dagame import check, solve, Market, Trade, Protocol, ForfeitRule, Routing, Scenario
from dagame.numerics import gt
from dagame.solver import joint_surplus, solve_all_selections


def test_checker_never_raises_and_invariants_hold_on_random_scenarios():
    """check() re-derives every best response by exhaustive enumeration when the space is small and raises
    AssertionError on any disagreement, so simply running it over many scenarios is itself a test."""
    rng = random.Random(99)
    n_exploitable = n_clean = 0
    for _ in range(60):
        sc = random_scenario(rng, kmax=4)
        rep = check(sc)
        # A coalition can always replicate any equilibrium, so the joint optimum is never below it.
        assert rep.joint_max >= rep.best_spe_total - 1e-8
        # Honest play is a feasible strategy: nobody's best response can be worse than it.
        assert rep.user_gain >= -1e-9 and rep.producer_gain >= -1e-9
        assert rep.exploitable == bool(rep.reasons)
        n_exploitable += rep.exploitable
        n_clean += (not rep.exploitable)
    assert n_exploitable > 5 and n_clean > 5       # the generator exercises both verdicts


def test_value_bounds_on_random_scenarios():
    rng = random.Random(5)
    for _ in range(200):
        sc = random_scenario(rng, kmax=9)
        sol = solve(sc)
        # producer never receives less than nothing it is owed, user never gets more than the best possible payoff
        gk = sc.g_at(sc.protocol.k)
        gmax = max(float(np.max(sc.g_at(t))) for t in range(1, sc.protocol.k + 1))
        assert sol.value_user <= gmax + 1e-9
        assert 0.0 <= sol.outcomes.reveal <= 1 + 1e-12
        o = sol.outcomes
        assert o.reveal + o.fallback + o.dropped + o.challenged == pytest.approx(1.0)
        assert joint_surplus(sc) >= sol.value_user + sol.value_producer - 1e-8


def test_large_k_runs_fast_and_is_finite():
    t0 = time.time()
    sol = solve(make(k=2000, sigma=0.005, f=5.0, fee=0.5, c=0.2, q=0.2, fallback=True))
    assert np.isfinite(sol.value_user) and np.isfinite(sol.value_producer)
    assert joint_surplus(make(k=2000, sigma=0.005, fallback=True)) > 0
    assert time.time() - t0 < 10.0


def test_extreme_but_legal_inputs():
    assert np.isfinite(solve(make(k=1, sigma=1e-8, f=0, fee=0)).value_user)
    assert np.isfinite(solve(make(k=50, sigma=1e-6, f=0, fee=0)).value_user)
    big = solve(make(k=10, sigma=0.05, f=1e12, fee=1.0, fallback=True, c=0.0, routing=Routing.burn()))
    assert np.isfinite(big.value_user)
    assert solve(make(k=5, size=1e6)).value_user > 0
    assert solve(make(k=5, fee=0.0, c=0.0, f=0.0, fallback=True, routing=Routing.producer())).ties.producer


def test_illegal_inputs_are_rejected_with_clear_errors():
    with pytest.raises(ValueError):
        solve(make(), user_tie="sometimes")
    with pytest.raises(ValueError):
        solve(make(), producer_tie="no")
    with pytest.raises(ValueError):
        Scenario(Market(100, 0.5), Trade(), Protocol(2000))       # sigma*k > 700
    with pytest.raises(TypeError):
        Scenario(Market(100, 0.1), Trade())                        # missing protocol


def test_selection_helper_reports_whether_ties_matter():
    sels = solve_all_selections(make(k=1, f=0.5, routing=Routing.producer(), c=0.5, fee=1, fallback=True))
    values = {round(s.value_user, 9) for _, s in sels}
    assert values == {-1.0, -0.5}                                  # the tie changes the user's payoff
    sels2 = solve_all_selections(make(k=3, f=10))
    assert len({round(s.value_user, 9) for _, s in sels2}) == 1    # ... and here it provably does not
