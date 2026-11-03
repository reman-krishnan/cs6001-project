import math
import random

import numpy as np
import pytest

from dagame import Market, Trade, PayoffKind, ForfeitRule, Routing, Protocol, Scenario, solve
from dagame.observations import (pi_call_form, pi_put_form, check_observation_1, check_observation_2,
                                 check_observation_3)


def test_pi_formula_matches_solver_random():
    rng = random.Random(1)
    for _ in range(60):
        k = rng.randint(2, 40)
        sigma = rng.choice([0.01, 0.05, 0.3])
        m, tr = Market(100, sigma), Trade(rng.choice([PayoffKind.CALL, PayoffKind.PUT]), rng.choice([1, 3.5]))
        fee, f = rng.choice([0, 0.3, 1, 5]), rng.choice([0, 0.5, 2, 10, 100])
        sol = solve(Scenario(m, tr, Protocol(k, ForfeitRule(f), Routing.burn(), fee)))
        assert sol.value_user == pytest.approx(pi_call_form(m, tr, k, fee, f), abs=1e-9, rel=1e-9)
        # E[g] = 0 => the put form is the same number (put-call parity)
        assert pi_call_form(m, tr, k, fee, f) == pytest.approx(pi_put_form(m, tr, k, fee, f), abs=1e-9)


def test_observation_2_rows_all_consistent():
    rows = check_observation_2(Market(100, 0.05), Trade(size=2.0), k=20, fee=0.7,
                               f_values=np.linspace(0, 400, 81))
    assert all(r["match_call"] and r["match_put"] for r in rows)
    assert all(r["deterred_solver"] == r["deterred_formula"] for r in rows)
    flags = [r["deterred_solver"] for r in rows]
    assert not flags[0] and flags[-1]                       # deterrence kicks in as f grows ...
    assert flags == sorted(flags)                           # ... and, once it holds, it keeps holding


def test_observation_1_self_routing_inert():
    rows = check_observation_1(Market(100, 0.05), Trade(), k=8, fee=1.0, f_values=[0, 1, 10, 1e4])
    assert all(r["independent_of_f"] for r in rows)
    assert rows[0]["v0"] > 0            # ... and hence speculation is never deterred


def test_observation_3_threshold():
    fees = [0.0, 0.25, 0.5, 0.75, 1.0]   # threshold fee - c = 0.5
    rows = check_observation_3(Market(100, 0.05), Trade(), k=5, fee=1.0, c=0.5, f_values=fees)
    assert [r["predicted"] for r in rows] == ["publish", "publish", "tie", "withhold", "withhold"]
    assert all(r["consistent"] for r in rows)
    assert [r["tie_flagged"] for r in rows] == [False, False, True, False, False]


