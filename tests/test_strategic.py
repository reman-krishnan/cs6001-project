import itertools
import random

import pytest

from conftest import make, random_scenario
from dagame import build_game, check, Routing
from dagame import tree as T
from dagame.strategic import solve_strategic, solve_all_strategic, extortion_threshold, INCLUDE_TIES
from dagame.solver import USER_TIES, PRODUCER_TIES, SIGMA_TIES


def _eligible(rng, n):
    out = []
    while len(out) < n:
        sc = random_scenario(rng, kmax=7)
        if sc.protocol.q == 0.0 and sc.protocol.beta_eff == 0.0:
            out.append(sc)
    return out


def test_strategic_lattice_equals_tree_under_all_selections():
    rng = random.Random(2718)
    for sc in _eligible(rng, 80):
        game = build_game(sc, strategic=True)
        for ut, pt, st, it in itertools.product(USER_TIES, PRODUCER_TIES, SIGMA_TIES, INCLUDE_TIES):
            sol = solve_strategic(sc, user_tie=ut, producer_tie=pt, sigma_tie=st, include_tie=it)
            v, prof, _ = T.backward_induction(game, T.tie_rule(ut, pt, st, it))
            assert sol.value_user == pytest.approx(v[0], abs=1e-8, rel=1e-9), (sc, ut, pt, st, it)
            assert sol.value_producer == pytest.approx(v[1], abs=1e-8, rel=1e-9), (sc, ut, pt, st, it)
            assert T.verify_spe(game, prof) == []


def test_extortion_under_producer_routing():
    # fee = 1, c = 0.2: producer routing with f = 5 > fee => censor every attempt, withhold, collect f.
    sc = make(k=4, f=5.0, fee=1.0, c=0.2, routing=Routing.producer(), fallback=True, sigma=0.05)
    sol = solve_strategic(sc)
    assert sol.p_reveal == 0.0 and sol.p_dropped == pytest.approx(1.0)
    assert sol.value_producer == pytest.approx(5.0) and sol.value_user == pytest.approx(-5.0)
    assert extortion_threshold(sc) == pytest.approx(1.0)
    # just below the threshold the producer prefers the fee: honest reveal goes through
    ok = solve_strategic(make(k=4, f=0.9, fee=1.0, c=0.2, routing=Routing.producer(), fallback=True, sigma=0.05))
    assert ok.p_reveal == pytest.approx(1.0) or ok.p_fallback > 0 or ok.p_dropped < 1.0


def test_burn_routing_has_no_extortion_incentive():
    for f in (0.0, 1.0, 100.0, 1e6):
        sol = solve_strategic(make(k=5, f=f, fee=1.0, c=0.2, routing=Routing.burn(), fallback=True, sigma=0.05,
                                   kind=__import__("dagame").PayoffKind.TRANSFER, value=5.0))
        assert sol.p_dropped == 0.0                      # honest transfer always completes
        assert sol.value_user == pytest.approx(4.0)


def test_without_fallback_or_with_self_routing_the_producer_cannot_gain_by_censoring():
    sol = solve_strategic(make(k=4, f=50.0, fee=1.0, routing=Routing.to_self(), sigma=0.05))
    assert sol.value_producer == pytest.approx(sol.scenario.protocol.fee * sol.p_reveal)


def test_preconditions_and_checker_integration():
    with pytest.raises(ValueError):
        solve_strategic(make(k=3, q=0.2))
    with pytest.raises(ValueError):
        solve_strategic(make(k=3, fallback=True, beta=0.5, ccost=0.1))
    with pytest.raises(ValueError):
        build_game(make(k=3, q=0.2), strategic=True)
    with pytest.raises(ValueError):
        solve_strategic(make(k=3), include_tie="maybe")
    assert len(solve_all_strategic(make(k=3, fallback=True))) == 16
