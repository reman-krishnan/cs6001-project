import itertools
import random

import pytest

from conftest import make, random_scenario
from dagame import Routing, PayoffKind
from dagame import tree as T
from dagame.rotating import (solve_rotating, solve_all_rotating, build_game_rotating, extortion_threshold_scan, INCLUDE_TIES)
from dagame.solver import solve, USER_TIES, PRODUCER_TIES
from dagame.strategic import solve_strategic


def _eligible(rng, n):
    out = []
    while len(out) < n:
        sc = random_scenario(rng, kmax=7)
        if sc.protocol.q == 0.0 and sc.protocol.beta_eff == 0.0 and sc.protocol.sigma_policy == "required":
            out.append(sc)
    return out


def test_lattice_equals_explicit_tree_under_all_selections():
    rng = random.Random(31415)
    for sc in _eligible(rng, 50):
        for alpha in (0.0, 0.3, 0.7, 1.0):
            game = build_game_rotating(sc, alpha)
            for ut, et, st in itertools.product(USER_TIES, INCLUDE_TIES, PRODUCER_TIES):
                sol = solve_rotating(sc, alpha, user_tie=ut, entity_tie=et, settle_tie=st)
                v, prof, _ = T.backward_induction(game, T.tie_rule(ut, st, "give", et))
                assert sol.value_user == pytest.approx(v[0], abs=1e-8, rel=1e-9), (alpha, ut, et, st, sc)
                assert sol.value_entity == pytest.approx(v[1], abs=1e-8, rel=1e-9), (alpha, ut, et, st, sc)
                assert T.verify_spe(game, prof) == []


def test_alpha_one_is_the_single_strategic_producer_and_alpha_zero_is_the_honest_game():
    rng = random.Random(7)
    for sc in _eligible(rng, 25):
        a1 = solve_rotating(sc, 1.0)
        s1 = solve_strategic(sc)
        assert a1.value_user == pytest.approx(s1.value_user, abs=1e-8) and a1.value_entity == pytest.approx(s1.value_producer, abs=1e-8)
        # alpha = 0: every producer is protocol-following (publishes sigma even when that does not pay). The base game's RATIONAL producer
        # does the same exactly when publishing is weakly optimal, so the two coincide only then (or when there is no fallback).
        pr = sc.protocol
        if (not pr.fallback) or (pr.fee - pr.publish_cost >= pr.routing.to_producer * sc.f):
            assert solve_rotating(sc, 0.0).value_user == pytest.approx(solve(sc).value_user, abs=1e-8)


def test_probabilities_sum_to_one_and_match_tree_expectation():
    sc = make(k=5, f=1.0, fee=1.0, c=0.3, routing=Routing.producer(), fallback=True, sigma=0.05)
    for alpha in (0.2, 0.9):
        s = solve_rotating(sc, alpha)
        assert s.p_reveal + s.p_fallback + s.p_dropped == pytest.approx(1.0)


def test_extortion_needs_the_entity_to_control_the_whole_window_so_rotation_disarms_it():
    mk = lambda f: make(k=8, f=f, fee=1.0, c=0.2, routing=Routing.producer(), fallback=True, sigma=0.02, kind=PayoffKind.TRANSFER, value=3.0)
    grid = [0.5 * 1.3 ** i for i in range(40)]
    thr = {a: extortion_threshold_scan(mk, a, grid)[0] for a in (0.2, 0.5, 0.9, 1.0)}
    assert thr[1.0] < 1.3 * 1.0 * 1.0 + 1e-9 and thr[1.0] >= 1.0 - 1e-9            # single producer: threshold fee / s_P = 1
    assert thr[0.2] > thr[0.5] > thr[0.9] >= thr[1.0]                               # a smaller stake share needs a far larger forfeit
    assert thr[0.2] > 50 * thr[1.0]


def test_burn_routing_is_never_extorted_at_any_stake_share():
    for alpha in (0.3, 1.0):
        sc = make(k=6, f=1e6, fee=1.0, c=0.2, routing=Routing.burn(), fallback=True, sigma=0.02, kind=PayoffKind.TRANSFER, value=3.0)
        assert all(s.value_user == pytest.approx(2.0) for _, s in solve_all_rotating(sc, alpha))


def test_preconditions():
    with pytest.raises(ValueError):
        solve_rotating(make(k=3, q=0.2), 0.5)
    with pytest.raises(ValueError):
        solve_rotating(make(k=3, fallback=True, sigma_policy="optional"), 0.5)
    with pytest.raises(ValueError):
        solve_rotating(make(k=3), 1.5)
    with pytest.raises(ValueError):
        solve_rotating(make(k=3), 0.5, user_tie="x")
    with pytest.raises(ValueError):
        build_game_rotating(make(k=15), 0.5)


@pytest.mark.parametrize("alpha,k", [(0.5, 4), (0.5, 6), (0.25, 4), (0.8, 6), (0.9, 8)])
def test_extortion_threshold_is_fee_over_share_times_alpha_to_the_window(alpha, k):
    """Closed form found numerically and explained by the argument: to extort, the entity must censor at EVERY block of the window (an honest producer
    would include a retry), which pays only if alpha^(k-1) * s_P * f > fee. So the threshold is fee / (s_P alpha^(k-1))."""
    from dagame.rotating import extortion_threshold
    for routing, s_p in ((Routing.producer(), 1.0), (Routing.pool(2), 0.5)):
        mk = lambda f: make(k=k, f=f, fee=1.0, c=0.2, routing=routing, fallback=True, sigma=0.02, kind=PayoffKind.TRANSFER, value=3.0)
        thr = extortion_threshold(mk, alpha, 0.1, 1e9)
        assert thr == pytest.approx(1.0 / (s_p * alpha ** (k - 1)), rel=2e-3)
