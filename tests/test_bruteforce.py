"""Exhaustive enumeration over ALL pure strategies / profiles on small games. No dynamic programming
is used to produce the reference values here."""
import itertools

import pytest

from conftest import make
from dagame import build_game, solve, Routing
from dagame import tree as T
from dagame.solver import USER_TIES, PRODUCER_TIES, SIGMA_TIES

SELECTIONS = list(itertools.product(USER_TIES, PRODUCER_TIES, SIGMA_TIES))


def _all_selection_payoffs(sc):
    out = set()
    for ut, pt, st in SELECTIONS:
        s = solve(sc, user_tie=ut, producer_tie=pt, sigma_tie=st)
        out.add((round(s.value_user, 9), round(s.value_producer, 9)))
    return out


@pytest.mark.parametrize("kw", [
    dict(k=2, f=10, fee=1, c=0.5, fallback=True, sigma_policy="optional"),
    dict(k=2, f=0.5, fee=1, c=0.5, fallback=True, routing=Routing.producer(), sigma_policy="optional"),  # producer tie
    dict(k=2, f=3, fee=1, c=2.0, fallback=True, routing=Routing.pool(2), sigma_policy="optional", q=0.3),
    dict(k=2, f=3, fee=0, c=0, fallback=True, routing=Routing.to_self(), sigma_policy="optional"),
    dict(k=2, f=5, fee=1, fallback=False),
])
def test_every_pure_spe_payoff_is_covered_by_the_solver_selections(kw):
    sc = make(**kw)
    game = build_game(sc)
    enumerated = set(T.enumerate_spe_payoffs(game))
    sel = _all_selection_payoffs(sc)
    assert sel <= enumerated                      # every selection the solver reports is a true SPE
    if not any(solve(sc, user_tie=u, producer_tie=p, sigma_tie=s).ties.any
               for u, p, s in SELECTIONS):
        assert len(enumerated) == 1               # no ties anywhere  =>  the SPE payoff is unique


def test_k3_exhaustive_spe_enumeration_with_producer():
    sc = make(k=3, f=10, fee=1, c=0.5, fallback=True)          # 14 decision nodes -> 16384 pure profiles
    game = build_game(sc)
    enumerated = set(T.enumerate_spe_payoffs(game))
    assert _all_selection_payoffs(sc) <= enumerated


@pytest.mark.parametrize("kw", [
    dict(k=4, f=10, fee=1),
    dict(k=4, f=2, fee=1, q=0.4),
    dict(k=4, f=10, fee=1, c=0.5, fallback=True),
    dict(k=4, f=50, fee=0.5, c=0.2, fallback=True, routing=Routing.producer()),   # fallback dead: option alive
])
def test_user_best_response_dp_equals_exhaustive_enumeration(kw):
    sc = make(**kw)
    game = build_game(sc)
    _, prof, _ = T.backward_induction(game)
    dp, _ = T.best_response(game, T.USER, prof)
    bf = T.brute_force_best_response(game, T.USER, prof)
    assert bf == pytest.approx(dp, abs=1e-9)
    assert bf == pytest.approx(solve(sc, user_tie="attempt", producer_tie="publish").value_user, abs=1e-9)


def test_producer_best_response_dp_equals_exhaustive_enumeration():
    sc = make(k=3, f=2, fee=1, c=0.2, fallback=True, routing=Routing.producer())
    game = build_game(sc)
    honest = {k: {"reveal": "attempt", "publish": "publish"}[k[0]] for k in game.decisions}
    dp, _ = T.best_response(game, T.PRODUCER, honest)
    assert T.brute_force_best_response(game, T.PRODUCER, honest) == pytest.approx(dp, abs=1e-9)


def test_guards_refuse_huge_enumerations():
    game = build_game(make(k=6, fallback=True))
    with pytest.raises(ValueError):
        T.brute_force_best_response(game, T.USER, {}, max_strategies=1000)
    with pytest.raises(ValueError):
        T.enumerate_spe_payoffs(game, max_profiles=1000)
    with pytest.raises(ValueError):
        build_game(make(k=17))
