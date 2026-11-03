"""The heart of the audit: the lattice solver and the explicit game tree are independent programs.
They must agree on every scenario, under each uniform tie-breaking selection."""
import itertools
import random

import numpy as np
import pytest

from conftest import make, random_scenario
from dagame import solve, build_game
from dagame import tree as T
from dagame.solver import _backward, evaluate, joint_surplus, USER_TIES, PRODUCER_TIES, SIGMA_TIES

SELECTIONS = list(itertools.product(USER_TIES, PRODUCER_TIES, SIGMA_TIES))


def dp_profile(sc, sel):
    """Lift the Markov lattice solution to a complete profile of the history-dependent tree game."""
    ut, pt, st = sel
    game = build_game(sc)
    sol = solve(sc, user_tie=ut, producer_tie=pt, sigma_tie=st)
    prof = {}
    hs = [1, 0] if (sc.protocol.fallback and sc.protocol.sigma_policy == "optional") else [int(sol.gave_sigma)]
    for h in hs:
        _, attempt, publish, _, _ = _backward(sc, bool(h), ut, pt)
        for key in game.decisions:
            if key[1] != h:
                continue
            if key[0] == "reveal":
                prof[key] = "attempt" if attempt[len(key[2])][key[2].count("u")] else "wait"
            elif key[0] == "publish":
                prof[key] = "publish" if publish[key[2].count("u")] else "withhold"
    for key in game.decisions:
        if key[0] == "sigma":
            prof[key] = "give" if sol.gave_sigma else "keep"
    return game, sol, prof


def test_solver_equals_tree_on_random_scenarios():
    rng = random.Random(20260929)
    n_ties = 0
    for i in range(150):
        sc = random_scenario(rng)
        game = build_game(sc)
        for sel in SELECTIONS:
            ut, pt, st = sel
            sol = solve(sc, user_tie=ut, producer_tie=pt, sigma_tie=st)
            v, prof, ties = T.backward_induction(game, T.tie_rule(ut, pt, st))
            assert sol.value_user == pytest.approx(v[0], abs=1e-8, rel=1e-9), (i, sel, sc)
            assert sol.value_producer == pytest.approx(v[1], abs=1e-8, rel=1e-9), (i, sel, sc)
        n_ties += sol.ties.any
    assert n_ties > 10     # the generator really does exercise ties


def test_lattice_profile_is_subgame_perfect_in_history_dependent_game():
    rng = random.Random(7)
    for i in range(80):
        sc = random_scenario(rng, kmax=6)
        for sel in rng.sample(SELECTIONS, 3):
            game, sol, prof = dp_profile(sc, sel)
            assert T.verify_spe(game, prof) == [], (i, sel, sc)
            got = T.expected_payoffs(game, prof)
            assert got[0] == pytest.approx(sol.value_user, abs=1e-8, rel=1e-9)
            assert got[1] == pytest.approx(sol.value_producer, abs=1e-8, rel=1e-9)


def test_joint_surplus_matches_tree():
    rng = random.Random(11)
    for _ in range(120):
        sc = random_scenario(rng)
        assert joint_surplus(sc) == pytest.approx(T.max_joint(build_game(sc))[0], abs=1e-8, rel=1e-9), sc


def test_forward_evaluation_matches_tree_for_arbitrary_profiles():
    rng = random.Random(3)
    for _ in range(80):
        sc = random_scenario(rng, kmax=6)
        game = build_game(sc)
        k = sc.protocol.k
        h = bool(sc.protocol.fallback)
        attempt = {t: np.array([rng.random() < 0.5 for _ in range(t + 1)]) for t in range(1, k)}
        publish = np.array([rng.random() < 0.5 for _ in range(k + 1)]) if h else None
        prof = {}
        for key in game.decisions:
            if key[0] == "reveal":
                prof[key] = "attempt" if attempt[len(key[2])][key[2].count("u")] else "wait"
            elif key[0] == "publish":
                prof[key] = "publish" if publish[key[2].count("u")] else "withhold"
            else:
                prof[key] = "give"
        if sc.protocol.sigma_policy == "optional" and h:
            pass  # 'give' subtree is the one scored; matches gave_sigma=True
        u, p, out = evaluate(sc, attempt, publish, h)
        got = T.expected_payoffs(game, prof)
        assert got[0] == pytest.approx(u, abs=1e-9) and got[1] == pytest.approx(p, abs=1e-9)
        assert out.reveal + out.fallback + out.dropped + out.challenged == pytest.approx(1.0)


def test_evaluate_rejects_malformed_profiles():
    sc = make(k=3)
    with pytest.raises(ValueError):
        evaluate(sc, {1: np.array([True, True])}, None, False)                                   # missing t=2
    with pytest.raises(ValueError):
        evaluate(sc, {1: np.array([True]), 2: np.array([True] * 3)}, None, False)               # wrong length
