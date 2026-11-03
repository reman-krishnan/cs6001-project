import pytest

import numpy as np

from dagame import bayes_private as B
from dagame.solver import minimal_deterring_forfeit
from dagame import Market, Trade, Protocol, ForfeitRule, Routing

S, K, FEE, Q = 0.0012, 6, 5e-4, 0.3


def test_type_zero_is_the_plain_speculator_and_value_is_increasing_in_type():
    z = B.interim(0.0, S, K, 0.0, FEE, Q)
    m = Market(100.0, S / (K - 1) ** 0.5)
    from dagame.solver import solve
    from dagame import Scenario
    base = solve(Scenario(m, Trade(size=1.0), Protocol(K, ForfeitRule(), Routing.burn(), FEE * 100, 0.0, Q, False))).value_user / 100
    assert z["v_br"] == pytest.approx(base, abs=1e-12)
    vs = [B.interim(t, S, K, 0.002, FEE, Q)["v_br"] for t in (0.0, 0.0005, 0.002, 0.01)]
    assert all(a < b for a, b in zip(vs, vs[1:]))


def test_best_response_never_below_honest_play_and_extraction_vanishes_for_deep_in_the_money_types():
    for th in (0.0, 0.0005, 0.005, 0.05):
        r = B.interim(th, S, K, 0.001, FEE, Q)
        assert r["v_br"] >= r["v_hon"] - 1e-12 and r["extraction"] >= 0
    assert B.interim(0.05, S, K, 0.001, FEE, Q)["extraction"] < 1e-9          # exposure so large the option is never exercised against
    assert B.interim(0.0, S, K, 0.0, FEE, Q)["extraction"] > 1e-4


def test_commit_decision_follows_the_sign_of_the_value():
    f_star = minimal_deterring_forfeit(Market(100.0, S / (K - 1) ** 0.5), Trade(size=1.0), Protocol(K, ForfeitRule(), Routing.burn(), fee=FEE * 100)) / 100
    assert B.interim(0.0, S, K, 0.0, FEE, Q)["commit"]                         # the speculator commits when the option is free
    assert not B.interim(0.0, S, K, 1.2 * f_star, FEE, Q)["commit"]            # a forfeit above f* deters the pure speculator
    assert B.interim(0.01, S, K, 1.2 * f_star, FEE, Q)["commit"]               # a genuinely exposed user still commits


def test_participation_cutoff_rises_with_the_forfeit():
    rows, _ = B.design_table(S, K, FEE, Q, f_grid=[0.0, 0.0004, 0.0008, 0.002])
    cuts = [B.participation_threshold(rows, f) for f in (0.0, 0.0004, 0.0008, 0.002)]
    assert all(c is not None for c in cuts) and all(a <= b for a, b in zip(cuts, cuts[1:]))


def test_designer_loss_decomposition_and_regimes():
    grid = [i * 1e-4 for i in range(0, 25)]
    rows, agg = B.design_table(S, K, FEE, 0.05, f_grid=grid)
    assert (agg.loss - (agg.exploitation + agg.friction + agg.lost_surplus)).abs().max() < 1e-12
    assert agg.participation.iloc[0] >= agg.participation.iloc[-1]             # a bigger forfeit never raises participation
    opt = B.optimal_forfeit(agg)
    assert opt["f_opt"] > 0 and opt["loss_opt"] < opt["loss_at_zero"]           # with light censoring a forfeit helps
    # heavy censoring and few speculators: the optimum is no forfeit
    prior = {0.0: 0.02, 0.01: 0.98}
    rows2, agg2 = B.design_table(S, K, FEE, 0.8, prior=prior, f_grid=grid)
    assert B.optimal_forfeit(agg2)["f_opt"] == 0.0


def test_validation():
    with pytest.raises(ValueError):
        B.design_table(S, K, FEE, Q, prior={0.0: 0.5, 0.01: 0.4})
    with pytest.raises(ValueError):
        B.type_scenario(1.2, S, K, 0.0, FEE, Q)


def test_eps_rule_is_the_smallest_forfeit_within_eps_of_the_best_reduction():
    import pandas as pd
    agg = pd.DataFrame({"f": [0.0, 1.0, 2.0, 3.0, 4.0], "loss": [10.0, 4.0, 1.2, 1.0, 1.0001]})
    o = B.eps_optimal_forfeit(agg, 0.01)
    assert o["f_opt"] == 3.0 and o["loss_min"] == 1.0          # 1.2 is not within 1% (0.09) of the 9.0 reduction, 1.0 is
    assert B.eps_optimal_forfeit(agg, 0.05)["f_opt"] == 2.0     # 1.2 is within 5% (0.45) of the reduction
    assert B.eps_optimal_forfeit(agg, 0.0)["f_opt"] == 3.0      # eps = 0 is the first exact minimiser
    with pytest.raises(ValueError):
        B.eps_optimal_forfeit(pd.DataFrame({"f": [1.0, 2.0], "loss": [1.0, 0.5]}))


def test_eps_rule_never_exceeds_the_exact_minimiser_on_a_real_designer_grid():
    grid = [i * 1e-4 for i in range(0, 41)]
    for q in (0.05, 0.6):
        _, agg = B.design_table(S, K, FEE, q, f_grid=grid)
        assert B.eps_optimal_forfeit(agg, 0.01)["f_opt"] <= B.optimal_forfeit(agg)["f_opt"]


def test_refined_eps_forfeit_is_not_limited_by_the_coarse_grid_spacing():
    f_star = minimal_deterring_forfeit(Market(100.0, S / (K - 1) ** 0.5), Trade(size=1.0), Protocol(K, ForfeitRule(), Routing.burn(), fee=FEE * 100)) / 100
    coarse = [i * 0.5 * f_star for i in range(41)]
    fine = [i * 0.05 * f_star for i in range(401)]
    _, agg_c = B.design_table(S, K, FEE, 0.05, f_grid=coarse)
    _, agg_f = B.design_table(S, K, FEE, 0.05, f_grid=fine)
    refined = B.refine_eps_forfeit(agg_c, S, K, FEE, 0.05)
    exact_fine = B.eps_optimal_forfeit(agg_f, 0.01)["f_opt"]
    assert refined <= B.eps_optimal_forfeit(agg_c, 0.01)["f_opt"] + 1e-15           # never worse than the coarse answer
    assert abs(refined - exact_fine) <= 0.1 * f_star                                  # and close to the answer on a ten times finer grid
