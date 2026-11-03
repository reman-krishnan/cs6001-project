import math

import numpy as np
import pytest

from dagame import contest as C


@pytest.mark.parametrize("V1,V2", [(1, 1), (2, 1), (1, 3), (5, 4.9), (0.2, 7)])
def test_allpay_mixed_equilibrium_satisfies_indifference_and_no_profitable_deviation(V1, V2):
    r = C.verify_allpay_equilibrium(V1, V2)
    assert r["is_equilibrium"] and r["max_dev_in_support"] < 1e-9
    assert r["best_deviation_above_strong"] < 0 and r["best_deviation_above_weak"] < 0


def test_allpay_closed_forms_match_monte_carlo():
    rng = np.random.default_rng(0)
    for Vs, Vw in [(1.0, 1.0), (2.0, 1.0), (3.0, 0.5)]:
        n = 2_000_000
        strong = rng.uniform(0, Vw, n)
        weak = np.where(rng.random(n) < 1 - Vw / Vs, 0.0, rng.uniform(0, Vw, n))
        eq = C.allpay_two_player(Vs, Vw)
        assert strong.mean() == pytest.approx(eq["exp_bid_strong"], rel=3e-3)
        assert weak.mean() == pytest.approx(eq["exp_bid_weak"], rel=5e-3, abs=2e-4)
        assert (strong > weak).mean() == pytest.approx(eq["p_strong_wins"], abs=3e-3)
        assert (Vs * (strong > weak) - strong).mean() == pytest.approx(eq["payoff_strong"], abs=4e-3)
        assert (Vw * (weak > strong) - weak).mean() == pytest.approx(0.0, abs=4e-3)
        assert eq["total_waste"] == pytest.approx(eq["exp_bid_strong"] + eq["exp_bid_weak"])


def test_equal_values_dissipate_the_whole_prize():
    eq = C.allpay_two_player(1.0, 1.0)
    assert eq["payoff_strong"] == 0.0 and eq["total_waste"] == pytest.approx(1.0)


def test_race_deterrence_threshold_is_exact():
    for a, r in [(0.5, 0.5), (0.9, 1.0), (0.8, 2.0), (0.99, 1.0)]:
        Vo, Va = 1.0, r
        thr = (1 - a) * Va + a * max(Va - Vo, 0.0)
        assert C.race_outcome(Vo, Va, thr * 1.001, a)["deterred"]
        assert not C.race_outcome(Vo, Va, thr * 0.999, a)["deterred"]
        assert C.race_outcome(Vo, Va, 0.0, a)["deterrence_threshold_cost"] == pytest.approx(thr)


def test_race_cannot_deter_an_attacker_who_values_the_index_more_than_the_owner():
    # with a small attack cost no availability below 1 deters a stronger attacker
    for a in (0.5, 0.9, 0.99):
        assert not C.race_outcome(1.0, 2.0, 0.01, a)["deterred"]


def test_min_deterring_bond_matches_the_game_solution():
    Vo, kappa, cr = 1.0, 0.01, 0.002
    for a in (0.5, 0.8, 0.9):
        for Va in (0.5, 1.0, 2.0):
            b = C.min_deterring_bond(Va, kappa, a)
            assert math.isfinite(b) and b >= 0
            if b > 1e-6:
                assert not C.bond_outcome(Vo, Va, kappa, a, b * 0.99, cr)["deterred"]
            assert C.bond_outcome(Vo, Va, kappa, a, b * 1.0001 + 1e-9, cr)["deterred"]


def test_bond_payoffs_follow_the_extensive_form():
    o = C.bond_outcome(1.0, 1.0, 0.01, 0.9, 0.05, 0.002)       # b < b* = 0.1: the attacker attacks
    assert o["attacks"]
    assert o["attacker_payoff"] == pytest.approx(0.9 * (-0.01 - 0.05) + 0.1 * (1.0 - 0.01))
    assert o["owner_payoff"] == pytest.approx(0.9 * (-0.002 + 0.05) + 0.1 * (-1.0))
    d = C.bond_outcome(1.0, 1.0, 0.01, 0.9, 0.2, 0.002)       # b > b*: deterred, nobody pays anything on the path
    assert d["deterred"] and d["attacker_payoff"] == 0 and d["owner_payoff"] == 0


def test_bond_needs_capital_and_availability():
    assert C.min_deterring_bond(2.0, 0.01, 0.5, owner_capital=1.0) == math.inf          # b* = 3.98 > capital
    assert C.min_deterring_bond(1.0, 0.01, 0.0) == math.inf                              # owner never present
    assert C.min_deterring_bond(1.0, 2.0, 0.0) == 0.0                                    # attack costs more than it can earn
    assert C.min_deterring_bond(1.0, 0.01, 1.0) == 0.0                                   # owner always present
    # an owner who cannot post the bond cannot respond, so the bond does not deter
    assert not C.bond_outcome(1.0, 1.0, 0.01, 0.9, 5.0, 0.002, owner_capital=1.0)["deterred"]


def test_b_star_is_monotone():
    bs = [C.min_deterring_bond(1.0, 0.01, a) for a in (0.3, 0.5, 0.7, 0.9)]
    assert all(x > y for x, y in zip(bs, bs[1:]))                                       # more availability, smaller bond
    vs = [C.min_deterring_bond(v, 0.01, 0.8) for v in (0.5, 1.0, 2.0)]
    assert all(x < y for x, y in zip(vs, vs[1:]))                                       # bigger prize, bigger bond


def test_comparison_table_headline():
    t = C.compare_mechanisms()
    feasible = t[t.bond_feasible]
    assert feasible.bond_deterred.all()                                                  # wherever a bond within capital exists, it deters
    assert not t[(t.value_ratio == 2.0) & (t.availability < 0.99)].race_deterred.any()   # race fails against a stronger attacker
    assert t[(t.value_ratio == 2.0) & (t.availability >= 0.8)].bond_deterred.all()


def test_input_validation():
    with pytest.raises(ValueError):
        C.allpay_two_player(0.0, 1.0)


def test_compare_mechanisms_flags_the_exact_tie_in_the_race():
    from dagame.contest import compare_mechanisms
    d = compare_mechanisms()
    assert "race_tie" in d.columns
    row = d[(d.availability == 0.99) & (d.value_ratio == 1.0)].iloc[0]
    assert bool(row.race_tie) and bool(row.race_deterred)          # attack cost equals the race threshold exactly
    assert int(d.race_tie.sum()) == 1


def test_min_feasible_bond_when_the_owner_needs_a_larger_bond_to_respond():
    # deterrence needs only b >= 0.1, but the owner answers only if V_o + b > c_r, i.e. b > 0.49
    Vo, Va, kappa, a, cr, K = 0.01, 1.0, 0.01, 0.9, 0.5, 1.0
    assert C.min_deterring_bond(Va, kappa, a) == pytest.approx(0.1)
    b = C.min_feasible_bond(Vo, Va, kappa, a, cr, K)
    assert 0.49 < b <= 0.49 + 1e-6
    out = C.bond_outcome(Vo, Va, kappa, a, b * (1 + 1e-6) + 1e-9, cr, K)
    assert out["deterred"]                                                      # the larger bond works in the game itself
    assert not C.bond_outcome(Vo, Va, kappa, a, 0.1 * (1 + 1e-6) + 1e-9, cr, K)["deterred"]    # the deterrence-only bond does not


def test_min_feasible_bond_at_the_capital_boundary_uses_the_whole_capital():
    Vo, Va, kappa, a, cr = 1.0, 2.0, 0.01, 0.5, 0.002
    b_min = ((1 - a) * Va - kappa) / a                                          # 1.98
    assert C.min_feasible_bond(Vo, Va, kappa, a, cr, owner_capital=b_min) == pytest.approx(b_min)
    d = C.compare_mechanisms(V_owner=Vo, attack_cost=kappa, response_cost=cr, owner_capital=b_min, availabilities=(a,), value_ratios=(Va / Vo,))
    assert bool(d.bond_feasible.iloc[0]) and bool(d.bond_deterred.iloc[0])      # the old nudge above b_min exceeded the capital here


def test_min_feasible_bond_is_infeasible_when_the_response_bond_exceeds_the_capital():
    # the owner answers only above 0.49 but can post at most 0.3
    assert C.min_feasible_bond(0.01, 1.0, 0.01, 0.9, 0.5, owner_capital=0.3) == math.inf
    assert not C.bond_outcome(0.01, 1.0, 0.01, 0.9, 0.3, 0.5, 0.3)["deterred"]
    assert C.min_feasible_bond(1.0, 1.0, 0.01, 0.0, 0.002, 5.0) == math.inf     # owner never present: no bond helps


def test_min_feasible_bond_reduces_to_the_deterrence_bond_when_the_owner_always_responds():
    for a in (0.3, 0.6, 0.9):
        assert C.min_feasible_bond(1.0, 1.0, 0.01, a, 0.002) == pytest.approx(C.min_deterring_bond(1.0, 0.01, a))


def test_min_feasible_bond_accepts_a_capital_just_above_the_response_threshold():
    """Regression: a fixed 1e-9 increment above the strict threshold reported infeasible here although b = K works. A capital
    that exceeds the threshold by less than the solver's tolerance is an exact tie for the solver and counts as infeasible."""
    Vo, Va, kappa, a, cr, K = 0.01, 1.0, 0.01, 0.9, 0.5, 0.4900001
    b = C.min_feasible_bond(Vo, Va, kappa, a, cr, K)
    assert math.isfinite(b) and 0.49 < b <= K
    assert C.bond_outcome(Vo, Va, kappa, a, K, cr, K)["deterred"]
    assert not math.isfinite(C.min_feasible_bond(Vo, Va, kappa, a, cr, 0.4900000001))


def test_bond_diagnostics_follow_the_owner_who_ignores():
    """If answering costs more than the loss it avoids, the owner ignores and the 'if attacked' fields must say so."""
    o = C.bond_outcome(0.1, 1.0, 0.01, 0.9, 0.0, 1.0)
    assert not o["owner_responds"] and o["attacks"]
    assert o["attacker_payoff_if_attack"] == pytest.approx(o["attacker_payoff"]) == pytest.approx(0.99)
    assert o["owner_payoff_if_attacked"] == pytest.approx(o["owner_payoff"]) == pytest.approx(-0.1)
    assert o["waste_if_attacked"] == pytest.approx(0.01)
    r = C.bond_outcome(1.0, 1.0, 0.01, 0.9, 0.05, 0.002)       # a responding owner keeps the original formulas
    assert r["owner_responds"] and r["owner_payoff_if_attacked"] == pytest.approx(0.9 * (-0.002 + 0.05) + 0.1 * -1.0)


def test_minimal_feasible_bond_is_strictly_answered_in_the_solvers_sense():
    """The helper's bond must make responding strictly better beyond the solver's tolerance, so no tie rule is involved."""
    from dagame.numerics import gt
    b = C.min_feasible_bond(0.5, 0.2, 0.01, 0.9, response_cost=0.5)
    assert gt(0.5 + b, 0.5) and b > 0


def test_min_feasible_bond_returns_the_capital_when_only_the_margin_overshoots():
    """K = 0.4900000015 is answered strictly in the solver's sense, so the helper must not call the design infeasible."""
    Vo, Va, kappa, a, cr, K = 0.01, 1.0, 0.01, 0.9, 0.5, 0.4900000015
    b = C.min_feasible_bond(Vo, Va, kappa, a, cr, K)
    assert b == K
    o = C.bond_outcome(Vo, Va, kappa, a, K, cr, K)
    assert o["owner_responds"] and o["deterred"] and not o["tie"]
