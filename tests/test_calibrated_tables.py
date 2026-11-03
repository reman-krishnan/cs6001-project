import math

import pytest

from dagame import course_results as C

V = 0.0457          # a measured-looking daily volatility


def test_window_vol_scaling():
    assert C.window_vol(0.04, 100) == pytest.approx(0.04 * math.sqrt(99 * 12 / 86400))
    assert C.window_vol(0.04, 5) < C.window_vol(0.04, 50)


def test_calibrated_designs_keep_the_qualitative_pattern_and_scale_with_f_star():
    d = C.designs_table(daily_vol=V, k=100).set_index("design")
    assert d.forfeit_over_fstar["forfeit only, small"] == pytest.approx(0.3)
    assert d.forfeit_over_fstar["forfeit only, large"] == pytest.approx(1.5)
    assert not d.speculation_deterred["plain commit-reveal (no forfeit)"]
    assert not d.speculation_deterred["forfeit only, small"] and d.speculation_deterred["forfeit only, large"]
    assert d.speculation_deterred["fallback + burn"] and d.speculator_value_bp["fallback + burn"] == pytest.approx(-5.0)
    assert not d.speculation_deterred["fallback + producer routing (f > fee - c)"]
    assert d.producer_prefers_publish["fallback + producer routing (f > fee - c)"] == False      # noqa: E712
    assert not d.speculation_deterred["fallback + burn, sigma optional"] and not d.speculation_deterred["self routing"]


def test_calibrated_designs_move_with_volatility():
    lo, hi = C.designs_table(daily_vol=0.035, k=100), C.designs_table(daily_vol=0.056, k=100)
    assert hi.f_star.iloc[0] > lo.f_star.iloc[0]
    assert hi.window_vol.iloc[0] > lo.window_vol.iloc[0]


def test_calibrated_coalition_table_shapes():
    t = C.coalition_table(daily_vol=V, k=60)
    burn = t[t.routing == "burn"].sort_values("forfeit")
    assert burn.coalition_gain_bp.is_monotonic_decreasing and burn.coalition_gain_bp.iloc[-1] < 0.5 * burn.coalition_gain_bp.iloc[0]
    selfr = t[t.routing == "self"]
    assert selfr.coalition_gain_bp.max() - selfr.coalition_gain_bp.min() < 1e-6          # the forfeit is inert when routed to the user
    assert t[t.routing == "burn"].producer_publishes.all() and not t[t.routing == "producer"].producer_publishes.any()


def test_calibrated_extortion_thresholds():
    import numpy as np
    e = C.extortion_table(daily_vol=V, k=20, fee=5e-4, c=1e-4, surplus=1e-3, f_grid=np.array([2e-4, 8e-4, 3e-3]))
    prod = e[e.routing == "producer"].set_index("f").loss_worst
    assert prod[2e-4] == pytest.approx(0.0, abs=1e-12) and prod[8e-4] == pytest.approx(1e-3 + 8e-4, abs=1e-9)
    assert e[e.routing.isin(["burn", "self"])].loss_worst.abs().max() < 1e-12


def test_calibrated_designer_has_grid_relative_to_f_star():
    d = C.designer_table(daily_vol=V, k=6, fee=5e-4, surplus=1e-3, qs=(0.1, 0.7), pis=(0.02, 0.9))
    assert (d.f_opt <= 1.5 * d.f_star + 1e-12).all() and set(d.regime) <= {"no forfeit", "partial", "near threshold", "full deterrence"}
    assert d[(d.q == 0.1) & (d.pi == 0.9)].regime.iloc[0] == "full deterrence"
