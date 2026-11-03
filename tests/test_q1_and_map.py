import math

import numpy as np
import pandas as pd
import pytest

from dagame import course_results as C


@pytest.fixture(scope="module")
def sweeps():
    return C.q1_sweeps(ks=(10, 50, 200), vols=(0.02, 0.04, 0.08), sizes=(0.1, 1.0, 10.0), base_k=50)


def test_q1_f_star_increases_in_window_volatility_and_size(sweeps):
    for fac in ("k", "volatility", "size"):
        d = sweeps[sweeps.factor == fac]
        assert d.f_star.is_monotonic_increasing and (d.f_star > 0).all(), fac


def test_q1_size_effect_is_more_than_proportional_with_a_flat_fee(sweeps):
    d = sweeps[sweeps.factor == "size"].set_index("value")
    assert d.f_star_over_notional.is_monotonic_increasing               # a flat fee covers less of a bigger trade's option


def test_q1_matches_direct_solver_call(sweeps):
    from dagame import Market, Trade, Protocol, ForfeitRule, Routing
    from dagame.solver import minimal_deterring_forfeit
    row = sweeps[(sweeps.factor == "k") & (sweeps.value == 50)].iloc[0]
    direct = minimal_deterring_forfeit(Market(100.0, C.block_sigma(0.04)), Trade(size=1.0),
                                       Protocol(50, ForfeitRule(), Routing.burn(), fee=0.05))
    assert row.f_star == pytest.approx(direct, rel=1e-12)


def test_feasibility_map_structure_and_participation_check():
    m = C.feasibility_map(ks=(5, 25, 100), sizes=(0.1, 10.0, 1000.0), kappa=0.01)
    assert len(m) == 9
    assert (m.feasible == (m.f_lo <= np.minimum(m.f_hi_collateral, m.f_hi_participation))).all()
    assert m.participation_check_ok.dropna().all()                        # the solver agrees with the IR formula
    p = m.pivot(index="size", columns="k", values="kappa_min")
    assert (p.diff(axis=1).iloc[:, 1:] > 0).all().all() and (p.diff(axis=0).iloc[1:, :] > 0).all().all()
    assert m[(m["size"] == 0.1) & (m.k == 5)].feasible.iloc[0] and not m[(m["size"] == 1000.0) & (m.k == 100)].feasible.iloc[0]


def test_coalition_table_matches_incentive_thresholds():
    t = C.coalition_table(f=0.0002).set_index("routing")     # fee - c = 4 bp of notional; pool(4) keeps a quarter
    assert t.loc["burn", "producer_publishes"] and t.loc["self", "producer_publishes"]
    assert t.loc["producer", "producer_publishes"] and t.loc["pool(4)", "producer_publishes"]
    t2 = C.coalition_table(f=0.01).set_index("routing")
    assert not t2.loc["producer", "producer_publishes"] and t2.loc["burn", "producer_publishes"]
    assert t2.loc["burn", "speculator_value_bp"] == pytest.approx(-5.0)
    assert t2.loc["producer", "speculator_value_bp"] > 0             # fallback dead => free option
