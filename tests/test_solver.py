

def test_user_value_range_equals_exhaustive_spe_range():
    """Node-specific producer ties: the speculator's extreme values match the full enumeration of pure SPE, and the eight
    uniform selections can miss them (a documented case with zero fee, forfeit and cost)."""
    import random
    import pytest
    from conftest import make, random_scenario
    from dagame import Routing, build_game, tree as T
    from dagame.solver import solve_all_selections, user_value_range
    sc = make(k=2, sigma=0.1, f=0.0, fee=0.0, c=0.0, q=0.5, fallback=True, routing=Routing.burn())
    lo, hi = user_value_range(sc)
    us = [v[0] for v in T.enumerate_spe_payoffs(build_game(sc))]
    assert (lo, hi) == (pytest.approx(min(us), abs=1e-7), pytest.approx(max(us), abs=1e-7))
    uni = [s.value_user for _, s in solve_all_selections(sc)]
    assert min(uni) > lo + 1.0 and max(uni) < hi - 1.0
    rng, n = random.Random(5), 0
    while n < 40:
        sc = random_scenario(rng, kmax=3)
        try:
            us = [v[0] for v in T.enumerate_spe_payoffs(build_game(sc))]
        except ValueError:
            continue
        n += 1
        lo, hi = user_value_range(sc)
        assert lo == pytest.approx(min(us), abs=1e-7) and hi == pytest.approx(max(us), abs=1e-7)
