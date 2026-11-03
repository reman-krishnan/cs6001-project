"""Numerical verification of the coalition impossibility theorem and of its two proof ingredients."""
import pytest

from dagame import course_results as C


@pytest.fixture(scope="module")
def df():
    return C.impossibility_search(n_trials=40, seed=11)


def test_theorem_strict_credibility_with_no_forfeit_leaving_the_coalition_gives_a_strictly_positive_coalition_gain(df):
    d = df[(df.s_other == 0.0) & df.p2_strict]
    assert len(d) > 20                                                          # the random designs really exercise the hypothesis
    assert (d.gain_bp > 0).all()                                                # P3 fails
    assert (d.gain_bp >= d.bound_bp - 1e-9).all()                               # ... by at least (1 - q) E[g_1^+]
    assert not d.p3_exact.any() and not d.p123.any()


def test_ingredient_one_equilibrium_total_is_nonpositive_when_the_producer_strictly_publishes(df):
    d = df[(df.s_other == 0.0) & df.p2_strict]
    assert (d.total_spne_bp <= 1e-7).all()                                      # optional stopping: E[g_tau] - c P(fallback) <= 0


def test_the_theorem_does_not_extend_to_burning_routing_or_to_ties(df):
    burn = df[df.routing == "burn"]
    assert burn.p123.any()                                                      # burning CAN satisfy all three
    assert (df[df.routing == "pool(4)"].p123.sum()) <= (burn.p123.sum())        # and a pool is no better than burning


def test_the_hypothesis_matters_without_strict_credibility_the_search_may_find_collusion_proof_designs(df):
    # P3 can hold when the producer does not publish (the fallback is dead): the theorem is about designs where it does
    d = df[(df.s_other == 0.0) & ~df.p2_strict]
    assert d.p3_exact.any()


def test_bound_formula_matches_the_first_block_distribution():
    import math
    from dagame.course_results import scenario, NOTIONAL
    sc = scenario(0.02, 11, f=0.01, fee=5e-4, c=1e-4, q=0.25)
    p, sig = sc.market.p, sc.market.sigma
    g1 = sc.g_at(1)
    assert float((p * max(g1[1], 0) + (1 - p) * max(g1[0], 0))) == pytest.approx(p * NOTIONAL * (math.exp(sig) - 1.0))
