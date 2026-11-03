import pytest

from conftest import make
from dagame import solve, check, Routing, Protocol
from dagame.solver import solve_all_selections


def test_certain_challenger_makes_fallback_credible_even_with_producer_routing():
    # k=1, fee - c = 0.5. Without a challenger the producer withholds for f=5 (5 > 0.5). With a sure challenger,
    # withholding yields the producer nothing, so it publishes.
    no = solve(make(k=1, f=5, routing=Routing.producer(), c=0.5, fee=1, fallback=True))
    yes = solve(make(k=1, f=5, routing=Routing.producer(), c=0.5, fee=1, fallback=True, beta=1.0, ccost=0.2))
    assert no.outcomes.dropped == 1.0
    assert yes.outcomes.fallback == 1.0 and yes.value_user == pytest.approx(-1.0)


def test_partial_challenger_hand_value():
    # beta = 0.5: producer's withhold value = 0.5 * 5 = 2.5 > 0.5 -> withholds.
    # user: 0.5 * (E[g_1] - fee) + 0.5 * (-5) = -0.5 - 2.5 = -3.0
    sol = solve(make(k=1, f=5, routing=Routing.producer(), c=0.5, fee=1, fallback=True, beta=0.5, ccost=0.2))
    assert sol.value_user == pytest.approx(-3.0)
    assert sol.outcomes.challenged == pytest.approx(0.5) and sol.outcomes.dropped == pytest.approx(0.5)
    assert sol.value_producer == pytest.approx(2.5)


def test_unprofitable_challenger_never_shows_up():
    a = solve(make(k=1, f=5, routing=Routing.producer(), c=0.5, fee=1, fallback=True, beta=1.0, ccost=1.0))  # fee == cost
    b = solve(make(k=1, f=5, routing=Routing.producer(), c=0.5, fee=1, fallback=True))
    assert a.value_user == pytest.approx(b.value_user) and a.outcomes.challenged == 0.0


def test_challenger_needs_sigma():
    # sigma optional and forfeit small: user keeps sigma, so no challenger can ever help.
    sol = solve(make(k=3, f=1, fee=1, c=0.5, fallback=True, sigma_policy="optional", beta=1.0, ccost=0.1))
    assert sol.outcomes.challenged == 0.0


def test_challenge_threshold_shifts_credibility_boundary():
    # producer publishes iff fee - c > (1-beta) * f  <=>  f < (fee - c) / (1 - beta) = 1.0 for beta = 0.5
    for f, expect_pub in [(0.9, True), (1.1, False)]:
        sol = solve(make(k=1, f=f, routing=Routing.producer(), c=0.5, fee=1, fallback=True, beta=0.5, ccost=0.2))
        assert (sol.outcomes.fallback == 1.0) == expect_pub


def test_checker_handles_contestation():
    rep = check(make(k=2, f=5.0, routing=Routing.producer(), c=0.5, fee=1, fallback=True, beta=1.0, ccost=0.2))
    assert rep.fallback_credible is True
