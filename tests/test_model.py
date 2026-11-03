import math

import numpy as np
import pytest

from conftest import make
from dagame import Market, Trade, PayoffKind, ForfeitRule, Routing, Protocol, Scenario


def test_martingale_probability_ln2():
    m = Market(100, math.log(2))
    assert m.p == pytest.approx(1 / 3, abs=1e-15)
    assert m.u * m.d == pytest.approx(1.0)


def test_p_is_stable_for_tiny_sigma():
    assert Market(100, 1e-9).p == pytest.approx(0.5, abs=1e-8)


def test_prices_are_recombining_and_correct():
    m = Market(100, math.log(2))
    np.testing.assert_allclose(m.prices_at(2), [25.0, 100.0, 400.0])
    np.testing.assert_allclose(m.prices_at(0), [100.0])


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), True, "0.1", None])
def test_market_rejects_bad_sigma(bad):
    with pytest.raises(ValueError):
        Market(100, bad)


def test_market_rejects_arbitrage_drift():
    with pytest.raises(ValueError):
        Market(100, 0.01, drift=0.5)


def test_overflow_guard():
    with pytest.raises(ValueError):
        Market(100, 1.0).prices_at(701)
    with pytest.raises(ValueError):
        make(k=701, sigma=1.0)


def test_trade_payoffs():
    S = np.array([50.0, 100.0, 200.0])
    np.testing.assert_allclose(Trade(PayoffKind.CALL, 2.0).g(S, 100), [-100, 0, 200])
    np.testing.assert_allclose(Trade(PayoffKind.PUT, 2.0).g(S, 100), [100, 0, -200])
    np.testing.assert_allclose(Trade(PayoffKind.TRANSFER, value=7.0).g(S, 100), [7, 7, 7])
    np.testing.assert_allclose(Trade(PayoffKind.CALL, 1.0, strike=150).g(S, 100), [-100, -50, 50])


@pytest.mark.parametrize("kw", [dict(size=0), dict(size=-1), dict(strike=0), dict(value=float("nan"))])
def test_trade_validation(kw):
    with pytest.raises(ValueError):
        Trade(**kw)


def test_forfeit_rule():
    assert ForfeitRule(2.0, 0.5)(Trade(size=10)) == 7.0
    with pytest.raises(ValueError):
        ForfeitRule(-1.0)
    with pytest.raises(ValueError):
        ForfeitRule(0.0, -0.1)


def test_routing_shares():
    assert Routing.pool(1).to_producer == 1.0
    assert Routing.pool(4).to_producer == 0.25 and Routing.pool(4).to_other == 0.75
    for bad in (0, -2, 1.5, True):
        with pytest.raises(ValueError):
            Routing.pool(bad)
    with pytest.raises(ValueError):
        Routing(0.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        Routing(-0.1, 0.6, 0.5)


@pytest.mark.parametrize("kw", [dict(k=0), dict(k=-3), dict(k=2.5), dict(k=True), dict(k=2, q=-0.1),
                                dict(k=2, q=1.1), dict(k=2, q=float("nan")), dict(k=2, fee=-1),
                                dict(k=2, publish_cost=-1), dict(k=2, sigma_policy="maybe")])
def test_protocol_validation(kw):
    with pytest.raises(ValueError):
        Protocol(**kw)


def test_drop_payoff_routing():
    sc = make(routing=Routing.producer(), f=8.0)
    assert sc.drop_payoff() == (-8.0, 8.0)
    assert make(routing=Routing.to_self(), f=8.0).drop_payoff() == (0.0, 0.0)
    assert make(routing=Routing.burn(), f=8.0).drop_payoff() == (-8.0, 0.0)
