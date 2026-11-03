import math

import pytest

from conftest import make
from dagame import check, assess_design, Market, Routing, Protocol, ForfeitRule, PayoffKind
from dagame.checker import DesignAssessment


def test_plain_commit_reveal_is_flagged_exploitable():
    rep = check(make(k=4, f=0.0, fee=0.5, fallback=False, sigma=0.1))
    assert rep.exploitable and rep.speculation_deterred is False
    assert rep.user_gain > 0
    assert rep.speculation_value_max > 0
    assert "brute-force" in rep.summary() and "EXPLOITABLE" in rep.summary()
    assert "user" in rep.brute_force_checked            # DP best response confirmed by full enumeration


def test_plain_commit_reveal_with_forfeit_but_no_fallback_still_exploitable_when_forfeit_too_small():
    assert check(make(k=4, f=1.0, fee=0.5, sigma=0.1)).exploitable


def test_forfeit_alone_can_deter_if_large_enough_but_still_flags_honest_deviation():
    rep = check(make(k=3, f=500.0, fee=1.0, sigma=0.1))
    assert rep.speculation_deterred is True             # forfeit exceeds any possible loss
    assert rep.user_gain > 0 or rep.exploitable is False


def test_fallback_burn_deters_speculator_and_fallback_is_credible():
    rep = check(make(k=3, f=10, fee=1, c=0.5, fallback=True, sigma=0.1))
    assert rep.speculation_deterred is True
    assert rep.fallback_credible is True
    assert rep.honest_spe_violations == []
    assert rep.user_gain == pytest.approx(0.0, abs=1e-9)


def test_collusion_exists_when_losses_exceed_forfeit_plus_cost():
    # u=2: worst loss over 2 blocks is 75; forfeit 10 << 75 + c
    rep = check(make(k=2, f=10, fee=1, c=0.5, fallback=True))
    assert rep.coalition_gain > 1.0 and rep.exploitable
    assert any("coalition" in r for r in rep.reasons)


def test_collusion_disappears_when_forfeit_covers_worst_loss():
    # coalition withholds only if g_k - c < -f, i.e. loss > f + c. Worst loss = 75 (k=2). f = 100 covers it.
    rep = check(make(k=2, f=100, fee=1, c=0.5, fallback=True))
    assert rep.coalition_gain == pytest.approx(0.0, abs=1e-9)
    assert rep.speculation_deterred is True
    assert not rep.exploitable, rep.summary()


def test_collusion_threshold_hand_derived():
    """k=2, sigma=ln2, fee=1, c=0.5, burn. g_2 in {300, 0, 0, -75}.
    The coalition only gains if WAITING beats revealing early at the down node (g_1 = -50):
        wait value C(f) = 1/3 * (-0.5) + 2/3 * max(-75.5, -f)  >  -50   <=>   f < 74.75.
    So 'f >= worst loss + c = 75.5' is SUFFICIENT for collusion-proofness but not necessary."""
    c = 0.5
    below = check(make(k=2, f=74.75 - 0.01, fee=1, c=c, fallback=True))
    above = check(make(k=2, f=74.75 + 0.01, fee=1, c=c, fallback=True))
    sufficient = check(make(k=2, f=75.5, fee=1, c=c, fallback=True))
    assert below.coalition_gain > 0
    assert above.coalition_gain == pytest.approx(0.0, abs=1e-9)
    assert sufficient.coalition_gain == pytest.approx(0.0, abs=1e-9)
    # exact size of the gain just below the threshold: 2/3 * (1/3-weighted improvement) = 2/3 * 0.01 * 2/3
    assert below.coalition_gain == pytest.approx((2 / 3) * (2 / 3) * 0.01, rel=1e-6)


def test_observation_3_producer_routing_kills_fallback():
    rep = check(make(k=3, f=2.0, fee=1, c=0.5, fallback=True, routing=Routing.producer(), sigma=0.1))
    assert rep.fallback_credible is False                    # publishing is not a best response for the producer
    assert any(v["key"][0] == "publish" for v in rep.honest_spe_violations)
    assert rep.speculation_deterred is False                 # ... so the free option is back
    assert rep.exploitable


def test_optional_sigma_lets_the_user_bypass_the_fallback():
    req = check(make(k=3, f=10, fee=1, c=0.5, fallback=True, sigma_policy="required", sigma=0.1))
    opt = check(make(k=3, f=10, fee=1, c=0.5, fallback=True, sigma_policy="optional", sigma=0.1))
    assert req.speculation_deterred is True
    assert opt.speculation_deterred is False


@pytest.mark.parametrize("f", [0.0, 1.0, 1e6])
def test_self_routing_never_deters(f):
    rep = check(make(k=3, f=f, fee=0.5, routing=Routing.to_self(), sigma=0.1))
    assert rep.speculation_deterred is False


def test_pool_routing_interpolates_between_burn_and_producer():
    # fee - c = 0.5. Producer keeps f/n of the forfeit. Publishes strictly iff 0.5 > f/n.
    f = 2.0
    creds = {n: check(make(k=2, f=f, fee=1, c=0.5, fallback=True, routing=Routing.pool(n), sigma=0.1)).fallback_credible
             for n in (1, 2, 3, 4, 5, 100)}
    assert creds == {1: False, 2: False, 3: False, 4: True, 5: True, 100: True}   # tie at n=4 is still "credible"


def test_transfer_trade_is_not_reported_as_speculative():
    rep = check(make(k=3, kind=PayoffKind.TRANSFER, value=5.0, f=3, fee=1, c=0.5, fallback=True))
    assert rep.speculation_deterred is None and rep.speculation_value_max is None
    assert not rep.exploitable


def test_assess_design_reports_the_three_q2_properties():
    m = Market(100, 0.1)
    good = assess_design(m, Protocol(3, ForfeitRule(flat=60), Routing.burn(), fee=1, publish_cost=0.5, fallback=True))
    assert isinstance(good, DesignAssessment)
    assert good.deters_speculation and good.producer_willing and good.collusion_proof and good.all_good
    assert good.honest_loss == pytest.approx(0.0, abs=1e-9)        # fallback rescues censored honest users
    bad = assess_design(m, Protocol(3, ForfeitRule(flat=2.0), Routing.producer(), fee=1, publish_cost=0.5, fallback=True))
    assert bad.producer_willing is False and not bad.deters_speculation and not bad.all_good
    plain = assess_design(m, Protocol(3, ForfeitRule(flat=2.0), Routing.burn(), fee=1))
    assert plain.producer_willing is None and not plain.deters_speculation


def test_honest_user_loss_from_censoring_without_fallback():
    from dagame.solver import honest_user_loss
    m = Market(100, 0.1)
    pr = Protocol(4, ForfeitRule(flat=6.0), Routing.burn(), fee=1.0, q=0.4)
    r = honest_user_loss(m, pr, value=5.0)
    # honest user attempts at every block; never authorized w.p. q^(k-1); then pays the forfeit.
    p_drop = 0.4 ** 3
    assert r["p_dropped"] == pytest.approx(p_drop)
    assert r["loss"] == pytest.approx(p_drop * ((5 - 1) + 6.0))
    with pytest.raises(ValueError):
        honest_user_loss(m, pr, value=0.5)


def test_checker_deterrence_uses_node_specific_producer_ties():
    """With q = 1 and an indifferent producer, a pure equilibrium with profitable speculation exists although the eight uniform
    selections all give zero. The checker must not report deterrence."""
    from conftest import make
    from dagame import Routing
    from dagame.checker import check
    sc = make(k=2, sigma=0.1, f=0.0, fee=0.0, c=0.0, q=1.0, fallback=True, routing=Routing.burn())
    rep = check(sc)
    assert rep.speculation_value_max > 1.0 and not rep.speculation_deterred
