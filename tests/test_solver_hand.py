"""Cases small enough to compute by hand. u=2, d=1/2, p=1/3, S0=K=100."""
import pytest

from conftest import make
from dagame import solve, Routing, PayoffKind
from dagame.solver import solve_all_selections


def test_k2_no_fallback_hand_value():
    # t=1 up: g=100, attempt -> 99 (> forfeit -10). t=1 down: g=-50, attempt -> -51 < -10, so wait, pay 10.
    # V0 = 1/3 * 99 + 2/3 * (-10) = 79/3
    sol = solve(make(k=2, f=10, fee=1))
    assert sol.value_user == pytest.approx(79 / 3, abs=1e-12)
    assert sol.outcomes.reveal == pytest.approx(1 / 3)
    assert sol.outcomes.dropped == pytest.approx(2 / 3)
    assert sol.outcomes.fallback == 0.0
    assert not sol.ties.any
    assert sol.attempt[1].tolist() == [False, True]   # down: wait, up: attempt


def test_k2_fallback_removes_option_value():
    # Producer publishes (fee-c = 0.5 > 0), so waiting never helps: V0 = E[g_1] - fee = -1, with ties everywhere.
    sc = make(k=2, f=10, fee=1, c=0.5, fallback=True)
    sol = solve(sc)
    assert sol.value_user == pytest.approx(-1.0, abs=1e-12)
    assert sol.ties.user_nodes and not sol.ties.producer
    for _, s in solve_all_selections(sc):
        assert s.value_user == pytest.approx(-1.0, abs=1e-12)


def test_k1_no_window_no_fallback():
    assert solve(make(k=1, f=7, routing=Routing.burn())).value_user == pytest.approx(-7.0)
    assert solve(make(k=1, f=7, routing=Routing.producer())).value_user == pytest.approx(-7.0)
    assert solve(make(k=1, f=7, routing=Routing.to_self())).value_user == pytest.approx(0.0)
    assert solve(make(k=1, f=7)).outcomes.dropped == 1.0


def test_k1_fallback_credibility_threshold():
    # fee - c = 0.5. Producer routing: publishes iff 0.5 > f.
    pub = solve(make(k=1, f=0.4, routing=Routing.producer(), c=0.5, fee=1, fallback=True))
    assert pub.outcomes.fallback == 1.0 and pub.value_user == pytest.approx(-1.0)   # E[g_1] - fee
    wh = solve(make(k=1, f=0.6, routing=Routing.producer(), c=0.5, fee=1, fallback=True))
    assert wh.outcomes.dropped == 1.0 and wh.value_user == pytest.approx(-0.6)


def test_producer_tie_is_flagged_and_matters():
    sc = make(k=1, f=0.5, routing=Routing.producer(), c=0.5, fee=1, fallback=True)   # fee - c == f exactly
    a = solve(sc, producer_tie="publish")
    b = solve(sc, producer_tie="withhold")
    assert a.ties.producer and b.ties.producer
    assert a.value_user == pytest.approx(-1.0) and b.value_user == pytest.approx(-0.5)   # different outcomes!


def test_q_equal_one_reveal_never_lands():
    sol = solve(make(k=3, q=1.0, f=10))
    assert sol.outcomes.reveal == 0.0 and sol.outcomes.dropped == pytest.approx(1.0)
    assert sol.value_user == pytest.approx(-10.0)
    assert len(sol.ties.user_nodes) == 2 + 3     # attempting is pointless at every node (t=1: 2 nodes, t=2: 3)


def test_zero_forfeit_zero_fee_is_pure_call():
    # V0 = E[(g_{k-1})^+]. k=2: g_1 in {100,-50}: 1/3 * 100
    assert solve(make(k=2, f=0, fee=0)).value_user == pytest.approx(100 / 3)


def test_transfer_no_option_value():
    # constant value 5, fee 1, q=0: attempts at t=1 always; no waiting incentive; V0 = 4.
    sol = solve(make(k=4, kind=PayoffKind.TRANSFER, value=5.0, fee=1.0))
    assert sol.value_user == pytest.approx(4.0)
    assert sol.outcomes.reveal == pytest.approx(1.0)


def test_sigma_optional_lets_user_bypass_fallback():
    req = solve(make(k=3, f=10, fee=1, c=0.5, fallback=True, sigma_policy="required"))
    opt = solve(make(k=3, f=10, fee=1, c=0.5, fallback=True, sigma_policy="optional"))
    nofb = solve(make(k=3, f=10, fee=1))
    assert req.gave_sigma and req.value_user == pytest.approx(-1.0)
    assert (not opt.gave_sigma) and opt.value_user == pytest.approx(nofb.value_user)
    assert opt.value_user > req.value_user


def test_self_routing_forfeit_is_inert():
    a = solve(make(k=3, f=0, routing=Routing.to_self())).value_user
    b = solve(make(k=3, f=1e6, routing=Routing.to_self())).value_user
    assert a == pytest.approx(b)
