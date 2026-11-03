"""Textbook validation of the engine (Centipede game) and the Nash-vs-SPNE comparison (non-credible threats)."""
import pytest

from conftest import make
from dagame import build_game, Routing
from dagame import tree as T
from dagame.normal_form import to_normal_form, pure_nash, nash_vs_spe


def centipede():
    # P1 (player 0) moves at n1 and n3, P2 (player 1) at n2 and n4. 'S' = stop, 'C' = continue.
    n4 = T.Decision(1, ("reveal", 0, "n4"), {"S": T.Terminal((2.0, 4.0)), "C": T.Terminal((4.0, 3.0))})
    n3 = T.Decision(0, ("reveal", 0, "n3"), {"S": T.Terminal((3.0, 1.0)), "C": n4})
    n2 = T.Decision(1, ("reveal", 0, "n2"), {"S": T.Terminal((0.0, 2.0)), "C": n3})
    n1 = T.Decision(0, ("reveal", 0, "n1"), {"S": T.Terminal((1.0, 0.0)), "C": n2})
    return T.game_from_root(n1)


def test_engine_reproduces_centipede_subgame_perfect_equilibrium():
    g = centipede()
    v, prof, ties = T.backward_induction(g)
    assert v.tolist() == [1.0, 0.0] and ties == []                 # the textbook SPNE: everybody stops immediately
    assert all(a == "S" for a in prof.values())
    assert T.verify_spe(g, prof) == []
    assert T.verify_spe(g, {**prof, ("reveal", 0, "n3"): "C"})       # a non-optimal action at an off-path node is caught


def test_centipede_normal_form_has_more_nash_equilibria_than_spne():
    r = nash_vs_spe(centipede())
    assert r["shape"] == (4, 4) and r["n_spe"] == 1 and r["n_nash"] > r["n_spe"]
    assert all(pay == (1.0, 0.0) for pay, _, _ in r["nash"])       # all Nash equilibria give the same outcome here


def test_every_spe_is_nash_but_not_conversely_in_the_protocol_game():
    """Burn routing, credible fallback, large forfeit: the user attempts at once out of fear of the producer's threat to
    withhold. That threat is not credible (publishing pays the producer more), so this Nash equilibrium is not subgame perfect."""
    sc = make(k=2, f=50.0, fee=1.0, c=0.2, fallback=True, routing=Routing.burn(), sigma=0.1)
    g = build_game(sc)
    r = nash_vs_spe(g)
    assert r["n_spe"] >= 1 and r["n_nash"] > r["n_spe"]
    keys, plans, U, P = to_normal_form(g)
    assert len(pure_nash(U, P)) == r["n_nash"]
    bad = r["non_credible"][0][2]
    assert any(k[0] == "publish" and a == "withhold" for k, a in bad.items())    # the non-credible threat
    good = [row for row in r["nash"] if row[1]]
    assert all(any(k[0] == "publish" and a == "publish" for k, a in row[2].items()) for row in good)


def test_producer_routing_makes_the_threat_credible():
    # f > fee - c: withholding pays the producer, so the threat becomes credible and every Nash equilibrium is SPNE here
    sc = make(k=2, f=2.0, fee=1.0, c=0.2, fallback=True, routing=Routing.producer(), sigma=0.1)
    r = nash_vs_spe(build_game(sc))
    assert r["n_spe"] >= 1
    assert all(a == "withhold" for row in r["nash"] if row[1] for k, a in row[2].items() if k[0] == "publish")


def test_guard():
    with pytest.raises(ValueError):
        to_normal_form(build_game(make(k=4, fallback=True)), max_profiles=1000)
