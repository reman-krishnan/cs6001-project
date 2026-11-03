import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dagame import Market, Trade, PayoffKind, ForfeitRule, Routing, Protocol, Scenario  # noqa: E402


def make(k=3, sigma=math.log(2), S0=100.0, drift=0.0, kind=PayoffKind.CALL, size=1.0, value=0.0,
         f=10.0, per_unit=0.0, routing=None, fee=1.0, c=0.0, q=0.0, fallback=False, sigma_policy="required", beta=0.0, ccost=0.0):
    return Scenario(Market(S0, sigma, drift), Trade(kind, size, None, value),
                    Protocol(k, ForfeitRule(f, per_unit), routing or Routing.burn(), fee, c, q, fallback,
                             sigma_policy, beta, ccost))


def random_scenario(rng: random.Random, kmax=7):
    k = rng.randint(1, kmax)
    sigma = rng.choice([0.05, 0.2, 0.5, math.log(2)])
    drift = rng.choice([0.0, 0.0, 0.01, -0.01])
    try:
        Market(100.0, sigma, drift)
    except ValueError:
        drift = 0.0
    kind = rng.choice([PayoffKind.CALL, PayoffKind.PUT, PayoffKind.TRANSFER])
    routing = rng.choice([Routing.burn(), Routing.to_self(), Routing.producer(), Routing.pool(3),
                          Routing(0.25, 0.25, 0.5, "mixed")])
    fee = rng.choice([0.0, 0.5, 1.0, 2.0])
    c = rng.choice([0.0, 0.5, 1.0, 2.0])
    f = rng.choice([0.0, 0.5, 1.0, 3.0, 10.0, 50.0])       # round numbers on purpose: ties are common
    q = rng.choice([0.0, 0.0, 0.3, 1.0])
    fallback = rng.random() < 0.6
    sp = rng.choice(["required", "optional"])
    beta = rng.choice([0.0, 0.0, 0.5, 1.0])
    ccost = rng.choice([0.0, 0.5, 5.0])
    return make(k=k, sigma=sigma, drift=drift, kind=kind, size=rng.choice([1.0, 2.5]),
                value=rng.choice([0.0, 5.0, 20.0]), f=f, routing=routing, fee=fee, c=c, q=q,
                fallback=fallback, sigma_policy=sp, beta=beta, ccost=ccost)
