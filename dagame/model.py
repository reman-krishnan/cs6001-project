"""Model primitives: market lattice, trade payoff, forfeit rule, routing rule, protocol, scenario.

Conventions (fixed once, used everywhere)
-----------------------------------------
* Block 0: the user commits. Prices S_0..S_k live on a recombining binomial lattice.
* Blocks 1..k-1: the *authorization window*. At each block t the user sees S_t and may attempt to
  reveal. An attempt is included with probability 1-q, and then the trade executes at S_t. A censored
  attempt (probability q) may be retried at the next block.
* Block k: *settlement*. If nothing was authorized, the producer (if it holds the signature sigma
  and the protocol has a fallback path) chooses to publish sigma, executing the trade at S_k.
  Otherwise the transaction is dropped and the forfeit f is charged and routed.
* Payoffs (user, producer), quasi-linear:
    user reveals at block t : (g_t - fee, fee)
    producer publishes at k : (g_k - fee, fee - publish_cost)
    dropped                 : (-(1 - share_user) * f, share_producer * f)
  where g_t = g(trade, S_t) is the trade's value to the user at execution.
* The price lattice uses a per-block log-volatility ``sigma``: u = e^sigma, d = 1/u. The up
  probability is p = (e^drift - d)/(u - d). With drift = 0 the price is a martingale (r = 0), which is
  the benchmark under which E[g] = 0 for the call/put trades.

Modelling limits (deliberate, documented in README):
* The producer's payoff does not depend on the price path (it is indifferent to g). Producer-side
  MEV or a producer that is the trade counterparty is NOT modelled.
* Censoring is exogenous and i.i.d. across attempts. A strategic censoring producer is NOT modelled.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

_MAX_EXPONENT = 700.0  # exp(700) ~ 1e304: stay clear of float overflow


class PayoffKind(str, Enum):
    CALL = "call"          # g = size * (S - K): user buys at K, gains if price rises
    PUT = "put"            # g = size * (K - S)
    TRANSFER = "transfer"  # g = value (constant): a plain payment, no exposure to the price


def _finite(name, x):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"{name} must be a finite real number, got {x!r}")
    return float(x)


def _nonneg(name, x):
    x = _finite(name, x)
    if x < 0:
        raise ValueError(f"{name} must be >= 0, got {x}")
    return x


def _pos(name, x):
    x = _finite(name, x)
    if x <= 0:
        raise ValueError(f"{name} must be > 0, got {x}")
    return x


@dataclass(frozen=True)
class Market:
    S0: float = 100.0
    sigma: float = 0.01   # per-block log volatility
    drift: float = 0.0    # log of expected gross return per block; 0 => martingale

    def __post_init__(self):
        _pos("S0", self.S0)
        _pos("sigma", self.sigma)
        _finite("drift", self.drift)
        _ = self.p  # validates the no-arbitrage range

    @property
    def u(self) -> float:
        return math.exp(self.sigma)

    @property
    def d(self) -> float:
        return math.exp(-self.sigma)

    @property
    def p(self) -> float:
        # p = (e^mu - d)/(u - d), rewritten with expm1 so it stays accurate for tiny sigma.
        p = math.expm1(self.drift + self.sigma) / math.expm1(2.0 * self.sigma)
        if not (0.0 < p < 1.0):
            raise ValueError(
                f"up-probability p={p:.6g} outside (0,1): |drift|={abs(self.drift)} is too large for "
                f"sigma={self.sigma} (would imply arbitrage on the lattice)")
        return p

    def prices_at(self, t: int) -> np.ndarray:
        """Prices at block t for j = 0..t up-moves (length t+1)."""
        if t < 0:
            raise ValueError("t must be >= 0")
        if self.sigma * t > _MAX_EXPONENT:
            raise ValueError(f"sigma*t = {self.sigma * t:.1f} would overflow the price lattice")
        j = np.arange(t + 1, dtype=float)
        return self.S0 * np.exp(self.sigma * (2.0 * j - t))


@dataclass(frozen=True)
class Trade:
    kind: PayoffKind = PayoffKind.CALL
    size: float = 1.0                 # m: number of units (scales g and size-dependent forfeits)
    strike: float | None = None       # default: at the money, K = S0
    value: float = 0.0                # TRANSFER only: constant value v to the user

    def __post_init__(self):
        if not isinstance(self.kind, PayoffKind):
            raise ValueError(f"kind must be a PayoffKind, got {self.kind!r}")
        _pos("size", self.size)
        if self.strike is not None:
            _pos("strike", self.strike)
        _finite("value", self.value)

    def g(self, S, S0: float):
        S = np.asarray(S, dtype=float)
        K = S0 if self.strike is None else self.strike
        if self.kind is PayoffKind.CALL:
            return self.size * (S - K)
        if self.kind is PayoffKind.PUT:
            return self.size * (K - S)
        return np.full_like(S, self.value)


@dataclass(frozen=True)
class ForfeitRule:
    """f(m) = flat + per_unit * m. Both parts non-negative."""
    flat: float = 0.0
    per_unit: float = 0.0

    def __post_init__(self):
        _nonneg("flat", self.flat)
        _nonneg("per_unit", self.per_unit)

    def __call__(self, trade: Trade) -> float:
        return self.flat + self.per_unit * trade.size


@dataclass(frozen=True)
class Routing:
    """Where a charged forfeit goes, as shares of f. Shares must be >= 0 and sum to 1.

    ``to_other`` is money that reaches neither player: burned, or paid to third-party validators.
    """
    to_user: float
    to_producer: float
    to_other: float
    label: str = "custom"

    def __post_init__(self):
        for n in ("to_user", "to_producer", "to_other"):
            _nonneg(n, getattr(self, n))
        if abs(self.to_user + self.to_producer + self.to_other - 1.0) > 1e-12:
            raise ValueError("routing shares must sum to 1")

    @staticmethod
    def to_self() -> "Routing":
        return Routing(1.0, 0.0, 0.0, "self")

    @staticmethod
    def burn() -> "Routing":
        return Routing(0.0, 0.0, 1.0, "burn")

    @staticmethod
    def producer() -> "Routing":
        return Routing(0.0, 1.0, 0.0, "producer")

    @staticmethod
    def pool(n_validators: int) -> "Routing":
        """Equal split among n validators of which the producer is one: it keeps a 1/n share.
        n = 1 is producer routing; n -> infinity approaches burning."""
        if isinstance(n_validators, bool) or not isinstance(n_validators, int) or n_validators < 1:
            raise ValueError("n_validators must be an integer >= 1")
        s = 1.0 / n_validators
        return Routing(0.0, s, 1.0 - s, f"pool({n_validators})")


@dataclass(frozen=True)
class Protocol:
    k: int
    forfeit: ForfeitRule = ForfeitRule()
    routing: Routing = Routing.burn()
    fee: float = 0.0             # phi: paid by the user on execution, received by the producer
    publish_cost: float = 0.0    # c: producer's extra cost of publishing sigma (block space)
    q: float = 0.0               # censoring probability of each reveal attempt
    fallback: bool = False       # is "producer publishes sigma" in the settlement-path set A?
    sigma_policy: str = "required"  # "required": user must hand sigma over; "optional": user may keep it
    # Contestation (Q4). If the producer withholds sigma at block k, an independent challenger is available with
    # probability challenge_prob and publishes it iff it is paid: fee - challenge_cost > 0. The challenger then
    # collects the fee, the trade executes for the user, and the producer receives nothing (no forfeit is charged).
    challenge_prob: float = 0.0
    challenge_cost: float = 0.0

    def __post_init__(self):
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 1:
            raise ValueError(f"k must be an integer >= 1, got {self.k!r}")
        _nonneg("fee", self.fee)
        _nonneg("publish_cost", self.publish_cost)
        q = _finite("q", self.q)
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"q must lie in [0,1], got {q}")
        if self.sigma_policy not in ("required", "optional"):
            raise ValueError("sigma_policy must be 'required' or 'optional'")
        b = _finite("challenge_prob", self.challenge_prob)
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"challenge_prob must lie in [0,1], got {b}")
        _nonneg("challenge_cost", self.challenge_cost)

    @property
    def beta_eff(self) -> float:
        """Probability that a withheld sigma is actually published by a challenger."""
        return float(self.challenge_prob) if self.fee > self.challenge_cost else 0.0


@dataclass(frozen=True)
class Scenario:
    market: Market
    trade: Trade
    protocol: Protocol

    def __post_init__(self):
        # Fail early (with a clear message) rather than deep inside a solver.
        self.market.prices_at(self.protocol.k)

    @property
    def f(self) -> float:
        return self.protocol.forfeit(self.trade)

    def g_at(self, t: int) -> np.ndarray:
        return self.trade.g(self.market.prices_at(t), self.market.S0)

    # --- payoff building blocks (single source of truth for solver, tree and checker) ---
    def drop_payoff(self) -> tuple[float, float]:
        r, f = self.protocol.routing, self.f
        return (-(1.0 - r.to_user) * f, r.to_producer * f)

    def withhold_payoff(self, g):
        """Expected (user, producer) payoff when the producer withholds sigma at block k, including contestation."""
        b = self.protocol.beta_eff
        du, dp = self.drop_payoff()
        return (b * (g - self.protocol.fee) + (1.0 - b) * du, (1.0 - b) * dp)

    def reveal_payoff(self, g):
        return (g - self.protocol.fee, self.protocol.fee)

    def publish_payoff(self, g):
        return (g - self.protocol.fee, self.protocol.fee - self.protocol.publish_cost)
