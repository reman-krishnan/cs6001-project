"""Fast lattice solver: subgame-perfect equilibrium by backward induction on the recombining lattice.

State at block t is (t, j) with j = number of up-moves. Because payoffs depend on the path only through
S_t, and a censored attempt leaves the user in exactly the position of a user who waited, the
equilibrium is Markov in (t, j). ``tree.py`` does NOT assume this (its strategies depend on the full
price history) and is used to audit it.

Ties are never broken silently. Every indifference is detected with the shared tolerance, reported in
``Solution.ties``, and resolved by an explicit, caller-chosen rule. ``solve_all_selections`` runs every
combination so the caller can see whether the choice matters.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .model import PayoffKind, Protocol, Scenario, ForfeitRule
from .numerics import close, gt, lt

USER_TIES = ("attempt", "wait")
PRODUCER_TIES = ("publish", "withhold")
NODE_TIES = ("best_for_user", "worst_for_user")      # node-specific producer tie rules, used only by ``user_value_range``
SIGMA_TIES = ("give", "keep")


@dataclass(frozen=True)
class Outcomes:
    """Probabilities of the three ways the game can end (they sum to 1)."""
    reveal: float      # user's own reveal was included
    fallback: float    # producer published sigma at block k
    dropped: float     # forfeit charged
    challenged: float = 0.0  # producer withheld but a challenger published sigma


@dataclass
class Ties:
    user_nodes: list = field(default_factory=list)   # [(t, j), ...] where attempt == wait for the user
    producer: bool = False                           # producer indifferent between publish/withhold
    sigma: bool = False                              # user indifferent between giving and keeping sigma

    @property
    def any(self) -> bool:
        return bool(self.user_nodes) or self.producer or self.sigma


@dataclass
class Solution:
    scenario: Scenario
    gave_sigma: bool
    value_user: float          # expected user payoff conditional on committing (V_0)
    value_producer: float
    attempt: dict              # t -> bool array over j: user attempts to reveal at (t, j)
    publish: np.ndarray | None  # bool array over j at block k, or None if the producer holds no sigma
    outcomes: Outcomes
    ties: Ties

    @property
    def speculation_profitable(self) -> bool:
        """Does committing beat not committing (payoff 0), beyond numerical noise?"""
        return bool(gt(self.value_user, 0.0))


def _validate_ties(user_tie, producer_tie, sigma_tie):
    if user_tie not in USER_TIES:
        raise ValueError(f"user_tie must be one of {USER_TIES}")
    if producer_tie not in PRODUCER_TIES + NODE_TIES:
        raise ValueError(f"producer_tie must be one of {PRODUCER_TIES + NODE_TIES}")
    if sigma_tie not in SIGMA_TIES:
        raise ValueError(f"sigma_tie must be one of {SIGMA_TIES}")


def _settlement(sc: Scenario, h: bool, producer_tie: str):
    """User's terminal payoff array over j at block k, the producer decision array, and a tie flag."""
    k, pr = sc.protocol.k, sc.protocol
    drop_u, drop_p = sc.drop_payoff()
    if not h:
        return np.full(k + 1, drop_u), None, False
    gk = sc.g_at(k)
    pub_u, pub_p = gk - pr.fee, pr.fee - pr.publish_cost
    wh_u, wh_p = sc.withhold_payoff(gk)          # wh_u is an array over j; wh_p a constant
    if gt(pub_p, wh_p):
        publish, tie = np.ones(k + 1, dtype=bool), False
    elif lt(pub_p, wh_p):
        publish, tie = np.zeros(k + 1, dtype=bool), False
    elif producer_tie == "best_for_user":          # the producer is indifferent at every state, so it may pick state by state
        publish, tie = np.asarray(pub_u >= wh_u) & np.ones(k + 1, dtype=bool), True
    elif producer_tie == "worst_for_user":
        publish, tie = np.asarray(pub_u <= wh_u) & np.ones(k + 1, dtype=bool), True
    else:
        publish, tie = np.full(k + 1, producer_tie == "publish"), True
    return np.where(publish, pub_u, wh_u), publish, tie


def _backward(sc: Scenario, h: bool, user_tie: str, producer_tie: str):
    k, p, q, fee = sc.protocol.k, sc.market.p, sc.protocol.q, sc.protocol.fee
    Uk, publish, producer_tie_flag = _settlement(sc, h, producer_tie)
    V = {k: Uk}
    attempt, tie_nodes = {}, []
    for t in range(k - 1, 0, -1):
        Vn = V[t + 1]
        C = p * Vn[1:] + (1.0 - p) * Vn[:-1]          # continuation value at (t, j), length t+1
        A = (1.0 - q) * (sc.g_at(t) - fee) + q * C     # value of attempting now
        tie = close(A, C)
        att = np.where(tie, user_tie == "attempt", gt(A, C))
        V[t] = np.where(att, A, C)
        attempt[t] = att
        tie_nodes += [(t, int(j)) for j in np.flatnonzero(tie)]
    V1 = V[1]
    V0 = p * V1[1] + (1.0 - p) * V1[0]
    return float(V0), attempt, publish, tie_nodes, producer_tie_flag


def evaluate(sc: Scenario, attempt: dict, publish, gave_sigma: bool):
    """Exact expected payoffs (user, producer) and outcome probabilities of a given Markov profile.

    Independent of ``_backward``: a forward pass that propagates the probability of still being
    unauthorized. Used to audit the DP value and to score arbitrary (e.g. honest) profiles.
    """
    k, p, q = sc.protocol.k, sc.market.p, sc.protocol.q
    fee, c = sc.protocol.fee, sc.protocol.publish_cost
    for t in range(1, k):
        if t not in attempt or np.shape(attempt[t]) != (t + 1,):
            raise ValueError(f"attempt[{t}] must be a boolean array of length {t + 1}")
    if gave_sigma and publish is not None and np.shape(publish) != (k + 1,):
        raise ValueError(f"publish must be a boolean array of length {k + 1}")
    drop_u, drop_p = sc.drop_payoff()
    mass = np.array([1.0])
    u_val = p_val = pr_reveal = pr_fallback = 0.0
    for t in range(1, k + 1):
        nxt = np.zeros(t + 1)
        nxt[1:] += p * mass
        nxt[:-1] += (1.0 - p) * mass
        mass = nxt
        if t < k:
            att = np.asarray(attempt[t], dtype=bool)
            done = mass * att * (1.0 - q)
            u_val += float(np.sum(done * (sc.g_at(t) - fee)))
            p_val += fee * float(np.sum(done))
            pr_reveal += float(np.sum(done))
            mass = mass - done
    challenged = 0.0
    if gave_sigma and publish is not None:
        pub = np.asarray(publish, dtype=bool)
        done = mass * pub
        u_val += float(np.sum(done * (sc.g_at(k) - fee)))
        p_val += (fee - c) * float(np.sum(done))
        pr_fallback = float(np.sum(done))
        rest = mass - done
        wh_u, wh_p = sc.withhold_payoff(sc.g_at(k))
        u_val += float(np.sum(rest * wh_u))
        p_val += wh_p * float(np.sum(rest))
        b = sc.protocol.beta_eff
        challenged, dropped = b * float(np.sum(rest)), (1.0 - b) * float(np.sum(rest))
    else:
        dropped = float(np.sum(mass))
        u_val += drop_u * dropped
        p_val += drop_p * dropped
    return u_val, p_val, Outcomes(pr_reveal, pr_fallback, dropped, challenged)


def solve(sc: Scenario, *, user_tie: str = "attempt", producer_tie: str = "publish",
          sigma_tie: str = "give") -> Solution:
    """Subgame-perfect equilibrium under one explicit tie-breaking selection."""
    _validate_ties(user_tie, producer_tie, sigma_tie)
    pr = sc.protocol
    ties = Ties()
    if pr.fallback and pr.sigma_policy == "optional":
        outs = {h: _backward(sc, h, user_tie, producer_tie) for h in (True, False)}
        v1, v0 = outs[True][0], outs[False][0]
        if close(v1, v0):
            ties.sigma, h = True, (sigma_tie == "give")
        else:
            h = bool(gt(v1, v0))
    else:
        h = bool(pr.fallback)
    V0, attempt, publish, tie_nodes, ptie = _backward(sc, h, user_tie, producer_tie)
    ties.user_nodes, ties.producer = tie_nodes, ptie
    u_fw, p_fw, outcomes = evaluate(sc, attempt, publish, h)
    # Self-audit: the backward DP and the independent forward pass must agree.
    if not close(V0, u_fw):
        raise RuntimeError(f"internal inconsistency: DP value {V0!r} != forward value {u_fw!r}")
    return Solution(sc, h, V0, p_fw, attempt, publish, outcomes, ties)


def solve_all_selections(sc: Scenario) -> list[tuple[dict, Solution]]:
    """The (at most 8) UNIFORM tie-breaking selections, each tie broken the same way at every node. Node-specific choices can give
    other equilibrium payoffs, see ``user_value_range`` for the exact range of the speculator's value."""
    out = []
    for ut, pt, st in itertools.product(USER_TIES, PRODUCER_TIES, SIGMA_TIES):
        out.append(({"user_tie": ut, "producer_tie": pt, "sigma_tie": st},
                    solve(sc, user_tie=ut, producer_tie=pt, sigma_tie=st)))
    return out


def user_value_range(sc: Scenario) -> tuple[float, float]:
    """Smallest and largest speculator value over ALL pure subgame-perfect profiles of the lattice game.

    ``solve_all_selections`` breaks each kind of tie the same way at every node. A producer who is indifferent can in fact
    choose differently at each settlement state, and the user's value is the same whichever way the user breaks its own
    ties, so the extreme values come from the producer picking, state by state, the best or the worst outcome for the user."""
    lo = solve(sc, producer_tie="worst_for_user").value_user
    hi = solve(sc, producer_tie="best_for_user").value_user
    return lo, hi


# ---------------------------------------------------------------------------------------------
# Derived quantities used by the proposal's questions
# ---------------------------------------------------------------------------------------------

def speculator_value(sc: Scenario) -> float:
    return solve(sc).value_user


def honest_user_loss(market, protocol: Protocol, *, value: float, **solve_kw) -> dict:
    """Expected shortfall of an honest user (constant-value transfer, no price exposure) versus the
    ideal outcome "executes for sure" (value - fee). Includes losses from censoring and forfeits."""
    from .model import Trade
    if value <= protocol.fee:
        raise ValueError("an honest user must value the transfer above the fee (value > fee)")
    sc = Scenario(market, Trade(PayoffKind.TRANSFER, value=value), protocol)
    sol = solve(sc, **solve_kw)
    ideal = value - protocol.fee
    return {"loss": ideal - sol.value_user, "ideal": ideal, "expected": sol.value_user,
            "p_dropped": sol.outcomes.dropped, "p_fallback": sol.outcomes.fallback,
            "solution": sol}


def minimal_deterring_forfeit(market, trade, protocol: Protocol, *, f_max: float | None = None,
                              iters: int = 200) -> float | None:
    """Smallest flat forfeit f with V_0(f) <= 0 (speculation not profitable), by bisection.

    Valid ONLY without a fallback path: then V_0 is continuous and non-increasing in f, so bisection is
    sound. With a fallback, V_0 can jump when the producer's decision flips, so this function refuses
    and the caller must use ``deterrence_scan``. Returns None if no finite forfeit deters (e.g. the
    forfeit is routed back to the user).
    """
    if protocol.fallback:
        raise ValueError("bisection is unsound with a fallback path (V_0 is not monotone in f); "
                         "use deterrence_scan")

    def v0(f):
        pr = Protocol(protocol.k, ForfeitRule(flat=f), protocol.routing, protocol.fee,
                      protocol.publish_cost, protocol.q, protocol.fallback, protocol.sigma_policy,
                      protocol.challenge_prob, protocol.challenge_cost)
        return solve(Scenario(market, trade, pr)).value_user

    if not gt(v0(0.0), 0.0):
        return 0.0
    if f_max is None:
        gmax = float(np.max(np.abs(Scenario(market, trade, protocol).g_at(protocol.k))))
        f_max = 4.0 * (gmax + protocol.fee + 1.0)
    if gt(v0(f_max), 0.0):
        return None
    lo, hi = 0.0, f_max
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if gt(v0(mid), 0.0):
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-12 * max(1.0, hi):
            break
    return hi


def deterrence_scan(market, trade, protocol: Protocol, f_values) -> list[dict]:
    """V_0 and the producer's SPE behaviour on a grid of flat forfeits (valid with a fallback too).
    Reports the min and max V_0 across tie-breaking selections."""
    rows = []
    for f in f_values:
        pr = Protocol(protocol.k, ForfeitRule(flat=float(f)), protocol.routing, protocol.fee,
                      protocol.publish_cost, protocol.q, protocol.fallback, protocol.sigma_policy,
                      protocol.challenge_prob, protocol.challenge_cost)
        sels = solve_all_selections(Scenario(market, trade, pr))
        vals = [s.value_user for _, s in sels]
        lo, hi = user_value_range(Scenario(market, trade, pr))
        rows.append({"f": float(f), "v0_min": min(vals + [lo]), "v0_max": max(vals + [hi]),
                     "ties": any(s.ties.any for _, s in sels),
                     "p_fallback_max": max(s.outcomes.fallback for _, s in sels),
                     "p_dropped_max": max(s.outcomes.dropped for _, s in sels)})
    return rows


def joint_surplus(sc: Scenario) -> float:
    """Maximum expected (user + producer) payoff over all joint strategies, on the lattice.

    This is what a coalition with enforceable side payments can achieve (transferable utility: any gain
    in the total can be split so both strictly gain). Fee transfers cancel inside the coalition; only
    money that leaves it matters: the publish cost c, the forfeit share routed elsewhere, and (with
    contestation) the fee a challenger collects. If sigma is optional the coalition also chooses whether
    to hand sigma over: with challengers around, keeping it can be strictly better for the coalition.
    Scales like the solver; ``tree.max_joint`` is the independent exact check for small k.
    """
    pr = sc.protocol
    if pr.fallback and pr.sigma_policy == "optional":
        return max(_joint(sc, True), _joint(sc, False))
    return _joint(sc, bool(pr.fallback))


def _joint(sc: Scenario, h: bool) -> float:
    k, p, q, pr = sc.protocol.k, sc.market.p, sc.protocol.q, sc.protocol
    drop = -pr.routing.to_other * sc.f
    if h:
        b = pr.beta_eff   # a challenger publishes for the user: total g - fee (the fee goes outside the coalition)
        gk = sc.g_at(k)
        Wk = np.maximum(gk - pr.publish_cost, b * (gk - pr.fee) + (1.0 - b) * drop)
    else:
        Wk = np.full(k + 1, drop)
    V = Wk
    for t in range(k - 1, 0, -1):
        C = p * V[1:] + (1.0 - p) * V[:-1]
        A = (1.0 - q) * sc.g_at(t) + q * C     # fee cancels: g - fee + fee
        V = np.maximum(A, C)
    return float(p * V[1] + (1.0 - p) * V[0])
