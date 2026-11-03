"""The exhaustive deviation checker.

Given a scenario it answers four questions, using the explicit game tree:

1. What are the subgame-perfect equilibrium payoffs (under each uniform tie-breaking selection)?
   Is committing purely for the option profitable (speculation not deterred)?
2. Is the *intended honest behaviour* (always try to reveal at the first block; hand over sigma;
   producer publishes whenever it holds sigma) an equilibrium? We report the best unilateral gain of
   each player and every subgame-perfection violation (one-shot deviations at every node).
3. Can a user-producer coalition with enforceable side payments do strictly better than every
   equilibrium selection? (Transferable utility: the coalition can reach any point on its joint
   frontier, so a profitable deviation exists iff the best joint total exceeds the equilibrium total.)
4. Did the two solution methods agree? Every best response computed by dynamic programming is
   re-derived by enumerating ALL pure strategies whenever the strategy space is small enough.

"Exploitable" is a precise, minimal statement: some player (or the coalition) has a strictly
profitable deviation from the protocol's intended outcome, beyond numerical tolerance.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .model import PayoffKind, Protocol, Scenario, Market, Trade, ForfeitRule
from .numerics import gt, lt
from . import tree as T
from .solver import honest_user_loss, solve_all_selections, user_value_range, USER_TIES, PRODUCER_TIES, SIGMA_TIES


@dataclass
class CheckReport:
    scenario: Scenario
    spe_payoffs: list                       # distinct (user, producer) payoffs across tie-breaking selections
    tie_keys: list                          # decision keys where some player is indifferent
    speculation_value_min: float | None     # V_0 across selections (None for TRANSFER trades)
    speculation_value_max: float | None
    honest_payoffs: tuple
    user_gain: float                        # best unilateral gain of the user vs the honest profile
    producer_gain: float
    honest_spe_violations: list
    joint_max: float
    best_spe_total: float
    coalition_gain: float
    brute_force_checked: list = field(default_factory=list)
    reasons: list = field(default_factory=list)

    @property
    def speculation_deterred(self) -> bool | None:
        if self.speculation_value_max is None:
            return None
        return not gt(self.speculation_value_max, 0.0)

    @property
    def fallback_credible(self) -> bool | None:
        if not self.scenario.protocol.fallback:
            return None
        return not any(v["key"][0] == "publish" for v in self.honest_spe_violations)

    @property
    def exploitable(self) -> bool:
        return bool(self.reasons)

    def summary(self) -> str:
        pr = self.scenario.protocol
        lines = [f"k={pr.k} f={self.scenario.f:g} routing={pr.routing.label} fee={pr.fee:g} "
                 f"c={pr.publish_cost:g} q={pr.q:g} fallback={pr.fallback} sigma={pr.sigma_policy}"]
        if self.speculation_value_max is not None:
            lines.append(f"  speculation value V0: [{self.speculation_value_min:.6g}, "
                         f"{self.speculation_value_max:.6g}]  deterred={self.speculation_deterred}")
        lines.append(f"  honest-profile gains: user={self.user_gain:.6g} producer={self.producer_gain:.6g}  "
                     f"SPE violations={len(self.honest_spe_violations)}")
        lines.append(f"  coalition: joint max={self.joint_max:.6g} vs best SPE total={self.best_spe_total:.6g} "
                     f"-> gain={self.coalition_gain:.6g}")
        lines.append(f"  brute-force cross-checks passed: {self.brute_force_checked or 'none (space too large)'}")
        lines.append("  VERDICT: " + ("EXPLOITABLE - " + "; ".join(self.reasons) if self.exploitable
                                      else "no profitable deviation found"))
        return "\n".join(lines)


def _honest_profile(game: T.Game) -> dict:
    prof = {}
    for key in game.decisions:
        prof[key] = {"reveal": "attempt", "publish": "publish", "sigma": "give", "include": "include"}[key[0]]
    return prof


def check(sc: Scenario, *, brute_force_max_strategies: int = 2 ** 15) -> CheckReport:
    game = T.build_game(sc)
    kind = sc.trade.kind

    # 1. equilibria under every tie-breaking selection
    spe, tie_keys, totals, seen = [], set(), [], set()
    for ut, pt, st in itertools.product(USER_TIES, PRODUCER_TIES, SIGMA_TIES):
        v, prof, ties = T.backward_induction(game, T.tie_rule(ut, pt, st))
        if T.verify_spe(game, prof):
            raise AssertionError("backward induction produced a profile that fails one-shot deviation check")
        key = (round(float(v[0]), 9), round(float(v[1]), 9))
        if key not in seen:
            seen.add(key)
            spe.append(key)
        tie_keys.update(ties)
        totals.append(float(v[0] + v[1]))
    v_users = [x[0] for x in spe]
    spec_min = spec_max = None
    if kind is not PayoffKind.TRANSFER:
        spec_min, spec_max = min(v_users), max(v_users)
        lo, hi = user_value_range(sc)                  # node-specific producer ties can give values the uniform selections miss
        spec_min, spec_max = min(spec_min, lo), max(spec_max, hi)

    # 2. the honest profile
    honest = _honest_profile(game)
    hp = T.expected_payoffs(game, honest)
    checked = []
    gains = {}
    for player, name in ((T.USER, "user"), (T.PRODUCER, "producer")):
        if not game.keys(player):
            gains[name] = 0.0
            continue
        br, _ = T.best_response(game, player, honest)
        n_strat = math.prod(len(game.decisions[k].actions) for k in game.keys(player))
        if n_strat <= brute_force_max_strategies:
            bf = T.brute_force_best_response(game, player, honest, max_strategies=brute_force_max_strategies)
            if abs(bf - br) > 1e-8 * max(1.0, abs(bf)):
                raise AssertionError(f"DP best response {br} != brute force {bf} for {name}")
            checked.append(name)
        gains[name] = br - float(hp[player])
    violations = T.verify_spe(game, honest)

    # 3. coalition
    joint, _ = T.max_joint(game)
    best_total = max(totals)
    coal = joint - best_total

    reasons = []
    if spec_max is not None and gt(spec_max, 0.0):
        reasons.append(f"speculation is profitable (V0 up to {spec_max:.6g} > 0 for a fairly priced trade)")
    if gt(gains["user"], 0.0):
        reasons.append(f"user gains {gains['user']:.6g} by deviating from honest behaviour")
    if gt(gains["producer"], 0.0):
        reasons.append(f"producer gains {gains['producer']:.6g} by deviating from honest behaviour")
    if gt(joint, best_total):
        reasons.append(f"user-producer coalition gains {coal:.6g} with side payments")

    def clean(x):   # snap floating-point dust to an exact zero so reports do not show 6e-15
        return 0.0 if abs(x) <= 1e-9 * max(1.0, abs(joint), abs(best_total)) else float(x)

    return CheckReport(sc, spe, sorted(tie_keys, key=repr), spec_min, spec_max,
                       (float(hp[0]), float(hp[1])), clean(gains["user"]), clean(gains["producer"]),
                       violations, float(joint), float(best_total), clean(coal), checked, reasons)


@dataclass
class DesignAssessment:
    """The three properties Q2 compares routing rules on, plus collusion, for one design."""
    deters_speculation: bool
    speculator_value: float          # worst case (max) over tie-breaking selections
    producer_willing: bool | None    # would the producer publish sigma (weakly)? None without fallback
    producer_strictly_prefers: bool | None
    collusion_gain: float
    collusion_proof: bool
    honest_loss: float               # worst case over selections
    report: CheckReport

    @property
    def all_good(self) -> bool:
        return (self.deters_speculation and (self.producer_willing is not False)
                and self.collusion_proof)


def assess_design(market: Market, protocol: Protocol, *, size: float = 1.0,
                  honest_value: float | None = None) -> DesignAssessment:
    rep = check(Scenario(market, Trade(PayoffKind.CALL, size=size), protocol))
    willing = strict = None
    if protocol.fallback:
        pub_p = protocol.fee - protocol.publish_cost
        drop_p = (1.0 - protocol.beta_eff) * protocol.routing.to_producer * protocol.forfeit(Trade(size=size))
        willing, strict = (not lt(pub_p, drop_p)), bool(gt(pub_p, drop_p))
    hv = honest_value if honest_value is not None else protocol.fee + 10.0 * market.S0 * market.sigma
    losses = []
    for _, kw in [(0, dict(user_tie=a, producer_tie=b, sigma_tie=c))
                  for a, b, c in itertools.product(USER_TIES, PRODUCER_TIES, SIGMA_TIES)]:
        losses.append(honest_user_loss(market, protocol, value=hv, **kw)["loss"])
    return DesignAssessment(bool(rep.speculation_deterred), float(rep.speculation_value_max), willing, strict,
                            rep.coalition_gain, not gt(rep.joint_max, rep.best_spe_total),
                            float(max(losses)), rep)
