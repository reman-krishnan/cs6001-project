"""Q4: contesting a disputed index (design document, Section 8.4).

Situation. After a reorg two bindings verify at one index. Rule C says a contested index can only be resolved by the signature
sigma, which only the OWNER holds, so a rebinding ATTACKER cannot win a resolved contest. The design document leaves two ways to
raise and settle the contest open:
  * BOND: the contester posts a bond; the loser pays it ("race-free, but requires bonded contesters");
  * RACE: contest by exhibiting h_v, "self-authenticating, keyless, but needs a precedence rule that creates a race".
Our formalization (ours, not the document's; every assumption is a parameter):
  Players: attacker A (value V_a if its binding stands, cost kappa to attack) and owner O (loses V_o if it does).
  The owner is present within the contest window with probability a (availability of the owner or a watchtower).
  BOND  : A attacks and posts bond b. If O is present and can post the same bond (capital K >= b), O responds with sigma at cost
          c_r and wins; A loses b to O. Otherwise A's binding stands. Solved as an extensive-form game (SPNE) with the
          course's game-tree engine, and checked against exhaustive profile enumeration.
  RACE  : if O is present, both exhibit h_v and the earlier inclusion wins; priority fees are paid by both (all-pay), a
          complete-information all-pay contest with a mixed equilibrium (Baye-Kovenock-de Vries). If O is absent A wins unopposed.
Payoffs are measured against the no-attack baseline (0 for both).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import tree as T
from .numerics import TOL as NUM_TOL, gt, lt


# ------------------------------------------------------------------ all-pay contest (the race) -------------------------
def allpay_two_player(V1: float, V2: float) -> dict:
    """Complete-information all-pay contest: player i values winning at V_i and pays its bid whether or not it wins.
    The (unique) equilibrium has the stronger player (V_s >= V_w) mix uniformly on [0, V_w] and the weaker mix with an atom at
    0 of mass 1 - V_w/V_s and density 1/V_s on (0, V_w]. Equilibrium payoffs: V_s - V_w for the stronger, 0 for the weaker."""
    if V1 <= 0 or V2 <= 0:
        raise ValueError("values must be > 0")
    Vs, Vw = max(V1, V2), min(V1, V2)
    out = dict(V_strong=Vs, V_weak=Vw, payoff_strong=Vs - Vw, payoff_weak=0.0,
               exp_bid_strong=Vw / 2.0, exp_bid_weak=Vw ** 2 / (2.0 * Vs), p_strong_wins=1.0 - Vw / (2.0 * Vs))
    out["total_waste"] = out["exp_bid_strong"] + out["exp_bid_weak"]
    return out


def verify_allpay_equilibrium(V1: float, V2: float, n: int = 2001, tol: float = 1e-9) -> dict:
    """Numerically check the mixed equilibrium of ``allpay_two_player``: against the opponent's mixed strategy every bid in the
    support must earn exactly the equilibrium payoff and no bid above V_w may earn more (the indifference conditions)."""
    Vs, Vw = max(V1, V2), min(V1, V2)
    eq = allpay_two_player(V1, V2)
    x = np.linspace(1e-9, Vw, n)
    F_weak = (Vs - Vw) / Vs + x / Vs          # CDF of the weaker player's bid (atom at 0 included)
    F_strong = x / Vw                         # CDF of the stronger player's bid
    pay_strong = Vs * F_weak - x              # stronger's payoff from bidding x in the support
    pay_weak = Vw * F_strong - x
    above = np.linspace(Vw, 3 * Vw, 400)[1:]
    dev_strong = Vs * 1.0 - above             # any bid above V_w wins for sure against the weaker player
    dev_weak = Vw * np.ones_like(above) - above
    return dict(max_dev_in_support=float(max(np.abs(pay_strong - eq["payoff_strong"]).max(), np.abs(pay_weak - eq["payoff_weak"]).max())),
                best_deviation_above_strong=float(dev_strong.max() - eq["payoff_strong"]),
                best_deviation_above_weak=float(dev_weak.max() - eq["payoff_weak"]),
                is_equilibrium=bool(np.abs(pay_strong - eq["payoff_strong"]).max() < 1e-6 and np.abs(pay_weak).max() < 1e-6
                                    and dev_strong.max() <= eq["payoff_strong"] + tol and dev_weak.max() <= tol))


def race_outcome(V_owner, V_attacker, attack_cost, availability) -> dict:
    """Subgame-perfect outcome of the race mechanism: the attacker attacks iff its expected payoff is positive (an attack that
    earns exactly zero is reported as a tie)."""
    a = availability
    contested = allpay_two_player(V_owner, V_attacker)
    att_if_present = max(V_attacker - V_owner, 0.0)                       # attacker's equilibrium payoff in the contest
    own_if_present = -V_owner + max(V_owner - V_attacker, 0.0)
    att = a * att_if_present + (1 - a) * V_attacker - attack_cost
    own = a * own_if_present + (1 - a) * (-V_owner)
    attacks = bool(gt(att, 0.0))
    return dict(mechanism="race", attacker_payoff_if_attack=att, owner_payoff_if_attacked=own, attacks=attacks,
                tie=bool(not gt(att, 0.0) and not lt(att, 0.0)),
                waste_if_attacked=a * contested["total_waste"] + attack_cost,
                owner_payoff=own if attacks else 0.0, attacker_payoff=att if attacks else 0.0,
                deterred=not attacks, deterrence_threshold_cost=(1 - a) * V_attacker + a * att_if_present)


# ------------------------------------------------------------------ bonded contest (extensive form) ---------------------
def build_bond_game(V_owner, V_attacker, attack_cost, availability, bond, response_cost, owner_capital=math.inf) -> T.Game:
    """Extensive-form bonded contest. Player 0 = attacker, player 1 = owner. Chance decides whether the owner is present."""
    can_bond = bond <= owner_capital
    p_present = availability if can_bond else 0.0
    decisions = {}
    own = T.Decision(1, ("include", 0, "owner"), {
        "respond": T.Terminal((-attack_cost - bond, -response_cost + bond)),
        "ignore": T.Terminal((V_attacker - attack_cost, -V_owner))})
    decisions[own.key] = own
    absent = T.Terminal((V_attacker - attack_cost, -V_owner))
    root = T.Decision(0, ("reveal", 0, "attacker"), {          # 'abstain' first: a tie resolves to NOT attacking
        "abstain": T.Terminal((0.0, 0.0)),
        "attack": T.Chance([(p_present, own), (1 - p_present, absent)])})
    decisions[root.key] = root
    return T.Game(root, decisions, None)


def bond_outcome(V_owner, V_attacker, attack_cost, availability, bond, response_cost=0.0, owner_capital=math.inf) -> dict:
    g = build_bond_game(V_owner, V_attacker, attack_cost, availability, bond, response_cost, owner_capital)
    v, prof, ties = T.backward_induction(g, T.tie_rule("wait", "publish", "give", "censor"))   # tie: attacker abstains
    attacks = prof[("reveal", 0, "attacker")] == "attack"
    if T.verify_spe(g, prof):
        raise AssertionError("backward induction profile fails the subgame-perfection check")
    enumerated = T.enumerate_spe_payoffs(g)                                                     # exhaustive cross-check
    if (round(float(v[0]), 9), round(float(v[1]), 9)) not in enumerated:
        raise AssertionError("bond game: backward-induction payoffs not found among enumerated SPNE")
    p_present = availability if bond <= owner_capital else 0.0
    responds = prof[("include", 0, "owner")] == "respond"      # the owner's equilibrium choice, which is not always to respond
    if responds:
        att = p_present * (-attack_cost - bond) + (1 - p_present) * (V_attacker - attack_cost)
        own = p_present * (-response_cost + bond) + (1 - p_present) * (-V_owner)
        waste = attack_cost + p_present * response_cost
    else:                                                       # a present owner who ignores behaves like an absent one
        att, own, waste = V_attacker - attack_cost, -V_owner, attack_cost
    return dict(mechanism="bond", bond=bond, attacker_payoff_if_attack=att, owner_payoff_if_attacked=own, attacks=bool(attacks),
                owner_responds=bool(responds), tie=bool(ties), waste_if_attacked=waste,
                owner_payoff=float(v[1]), attacker_payoff=float(v[0]), deterred=not attacks)


def min_deterring_bond(V_attacker, attack_cost, availability, owner_capital=math.inf) -> float:
    """Smallest bond b with attacker payoff <= 0: (1-a) V_a - kappa <= a b, i.e. b* = ((1-a) V_a - kappa) / a. Returns 0 if
    the attack is unprofitable without a bond, inf if no bond within the owner's capital works (or the owner is never present)."""
    need = (1 - availability) * V_attacker - attack_cost
    if need <= 0:
        return 0.0
    if availability <= 0:
        return math.inf
    b = need / availability
    return b if b <= owner_capital else math.inf


def min_feasible_bond(V_owner, V_attacker, attack_cost, availability, response_cost=0.0, owner_capital=math.inf) -> float:
    """Smallest bond that both deters the attacker and is answered by a present owner, or inf if none fits the owner's capital.

    Deterrence (the owner answers): a (-kappa - b) + (1 - a)(V_a - kappa) <= 0, i.e. b >= b_min = max(0, ((1-a) V_a - kappa) / a).
    The owner answers when V_o + b > c_r, i.e. b > c_r - V_o (strictly). Both need b <= K, so a feasible bond exists iff
    K >= b_min and V_o + K > c_r, and the smallest one is max(b_min, a margin of two solver tolerances above c_r - V_o), or K itself when only the margin overshoots K. Strictly the answer condition has an infimum and no minimum, so this is a numerically selected bond."""
    need = (1 - availability) * V_attacker - attack_cost
    if need > 0 and availability <= 0:
        return math.inf
    b_min = 0.0 if need <= 0 else need / availability
    r = response_cost - V_owner
    r_strict = r + 2 * NUM_TOL * max(1.0, abs(r), abs(V_owner), abs(response_cost))   # strictly above c_r - V_o in the solver's sense (numerics.TOL)
    b = max(b_min, r_strict)
    if b > owner_capital and owner_capital >= b_min and gt(V_owner + owner_capital, response_cost):
        b = owner_capital                                     # the margin overshoots K, yet K itself deters and is answered
    return b if b <= owner_capital else math.inf


def compare_mechanisms(V_owner=1.0, attack_cost=0.01, response_cost=0.002, owner_capital=1.0,
                       availabilities=(0.5, 0.8, 0.9, 0.99), value_ratios=(0.5, 1.0, 2.0)) -> pd.DataFrame:
    """Race vs bonded contest across owner availability a and the attacker's relative valuation V_a / V_o. For the bond the
    designer picks the smallest deterring bond within the owner's capital (if one exists)."""
    rows = []
    for a in availabilities:
        for r in value_ratios:
            Va = r * V_owner
            race = race_outcome(V_owner, Va, attack_cost, a)
            b_star = min_feasible_bond(V_owner, Va, attack_cost, a, response_cost, owner_capital)
            if math.isfinite(b_star):
                bond = bond_outcome(V_owner, Va, attack_cost, a, min(owner_capital, b_star * (1 + 1e-6) + 1e-9), response_cost, owner_capital)
            else:
                bond = bond_outcome(V_owner, Va, attack_cost, a, owner_capital if math.isfinite(owner_capital) else 1e6, response_cost, owner_capital)
            rows.append(dict(V_owner=V_owner, attack_cost=attack_cost, response_cost=response_cost, owner_capital=owner_capital,
                             availability=a, value_ratio=r, race_deterred=race["deterred"], race_tie=race["tie"], race_owner_loss_if_attacked=-race["owner_payoff_if_attacked"],
                             race_attacker_gain_if_attack=race["attacker_payoff_if_attack"],
                             race_min_attack_cost_to_deter=race["deterrence_threshold_cost"], bond_star=b_star,
                             bond_feasible=bool(math.isfinite(b_star)), bond_deterred=bond["deterred"],
                             bond_owner_loss_if_attacked=-bond["owner_payoff_if_attacked"]))
    return pd.DataFrame(rows)
