"""Explicit extensive-form game and a generic solution engine (independent of ``solver.py``).

The game is a DAG rather than a tree: a censored attempt and a "wait" lead to the *same* continuation
node, because the continuation game is identical. Decision nodes are keyed by the price history (a
string of 'u'/'d'), so strategies may depend on the whole path, NOT only on the lattice state (t, j).
That makes agreement with the Markov lattice solver a genuine test, not an assumption.

Players: 0 = user, 1 = producer. Payoff vectors are (user, producer).
Decision-node keys: ('reveal', h, path) | ('publish', h, path) | ('sigma', 0, '')  where h in {0,1}
says whether the producer holds sigma in that subtree.

Size is exponential in k (about 2^k nodes). Intended for k <= ~14.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from .model import Scenario
from .numerics import close, gt, lt

USER, PRODUCER = 0, 1
Profile = Mapping[tuple, str]


@dataclass(eq=False)
class Terminal:
    payoff: tuple


@dataclass(eq=False)
class Chance:
    branches: list  # [(probability, node), ...]


@dataclass(eq=False)
class Decision:
    player: int
    key: tuple
    actions: dict  # action name -> node


@dataclass
class Game:
    root: object
    decisions: dict  # key -> Decision (unique)
    scenario: Scenario

    def keys(self, player: int) -> list:
        return sorted((k for k, d in self.decisions.items() if d.player == player), key=repr)


def build_game(sc: Scenario, strategic: bool = False) -> Game:
    k, pr = sc.protocol.k, sc.protocol
    p, q, S0, sigma = sc.market.p, pr.q, sc.market.S0, sc.market.sigma
    if k > 16:
        raise ValueError("tree engine is exponential in k; refusing k > 16 (use solver.py)")
    if strategic and (q != 0.0 or pr.beta_eff != 0.0):
        raise ValueError("strategic censoring requires q = 0 and no contestation")
    decisions: dict = {}

    def gval(path: str) -> float:
        S = S0 * math.exp(sigma * (path.count("u") - path.count("d")))
        return float(sc.trade.g(np.array([S]), S0)[0])

    def register(node: Decision) -> Decision:
        assert node.key not in decisions, f"duplicate decision key {node.key}"
        decisions[node.key] = node
        return node

    def build(h: int) -> object:
        block_memo: dict = {}
        cont_memo: dict = {}

        def cont(path: str):
            if path not in cont_memo:
                cont_memo[path] = Chance([(p, block(path + "u")), (1.0 - p, block(path + "d"))])
            return cont_memo[path]

        def block(path: str):
            if path in block_memo:
                return block_memo[path]
            t = len(path)
            g = gval(path)
            if t == k:
                drop = Terminal(tuple(sc.drop_payoff()))
                if h:
                    pub = Terminal(tuple(float(x) for x in sc.publish_payoff(g)))
                    b = pr.beta_eff
                    wh = drop if b == 0.0 else Chance([(b, Terminal((g - pr.fee, 0.0))), (1.0 - b, drop)])
                    node = register(Decision(PRODUCER, ("publish", h, path), {"publish": pub, "withhold": wh}))
                else:
                    node = drop
            else:
                done = Terminal(tuple(float(x) for x in sc.reveal_payoff(g)))
                nxt = cont(path)  # shared by 'wait' and by the censored branch of 'attempt'
                if strategic:    # the (single) producer decides whether to include the attempt
                    att = register(Decision(PRODUCER, ("include", h, path), {"include": done, "censor": nxt}))
                else:
                    att = Chance([(1.0 - q, done), (q, nxt)])
                node = register(Decision(USER, ("reveal", h, path), {"attempt": att, "wait": nxt}))
            block_memo[path] = node
            return node

        return cont("")

    if pr.fallback and pr.sigma_policy == "optional":
        root = register(Decision(USER, ("sigma", 0, ""), {"give": build(1), "keep": build(0)}))
    else:
        root = build(1 if pr.fallback else 0)
    return Game(root, decisions, sc)


def game_from_root(root) -> Game:
    """Wrap ANY perfect-information game (Terminal / Chance / Decision nodes with unique decision keys) as a Game, so the
    engine can be validated on textbook games such as the Centipede game."""
    decisions, seen = {}, set()

    def walk(n):
        if id(n) in seen:
            return
        seen.add(id(n))
        if isinstance(n, Chance):
            for _, ch in n.branches:
                walk(ch)
        elif isinstance(n, Decision):
            assert n.key not in decisions, f"duplicate key {n.key}"
            decisions[n.key] = n
            for ch in n.actions.values():
                walk(ch)

    walk(root)
    return Game(root, decisions, None)


# ---------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------

def _eval(node, profile: Profile, memo: dict) -> np.ndarray:
    nid = id(node)
    if nid in memo:
        return memo[nid]
    if isinstance(node, Terminal):
        v = np.array(node.payoff, dtype=float)
    elif isinstance(node, Chance):
        v = sum(pr * _eval(ch, profile, memo) for pr, ch in node.branches)
    else:
        if node.key not in profile:
            raise KeyError(f"profile has no action for decision {node.key}")
        v = _eval(node.actions[profile[node.key]], profile, memo)
    memo[nid] = v
    return v


def expected_payoffs(game: Game, profile: Profile) -> np.ndarray:
    """Expected (user, producer) payoffs. The profile only needs to cover the nodes it reaches."""
    return _eval(game.root, profile, {})


def node_values(game: Game, profile: Profile) -> dict:
    """Continuation payoffs at every node; requires a complete profile."""
    memo: dict = {}
    for d in game.decisions.values():
        if d.key not in profile:
            raise KeyError(f"incomplete profile: missing {d.key}")
    _eval(game.root, profile, memo)
    for d in game.decisions.values():        # also cover every branch NOT chosen by the profile,
        for ch in d.actions.values():        # which one-shot deviation checks need
            _eval(ch, profile, memo)
        _eval(d, profile, memo)
    return memo


TieRule = Callable[[tuple, list], str]


def tie_rule(user_tie: str = "attempt", producer_tie: str = "publish", sigma_tie: str = "give",
             include_tie: str = "include") -> TieRule:
    pref = {"include": ("include", "censor") if include_tie == "include" else ("censor", "include"),
            "reveal": ("attempt", "wait") if user_tie == "attempt" else ("wait", "attempt"),
            "publish": ("publish", "withhold") if producer_tie == "publish" else ("withhold", "publish"),
            "sigma": ("give", "keep") if sigma_tie == "give" else ("keep", "give")}

    def rule(key, candidates):
        for a in pref.get(key[0], ()):
            if a in candidates:
                return a
        return candidates[0]
    return rule


def backward_induction(game: Game, rule: TieRule | None = None):
    """One subgame-perfect equilibrium, selected by ``rule``. Returns (payoffs, complete profile, tied keys)."""
    rule = rule or tie_rule()
    memo: dict = {}
    profile: dict = {}
    ties: list = []

    def val(node):
        nid = id(node)
        if nid in memo:
            return memo[nid]
        if isinstance(node, Terminal):
            v = np.array(node.payoff, dtype=float)
        elif isinstance(node, Chance):
            v = sum(pr * val(ch) for pr, ch in node.branches)
        else:
            vals = {a: val(ch) for a, ch in node.actions.items()}
            best = max(v[node.player] for v in vals.values())
            cands = [a for a, v in vals.items() if not gt(best, v[node.player])]
            if len(cands) > 1:
                ties.append(node.key)
            choice = rule(node.key, cands)
            profile[node.key] = choice
            v = vals[choice]
        memo[nid] = v
        return v

    v = val(game.root)
    for d in game.decisions.values():   # force off-path nodes into the profile as well
        val(d)
    return v, profile, ties


def best_response(game: Game, player: int, profile: Profile):
    """Exact best-response value for ``player`` (maximizing at all of its nodes) with the other player's
    actions fixed by ``profile``. Returns (value, argmax strategy dict)."""
    memo: dict = {}
    strat: dict = {}

    def val(node) -> float:
        nid = id(node)
        if nid in memo:
            return memo[nid]
        if isinstance(node, Terminal):
            v = node.payoff[player]
        elif isinstance(node, Chance):
            v = sum(pr * val(ch) for pr, ch in node.branches)
        elif node.player == player:
            vals = {a: val(ch) for a, ch in node.actions.items()}
            a_star = max(vals, key=lambda a: vals[a])
            strat[node.key] = a_star
            v = vals[a_star]
        else:
            v = val(node.actions[profile[node.key]])
        memo[nid] = v
        return v

    v = val(game.root)
    for d in game.decisions.values():
        val(d)
    return float(v), strat


def max_joint(game: Game):
    """Maximum of (user + producer) expected payoff over all joint strategies: what a coalition with
    enforceable side payments can achieve. Returns (value, argmax joint strategy)."""
    memo: dict = {}
    strat: dict = {}

    def val(node) -> float:
        nid = id(node)
        if nid in memo:
            return memo[nid]
        if isinstance(node, Terminal):
            v = node.payoff[0] + node.payoff[1]
        elif isinstance(node, Chance):
            v = sum(pr * val(ch) for pr, ch in node.branches)
        else:
            vals = {a: val(ch) for a, ch in node.actions.items()}
            a_star = max(vals, key=lambda a: vals[a])
            strat[node.key] = a_star
            v = vals[a_star]
        memo[nid] = v
        return v

    v = val(game.root)
    for d in game.decisions.values():
        val(d)
    return float(v), strat


# ---------------------------------------------------------------------------------------------
# Verification and exhaustive enumeration
# ---------------------------------------------------------------------------------------------

def verify_spe(game: Game, profile: Profile) -> list[dict]:
    """One-shot-deviation check at EVERY decision node (on and off the path). In a finite game a profile
    is subgame perfect iff this returns []. Each violation: {key, chosen, better, gain}."""
    memo = node_values(game, profile)
    bad = []
    for d in game.decisions.values():
        chosen = profile[d.key]
        v_chosen = memo[id(d.actions[chosen])][d.player]
        for a, ch in d.actions.items():
            gain = float(memo[id(ch)][d.player] - v_chosen)
            if gt(memo[id(ch)][d.player], v_chosen):
                bad.append({"key": d.key, "chosen": chosen, "better": a, "gain": gain})
    return bad


def _strategy_space(game: Game, player: int):
    keys = game.keys(player)
    return keys, [list(game.decisions[k].actions) for k in keys]


def brute_force_best_response(game: Game, player: int, profile: Profile, max_strategies: int = 2 ** 18):
    """Enumerate EVERY pure strategy of ``player`` (all nodes) against the others fixed by ``profile``.
    Returns the best expected payoff. Exponential: guarded by ``max_strategies``."""
    keys, spaces = _strategy_space(game, player)
    n = math.prod(len(s) for s in spaces) if spaces else 1
    if n > max_strategies:
        raise ValueError(f"{n} pure strategies exceeds max_strategies={max_strategies}")
    best = -math.inf
    for combo in itertools.product(*spaces):
        prof = dict(profile)
        prof.update(zip(keys, combo))
        best = max(best, float(expected_payoffs(game, prof)[player]))
    return best


def enumerate_spe_payoffs(game: Game, max_profiles: int = 2 ** 17):
    """Exact set of subgame-perfect equilibrium payoff vectors: enumerate EVERY pure profile of both
    players and keep those passing ``verify_spe``. Only for very small games."""
    keys = sorted(game.decisions, key=repr)
    spaces = [list(game.decisions[k].actions) for k in keys]
    n = math.prod(len(s) for s in spaces)
    if n > max_profiles:
        raise ValueError(f"{n} pure profiles exceeds max_profiles={max_profiles}")
    found = {}
    for combo in itertools.product(*spaces):
        prof = dict(zip(keys, combo))
        if not verify_spe(game, prof):
            v = expected_payoffs(game, prof)
            found[(round(float(v[0]), 9), round(float(v[1]), 9))] = dict(prof)
    return found
