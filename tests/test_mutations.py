"""Mutation tests: deliberately injected bugs must be caught by the independent checks.

Each mutant rewrites one line of the source of one function (an arithmetic slip, a swapped comparison, a dropped term), installs
the rewritten function for the duration of one test, and runs a battery of independent checks: the hand-computed game, the lattice
solver against the explicit game tree (base, strategic and rotating models, every uniform tie-breaking rule), the coalition optimum
against the tree, and the bond formula against the extensive-form bond game. A mutant counts as caught if any check fails or raises.
The unmutated code must pass every check, otherwise the battery would catch "mutants" for the wrong reason.
"""
import inspect
import itertools
import math
import random
import textwrap

import pytest

from conftest import make, random_scenario
from dagame import build_game, solve
from dagame import contest as Cn
from dagame import model as Mo
from dagame import rotating as Ro
from dagame import solver as So
from dagame import strategic as St
from dagame import tree as T
from dagame.model import Market, Routing, Scenario, Trade, Protocol, ForfeitRule
from dagame.solver import PRODUCER_TIES, SIGMA_TIES, USER_TIES, joint_surplus

TOL = dict(abs=1e-8, rel=1e-9)


def _scenarios(seed, n, kmax, pred=lambda sc: True):
    rng, out = random.Random(seed), []
    while len(out) < n:
        sc = random_scenario(rng, kmax=kmax)
        if pred(sc):
            out.append(sc)
    return out


BASE = _scenarios(11, 30, 5)
STRAT = _scenarios(12, 15, 5, lambda sc: sc.protocol.q == 0.0 and sc.protocol.beta_eff == 0.0)
ROT = _scenarios(13, 10, 5, lambda sc: sc.protocol.q == 0.0 and sc.protocol.beta_eff == 0.0 and sc.protocol.sigma_policy == "required")


def _close(a, b):
    return a == pytest.approx(b, **TOL)


def check_battery():
    """Names of the independent checks that fail. Empty list = all pass."""
    bad = []
    # 1. hand-computed game: u = 2, d = 1/2, k = 2, fee 1, burned forfeit 10 -> V0 = 79/3
    hand = Scenario(Market(100, math.log(2)), Trade(size=1), Protocol(2, ForfeitRule(flat=10), Routing.burn(), fee=1))
    if not _close(solve(hand).value_user, 79 / 3):
        bad.append("hand game")
    # 2. lattice solver against the explicit game tree under every uniform selection, and the coalition optimum
    for sc in BASE:
        game = build_game(sc)
        for ut, pt, st in itertools.product(USER_TIES, PRODUCER_TIES, SIGMA_TIES):
            sol = solve(sc, user_tie=ut, producer_tie=pt, sigma_tie=st)
            v, _, _ = T.backward_induction(game, T.tie_rule(ut, pt, st))
            if not (_close(sol.value_user, v[0]) and _close(sol.value_producer, v[1])):
                bad.append("solver vs tree")
                break
        if not _close(joint_surplus(sc), T.max_joint(game)[0]):
            bad.append("coalition vs tree")
    # 3. strategic producer against its tree
    for sc in STRAT:
        game = build_game(sc, strategic=True)
        for ut, pt, st, it in itertools.product(USER_TIES, PRODUCER_TIES, SIGMA_TIES, St.INCLUDE_TIES):
            sol = St.solve_strategic(sc, user_tie=ut, producer_tie=pt, sigma_tie=st, include_tie=it)
            v, _, _ = T.backward_induction(game, T.tie_rule(ut, pt, st, it))
            if not (_close(sol.value_user, v[0]) and _close(sol.value_producer, v[1])):
                bad.append("strategic vs tree")
                break
    # 4. rotating producers against their tree
    for sc in ROT:
        for alpha in (0.3, 1.0):
            game = Ro.build_game_rotating(sc, alpha)
            for ut, et, st in itertools.product(USER_TIES, Ro.INCLUDE_TIES, PRODUCER_TIES):
                sol = Ro.solve_rotating(sc, alpha, user_tie=ut, entity_tie=et, settle_tie=st)
                v, _, _ = T.backward_induction(game, T.tie_rule(ut, st, "give", et))
                if not (_close(sol.value_user, v[0]) and _close(sol.value_entity, v[1])):
                    bad.append("rotating vs tree")
                    break
    # 5. the smallest deterring bond deters in the extensive-form bond game, and 1% less does not
    for a, Va in ((0.5, 0.5), (0.8, 2.0), (0.9, 1.0)):
        b = Cn.min_deterring_bond(Va, 0.01, a)
        if not Cn.bond_outcome(1.0, Va, 0.01, a, b * (1 + 1e-6) + 1e-9).get("deterred") or \
                Cn.bond_outcome(1.0, Va, 0.01, a, 0.99 * b)["deterred"]:
            bad.append("bond formula vs bond game")
    return sorted(set(bad))


# (module, attribute path, exact source text, replacement): one injected bug each
MUTANTS = [
    (So, "_backward", "C = p * Vn[1:] + (1.0 - p) * Vn[:-1]", "C = (1.0 - p) * Vn[1:] + p * Vn[:-1]"),
    (So, "_backward", "A = (1.0 - q) * (sc.g_at(t) - fee) + q * C", "A = (1.0 - q) * sc.g_at(t) + q * C"),
    (So, "_backward", 'att = np.where(tie, user_tie == "attempt", gt(A, C))', 'att = np.where(tie, user_tie == "attempt", gt(C, A))'),
    (So, "_settlement", "if gt(pub_p, wh_p):", "if gt(wh_p, pub_p):"),
    (So, "_joint", "drop = -pr.routing.to_other * sc.f", "drop = -(1.0 - pr.routing.to_user) * sc.f"),
    (So, "_joint", "A = (1.0 - q) * sc.g_at(t) + q * C", "A = sc.g_at(t)"),
    (Mo, "Scenario.drop_payoff", "return (-(1.0 - r.to_user) * f, r.to_producer * f)", "return (-(r.to_user) * f, r.to_producer * f)"),
    (Mo, "Market.p", "p = math.expm1(self.drift + self.sigma) / math.expm1(2.0 * self.sigma)", "p = 0.5"),
    (St, "_bw", 'inc = np.where(tie_i, include_tie == "include", gt(fee, Cp))', 'inc = np.where(tie_i, include_tie == "include", gt(Cp, fee))'),
    (Ro, "_bw", "Au = (1 - alpha) * (g - fee) + alpha * np.where(inc, g - fee, Cu)", "Au = alpha * (g - fee) + (1 - alpha) * np.where(inc, g - fee, Cu)"),
    (Ro, "_bw", "Ek = np.full(k + 1, alpha * (pub_e if ent_pub else wh_e))", "Ek = np.full(k + 1, pub_e if ent_pub else wh_e)"),
    (Cn, "min_deterring_bond", "b = need / availability", "b = need"),
]


def _install(monkeypatch, module, path, old, new):
    owner_name, _, attr = path.rpartition(".")
    owner = getattr(module, owner_name) if owner_name else module
    obj = owner.__dict__[attr]
    fn = obj.fget if isinstance(obj, property) else obj
    src = textwrap.dedent(inspect.getsource(fn))
    assert src.count(old) == 1, f"mutant source text not found exactly once in {path}"
    ns = dict(vars(module))
    exec(compile(src.replace(old, new), f"<mutant {path}>", "exec"), ns)
    monkeypatch.setattr(owner, attr, ns[attr])


def test_unmutated_code_passes_every_check():
    assert check_battery() == []


@pytest.mark.parametrize("module,path,old,new", MUTANTS, ids=[f"{m.__name__.split('.')[-1]}.{p}#{i}" for i, (m, p, _, _) in enumerate(MUTANTS)])
def test_injected_bug_is_caught(monkeypatch, module, path, old, new):
    _install(monkeypatch, module, path, old, new)
    try:
        failed = check_battery()
    except Exception as e:          # a crash or a failed internal self-audit also counts as detection
        failed = [f"raised {type(e).__name__}"]
    assert failed, f"injected bug in {path} was NOT caught: {new!r}"
