"""Verification facts and illustrative results: extensive-form games, backward induction / SPNE, best response,
exhaustive deviation checking, normal form vs SPNE, quasi-linear payments, Bayesian designer.

    python scripts/run_course_core.py            # ~1 minute, writes ./results/illustrative (illustrative parameters)
"""
import math
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import numpy as np
import pandas as pd

from dagame import Market, Trade, ForfeitRule, Routing, Protocol, Scenario, solve, check, build_game
from dagame import course_results as C
from dagame import tree as T
from dagame.normal_form import nash_vs_spe
from dagame.observations import check_observation_1, check_observation_2, check_observation_3

import json
V = {}          # facts saved to verification.json for the report
OUT = os.path.join(ROOT, "results", "illustrative")   # separate folder: its parameters are illustrative, not measured
os.makedirs(OUT, exist_ok=True)


def banner(s):
    print("\n" + "=" * 96 + f"\n{s}\n" + "=" * 96)


banner("1. Sanity check: the engine on the CENTIPEDE game (textbook SPNE = stop at once, payoffs (1, 0))")
from test_normal_form import centipede
g = centipede()
v, prof, _ = T.backward_induction(g)
print("backward induction payoffs:", v.tolist(), " SPNE profile:", prof)
r = nash_vs_spe(g)
print(f"strategic form {r['shape']}: {r['n_nash']} pure Nash equilibria, of which {r['n_spe']} subgame perfect")
V['centipede'] = dict(nash=r['n_nash'], spne=r['n_spe'], payoff=v.tolist())

banner("2. Backward induction on the protocol game (hand check: u=2, d=1/2, k=2, fee 1, forfeit 10 -> V0 = 79/3)")
sc = Scenario(Market(100, math.log(2)), Trade(size=1), Protocol(2, ForfeitRule(flat=10), Routing.burn(), fee=1))
print(f"solver V0 = {solve(sc).value_user:.10f}   79/3 = {79 / 3:.10f}")
V['hand_v0'] = dict(solver=solve(sc).value_user, exact=79 / 3)

banner("3. Exhaustive deviation checker (project-document style): plain commit-reveal is manipulable")
plain = Scenario(Market(100, 0.1), Trade(size=1), Protocol(4, ForfeitRule(), Routing.burn(), fee=0.5))
rep_ = check(plain)
print(rep_.summary())
V['plain_commit_reveal'] = dict(exploitable=bool(rep_.exploitable), user_gain=float(rep_.user_gain), spec_value=float(rep_.speculation_value_max), violations=len(rep_.honest_spe_violations))

banner("4. Nash equilibria vs subgame-perfect equilibria: the producer's non-credible threat")
sc = Scenario(Market(100, 0.1), Trade(size=1), Protocol(2, ForfeitRule(flat=50.0), Routing.burn(), 1.0, 0.2, fallback=True))
r = nash_vs_spe(build_game(sc))
V['protocol_burn'] = dict(nash=r['n_nash'], spne=r['n_spe'], noncredible=len(r['non_credible']))
print(f"burn routing, forfeit 50: {r['n_nash']} pure Nash equilibria, {r['n_spe']} subgame perfect, "
      f"{len(r['non_credible'])} rest on non-credible threats")
ex = r["non_credible"][0]
print("  a non-credible-threat equilibrium pays (user, producer) =", tuple(round(x, 3) for x in ex[0]),
      "and has the producer play 'withhold' at:", [k[2] for k, a in ex[2].items() if k[0] == "publish" and a == "withhold"][:4], "...")
sc = Scenario(Market(100, 0.1), Trade(size=1), Protocol(2, ForfeitRule(flat=50.0), Routing.producer(), 1.0, 0.2, fallback=True))
r = nash_vs_spe(build_game(sc))
V['protocol_producer'] = dict(nash=r['n_nash'], spne=r['n_spe'])
print(f"producer routing, forfeit 50: withholding now pays the producer, so the threat is credible: "
      f"{r['n_nash']} Nash, {r['n_spe']} subgame perfect")

banner("5. Designs compared by the speculator's equilibrium value (ILLUSTRATIVE parameters: 2% window volatility; no coalitions)")
t = C.designs_table()
t.to_csv(os.path.join(OUT, "designs_table.csv"), index=False)
print(t.round(3).to_string(index=False))

banner("6. Observations 1-3 against closed forms")
o1 = check_observation_1(Market(100, 0.05), Trade(), 10, 1.0, [0, 1, 100, 1e6])
print("Obs 1 (self routing inert):", [(x["f"], round(x["v0"], 4)) for x in o1])
V['obs1'] = dict(independent_of_f=all(x['independent_of_f'] for x in o1), value=o1[0]['v0'])
o2 = check_observation_2(Market(100, 0.05), Trade(size=2.0), 20, 0.7, np.linspace(0, 400, 9))
V['obs2'] = dict(all_match=all(x['match_call'] and x['match_put'] for x in o2), rows=len(o2), max_abs_diff=max(max(abs(x['v0_solver'] - x['call_form']), abs(x['v0_solver'] - x['put_form'])) for x in o2))
print("Obs 2: solver = call form = put form on every row:", all(x["match_call"] and x["match_put"] for x in o2),
      "| deterrence flips at f =", next(x["f"] for x in o2 if x["deterred_solver"]))
o3 = check_observation_3(Market(100, 0.05), Trade(), 5, 1.0, 0.5, [0.0, 0.25, 0.5, 0.75, 1.0])
V['obs3'] = dict(all_consistent=all(x['consistent'] for x in o3), rows=len(o3))
print("Obs 3 (producer routing kills the fallback above fee - c = 0.5):",
      [(x["f"], x["predicted"], x["consistent"]) for x in o3])

banner("7. Strategic sequencer (extensive-form game with a second mover): extortion above f = fee / s_P")
e = C.extortion_table()
e.to_csv(os.path.join(OUT, "extortion.csv"), index=False)
for name in ("burn", "self", "pool(4)", "producer"):
    g_ = e[e.routing == name]
    hit = g_[g_.loss_worst > 1e-9]
    print(f"  {name:<9} first forfeit with a loss: {hit.f.min() if len(hit) else 'never':<24} threshold fee/s_P = {g_.threshold.iloc[0]}")

banner("8. Bayesian designer: speculators vs honest users, forfeit chosen ex ante")
d = C.designer_table()
d.to_csv(os.path.join(OUT, "designer.csv"), index=False)
print(d.pivot(index="q", columns="pi", values="regime").to_string())
C.save_figures(e, d, OUT)
json.dump(V, open(os.path.join(OUT, "verification.json"), "w"), indent=1)
print("\nwrote", OUT)
