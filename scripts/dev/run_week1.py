"""Week 1 deliverable: the model is fixed, and the deviation checker is written AND tested on plain commit-reveal.

Run:  python scripts/dev/run_week1.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from dagame import Market, Trade, ForfeitRule, Routing, Protocol, Scenario, check, build_game, solve
from dagame import tree as T


def banner(s):
    print("\n" + "=" * 100 + f"\n{s}\n" + "=" * 100)


banner("1. Sanity: the hand-computed game (u=2, d=1/2, p=1/3, K=S0=100, k=2, fee=1, forfeit=10, burn)")
sc = Scenario(Market(100, math.log(2)), Trade(size=1), Protocol(2, ForfeitRule(flat=10), Routing.burn(), fee=1))
sol = solve(sc)
print(f"solver V0 = {sol.value_user:.10f}   hand value 79/3 = {79 / 3:.10f}")
print(f"user attempts at (t=1, j=0 [down]): {bool(sol.attempt[1][0])}   at (t=1, j=1 [up]): {bool(sol.attempt[1][1])}")
g = build_game(sc)
print(f"game tree: {len(g.decisions)} decision nodes; backward-induction payoffs = "
      f"{T.backward_induction(g)[0].round(10).tolist()}")

banner("2. PLAIN COMMIT-REVEAL (no forfeit, no fallback) must be flagged as exploitable")
plain = Scenario(Market(100, 0.1), Trade(size=1), Protocol(4, ForfeitRule(), Routing.burn(), fee=0.5))
print(check(plain).summary())

banner("3. Other designs through the same checker")
designs = {
    "forfeit only, too small (f=1)": Protocol(4, ForfeitRule(flat=1.0), Routing.burn(), fee=0.5),
    "forfeit only, huge (f=500)": Protocol(4, ForfeitRule(flat=500.0), Routing.burn(), fee=0.5),
    "fallback + burn, f=10": Protocol(4, ForfeitRule(flat=10.0), Routing.burn(), 0.5, 0.2, fallback=True),
    "fallback + burn, f=500": Protocol(4, ForfeitRule(flat=500.0), Routing.burn(), 0.5, 0.2, fallback=True),
    "fallback + PRODUCER routing, f=10 (Obs. 3)": Protocol(4, ForfeitRule(flat=10.0), Routing.producer(), 0.5, 0.2, fallback=True),
    "fallback + burn, f=10, sigma REQUIRED": Protocol(4, ForfeitRule(flat=10.0), Routing.burn(), 0.5, 0.2,
                                                       fallback=True, sigma_policy="required"),
    "fallback + burn, f=10, sigma OPTIONAL (user can bypass the fallback)": Protocol(
        4, ForfeitRule(flat=10.0), Routing.burn(), 0.5, 0.2, fallback=True, sigma_policy="optional"),
    "self routing, f=500 (Obs. 1)": Protocol(4, ForfeitRule(flat=500.0), Routing.to_self(), fee=0.5),
}
for name, pr in designs.items():
    print(f"\n[{name}]")
    print(check(Scenario(Market(100, 0.1), Trade(size=1), pr)).summary())
