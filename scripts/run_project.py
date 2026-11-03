"""ONE script that regenerates every table and figure of the project.

    python scripts/run_project.py                                   # ETH, ./data/historical_prices.csv -> ./results/main   (~2 min)
    python scripts/run_project.py --data data/BTCUSDT_1d.csv --out results/assets/BTCUSDT --skip-slow
Q1 sweeps (k, volatility, size) + feasibility map + volatility interval; Q2 routing designs; Q3 extortion (single and rotating producers) and
coalitions + the impossibility search; Q4 contest mechanisms; Bayesian designer (mixture of types and a PRIVATE type); Han et al. check.
--skip-slow omits the two slow searches (impossibility, rotating) so period/asset runs stay fast.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import pandas as pd

from dagame import course_results as C
from dagame.han_check import han_table
from dagame.volatility import daily_volatility, daily_volatility_ci, load_close

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(ROOT, "data", "historical_prices.csv"))
ap.add_argument("--out", default=os.path.join(ROOT, "results", "main"))
ap.add_argument("--skip-slow", action="store_true")
ap.add_argument("--n-search", type=int, default=300, help="random designs per routing in the impossibility search")
a = ap.parse_args()
OUT = a.out
os.makedirs(OUT, exist_ok=True)
t0 = time.time()


def save(df, name):
    df.to_csv(os.path.join(OUT, name), index=False)
    print(f"  {name:<30}{len(df):>5} rows", flush=True)
    return df


print("Volatility")
if os.path.exists(a.data):
    sig, lo, hi, n = daily_volatility_ci(load_close(a.data))
    label = "DATA"
    save(pd.DataFrame([dict(source=label, daily_vol=sig, ci_low=lo, ci_high=hi, n_returns=n, data_file=os.path.basename(a.data))]), "calibration.csv")
else:
    sig, lo, hi, label = 0.04, None, None, "ANCHORED (no data file)"
    save(pd.DataFrame([dict(source=label, daily_vol=sig, ci_low=None, ci_high=None, n_returns=0, data_file="")]), "calibration.csv")
    print("  no data file: using anchored 4% daily volatility; results are labelled accordingly")

K_WIN, K_DES, FEE, COST, SURPLUS = 100, 41, 5e-4, 1e-4, 1e-3
print("Q1 characterization, interval and feasibility map")
q1 = save(C.q1_sweeps(daily_vol=sig), "q1_sweeps.csv")
if lo is not None:                       # f* is increasing in volatility, so the volatility interval maps to an interval for f*
    fs = [C._fstar(K_WIN, C.block_sigma(v), 1.0, FEE * 100.0) / 100.0 for v in (lo, sig, hi)]
    save(pd.DataFrame([dict(k=K_WIN, daily_vol_low=lo, daily_vol=sig, daily_vol_high=hi, fstar_low=fs[0], fstar=fs[1], fstar_high=fs[2])]), "q1_fstar_ci.csv")
fm = save(C.feasibility_map(daily_vol=sig), "q1_feasibility_map.csv")
print("Q2 routing designs; Q3 extortion and coalitions (measured volatility)")
save(C.designs_table(daily_vol=sig, k=K_WIN, fee=FEE, c=COST), "q2_routing_designs.csv")
ex = save(C.extortion_table(daily_vol=sig, k=K_WIN, fee=FEE, c=COST, surplus=SURPLUS), "q3_producer_incentives.csv")
save(C.coalition_table(daily_vol=sig, k=K_WIN, fee=FEE, c=COST), "q3_coalition.csv")
if not a.skip_slow:
    save(C.rotating_table(sig, fee=FEE, c=COST, surplus=SURPLUS), "q3_rotating.csv")
    save(C.impossibility_search(n_trials=a.n_search), "q3_impossibility.csv")
print("Q4 contest mechanisms")
save(C.contest_table(), "q4_contest.csv")
print("Bayesian designer (mixture of types; private type), pricing check")
dg = save(C.designer_table(daily_vol=sig, k=K_DES, fee=FEE, surplus=SURPLUS), "bayes_designer.csv")
bi, bd, bs = C.bayes_private_tables(sig, k=K_DES, fee=FEE)
save(bi, "bayes_private_interim.csv"); save(bd, "bayes_private_design.csv"); save(bs, "bayes_private_summary.csv")
save(han_table(), "han_check.csv")
C.save_figures(ex, dg, OUT, sweeps=q1, fmap=fm)
json.dump({"volatility_source": label, "daily_vol": sig, "q1_k": 100, "q2_q3_k": K_WIN, "designer_k": K_DES, "block_seconds": 12,
           "fee_bp": 5, "publish_cost_bp": 1, "honest_surplus_bp": 10, "slow_skipped": a.skip_slow, "seconds": round(time.time() - t0)},
          open(os.path.join(OUT, "run_info.json"), "w"), indent=1)
print(f"done in {time.time() - t0:.0f}s -> {OUT}  (volatility source: {label})")
