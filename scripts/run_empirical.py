"""Fat tails and the square-root-of-time rule on INTRADAY data (needs a 1-minute or 1-second CSV, see download_binance.py).

    python scripts/run_empirical.py --data data/ETHUSDT_1m.csv --window-minutes 20 --fee-bp 5
    python scripts/run_empirical.py --data data/ETHUSDT_1s.csv --resample 12s --out results/intraday_12s --save-resampled data/ETHUSDT_12s.csv
Writes results/intraday_1m/{empirical_fstar.csv, variance_ratio.csv, block_vol.csv}. block_vol.csv compares the realized per-step volatility with the one implied by the daily calibration. Compares f* under the empirical window-return distribution (with a
block-bootstrap interval) with the lognormal f* at the same window volatility, and tests for departures from square-root-of-time variance scaling (variance ratios).
"""
import argparse
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd

from dagame.empirical import bootstrap_empirical_fstar, lognormal_fstar, variance_ratio, window_returns
from dagame.volatility import load_close

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--window-minutes", type=float, default=20.0)
ap.add_argument("--fee-bp", type=float, default=5.0)
ap.add_argument("--n-boot", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--resample", default=None, help="e.g. 12s: resample a finer file (such as 1-second closes) to this step before analysis")
ap.add_argument("--block-windows", type=float, default=12.0, help="bootstrap block length in multiples of the window, so blocks are much longer than the overlap")
ap.add_argument("--save-resampled", default=None, help="write the resampled closes (date, close) to this CSV")
ap.add_argument("--vr-multiples", type=int, nargs="+", default=[2, 5, 10, 30, 60])
ap.add_argument("--calibration", default=os.path.join(ROOT, "data", "historical_prices.csv"), help="daily file used for the main calibration")
ap.add_argument("--out", default=os.path.join(ROOT, "results", "intraday_1m"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
prices = load_close(a.data)
if a.resample:
    # a second without trades has no close, so carry the last price forward (the price did not change)
    prices = prices.resample(a.resample).last().ffill()
    if a.save_resampled:
        prices.rename("close").rename_axis("date").to_csv(a.save_resampled)
N, fee = 100.0, a.fee_bp * 1e-4 * 100.0
R = window_returns(prices, a.window_minutes * 60.0)
s = float(np.std(R))
step_s = float(np.median(np.diff(pd.DatetimeIndex(prices.index).as_unit("ns").asi8)) / 1e9)
block = max(20, int(a.block_windows * a.window_minutes * 60.0 / step_s))
emp, lo, hi = bootstrap_empirical_fstar(R, N, fee, n_boot=a.n_boot, block=block, seed=a.seed)
ln = lognormal_fstar(s, N, fee)
kurt = float(((R - R.mean()) ** 4).mean() / R.var() ** 2)            # of the overlapping WINDOW returns
r_step = np.diff(np.log(prices.to_numpy(float)))
kurt_step = float(((r_step - r_step.mean()) ** 4).mean() / r_step.var() ** 2)   # of the one-step (12 s or 1 min) returns
zero_share = float(np.mean(r_step == 0.0))
pd.DataFrame([dict(window_minutes=a.window_minutes, n_windows=len(R), window_vol=s, excess_kurtosis=kurt - 3.0, step_excess_kurtosis=kurt_step - 3.0, zero_step_share=zero_share, f_empirical=emp / N, ci_low=lo / N,
                   ci_high=hi / N, f_lognormal=ln / N, ratio=emp / ln, n_boot=a.n_boot, block_steps=block, seed=a.seed)]).to_csv(os.path.join(a.out, "empirical_fstar.csv"), index=False)
vr = variance_ratio(prices, multiples=tuple(a.vr_multiples))
vr.to_csv(os.path.join(a.out, "variance_ratio.csv"), index=False)
print(f"window vol {s:.4%}, excess kurtosis of window returns {kurt - 3:.2f}, of one-step returns {kurt_step - 3:.2f}\nf* empirical {emp / N:.4%} [{lo / N:.4%}, {hi / N:.4%}] vs lognormal {ln / N:.4%} (ratio {emp / ln:.2f})")
print(vr.round(3).to_string(index=False))

# Block volatility: realized per-step volatility against the one implied by the daily calibration (sqrt-of-time scaling)
from dagame.volatility import daily_volatility
step = float(np.median(np.diff(pd.DatetimeIndex(prices.index).as_unit("ns").asi8)) / 1e9)
r1 = np.diff(np.log(prices.to_numpy(float)))
sig_step = float(np.sqrt(np.mean(r1 ** 2)))
cal_daily, _ = daily_volatility(load_close(a.calibration))
daily_same, n_days = daily_volatility(prices.resample("1D").last().dropna())
implied_step = cal_daily * np.sqrt(step / 86400.0)
pd.DataFrame([dict(step_seconds=step, n_steps=len(r1), sigma_step_realized=sig_step, sigma_step_implied_by_calibration=implied_step,
                   ratio_realized_to_implied=sig_step / implied_step, daily_equivalent_from_steps=sig_step * np.sqrt(86400.0 / step),
                   daily_from_daily_returns_same_sample=daily_same, n_daily_returns_same_sample=n_days, calibration_daily=cal_daily)]
             ).to_csv(os.path.join(a.out, "block_vol.csv"), index=False)
print(f"step {step:g}s: realized sigma {sig_step:.5%} vs {implied_step:.5%} implied by the {cal_daily:.2%} daily calibration "
      f"(daily equivalent {sig_step * np.sqrt(86400.0 / step):.2%}; daily returns in the same sample give {daily_same:.2%} from {n_days} days)")
