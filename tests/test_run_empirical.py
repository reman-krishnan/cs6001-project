"""run_empirical.py on a synthetic 1-second random walk with missing seconds: resampling to 12 s must give regular steps and the
realized per-step volatility must match the volatility the series was built with."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_resample_to_12s_recovers_the_true_step_volatility(tmp_path):
    rng = np.random.default_rng(5)
    n, daily = 6 * 3600, 0.04
    sig1 = daily * np.sqrt(1 / 86400.0)
    idx = pd.date_range("2026-01-01", periods=n, freq="1s", tz="UTC")
    price = 100 * np.exp(np.cumsum(rng.normal(0, sig1, n)))
    df = pd.DataFrame({"date": idx, "close": price})
    df = df.drop(rng.choice(np.arange(1, n - 1), size=300, replace=False))        # seconds without a close
    data = tmp_path / "x_1s.csv"
    df.to_csv(data, index=False)
    cal = tmp_path / "cal.csv"
    days = pd.date_range("2025-01-01", periods=400, freq="1D", tz="UTC")
    pd.DataFrame({"date": days, "close": 100 * np.exp(np.cumsum(rng.normal(0, daily, 400)))}).to_csv(cal, index=False)
    out, saved = tmp_path / "out", tmp_path / "x_12s.csv"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_empirical.py"), "--data", str(data), "--resample", "12s", "--window-minutes", "4",
                        "--out", str(out), "--save-resampled", str(saved), "--calibration", str(cal)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    bv = pd.read_csv(out / "block_vol.csv").iloc[0]
    assert bv.step_seconds == 12.0
    assert abs(bv.sigma_step_realized / (daily * np.sqrt(12 / 86400.0)) - 1) < 0.08
    s = pd.read_csv(saved, parse_dates=["date"])
    assert (s.date.diff().dropna() == pd.Timedelta(seconds=12)).all()              # regular grid, no holes
