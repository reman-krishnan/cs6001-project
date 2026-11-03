import math

import numpy as np
import pandas as pd
import pytest

from dagame import empirical as E
from dagame.volatility import daily_volatility_ci


def _prices(n, step=60.0, daily=0.04, kind="gbm", seed=0, rho=0.0):
    rng = np.random.default_rng(seed)
    s = daily * math.sqrt(step / 86400.0)
    z = rng.standard_normal(n) if kind == "gbm" else rng.standard_t(3, n) / math.sqrt(3.0)
    r = s * z
    if rho:
        for i in range(1, n):
            r[i] += rho * r[i - 1]
    idx = pd.to_datetime(np.arange(n + 1) * step, unit="s", utc=True)
    return pd.Series(100 * np.exp(np.concatenate([[0], np.cumsum(r)])), index=idx)


def test_variance_ratio_is_one_for_independent_increments_and_detects_trends():
    vr = E.variance_ratio(_prices(200_000, seed=1))
    assert (vr.variance_ratio.sub(1).abs() < 0.05).all() and (vr.z.abs() < 3.5).all()
    assert vr.implied_exponent.sub(0.5).abs().max() < 0.02
    trend = E.variance_ratio(_prices(200_000, seed=1, rho=0.3))
    assert (trend.variance_ratio > 1.25).all() and (trend.z > 5).all() and (trend.implied_exponent > 0.5).all()


def test_empirical_fstar_matches_the_lognormal_benchmark_for_gaussian_returns():
    p = _prices(300_000, seed=2)
    R = E.window_returns(p, 20 * 60)
    s = float(np.std(R))
    emp = E.empirical_fstar(R, 100.0, 0.05)
    ln = E.lognormal_fstar(s, 100.0, 0.05)
    assert emp == pytest.approx(ln, rel=0.04)


def test_fat_tails_move_the_forfeit_and_the_bootstrap_interval_brackets_the_estimate():
    pg, pt = _prices(300_000, seed=3), _prices(300_000, seed=3, kind="t")
    Rg, Rt = E.window_returns(pg, 60), E.window_returns(pt, 60)               # a 1-minute window keeps the t-tails (no aggregation)
    kg = float(((Rg - Rg.mean()) ** 4).mean() / Rg.var() ** 2)
    kt = float(((Rt - Rt.mean()) ** 4).mean() / Rt.var() ** 2)
    assert kt > 2 * kg
    assert E.empirical_fstar(Rt, 100.0, 0.05) != pytest.approx(E.empirical_fstar(Rg, 100.0, 0.05), rel=0.03)
    est, lo, hi = E.bootstrap_empirical_fstar(Rg[:20000], 100.0, 0.05, n_boot=60, block=30, seed=1)
    assert lo <= est <= hi


def test_window_returns_validation_and_zero_fee_edge():
    p = _prices(1000)
    assert len(E.window_returns(p, 600)) == 1001 - 10
    with pytest.raises(ValueError):
        E.window_returns(p, 90)
    assert E.empirical_fstar(E.window_returns(p, 60), 100.0, 1e6) == 0.0          # a fee above every gain already deters


def test_volatility_bootstrap_interval_covers_the_truth():
    p = _prices(40_000, step=3600.0, daily=0.05, seed=4)
    s, lo, hi, n = daily_volatility_ci(p, n_boot=200, seed=1)
    assert lo <= 0.05 <= hi and lo < s < hi and n == 40_000


def test_robust_variance_ratio_statistic_matches_the_classical_one_for_iid_and_is_smaller_under_volatility_clustering():
    import numpy as np
    import pandas as pd
    from dagame.empirical import variance_ratio
    rng = np.random.default_rng(3)
    idx = pd.date_range("2026-01-01", periods=60001, freq="1min", tz="UTC")
    iid = pd.Series(np.exp(np.cumsum(rng.normal(0, 1e-3, 60001))), index=idx)
    v = variance_ratio(iid, multiples=(2, 10, 30))
    assert (np.abs(v.z_robust - v.z) < 0.35 * np.maximum(1.0, np.abs(v.z))).all()                  # same answer when the classical assumptions hold
    vol = np.repeat(rng.choice([0.3e-3, 3e-3], size=600), 100)                                         # volatility clusters in blocks of 100
    ret = rng.normal(0, 1, 60000) * vol
    ret[1:] += 0.05 * ret[:-1]                                                                          # a small genuine autocorrelation
    clust = variance_ratio(pd.Series(np.exp(np.concatenate([[0.0], np.cumsum(ret)])), index=idx), multiples=(2,))
    assert abs(clust.z_robust.iloc[0]) < abs(clust.z.iloc[0])                                          # clustering makes the classical z too large
