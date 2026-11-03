import math

import numpy as np
import pandas as pd
import pytest

from dagame.volatility import daily_volatility, load_close


def _csv(tmp_path, sigma=0.04, n=3000, step_s=3600.0, cols=("date", "close"), seed=1):
    rng = np.random.default_rng(seed)
    r = sigma * math.sqrt(step_s / 86400.0) * rng.standard_normal(n)
    t = pd.to_datetime(np.arange(n + 1) * step_s, unit="s", utc=True)
    df = pd.DataFrame({cols[0]: t.astype(str), cols[1]: 100 * np.exp(np.concatenate([[0], np.cumsum(r)]))})
    p = tmp_path / "p.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_recovers_known_volatility(tmp_path):
    sig, n = daily_volatility(load_close(_csv(tmp_path)))
    assert abs(sig - 0.04) < 0.004 and n == 3000
    sig2, _ = daily_volatility(load_close(_csv(tmp_path, sigma=0.08, seed=4)))
    assert abs(sig2 - 0.08) < 0.008


def test_column_detection_and_validation(tmp_path):
    assert len(load_close(_csv(tmp_path, cols=("snapped_at", "price")))) > 100
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_close(str(bad))
    neg = tmp_path / "neg.csv"
    pd.DataFrame({"date": pd.date_range("2024-01-01", periods=40).astype(str), "close": [-1.0] * 40}).to_csv(neg, index=False)
    with pytest.raises(ValueError):
        load_close(str(neg))
