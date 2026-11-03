import pytest

from dagame.han_check import han_table, han_premium_pct, crr_american


def test_game_solver_reproduces_the_han_method():
    t = han_table()
    assert t.abs_diff.max() < 1e-9


def test_reported_bands():
    assert han_premium_pct(0.0075) == pytest.approx(0.30, abs=0.02)             # stocks / fiat
    assert 0.95 < han_premium_pct(0.03) < han_premium_pct(0.058) < 2.4          # crypto range in their Section 4.3.2


def test_premium_rises_with_strike_time_and_american_equals_european_at_zero_rate():
    v = [han_premium_pct(0.04, T) for T in (1, 5, 30, 100, 300)]
    assert all(a < b for a, b in zip(v, v[1:]))
    import math
    from scipy.stats import binom
    n, s = 36, 0.04 / 6
    p = (1 - math.exp(-s)) / (math.exp(s) - math.exp(-s))
    eur = sum(binom.pmf(j, n, p) * max(math.exp(s * (2 * j - n)) - 1, 0) for j in range(n + 1))
    assert crr_american(s, n) == pytest.approx(eur, rel=1e-12)
