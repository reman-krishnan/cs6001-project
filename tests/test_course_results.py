import pytest

from dagame import course_results as C


def test_designs_table_matches_the_theory():
    t = C.designs_table().set_index("design")
    assert t.loc["plain commit-reveal (no forfeit)", "speculation_deterred"] == False      # noqa: E712
    assert t.loc["fallback + burn", "speculation_deterred"] == True                        # noqa: E712
    assert t.loc["fallback + burn", "speculator_value_bp"] == pytest.approx(-5.0)          # V0 = -fee = -5 bp
    assert t.loc["forfeit only, small", "speculation_deterred"] == False                   # noqa: E712
    assert t.loc["forfeit only, large", "speculation_deterred"] == True                    # noqa: E712
    assert t.loc["fallback + producer routing (f > fee - c)", "speculation_deterred"] == False   # noqa: E712
    assert t.loc["fallback + burn, sigma optional", "speculation_deterred"] == False       # noqa: E712
    assert t.loc["self routing", "speculation_deterred"] == False                          # noqa: E712


def test_extortion_thresholds_and_designer_regimes():
    import numpy as np
    e = C.extortion_table(k=8, f_grid=np.array([5e-4, 2e-3, 5e-3]))
    prod = e[e.routing == "producer"].set_index("f").loss_worst
    assert prod[5e-4] == pytest.approx(0.0, abs=1e-12) and prod[2e-3] == pytest.approx(0.012, abs=1e-9)
    assert e[e.routing.isin(["burn", "self"])].loss_worst.abs().max() < 1e-12
    d = C.designer_table(qs=(0.1, 0.5), pis=(0.02, 0.3))
    assert {"full deterrence", "no forfeit"} <= set(d.regime)


def test_designer_table_evaluates_the_speculator_at_the_row_q():
    """Regression: the speculator's value used to be computed once at q = 0 and reused for every q."""
    from dagame.course_results import _speculator_value, designer_table
    from dagame import Market
    m = Market(100.0, 0.0005)
    assert _speculator_value(m, 40, 0.0, 5e-4, 0.9) < _speculator_value(m, 40, 0.0, 5e-4, 0.0) - 1e-5     # censoring lowers the option's value
    d = designer_table(daily_vol=0.0457, k=21, fee=5e-4, surplus=1e-3, qs=(0.0, 0.9), pis=(0.5,))
    fq = d.set_index("q").f_star_q
    assert fq[0.9] < fq[0.0]                                                                                # the q-specific threshold falls with q
    assert d.f_star.nunique() == 1                                                                          # f_star is the q = 0 benchmark


def test_full_deterrence_label_requires_a_nonpositive_speculator_value():
    """With the default parameters, q = 0.2 and pi = 0.01 give f_opt = 0.068 against f_star_q = 0.0708. That is within 5% of the
    threshold but the speculator still gains, so the row must not be called full deterrence."""
    d = C.designer_table(qs=(0.2,), pis=(0.01,))
    r = d.iloc[0]
    assert r.f_opt < r.f_star_q and r.regime == "near threshold"
    big = C.designer_table(qs=(0.05, 0.2, 0.5), pis=(0.01, 0.1, 0.9))
    for _, row in big[big.regime == "full deterrence"].iterrows():
        assert row.f_opt >= row.f_star_q * (1 - 1e-9)
