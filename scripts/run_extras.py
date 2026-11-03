"""Supplementary analyses for the report, written to ./results/main/extras_*.csv.

    python scripts/run_extras.py            # needs results/main/calibration.csv (run run_project.py first)
1. fee sensitivity of f* and the effect of a fee paid at commitment time;
2. closed-form (Gaussian) approximation of f* against the lattice value;
3. the range of burned forfeits that is credible and nearly collusion-proof, against k and volatility;
4. the honest user's expected cost of a plain commit-reveal protocol with a forfeit of f*, under censoring with retries;
5. lattice-resolution sensitivity of the private-type designer's forfeit (exact minimiser against the 99% and 95% rules, refined search)
   and, at a fixed lattice, its sensitivity to the spacing of the forfeit grid;
6. one-at-a-time sensitivity of the 99% forfeit to the prior over types and to the loss weights;
7. robustness of the race-versus-bond comparison to the attack cost and the owner capital;
8. forfeits that grow with the transaction size (f = a + b m), against a flat fee and against a fee proportional to size;
9. the smallest deterring forfeit against the censoring probability q (plain commit-reveal with a burned forfeit, no fallback);
10. the window frontier: the largest integer k whose nearly collusion-proof burned forfeit fits the collateral limit (every k from 2 to 400);
11. sensitivity of f_cp and of that frontier to the coalition-gain tolerance;
12. a mostly burned forfeit, pool(n): the smallest n that is strictly credible and nearly collusion-proof.
"""
import math
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from dagame import bayes_private as B
from dagame.course_results import NOTIONAL, ROUTINGS, _fstar, _fstar_fraction, block_sigma, scenario, window_vol
from dagame.model import ForfeitRule, Market, PayoffKind, Protocol, Routing, Scenario, Trade
from dagame.solver import honest_user_loss, joint_surplus, minimal_deterring_forfeit, solve, solve_all_selections

OUT = os.path.join(ROOT, "results", "main")
cal = pd.read_csv(os.path.join(OUT, "calibration.csv")).iloc[0]
DAILY = float(cal.daily_vol)
BP = 1e-4
K0 = 100


def market(daily, k):
    return Market(100.0, block_sigma(daily))


def fstar(daily, k, fee_bp, commit_bp=0.0):
    """Smallest forfeit (fraction of the notional) with V_0(f) <= commit fee, plain commit-reveal with burned forfeit."""
    mk, fee = market(daily, k), fee_bp * BP * NOTIONAL
    if commit_bp == 0.0:
        return minimal_deterring_forfeit(mk, Trade(size=1.0), Protocol(k, ForfeitRule(), Routing.burn(), fee=fee)) / NOTIONAL

    def v0(f):
        return solve(Scenario(mk, Trade(size=1.0), Protocol(k, ForfeitRule(flat=f * NOTIONAL), Routing.burn(), fee=fee))).value_user / NOTIONAL - commit_bp * BP
    return brentq(v0, 0.0, 1.0, xtol=1e-12)


def gaussian_fstar(s, fee_bp):
    """f solving E[(phi - f - g)^+] = phi for g ~ N(0, s^2): a put struck at phi - f costs phi."""
    phi = fee_bp * BP
    put = lambda f: (lambda a: a * norm.cdf(a / s) + s * norm.pdf(a / s))(phi - f) - phi
    return brentq(put, 0.0, 10 * s + phi)


# ---- 1 and 2: fee sensitivity, commit fee, closed form ----
rows = []
s_win = window_vol(DAILY, K0)
for fee in (1, 2.5, 5, 10, 20):
    rows.append(dict(fee_bp=fee, commit_fee_bp=0.0, f_star=fstar(DAILY, K0, fee), f_gauss=gaussian_fstar(s_win, fee), window_vol=s_win))
for cf in (2.5, 5.0, 10.0):
    rows.append(dict(fee_bp=5, commit_fee_bp=cf, f_star=fstar(DAILY, K0, 5, cf), f_gauss=float("nan"), window_vol=s_win))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "extras_fee_sensitivity.csv"), index=False)

# ---- 3: credible and nearly collusion-proof burned forfeits ----
EPS_BP = 0.1          # a coalition gain below this many basis points of the notional is treated as zero
FEE, C = 5e-4, 1e-4


def coalition_gain_bp(daily, k, f, routing=None):
    sc = scenario(window_vol(daily, k), k, f=f, fee=FEE, c=C, routing=routing or Routing.burn())
    tot = max(x.value_user + x.value_producer for _, x in solve_all_selections(sc))
    return (joint_surplus(sc) - tot) / NOTIONAL * 1e4


def f_cp(daily, k, eps_bp=EPS_BP, routing=None):
    """Smallest forfeit whose coalition gain is at most eps_bp. With burn (or any routing that keeps the producer publishing) the
    equilibrium total does not depend on f and the joint optimum falls with f, so the gain is monotone and brentq finds the threshold."""
    g0 = coalition_gain_bp(daily, k, 0.0, routing)
    if g0 <= eps_bp:
        return 0.0
    hi = 0.5
    return brentq(lambda f: coalition_gain_bp(daily, k, f, routing) - eps_bp, 0.0, hi, xtol=1e-9)


def window_scan(daily, kappa, eps_list=(EPS_BP,), k_lo=2, k_hi=400):
    """Evaluate EVERY integer window k in [k_lo, k_hi]: the burned forfeit kappa is nearly collusion-proof at k iff the coalition gain at
    f = kappa is at most eps (the gain falls with f under burning, so f_cp(k) <= kappa exactly then). Returns, per eps, the largest feasible
    k and whether the feasible windows form one unbroken run starting at k_lo.

    Why the run is unbroken (so no window beyond a failing one can pass): under burning with q = 0 the producer always publishes, the
    user is indifferent between authorizing and waiting, and the best equilibrium total is 0 at every k. The coalition's optimum is
    non-decreasing in k: with one more block, at block k it can authorize at g_k instead of publishing at g_k - c, and withholding later
    still costs -f, so every plan of the k-block game is matched or beaten. Hence the gain, and with it f_cp, is non-decreasing in k.
    The scan checks this rather than relying on it."""
    gains = {k: coalition_gain_bp(daily, k, kappa) for k in range(k_lo, k_hi + 1)}
    out = {}
    for eps in eps_list:
        ok = [k for k, gk in gains.items() if gk <= eps]
        kmax = max(ok) if ok else k_lo - 1
        out[eps] = dict(k_max=kmax, unbroken=bool(ok == list(range(k_lo, kmax + 1))),
                        gain_nondecreasing=bool(all(gains[k + 1] >= gains[k] - 1e-9 for k in range(k_lo, k_hi))))
    return out


rows = []
for k in (10, 25, 50, 100, 200):
    for daily in (0.02, DAILY, 0.08):
        mk = Market(100.0, block_sigma(daily))
        fs = minimal_deterring_forfeit(mk, Trade(size=1.0), Protocol(k, ForfeitRule(), Routing.burn(), fee=FEE * NOTIONAL)) / NOTIONAL
        rows.append(dict(k=k, daily_vol=daily, f_star=fs, f_cp=f_cp(daily, k)))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "extras_burn_range.csv"), index=False)

# ---- 4: the honest user's cost under a forfeit of f* ----
rows = []
fs = fstar(DAILY, K0, 5)
for q in (0.05, 0.3, 0.6):
    pr = Protocol(K0, ForfeitRule(flat=fs * NOTIONAL), Routing.burn(), 5e-4 * NOTIONAL, 0.0, q, False)
    r = honest_user_loss(market(DAILY, K0), pr, value=(5e-4 + 1e-3) * NOTIONAL)
    rows.append(dict(q=q, f_star=fs, honest_loss_bp=r["loss"] / NOTIONAL * 1e4, p_dropped=r["p_dropped"]))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "extras_honest_cost.csv"), index=False)

# ---- 5: resolution sensitivity of the private-type forfeit (same window volatility, more lattice steps) ----
# Each extra step is also an extra authorization opportunity with the same per-attempt censoring probability q, so this is a JOINT
# sensitivity to price resolution and to the number of attempts (at q = 0.6, every attempt fails with prob. 0.6^5 vs 0.6^80).
K_DES = 41
s_des = window_vol(DAILY, K_DES)
rows = []
for kk in (6, 11, 21, 41, 81):
    fs_ = _fstar_fraction(s_des, kk, 5e-4)
    grid = [40.0 * fs_ * i / 160 for i in range(161)]
    for q in (0.05, 0.3, 0.6):
        _, agg = B.design_table(s_des, kk, 5e-4, q, f_grid=grid)
        ex, c99 = B.optimal_forfeit(agg), B.eps_optimal_forfeit(agg, 0.01)
        f99 = B.refine_eps_forfeit(agg, s_des, kk, 5e-4, q, eps=0.01)
        f95 = B.refine_eps_forfeit(agg, s_des, kk, 5e-4, q, eps=0.05)
        rows.append(dict(steps=kk - 1, q=q, f_star=fs_, exact_over_fstar=ex["f_opt"] / fs_, eps99_over_fstar=f99 / fs_,
                         eps99_coarse_grid_over_fstar=c99["f_opt"] / fs_, eps95_over_fstar=f95 / fs_, loss_exact=ex["loss_opt"],
                         loss_at_zero=c99["loss_at_zero"]))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "extras_bayes_resolution.csv"), index=False)

# ---- 5b: forfeit-grid spacing at a fixed lattice (the 99% forfeit must not depend on the grid) ----
fs_ = _fstar_fraction(s_des, K_DES, 5e-4)
coarse_grid = [40.0 * fs_ * i / 160 for i in range(161)]
rows = []
for q in (0.05, 0.3, 0.6):
    _, agg = B.design_table(s_des, K_DES, 5e-4, q, f_grid=coarse_grid)
    ref = B.eps_optimal_forfeit(agg, 0.01)
    target = ref["loss_min"] + 0.01 * (ref["loss_at_zero"] - ref["loss_min"]) + 1e-18
    for h in (0.5, 0.25, 0.1, 0.05, 0.025):
        g = [h * fs_ * i for i in range(int(round(6.0 / h)) + 1)]
        _, a_ = B.design_table(s_des, K_DES, 5e-4, q, f_grid=g)
        ok = a_.loss.to_numpy() <= target
        rows.append(dict(q=q, spacing_over_fstar=h, eps99_over_fstar=float(a_.f.to_numpy()[int(np.argmax(ok))]) / fs_ if ok.any() else float("nan")))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "extras_bayes_gridspacing.csv"), index=False)

# ---- 6: one-at-a-time prior and weight sensitivity of the 99% forfeit (designer window) ----
PRIORS = {
    "baseline": B.DEFAULT_PRIOR,
    "few speculators": {0.0: 0.10, 0.0002: 0.15, 0.0005: 0.15, 0.001: 0.20, 0.002: 0.20, 0.005: 0.12, 0.01: 0.08},
    "many speculators": {0.0: 0.60, 0.0002: 0.10, 0.0005: 0.05, 0.001: 0.10, 0.002: 0.07, 0.005: 0.05, 0.01: 0.03},
    "no type at the fee": {0.0: 0.35, 0.0003: 0.20, 0.0007: 0.15, 0.0012: 0.12, 0.0025: 0.08, 0.005: 0.06, 0.01: 0.04},
}
WEIGHTS = {"lambda=nu=1": (1.0, 1.0), "lambda=1, nu=5": (1.0, 5.0), "lambda=5, nu=1": (5.0, 1.0)}
rows = []
for pn, pr_ in PRIORS.items():
    for wn, (lam, nu) in WEIGHTS.items():
        if pn != "baseline" and wn != "lambda=nu=1":
            continue                                           # one at a time: vary the prior or the weights, not both
        _, agg = B.design_table(s_des, K_DES, 5e-4, 0.3, prior=pr_, f_grid=coarse_grid, lam=lam, nu=nu)
        f99 = B.refine_eps_forfeit(agg, s_des, K_DES, 5e-4, 0.3, prior=pr_, eps=0.01, lam=lam, nu=nu)
        rows.append(dict(k=K_DES, prior=pn, weights=wn, q=0.3, f_star=fs_, exact_over_fstar=B.optimal_forfeit(agg)["f_opt"] / fs_,
                         f_opt_over_fstar=f99 / fs_))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "extras_bayes_sensitivity.csv"), index=False)

# ---- 7: race versus bond for several attack costs and owner capitals ----
from dagame.contest import compare_mechanisms
rows = []
for kap in (0.005, 0.01, 0.02, 0.05):
    for cap in (0.5, 1.0, 2.0):
        d_ = compare_mechanisms(attack_cost=kap, owner_capital=cap)
        st = d_[(d_.value_ratio > 1) & (d_.availability >= 0.8)]
        rows.append(dict(attack_cost=kap, owner_capital=cap, cells=len(d_), race_deters=int(d_.race_deterred.sum()), race_ties=int(d_.race_tie.sum()),
                         bond_feasible=int(d_.bond_feasible.sum()), bond_deters=int((d_.bond_feasible & d_.bond_deterred).sum()),
                         strong_cells=len(st), strong_race_deters=int(st.race_deterred.sum()), strong_bond_deters=int(st.bond_deterred.sum())))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "extras_contest_robustness.csv"), index=False)

# ---- 8: size-dependent forfeit rules ----
SIZES = (0.1, 1.0, 10.0, 100.0)
KS = 100
fee_flat = 5 * BP * NOTIONAL                                  # 5 bp of a one-unit trade, the same currency amount for every size
fstar_flat = {m: _fstar(KS, block_sigma(DAILY), m, fee_flat) for m in SIZES}
fstar_prop = {m: _fstar(KS, block_sigma(DAILY), m, 5 * BP * NOTIONAL * m) for m in SIZES}
ms = np.array(SIZES)
fs_arr = np.array([fstar_flat[m] for m in SIZES])
b_fit, a_fit = np.polyfit(ms, fs_arr, 1)                      # f = a + b m, least squares in currency units
a_fit = max(a_fit, 0.0)
prop_rate = fstar_flat[1.0] / NOTIONAL                        # proportional rule calibrated at size 1: f = rate * N


def v0_bp(m, flat, per_unit, fee):
    pr = Protocol(KS, ForfeitRule(flat=flat, per_unit=per_unit), Routing.burn(), fee=fee)
    return solve(Scenario(market(DAILY, KS), Trade(size=m), pr)).value_user / (NOTIONAL * m) * 1e4


rows = []
for m in SIZES:
    N_ = NOTIONAL * m
    rows.append(dict(size=m, fstar_flat_fee_over_N=fstar_flat[m] / N_, fstar_prop_fee_over_N=fstar_prop[m] / N_,
                     affine_rule=a_fit + b_fit * m, affine_over_fstar=(a_fit + b_fit * m) / fstar_flat[m],
                     v0_bp_proportional_rule=v0_bp(m, 0.0, prop_rate * NOTIONAL, fee_flat),
                     v0_bp_affine_rule=v0_bp(m, a_fit, b_fit, fee_flat),
                     a_fit=a_fit, b_fit=b_fit, prop_rate=prop_rate))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "extras_size_rules.csv"), index=False)

# ---- 9: f* against the censoring probability q ----
rows = []
mk9 = market(DAILY, KS)
for q in (0.0, 0.1, 0.3, 0.6, 0.9):
    f9 = minimal_deterring_forfeit(mk9, Trade(size=1.0), Protocol(KS, ForfeitRule(), Routing.burn(), fee=5 * BP * NOTIONAL, q=q)) / NOTIONAL
    sol0 = solve(Scenario(mk9, Trade(size=1.0), Protocol(KS, ForfeitRule(), Routing.burn(), fee=5 * BP * NOTIONAL, q=q)))
    rows.append(dict(q=q, k=KS, f_star=f9, v0_without_forfeit_bp=sol0.value_user / NOTIONAL * 1e4))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "extras_q_sweep.csv"), index=False)

# ---- 10 and 11: the window frontier, from a scan of EVERY integer window 2..400, and its sensitivity to the coalition-gain tolerance ----
KAPPA, EPS_LIST = 0.01, (1.0, 0.1, 0.01, 0.001)
rows, tol_rows = [], []
for daily in (0.02, DAILY, 0.08):
    scan = window_scan(daily, KAPPA, EPS_LIST if daily == DAILY else (EPS_BP,))
    for eps, r in scan.items():
        if not (r["unbroken"] and r["gain_nondecreasing"]):
            raise SystemExit(f"feasible windows are not one unbroken run at daily volatility {daily}, eps {eps}")
    r = scan[EPS_BP]
    rows.append(dict(daily_vol=daily, kappa=KAPPA, eps_bp=EPS_BP, k_max=r["k_max"], minutes=r["k_max"] * 12 / 60.0,
                     f_cp_at_k_max=f_cp(daily, r["k_max"]), f_cp_at_k_max_plus_1=f_cp(daily, r["k_max"] + 1), windows_checked="every k in 2..400"))
    if daily == DAILY:
        for eps, r in scan.items():
            tol_rows.append(dict(eps_bp=eps, k=K0, f_cp=f_cp(DAILY, K0, eps), k_max=r["k_max"], minutes=r["k_max"] * 12 / 60.0))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "extras_frontier.csv"), index=False)
pd.DataFrame(tol_rows).to_csv(os.path.join(OUT, "extras_tolerance.csv"), index=False)

# ---- 12: a mostly burned forfeit, pool(n): the producer keeps 1/n. Smallest n that is strictly credible and nearly collusion-proof ----
rows = []
for n in range(2, 81):
    r_ = Routing.pool(n)
    credible_cap = n * (FEE - C)                            # strict credibility: fee - c > f / n
    f_top = credible_cap * (1 - 1e-6)                       # strictly credible beyond the solver's tie tolerance (numerics.TOL)
    gain_top = coalition_gain_bp(DAILY, K0, f_top, r_)      # below the cap the producer publishes, so the gain falls with f
    works = bool(gain_top <= EPS_BP)
    f_needed = brentq(lambda f: coalition_gain_bp(DAILY, K0, f, r_) - EPS_BP, 0.0, f_top, xtol=1e-9) if works else float("nan")
    rows.append(dict(n=n, producer_share=1.0 / n, credible_cap=credible_cap, gain_at_cap_bp=gain_top, works=works, f_cp=f_needed,
                     extortion_threshold=n * FEE, extortion_free=bool(works and f_needed < n * FEE)))
    if works:
        break
pd.DataFrame(rows).to_csv(os.path.join(OUT, "extras_pool.csv"), index=False)
print("wrote extras_*.csv")
