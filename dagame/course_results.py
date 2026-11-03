"""Results that use ONLY course-aligned machinery (extensive-form games, backward induction / SPNE, best response, Nash vs SPNE,
quasi-linear payments, Bayesian type uncertainty). At module level it imports nothing from the finance / calibration extensions
(tests/test_core_isolation.py enforces this). The coalition analyses (``coalition_table``, ``impossibility_search``) and the
contest and private-type tables import their extension modules inside the function that needs them.

Units: S0 = 100, one unit traded, so the notional is 100 and money is quoted as a FRACTION of it (1e-4 = 1 basis point).
"""
from __future__ import annotations

import math
import os
import random

import numpy as np
import pandas as pd

from .model import ForfeitRule, Market, PayoffKind, Protocol, Routing, Scenario, Trade
from .numerics import gt
from .solver import honest_user_loss, solve, minimal_deterring_forfeit, user_value_range
from .strategic import solve_all_strategic

NOTIONAL = 100.0
ROUTINGS = {"self": Routing.to_self(), "burn": Routing.burn(), "pool(4)": Routing.pool(4), "producer": Routing.producer()}


def scenario(s_tot, k, *, f, fee, c=0.0, q=0.0, routing=Routing.burn(), fallback=True, sigma_policy="required",
             kind=PayoffKind.CALL, value=0.0):
    """Fractions of the notional; s_tot = total volatility over the k-1 step window (per-step sigma = s_tot / sqrt(k-1))."""
    return Scenario(Market(100.0, s_tot / math.sqrt(k - 1)), Trade(kind, 1.0, None, value * NOTIONAL),
                    Protocol(k, ForfeitRule(flat=f * NOTIONAL), routing, fee * NOTIONAL, c * NOTIONAL, q, fallback,
                             sigma_policy))


def window_vol(daily_vol: float, k: int, block_seconds: float = 12.0) -> float:
    """Total volatility over the k-1 price steps of a k-block window: daily * sqrt((k-1) * block_seconds / 86400)."""
    return block_sigma(daily_vol, block_seconds) * math.sqrt(k - 1)


def _fstar_fraction(s, k, fee):
    """Lattice f*/notional of plain commit-reveal + forfeit (burn routing, no fallback); fee is a fraction of the notional."""
    return minimal_deterring_forfeit(Market(100.0, s / math.sqrt(k - 1)), Trade(size=1.0),
                                     Protocol(k, ForfeitRule(), Routing.burn(), fee=fee * NOTIONAL)) / NOTIONAL


def designs_table(s=0.02, k=4, fee=5e-4, c=1e-4, big_f=0.5, small_f=0.005, daily_vol=None):
    """Q1/Q2 in one table: the speculator's equilibrium value and what the producer does, per design (no coalitions).
    With daily_vol the window volatility is MEASURED (k blocks of 12 s) and the forfeits are set relative to the measured
    f*: small = 0.3 f*, large = 1.5 f*. Without it the old illustrative parameters are used."""
    if daily_vol is not None:
        s = window_vol(daily_vol, k)
    fs = _fstar_fraction(s, k, fee)
    if daily_vol is not None:
        small_f, big_f = 0.3 * fs, 1.5 * fs
    rows = []
    specs = [("plain commit-reveal (no forfeit)", dict(f=0.0, fallback=False)),
             ("forfeit only, small", dict(f=small_f, fallback=False)),
             ("forfeit only, large", dict(f=big_f, fallback=False)),
             ("fallback + burn", dict(f=small_f, fallback=True)),
             ("fallback + producer routing (f > fee - c)", dict(f=small_f, fallback=True, routing=Routing.producer())),
             ("fallback + burn, sigma optional", dict(f=small_f, fallback=True, sigma_policy="optional")),
             ("self routing", dict(f=big_f, fallback=False, routing=Routing.to_self()))]
    for name, kw in specs:
        sc = scenario(s, k, fee=fee, c=c, **kw)
        sol = solve(sc)
        v_hi = max(sol.value_user, user_value_range(sc)[1])          # largest speculator value over all pure equilibria
        r = kw.get("routing", Routing.burn())
        rows.append(dict(design=name, window_vol=s, f_star=fs, forfeit=kw["f"], forfeit_over_fstar=kw["f"] / fs if fs > 0 else float("nan"),
                         speculator_value_bp=v_hi / NOTIONAL * 1e4, speculation_deterred=not gt(v_hi, 0.0),
                         producer_prefers_publish=bool(fee - c >= r.to_producer * kw["f"]) if kw["fallback"] else None,
                         p_fallback=sol.outcomes.fallback, p_dropped=sol.outcomes.dropped, ties=sol.ties.any))
    return pd.DataFrame(rows)


def extortion_table(k=16, fee=1e-3, c=2e-4, s=0.01, surplus=0.01, f_grid=None, daily_vol=None):
    """A strategic sequencer (a perfect-information EFG with a second mover): honest user's expected loss vs the forfeit."""
    if daily_vol is not None:
        s = window_vol(daily_vol, k)
    f_grid = np.concatenate([[0.0], np.geomspace(1e-5, 0.5, 40)]) if f_grid is None else f_grid
    rows = []
    for name, r in ROUTINGS.items():
        for f in f_grid:
            sc = scenario(s, k, f=float(f), fee=fee, c=c, routing=r, kind=PayoffKind.TRANSFER, value=fee + surplus)
            losses = [surplus * NOTIONAL - x.value_user for _, x in solve_all_strategic(sc)]
            rows.append(dict(routing=name, f=float(f), window_vol=s, loss_worst=max(losses) / NOTIONAL,
                             threshold=fee / r.to_producer if r.to_producer > 0 else math.inf))
    return pd.DataFrame(rows)


def _speculator_value(m, k, f, fee, q):
    """The pure speculator's equilibrium value (fraction of the notional) with forfeit f and censoring probability q."""
    pr = Protocol(k, ForfeitRule(flat=f * NOTIONAL), Routing.burn(), fee * NOTIONAL, 0.0, q, False)
    return solve(Scenario(m, Trade(size=1.0), pr)).value_user / NOTIONAL


def designer_table(s=0.05, k=6, fee=1e-3, surplus=0.02, qs=(0.05, 0.1, 0.2, 0.3, 0.5, 0.7),
                   pis=(0.01, 0.03, 0.1, 0.3, 0.6, 0.9), daily_vol=None):
    """Bayesian designer: a share pi of users are speculators, 1 - pi honest. The designer picks the forfeit of a
    plain commit-reveal protocol to minimise expected welfare loss  pi * max(V0, 0) + (1 - pi) * min(honest loss, surplus).
    Both the speculator's value and the honest user's loss are evaluated at the censoring probability q of the row.
    With daily_vol the window volatility is measured and the forfeit grid is 0 .. 1.5 f* (61 points), where f* is the
    q = 0 smallest deterring forfeit. ``f_star_q`` is the smallest deterring forfeit at the row's q, and a forfeit counts as
    full deterrence only when the speculator's computed value at that forfeit is not positive. A forfeit that reaches 95% of the
    q-specific value without deterring is labelled "near threshold"."""
    if daily_vol is not None:
        s = window_vol(daily_vol, k)
    m = Market(100.0, s / math.sqrt(k - 1))
    f_star = _fstar_fraction(s, k, fee)
    grid = [1.5 * f_star * i / 60 for i in range(61)] if daily_vol is not None else [i * 0.004 for i in range(61)]
    rows = []
    for q in qs:
        spec = {f: _speculator_value(m, k, f, fee, q) for f in grid}
        hon = {}
        for f in grid:
            pr = Protocol(k, ForfeitRule(flat=f * NOTIONAL), Routing.burn(), fee * NOTIONAL, 0.0, q, False)
            hon[f] = honest_user_loss(m, pr, value=(fee + surplus) * NOTIONAL)["loss"] / NOTIONAL
        f_star_q = minimal_deterring_forfeit(m, Trade(size=1.0), Protocol(k, ForfeitRule(), Routing.burn(), fee * NOTIONAL, 0.0, q, False)) / NOTIONAL
        for pi in pis:
            w = [pi * max(spec[f], 0.0) + (1 - pi) * min(hon[f], surplus) for f in grid]
            i = int(np.argmin(w))
            if i == 0:
                regime = "no forfeit"
            elif not gt(spec[grid[i]], 0.0):
                regime = "full deterrence"
            else:
                regime = "near threshold" if grid[i] >= 0.95 * f_star_q else "partial"
            rows.append(dict(q=q, pi=pi, window_vol=s, f_opt=grid[i], f_star=f_star, f_star_q=f_star_q, regime=regime))
    return pd.DataFrame(rows)


def block_sigma(daily_vol: float, block_seconds: float = 12.0) -> float:
    """Per-block volatility from a daily volatility (Brownian scaling): sigma_block = daily * sqrt(block_seconds / 86400)."""
    return daily_vol * math.sqrt(block_seconds / 86400.0)


def _fstar(k, sigma_block, size, fee_abs):
    """Minimal deterring forfeit (currency units) of plain commit-reveal + forfeit for a trade of `size` units and a FLAT fee."""
    pr = Protocol(k, ForfeitRule(), Routing.burn(), fee=fee_abs)
    return minimal_deterring_forfeit(Market(100.0, sigma_block), Trade(size=size), pr)


def q1_sweeps(daily_vol=0.04, block_seconds=12.0, fee_bp=5.0, base_k=100, base_size=1.0,
              ks=(10, 25, 50, 100, 200, 400), vols=(0.01, 0.02, 0.04, 0.06, 0.08), sizes=(0.1, 1.0, 10.0, 100.0)):
    """Q1: f* as a function of the window length k, the volatility and the transaction size, one factor at a time.
    The fee is a FLAT amount (fee_bp basis points of a one-unit trade), as on a real ledger; f* is reported in currency and as
    a fraction of the trade's notional (100 * size)."""
    fee = fee_bp * 1e-4 * 100.0
    rows = []
    for k in ks:
        rows.append(dict(factor="k", value=k, k=k, daily_vol=daily_vol, size=base_size))
    for v in vols:
        rows.append(dict(factor="volatility", value=v, k=base_k, daily_vol=v, size=base_size))
    for m in sizes:
        rows.append(dict(factor="size", value=m, k=base_k, daily_vol=daily_vol, size=m))
    for r in rows:
        f = _fstar(r["k"], block_sigma(r["daily_vol"], block_seconds), r["size"], fee)
        r.update(f_star=f, f_star_over_notional=f / (100.0 * r["size"]), window_seconds=r["k"] * block_seconds)
    return pd.DataFrame(rows)


def feasibility_map(daily_vol=0.04, block_seconds=12.0, fee_bp=5.0, q=0.3, surplus_bp=10.0, kappa=0.01,
                    ks=(5, 10, 25, 50, 100, 200, 400, 800), sizes=(0.1, 1.0, 10.0, 100.0, 1000.0)):
    """Where does a feasible forfeit exist (plain commit-reveal + forfeit, FLAT fee), across k and transaction size?
    Lower bound f_lo = f*(k, sigma, size), the smallest forfeit that deters speculation.
    Two upper bounds, both explicit:
      * collateral: a user can be asked to lock at most kappa * notional (an ASSUMPTION, not derived: sweep kappa);
      * participation (individual rationality) against censoring with retries: q^(k-1) (surplus + f) <= surplus. With retries
        this bound is huge, so it almost never binds; it is reported and cross-checked against the solver.
    kappa_min = f_lo / notional is the smallest collateral fraction that works: the map is really a map of kappa_min."""
    fee = fee_bp * 1e-4 * 100.0
    sig = block_sigma(daily_vol, block_seconds)
    rows = []
    for k in ks:
        for m in sizes:
            f_lo = _fstar(k, sig, m, fee)
            notional = 100.0 * m
            u = surplus_bp * 1e-4 * notional
            f_ir = u * (q ** (-(k - 1)) - 1.0) if k <= 60 else math.inf
            ir_ok = None
            if math.isfinite(f_ir) and f_ir < 1e9:
                loss = honest_user_loss(Market(100.0, sig), Protocol(k, ForfeitRule(flat=f_ir), Routing.burn(), fee, 0.0, q, False),
                                        value=fee + u)["loss"]
                ir_ok = bool(abs(loss - u) < 1e-6 * max(u, 1.0))     # at f = f_ir the honest user is exactly indifferent
            f_hi = min(kappa * notional, f_ir)
            rows.append(dict(k=k, size=m, f_lo=f_lo, kappa_min=f_lo / notional, f_hi_collateral=kappa * notional,
                             f_hi_participation=f_ir, feasible=bool(f_lo <= f_hi), participation_check_ok=ir_ok))
    return pd.DataFrame(rows)


def coalition_table(s=0.02, k=6, fee=5e-4, c=1e-4, f=0.01, daily_vol=None, forfeits_over_fstar=(0.5, 1.0, 3.0)):
    """Q3: for each routing rule, the producer's incentive to publish and whether a user-producer coalition with side payments
    beats every equilibrium total (joint deviation, transferable utility). Fallback design, sigma required.
    With daily_vol: measured window volatility and several forfeits expressed as multiples of the measured f*."""
    from .solver import joint_surplus, solve_all_selections, user_value_range
    if daily_vol is not None:
        s = window_vol(daily_vol, k)
    fs = _fstar_fraction(s, k, fee)
    forfeits = [x * fs for x in forfeits_over_fstar] if daily_vol is not None else [f]
    rows = []
    for ff in forfeits:
        for name, r in ROUTINGS.items():
            sc = scenario(s, k, f=ff, fee=fee, c=c, routing=r)
            sels = solve_all_selections(sc)
            tot = max(x.value_user + x.value_producer for _, x in sels)
            pub, wh = sc.protocol.fee - sc.protocol.publish_cost, r.to_producer * sc.f
            rows.append(dict(routing=name, window_vol=s, f_star=fs, forfeit=ff, forfeit_over_fstar=ff / fs if fs > 0 else float("nan"),
                             producer_publishes=bool(pub >= wh), producer_gain_from_publishing=(pub - wh) / NOTIONAL,
                             speculator_value_bp=max(max(x.value_user for _, x in sels), user_value_range(sc)[1]) / NOTIONAL * 1e4,
                             coalition_gain_bp=(joint_surplus(sc) - tot) / NOTIONAL * 1e4))
    return pd.DataFrame(rows)


def contest_table(**kw):
    """Q4: race vs bonded contest (see contest.py for the formalization of the design document's two open mechanisms)."""
    from .contest import compare_mechanisms
    return compare_mechanisms(**kw)


def rotating_table(daily_vol, ks=(4, 8, 12, 20), alphas=(0.1, 0.25, 0.5, 0.75, 0.9, 1.0), fee=5e-4, c=1e-4, surplus=1e-3):
    """Endogenous censoring by rotating producers: extortion threshold (as a multiple of the single-producer threshold fee / s_P) against the
    strategic entity's stake share alpha and the window length k. Producer routing, fallback design, honest transfer user."""
    from .rotating import extortion_threshold
    rows = []
    for k in ks:
        s = window_vol(daily_vol, k)
        mk = lambda f, s=s, k=k: scenario(s, k, f=f, fee=fee, c=c, routing=Routing.producer(), fallback=True,
                                          kind=PayoffKind.TRANSFER, value=fee + surplus)
        for a in alphas:
            thr = extortion_threshold(mk, a, 0.1 * fee, 1e6 * fee)
            rows.append(dict(k=k, alpha=a, window_vol=s, threshold=thr, threshold_over_single=thr / fee if math.isfinite(thr) else float("inf"),
                             single_producer_threshold=fee))
    return pd.DataFrame(rows)


def bayes_private_tables(daily_vol, k=41, fee=5e-4, qs=(0.05, 0.3, 0.6), n_grid=161, prior=None, eps=0.01):
    """Private user type theta (hidden exposure): interim participation/extraction and the designer's ex-ante choice of the forfeit.
    The reported forfeit is the smallest one within ``eps`` of the best achievable loss reduction, located by ``refine_eps_forfeit`` so
    it is not limited to the coarse grid. The exact minimiser, the 95% version and the coarse-grid value are kept for comparison."""
    from . import bayes_private as B
    s = window_vol(daily_vol, k)
    fs = _fstar_fraction(s, k, fee)
    grid = [40.0 * fs * i / (n_grid - 1) for i in range(n_grid)]            # 0 .. 40 f*
    inter, design, summ = [], [], []
    for q in qs:
        rows, agg = B.design_table(s, k, fee, q, prior=prior, f_grid=grid)
        rows["q"], agg["q"] = q, q
        inter.append(rows)
        design.append(agg)
        exact = B.optimal_forfeit(agg)
        coarse = B.eps_optimal_forfeit(agg, eps)
        f99 = B.refine_eps_forfeit(agg, s, k, fee, q, prior=prior, eps=eps)
        f95 = B.refine_eps_forfeit(agg, s, k, fee, q, prior=prior, eps=0.05)
        at, agg99 = B.design_table(s, k, fee, q, prior=prior, f_grid=[f99])
        summ.append(dict(q=q, window_vol=s, f_star=fs, f_opt=f99, f_opt_over_fstar=f99 / fs if fs > 0 else float("nan"),
                         f_eps95_over_fstar=f95 / fs, f_exact_over_fstar=exact["f_opt"] / fs, f_coarse_grid_over_fstar=coarse["f_opt"] / fs,
                         loss_opt=float(agg99.loss.iloc[0]), loss_at_zero=coarse["loss_at_zero"], participation_at_opt=float(agg99.participation.iloc[0]),
                         participation_at_zero=float(agg.participation[0]),
                         cutoff_type_at_zero=B.participation_threshold(rows, 0.0), cutoff_type_at_opt=B.participation_threshold(at, f99)))
    return pd.concat(inter, ignore_index=True), pd.concat(design, ignore_index=True), pd.DataFrame(summ)


def save_figures(extortion: pd.DataFrame, designer: pd.DataFrame, outdir: str, sweeps=None, fmap=None):
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for name, g in extortion.groupby("routing", sort=False):
        g = g[g.f > 0]
        ax.loglog(g.f, np.maximum(g.loss_worst, 1e-6), label=name)
    ax.set(xlabel="forfeit / notional", ylabel="honest user's expected loss / notional",
           title="Extortion by a strategic sequencer")
    ax.legend()
    fig.savefig(os.path.join(outdir, "course_fig_extortion.png"), bbox_inches="tight", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    piv = designer.pivot(index="q", columns="pi", values="f_opt") / designer.f_star.iloc[0]
    im = ax.imshow(piv.values, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1.05)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{c:g}" for c in piv.index])
    ax.set(xlabel="share of speculators", ylabel="censoring probability q", title="Designer's optimal forfeit / f*")
    fig.colorbar(im, ax=ax)
    fig.savefig(os.path.join(outdir, "course_fig_designer.png"), bbox_inches="tight", dpi=130)
    plt.close(fig)
    if sweeps is not None:
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
        for a, fac, lab in zip(ax, ("k", "volatility", "size"), ("window length k (blocks)", "daily volatility", "transaction size (units)")):
            d = sweeps[sweeps.factor == fac]
            a.loglog(d.value, d.f_star, "o-")
            a.set(xlabel=lab, ylabel="f* (currency)", title=f"f* vs {fac}")
            a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        fig.savefig(os.path.join(outdir, "course_fig_q1_sweeps.png"), bbox_inches="tight", dpi=130)
        plt.close(fig)
    if fmap is not None:
        fig, ax = plt.subplots(figsize=(6.0, 3.9))
        piv = fmap.pivot(index="size", columns="k", values="kappa_min")
        im = ax.imshow(np.log10(np.maximum(piv.values, 1e-9)), origin="lower", aspect="auto", cmap="viridis_r")
        for a, sz in enumerate(piv.index):
            for b, k in enumerate(piv.columns):
                ok = bool(fmap[(fmap["size"] == sz) & (fmap.k == k)].feasible.iloc[0])
                ax.text(b, a, "ok" if ok else "x", ha="center", va="center", fontsize=7, color="white" if ok else "black")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{v:g}" for v in piv.index])
        ax.set(xlabel="window length k (blocks)", ylabel="transaction size (units)",
               title="Smallest collateral fraction kappa_min (colour); ok = feasible at the assumed kappa")
        ax.title.set_fontsize(8)
        fig.colorbar(im, ax=ax, label="log10 kappa_min")
        fig.savefig(os.path.join(outdir, "course_fig_feasibility_map.png"), bbox_inches="tight", dpi=130)
        plt.close(fig)


def impossibility_search(n_trials=300, seed=2026, routings=None):
    """Random designs checked against the three properties of a fallback design, per routing rule (sigma required, no contestation):
      P1 deters speculation (V_0 <= 0 in every pure equilibrium, node-specific ties included),
      P2 the producer STRICTLY prefers to publish (fee - c > s_P f),
      P3 no user-producer coalition with side payments beats every equilibrium total (exact, gamma = 0).
    THEOREM (proved in the report; checked here): if the forfeit stays inside the coalition (s_other = 0: self or producer routing, or any
    mix of the two) then P2 implies the coalition gains at least (1 - q) E[g_1^+] > 0, so P3 fails.  Also records that bound."""
    from .solver import joint_surplus, solve_all_selections, user_value_range
    rng = random.Random(seed)
    routings = routings or {"self": Routing.to_self(), "producer": Routing.producer(), "mixed 50/50": Routing(0.5, 0.5, 0.0, "mixed"),
                            "pool(4)": Routing.pool(4), "burn": Routing.burn()}
    rows = []
    for name, r in routings.items():
        for _ in range(n_trials):
            s, k = 10 ** rng.uniform(-3, -0.7), rng.randint(2, 40)
            fee = 10 ** rng.uniform(-5, -2)
            c = fee * rng.choice([0.0, 0.2, 0.6, 0.95])
            f = rng.choice([0.0, 10 ** rng.uniform(-6, -0.3)])
            q = rng.choice([0.0, 0.1, 0.4])
            sc = scenario(s, k, f=f, fee=fee, c=c, q=q, routing=r)
            sels = solve_all_selections(sc)
            v0 = max(max(x.value_user for _, x in sels), user_value_range(sc)[1])
            tot = max(x.value_user + x.value_producer for _, x in sels)
            gain = joint_surplus(sc) - tot
            sigma_b = sc.market.sigma
            bound = (1 - q) * sc.market.p * NOTIONAL * (math.exp(sigma_b) - 1.0)          # (1-q) E[g_1^+] for the call
            p2s = bool(fee - c > r.to_producer * f)
            rows.append(dict(routing=name, s_other=r.to_other, window_vol=s, k=k, fee=fee, c=c, f=f, q=q, p1=bool(not gt(v0, 0.0)), p2_strict=p2s,
                             p3_exact=bool(not gt(gain, 0.0)), gain_bp=gain / NOTIONAL * 1e4, bound_bp=bound / NOTIONAL * 1e4,
                             total_spne_bp=tot / NOTIONAL * 1e4))
    df = pd.DataFrame(rows)
    df["p123"] = df.p1 & df.p2_strict & df.p3_exact
    return df
