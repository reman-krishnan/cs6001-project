"""Turn the result files into the report's LaTeX tables, number macros and figures. Nothing in the report's numbers is typed by hand.

    python scripts/make_report_assets.py                       # reads ./results/main, ./results/illustrative, optional ./results/periods/<period> folders
    python scripts/make_report_assets.py --periods results/periods   # folder holding all, 2018_2019, ... (from scripts/run_periods.sh)
Writes ./report/assets/{numbers.tex, tab_*.tex, figs/*.png}; then:  cd report && pdflatex report.tex (twice).
"""
import argparse
import json
import math
import os
import shutil

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ap = argparse.ArgumentParser()
ap.add_argument("--results", default=os.path.join(ROOT, "results", "main"))
ap.add_argument("--illustrative", default=os.path.join(ROOT, "results", "illustrative"))
ap.add_argument("--periods", default=os.path.join(ROOT, "results", "periods"))
ap.add_argument("--out", default=os.path.join(ROOT, "report", "assets"))
ap.add_argument("--tests", default=None, help="number of tests to print in the report. Default: counted by pytest collection")
a = ap.parse_args()
# The report says "all N tests pass" and "all M injected bugs caught", so both numbers come from an actual, successful test run,
# never from collection alone. --tests overrides the count only for drafts (the mutation count is then still taken from a run).
import re
import subprocess
import sys as _sys
_run = subprocess.run([_sys.executable, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
if _run.returncode != 0:
    raise SystemExit("the test suite failed, so the report cannot say that the tests pass:\n" + _run.stdout[-3000:])
_m = re.search(r"(\d+) passed", _run.stdout)
if not _m or re.search(r"\d+ (failed|error)", _run.stdout):
    raise SystemExit("could not confirm a clean test run from pytest's summary")
if a.tests is None:
    a.tests = _m.group(1)
n_mut = str(sum(l.startswith("PASSED tests/test_mutations.py::test_injected_bug_is_caught") for l in _run.stdout.splitlines()))
os.makedirs(os.path.join(a.out, "figs"), exist_ok=True)
R = lambda f: pd.read_csv(os.path.join(a.results, f))
macros = {}


def M(name, value):
    macros[name] = value


pct = lambda x, d=2: f"{x * 100:.{d}f}\\%"
# Display policy: every minimum forfeit (f*, f_cp) or bond is rounded UP, so the value shown suffices; interval ends are rounded
# outward (low down, high up); every other quantity is rounded to nearest.
pct_up = lambda x, d=2: f"{math.ceil(round(x * 100 * 10 ** d, 6)) / 10 ** d:.{d}f}\\%"
pct_dn = lambda x, d=2: f"{math.floor(round(x * 100 * 10 ** d, 6)) / 10 ** d:.{d}f}\\%"
bp_up = lambda x, d=1: f"{math.ceil(round(x * 10 ** d, 6)) / 10 ** d:.{d}f}"
num_up = lambda x, d=3: f"{math.ceil(round(x * 10 ** d, 6)) / 10 ** d:.{d}f}"
bp = lambda x, d=1: (f"$-{abs(x):.{d}f}$" if x < 0 and round(abs(x), d) > 0 else f"{abs(x) if round(abs(x), d) == 0 else x:.{d}f}")


def sci(x):
    e = int(math.floor(math.log10(x))) if x > 0 else 0
    return f"$10^{{{e}}}$" if x > 0 else "0"
yn = lambda b: "Yes" if b else "No"


def bordered(spec):
    """Fully bordered column spec: a vertical rule around every column."""
    return "|" + "|".join(c for c in spec if c in "lrc") + "|"


def table(path, spec, header, rows):
    """Write a fully bordered tabular with a shaded header row (needs xcolor with the table option)."""
    with open(os.path.join(a.out, path), "w") as fh:
        fh.write("\\begin{tabular}{" + bordered(spec) + "}\n\\hline\n\\rowcolor{gray!15}" + " & ".join(header) + " \\\\\n\\hline\n")
        for r in rows:
            fh.write(" & ".join(str(c) for c in r) + " \\\\\n\\hline\n")
        fh.write("\\end{tabular}\n")


info = json.load(open(os.path.join(a.results, "run_info.json")))
cal = R("calibration.csv").iloc[0]
M("DailyVol", pct(cal.daily_vol)); M("NReturns", f"{int(cal.n_returns):,}"); M("FeeBp", f"{info['fee_bp']}"); M("CostBp", f"{info['publish_cost_bp']}")
M("SurplusBp", f"{info['honest_surplus_bp']}"); M("QK", f"{info['q1_k']}"); M("DesignerK", f"{info['designer_k']}"); M("NTests", a.tests); M("NMutants", n_mut)
M("VolSource", info["volatility_source"])

# ---- Q1 ----
q1 = R("q1_sweeps.csv")
byf = {f: q1[q1.factor == f].reset_index(drop=True) for f in ("k", "volatility", "size")}
rows = []
for i in range(max(len(v) for v in byf.values())):
    row = []
    for f, fmt in (("k", lambda v: f"{int(v)}"), ("volatility", lambda v: pct(v, 0)), ("size", lambda v: f"{v:g}")):
        d = byf[f]
        row += [fmt(d.value[i]), pct_up(d.f_star_over_notional[i], 3)] if i < len(d) else ["", ""]
    rows.append(row)
table("tab_q1.tex", "rr|rr|rr", ["$k$", "$f^*$", "daily vol.", "$f^*$", "size", "$f^*$"], rows)
fk = byf["k"].set_index("value").f_star_over_notional
M("FstarKTen", pct_up(fk[10], 3)); M("FstarKHundred", pct_up(fk[100], 3)); M("FstarKFourHundred", pct_up(fk[400], 3))
fs = byf["size"].set_index("value").f_star_over_notional
M("FstarSizeSmall", pct_up(fs[0.1], 3)); M("FstarSizeOne", pct_up(fs[1.0], 3)); M("FstarSizeLarge", pct_up(fs[100.0], 3))
vv = byf["volatility"]
M("VolSweepFactor", f"{vv.value.iloc[-1] / vv.value.iloc[0]:.0f}"); M("VolSweepRatio", f"{vv.f_star.iloc[-1] / vv.f_star.iloc[0]:.0f}")
M("WindowMinutes", f"{info['q1_k'] * info['block_seconds'] / 60:.0f}")
fm = R("q1_feasibility_map.csv")
M("NFeasible", f"{int(fm.feasible.sum())}"); M("NCells", f"{len(fm)}")
M("KappaAssumed", pct((fm.f_hi_collateral / (100.0 * fm["size"])).iloc[0], 0))
kk = [5, 25, 100, 400, 800]
piv = fm.pivot(index="size", columns="k", values="kappa_min").sort_index(ascending=False)
table("tab_kappa.tex", "r" + "r" * len(kk), ["size \\textbackslash{} $k$"] + [str(k) for k in kk],
      [[f"{sz:g}"] + [pct(piv.loc[sz, k], 2) for k in kk] for sz in piv.index])
M("KappaMax", pct(fm.kappa_min.max(), 1))

# ---- periods (optional) ----
names = {"all": "Pooled 2017--2026", "2018_2019": "2018--2019", "2020_2021": "2020--2021", "2022_2023": "2022--2023", "2024_2026": "2024--2026"}
prow, have = [], True
for p in names:
    d = os.path.join(a.periods, p)
    if not os.path.isdir(d):
        have = False; break
    c = pd.read_csv(os.path.join(d, "calibration.csv")).iloc[0]
    s = pd.read_csv(os.path.join(d, "q1_sweeps.csv"))
    f100 = s[(s.factor == "k") & (s.value == 100)].f_star_over_notional.iloc[0]
    q2 = pd.read_csv(os.path.join(d, "q2_routing_designs.csv")).set_index("design")
    plain = q2.speculator_value_bp["plain commit-reveal (no forfeit)"]
    prow.append((names[p], c.daily_vol, f100, plain))
if have:
    table("tab_periods.tex", "lrrr", ["Period", "daily vol.", "$f^*$ ($k{=}100$)", "plain commit-reveal, speculator value (bp)"],
          [[n, pct(v), pct_up(f, 2), bp(pl)] for n, v, f, pl in prow])
    sub = [x for x in prow if x[0] != names["all"]]
    M("VolLow", pct(min(x[1] for x in sub))); M("VolHigh", pct(max(x[1] for x in sub)))
    M("FstarLow", pct_up(min(x[2] for x in sub))); M("FstarHigh", pct_up(max(x[2] for x in sub)))
    M("SpecPlainLow", bp(min(x[3] for x in sub))); M("SpecPlainHigh", bp(max(x[3] for x in sub)))
    M("FstarRatio", f"{max(x[2] for x in sub) / min(x[2] for x in sub):.1f}")
else:
    with open(os.path.join(a.out, "tab_periods.tex"), "w") as fh:
        fh.write("\\emph{(Run \\texttt{run\\_periods.sh} and re-run this script with \\texttt{--periods} to fill in the period comparison.)}\n")
    for k in ("VolLow", "VolHigh", "FstarLow", "FstarHigh", "SpecPlainLow", "SpecPlainHigh", "FstarRatio"):
        M(k, "n/a")

# ---- Q2 ----
q2 = R("q2_routing_designs.csv")
tex = lambda d: d.replace("(f > fee - c)", "($f>\\mathrm{fee}-c$)").replace("sigma optional", "signature $\\Sigma$ optional")
table("tab_q2.tex", "lrrc", ["Design", "$f/f^*$", "speculator value (bp)", "deterred?"],
      [[tex(r.design), f"{r.forfeit_over_fstar:.1f}", bp(r.speculator_value_bp), yn(r.speculation_deterred)] for r in q2.itertuples()])
v = q2.set_index("design").speculator_value_bp
M("SpecPlain", bp(v["plain commit-reveal (no forfeit)"])); M("SpecSmall", bp(v["forfeit only, small"])); M("SpecLarge", bp(v["forfeit only, large"]))
M("SpecFallback", bp(v["fallback + burn"])); M("SpecProducer", bp(v["fallback + producer routing (f > fee - c)"]))
M("SpecOptional", bp(v["fallback + burn, sigma optional"])); M("SpecSelf", bp(v["self routing"]))
M("WindowVol", pct(q2.window_vol.iloc[0], 2)); M("FstarWindow", pct_up(q2.f_star.iloc[0], 3))

# ---- Q3 coalition ----
c3 = R("q3_coalition.csv")
order = ["burn", "self", "pool(4)", "producer"]
mult = sorted(c3.forfeit_over_fstar.unique())
rows = []
for r in order:
    d = c3[c3.routing == r].sort_values("forfeit_over_fstar")
    pub = d.producer_publishes.all(); never = not d.producer_publishes.any()
    rows.append([r, "Yes" if pub else ("No" if never else "mixed")] + [bp(x, 2) for x in d.coalition_gain_bp])
table("tab_coalition.tex", "lc" + "r" * len(mult), ["routing", "producer publishes?"] + [f"gain at {m:g}$f^*$ (bp)" for m in mult], rows)
g = lambda r, i: c3[c3.routing == r].sort_values("forfeit_over_fstar").coalition_gain_bp.iloc[i]
M("CoalBurnHalf", bp(g("burn", 0))); M("CoalBurnOne", bp(g("burn", 1))); M("CoalBurnThree", f"{g('burn', 2):.3f}")
M("CoalSelfMin", bp(c3[c3.routing == "self"].coalition_gain_bp.min())); M("CoalSelfMax", bp(c3[c3.routing == "self"].coalition_gain_bp.max()))
M("CoalPoolMax", bp(c3[c3.routing == "pool(4)"].coalition_gain_bp.max(), 2)); M("CoalProdHalf", bp(g("producer", 0))); M("CoalProdThree", bp(g("producer", 2)))

# ---- Q3 extortion ----
e = R("q3_producer_incentives.csv")
rows = []
surplus = info["honest_surplus_bp"] * 1e-4
for r in ["self", "burn", "pool(4)", "producer"]:
    d = e[e.routing == r]
    hit = d[d.loss_worst > 1e-9]
    thr = d.threshold.iloc[0]
    rows.append([r, "none" if not math.isfinite(thr) else pct(thr, 3), "never" if not len(hit) else pct(hit.f.min(), 3),
                 "0" if not len(hit) else pct(hit.loss_worst.iloc[0], 3)])
table("tab_extortion.tex", "lrrr", ["routing", "threshold $\\phi/\\rho_P$", "first loss on grid at $f=$", "loss there"], rows)
M("ThreshProducer", pct(e[e.routing == "producer"].threshold.iloc[0], 3)); M("ThreshPool", pct(e[e.routing == "pool(4)"].threshold.iloc[0], 3))

# ---- designer ----
b = R("bayes_designer.csv")
rc = b.regime.value_counts().to_dict()
M("DesNoForfeit", f"{rc.get('no forfeit', 0)}"); M("DesPartial", f"{rc.get('partial', 0)}"); M("DesFull", f"{rc.get('full deterrence', 0)}")
M("DesWindowVol", pct(b.window_vol.iloc[0], 3)); M("DesFstarBp", bp_up(b.f_star.iloc[0] * 1e4, 1))

# ---- Han ----
h = R("han_check.csv")
table("tab_han.tex", "rrrr", ["daily vol.", "Han method (\\%)", "game solver (\\%)", "$|$diff$|$"],
      [[pct(r.daily_vol, 2), f"{r.han_method_pct:.4f}", f"{r.game_solver_pct:.4f}", f"{r.abs_diff:.1e}"] for r in h.itertuples()])
M("HanMaxDiff", sci(h.abs_diff.max()))

# ---- Q4 contest ----
has = lambda f: os.path.exists(os.path.join(a.results, f))
if has("q4_contest.csv"):
    c4 = R("q4_contest.csv")
    r0 = c4.iloc[0]
    M("ContestVo", f"{r0.V_owner:g}"); M("ContestKappa", f"{r0.attack_cost:g}"); M("ContestCr", f"{r0.response_cost:g}"); M("ContestCap", f"{r0.owner_capital:g}")
    table("tab_contest.tex", "rrccrc", ["$a$", "$V_a/V_o$", "race deters?", "attack cost to deter race", "smallest bond", "bond deters?"],
          [[f"{r.availability:g}", f"{r.value_ratio:g}", yn(r.race_deterred) + (" (tie)" if r.race_tie else ""), num_up(r.race_min_attack_cost_to_deter),
            "n/a" if not r.bond_feasible else num_up(r.bond_star), yn(r.bond_deterred) if r.bond_feasible else "no (bond $>K$)"] for r in c4.itertuples()])
    M("ContestRaceTies", f"{int(c4.race_tie.sum())}")
    M("ContestRaceDeterCells", f"{int(c4.race_deterred.sum())}"); M("ContestCells", f"{len(c4)}")
    M("ContestBondFeasible", f"{int(c4.bond_feasible.sum())}"); M("ContestBondDeters", f"{int((c4.bond_feasible & c4.bond_deterred).sum())}")
    st = c4[(c4.value_ratio > 1) & (c4.availability >= 0.8)]
    M("ContestStrongRaceDeter", f"{int(st.race_deterred.sum())}"); M("ContestStrongBondDeter", f"{int(st.bond_deterred.sum())}"); M("ContestStrongN", f"{len(st)}")

# ---- rotating producers ----
if has("q3_rotating.csv"):
    rt = R("q3_rotating.csv")
    ks_ = sorted(rt.k.unique())
    piv_r = rt.pivot(index="alpha", columns="k", values="threshold_over_single")
    fmt = lambda v: "$>10^{6}$" if not math.isfinite(v) else (f"{v:.1f}" if v < 100 else f"{v:,.0f}")
    table("tab_rotating.tex", "r" + "r" * len(ks_), ["stake share $\\alpha$"] + [f"$k={k}$" for k in ks_],
          [[f"{al:g}"] + [fmt(piv_r.loc[al, k]) for k in ks_] for al in piv_r.index])
    fin = rt[rt.threshold_over_single.apply(math.isfinite)]
    cf = 1.0 / (fin.alpha ** (fin.k - 1))
    M("RotCFdev", sci((fin.threshold_over_single / cf - 1).abs().max())); M("RotCells", f"{len(fin)}")
    g = lambda al, k: piv_r.loc[al, k]
    M("RotHalfFour", f"{g(0.5, 4):.0f}"); M("RotHalfTwelve", f"{g(0.5, 12):,.0f}")

# ---- impossibility ----
if has("q3_impossibility.csv"):
    im = R("q3_impossibility.csv")
    order_ = ["self", "producer", "mixed 50/50", "pool(4)", "burn"]
    table("tab_impossibility.tex", "lrrrrr", ["routing", "designs", "P1 deters", "P2 strict", "P3 exact", "all three"],
          [[r, int((im.routing == r).sum()), int(im[im.routing == r].p1.sum()), int(im[im.routing == r].p2_strict.sum()),
            int(im[im.routing == r].p3_exact.sum()), int(im[im.routing == r].p123.sum())] for r in order_])
    th = im[(im.s_other == 0.0) & im.p2_strict]
    M("ThmN", f"{len(th)}"); M("ThmAllPositive", "all" if bool((th.gain_bp > 0).all()) else "not all")
    mm = float(th.gain_bp.sub(th.bound_bp).min())          # Prop. 4: gain >= bound. A negative margin beyond rounding is a violation.
    if mm < -1e-9:
        raise SystemExit(f"impossibility bound violated: min(gain - bound) = {mm} bp")
    M("ThmMinMargin", f"{mm:.1e}"); M("ThmMaxTotal", sci(max(float(th.total_spne_bp.max()), 1e-15)))
    for nm, mac in (("burn", "ImpBurnAll"), ("pool(4)", "ImpPoolAll"), ("producer", "ImpProdAll"), ("self", "ImpSelfAll"), ("mixed 50/50", "ImpMixedAll")):
        M(mac, f"{int(im[im.routing == nm].p123.sum())}")
    M("ImpPerRouting", f"{int((im.routing == 'burn').sum())}")

# ---- private-type Bayesian designer ----
if has("bayes_private_summary.csv"):
    bp_ = R("bayes_private_summary.csv").sort_values("q")
    table("tab_bayespriv.tex", "rrrrrrr", ["$q$", "99\\% rule$/f^*$", "95\\% rule$/f^*$", "exact minimiser$/f^*$", "participation (99\\% rule)", "loss at $f{=}0$ (bp)", "loss at 99\\% rule (bp)"],
          [[f"{r.q:g}", num_up(r.f_opt_over_fstar, 2), num_up(r.f_eps95_over_fstar, 2), f"{r.f_exact_over_fstar:.2f}", f"{r.participation_at_opt:.2f}",
            bp(r.loss_at_zero * 1e4, 2), bp(r.loss_opt * 1e4, 2)] for r in bp_.itertuples()])
    M("BPLow", num_up(bp_.f_opt_over_fstar.min(), 2)); M("BPHigh", num_up(bp_.f_opt_over_fstar.max(), 2))
    M("BPEpsLooseLow", num_up(bp_.f_eps95_over_fstar.min(), 2)); M("BPEpsLooseHigh", num_up(bp_.f_eps95_over_fstar.max(), 2))
    M("BPExactLow", f"{bp_.f_exact_over_fstar.min():.1f}"); M("BPExactHigh", f"{bp_.f_exact_over_fstar.max():.1f}")
    M("BPPartOpt", f"{bp_.participation_at_opt.iloc[0]:.2f}"); M("BPCutoff", f"{bp_.cutoff_type_at_opt.iloc[0] * 1e4:.0f}")
    M("BPFstarBp", bp_up(bp_.f_star.iloc[0] * 1e4, 1)); M("BPWindowVol", pct(bp_.window_vol.iloc[0], 2))

# ---- optional: confidence interval, intraday robustness, other assets (only if the data were available) ----
if has("q1_fstar_ci.csv"):
    ci = R("q1_fstar_ci.csv").iloc[0]
    M("HaveCI", "1"); M("VolCI", f"{pct_dn(ci.daily_vol_low)}--{pct_up(ci.daily_vol_high)}"); M("FstarCI", f"{pct_dn(ci.fstar_low, 2)}--{pct_up(ci.fstar_high, 2)}")
emp_dir = os.path.join(ROOT, "results", "intraday_1m")
if os.path.exists(os.path.join(emp_dir, "empirical_fstar.csv")):
    e_ = pd.read_csv(os.path.join(emp_dir, "empirical_fstar.csv")).iloc[0]
    v_ = pd.read_csv(os.path.join(emp_dir, "variance_ratio.csv"))
    M("HaveEmp", "1"); M("EmpWindow", f"{e_.window_minutes:g}"); M("EmpKurt", f"{e_.excess_kurtosis:.1f}"); M("EmpFstar", pct_up(e_.f_empirical, 3))
    M("EmpStepKurt", f"{e_.step_excess_kurtosis:.1f}"); M("EmpCI", f"{pct_dn(e_.ci_low, 3)}--{pct_up(e_.ci_high, 3)}"); M("EmpWindowVol", pct(e_.window_vol, 2)); M("EmpNWindows", f"{int(e_.n_windows):,}")
    m1 = pd.read_csv(os.path.join(ROOT, "data", "ETHUSDT_1m.csv"), usecols=["date"]).date
    M("EmpFrom", pd.Timestamp(m1.iloc[0]).strftime("%-d %B %Y")); M("EmpTo", pd.Timestamp(m1.iloc[-1]).strftime("%-d %B %Y"))
    M("EmpImpliedDaily", pct(e_.window_vol / math.sqrt(e_.window_minutes * 60.0 / 86400.0), 1)); M("EmpLognormal", pct_up(e_.f_lognormal, 3)); M("EmpRatio", f"{e_.ratio:.2f}")
    table("tab_vr.tex", "rrrrr", ["multiple $m$", "VR($m$)", "$z$", "robust $z$", "implied exponent"], [[int(r.multiple), f"{r.variance_ratio:.3f}", f"{r.z:.1f}", f"{r.z_robust:.1f}", f"{r.implied_exponent:.3f}"] for r in v_.itertuples()])
tw_dir = os.path.join(ROOT, "results", "intraday_12s")
if os.path.exists(os.path.join(tw_dir, "empirical_fstar.csv")):
    t_ = pd.read_csv(os.path.join(tw_dir, "empirical_fstar.csv")).iloc[0]
    tv_ = pd.read_csv(os.path.join(tw_dir, "variance_ratio.csv"))
    tb_ = pd.read_csv(os.path.join(tw_dir, "block_vol.csv")).iloc[0]
    d12 = pd.read_csv(os.path.join(ROOT, "data", "ETHUSDT_12s.csv"), usecols=["date"]).date
    M("HaveTwelve", "1"); M("TwN", f"{int(t_.n_windows):,}"); M("TwWindowVol", pct(t_.window_vol, 2)); M("TwKurt", f"{t_.excess_kurtosis:.1f}")
    M("TwStepKurt", f"{t_.step_excess_kurtosis:.1f}"); M("TwZeroShare", pct(t_.zero_step_share, 0)); M("TwFstar", pct_up(t_.f_empirical, 3)); M("TwCI", f"{pct_dn(t_.ci_low, 3)}--{pct_up(t_.ci_high, 3)}"); M("TwLognormal", pct(t_.f_lognormal, 3)); M("TwRatio", f"{t_.ratio:.2f}")
    M("TwFrom", pd.Timestamp(d12.iloc[0]).strftime("%-d %B")); M("TwTo", pd.Timestamp(d12.iloc[-1]).strftime("%-d %B %Y"))
    M("TwSigmaStep", pct(tb_.sigma_step_realized, 4)); M("TwImpliedStep", pct(tb_.sigma_step_implied_by_calibration, 4))
    M("TwBlockRatio", f"{tb_.ratio_realized_to_implied:.2f}"); M("TwDailyEq", pct(tb_.daily_equivalent_from_steps, 1))
    M("TwVRTwo", f"{tv_[tv_.multiple == 2].variance_ratio.iloc[0]:.2f}")
    M("TwVRLow", f"{tv_[tv_.multiple >= 5].variance_ratio.min():.2f}"); M("TwVRHigh", f"{tv_[tv_.multiple >= 5].variance_ratio.max():.2f}")
    r100 = tv_[tv_.multiple == 100].iloc[0]
    M("TwVRWindow", f"{r100.variance_ratio:.2f}"); M("TwZWindow", f"{r100.z:.1f}"); M("TwUnderstate", f"{(math.sqrt(r100.variance_ratio) - 1) * 100:.0f}\\%")
    table("tab_vr12.tex", "rrrrr", ["multiple $m$ (12 s)", "VR($m$)", "$z$", "robust $z$", "implied exponent"], [[int(r.multiple), f"{r.variance_ratio:.3f}", f"{r.z:.1f}", f"{r.z_robust:.1f}", f"{r.implied_exponent:.3f}"] for r in tv_.itertuples()])
    sig_ = tv_[tv_.z_robust > 1.96]
    M("TwSigMinutes", f"{int(sig_.multiple.max()) * float(tb_.step_seconds) / 60.0:.0f}" if len(sig_) else "0")
    M("TwZRobWindow", f"{tv_[tv_.multiple == 100].z_robust.iloc[0]:.1f}"); M("TwZRobSig", f"{sig_.z_robust.min():.1f}" if len(sig_) else "n/a")
    M("TwZRobShort", f"{tv_[tv_.multiple == 2].z_robust.iloc[0]:.1f}")
    M("TwNBoot", f"{int(t_.n_boot):,}"); M("TwSeed", f"{int(t_.seed)}")
asset_rows = []
for sym in ("ETHUSDT", "BTCUSDT", "SOLUSDT", "BNBUSDT"):
    d_ = os.path.join(ROOT, "results", "assets", sym)
    if os.path.isdir(d_):
        c_ = pd.read_csv(os.path.join(d_, "calibration.csv")).iloc[0]
        q_ = pd.read_csv(os.path.join(d_, "q1_sweeps.csv"))
        f_ = q_[(q_.factor == "k") & (q_.value == 100)].f_star_over_notional.iloc[0]
        asset_rows.append([sym, pct(c_.daily_vol), f"{pct_dn(c_.ci_low)}--{pct_up(c_.ci_high)}", pct_up(f_, 2)])
if asset_rows:
    M("HaveAssets", "1")
    fvals = [float(r[3].replace("\\%", "")) for r in asset_rows]
    M("AssetFLow", f"{min(fvals):.2f}\\%"); M("AssetFHigh", f"{max(fvals):.2f}\\%")
    table("tab_assets.tex", "lrrr", ["asset", "daily vol.", "95\\% interval", "$f^*$ ($k{=}100$)"], asset_rows)

# ---- supplementary analyses (scripts/run_extras.py) ----
if has("extras_fee_sensitivity.csv"):
    fe = R("extras_fee_sensitivity.csv")
    table("tab_fee.tex", "rrrr", ["fee (bp)", "fee at commitment (bp)", "$f^*$ (lattice)", "$f^*$ (closed form)"],
          [[f"{r.fee_bp:g}", f"{r.commit_fee_bp:g}", pct_up(r.f_star, 3), "" if math.isnan(r.f_gauss) else pct_up(r.f_gauss, 3)] for r in fe.itertuples()])
    base = fe[(fe.fee_bp == 5) & (fe.commit_fee_bp == 0)].iloc[0]
    M("GaussFstar", pct_up(base.f_gauss, 3)); M("FeeLowF", pct_up(fe[fe.fee_bp == 1].f_star.iloc[0], 2)); M("FeeHighF", pct_up(fe[fe.fee_bp == 20].f_star.iloc[0], 2))
    M("CommitFeeFive", pct_up(fe[(fe.commit_fee_bp == 5)].f_star.iloc[0], 2))
    M("GaussMaxRelErr", f"{((fe.f_gauss - fe.f_star).abs() / fe.f_star).max() * 100:.1f}\\%")
if has("extras_burn_range.csv"):
    br = R("extras_burn_range.csv")
    vols = sorted(br.daily_vol.unique())
    kap = float((fm.f_hi_collateral / (100.0 * fm["size"])).iloc[0])
    cell = lambda r: f"{pct_up(r.f_star, 2)[:-2]} / {pct_up(r.f_cp, 2)[:-2]}" + ("$^{\\ast}$" if r.f_cp > kap else "")
    table("tab_burnrange.tex", "r" + "c" * len(vols), ["$k$"] + [f"daily vol.\\ {pct(v, 2)}" for v in vols],
          [[f"{k}"] + [cell(br[(br.k == k) & (br.daily_vol == v)].iloc[0]) for v in vols] for k in sorted(br.k.unique())])
    pooled = br[br.daily_vol == vols[1]].set_index("k")
    M("CpRatioHundred", f"{pooled.f_cp[100] / pooled.f_star[100]:.1f}"); M("CpHundred", pct_up(pooled.f_cp[100], 2)); M("CpFifty", pct_up(pooled.f_cp[50], 2))
    M("CpTwoHundred", pct_up(pooled.f_cp[200], 2)); M("CpTen", pct_up(pooled.f_cp[10], 2))
    M("NBurnInfeasible", f"{int((br.f_cp > kap).sum())}"); M("NBurnCells", f"{len(br)}")
if has("extras_frontier.csv"):
    fr = R("extras_frontier.csv").sort_values("daily_vol")
    base_fr = fr.iloc[1]
    M("FrontierK", f"{int(base_fr.k_max)}"); M("FrontierMin", f"{base_fr.minutes:.1f}")
    M("FrontierEightK", f"{int(fr.iloc[2].k_max)}"); M("FrontierEightMin", f"{fr.iloc[2].minutes:.1f}")
    M("FrontierTwoK", f"{int(fr.iloc[0].k_max)}"); M("FrontierTwoMin", f"{fr.iloc[0].minutes:.1f}")
if has("extras_tolerance.csv"):
    tl = R("extras_tolerance.csv").sort_values("eps_bp", ascending=False)
    table("tab_tolerance.tex", "rrrr", ["tolerance (bp)", "$f_{\\mathrm{cp}}$ at $k{=}100$", "largest $k$ within 1\\%", "minutes"],
          [[f"{r.eps_bp:g}", pct_up(r.f_cp, 2), f"{int(r.k_max)}", f"{r.minutes:.1f}"] for r in tl.itertuples()])
    t_ = tl.set_index("eps_bp")
    M("CpTolOne", pct_up(t_.f_cp[1.0], 2)); M("CpTolHundredth", pct_up(t_.f_cp[0.01], 2)); M("CpTolThousandth", pct_up(t_.f_cp[0.001], 2))
    M("FrontierTolOneMin", f"{t_.minutes[1.0]:.1f}"); M("FrontierTolHundredthMin", f"{t_.minutes[0.01]:.1f}")
if has("extras_pool.csv"):
    po = R("extras_pool.csv")
    ok_ = po[po.works]
    if len(ok_):
        r_ = ok_.iloc[0]
        M("PoolN", f"{int(r_.n)}"); M("PoolF", pct_up(r_.f_cp, 2)); M("PoolExt", pct(r_.extortion_threshold, 2))
        M("PoolNoExt", "1" if bool(r_.extortion_free) else "0")
if has("extras_bayes_resolution.csv"):
    br_ = R("extras_bayes_resolution.csv")
    lo_, hi_ = br_.steps.min(), br_.steps.max()
    M("BPStepsLow", f"{int(lo_)}"); M("BPStepsHigh", f"{int(hi_)}")
    M("BPEpsCoarse", num_up(br_[br_.steps == lo_].eps99_over_fstar.min(), 2))
    M("BPEpsFine", num_up(br_[br_.steps == hi_].eps99_over_fstar.max(), 2)); M("BPEpsFineLow", num_up(br_[br_.steps == hi_].eps99_over_fstar.min(), 2))
    M("BPExactCoarse", f"{br_[br_.steps == lo_].exact_over_fstar.min():.1f}"); M("BPExactFine", f"{br_[br_.steps == hi_].exact_over_fstar.max():.1f}")
if has("extras_bayes_gridspacing.csv"):
    gs_ = R("extras_bayes_gridspacing.csv")
    g3 = gs_[gs_.q == 0.3].set_index("spacing_over_fstar").eps99_over_fstar
    M("BPGridCoarse", num_up(g3[0.25], 2)); M("BPGridFine", num_up(g3[0.025], 2)); M("BPGridFineLowQ", num_up(gs_[gs_.spacing_over_fstar == 0.025].eps99_over_fstar.min(), 2))
if has("extras_bayes_sensitivity.csv"):
    bs_ = R("extras_bayes_sensitivity.csv")
    M("BPSensLo", num_up(bs_.f_opt_over_fstar.min(), 2)); M("BPSensHi", num_up(bs_.f_opt_over_fstar.max(), 2))
    M("BPSensPriors", f"{bs_.prior.nunique()}"); M("BPSensWeights", f"{bs_.weights.nunique()}")
if has("extras_contest_robustness.csv"):
    cr_ = R("extras_contest_robustness.csv")
    M("RobSettings", f"{len(cr_)}"); M("RobStrongRace", f"{int(cr_.strong_race_deters.max())}"); M("RobStrongBond", f"{int((cr_.strong_bond_deters == cr_.strong_cells).sum())}")
    M("RobRaceTiesMax", f"{int(cr_.race_ties.max())}")
if has("extras_size_rules.csv"):
    sz = R("extras_size_rules.csv")
    table("tab_size.tex", "rrrrr", ["size", "$f^*/N$, flat fee", "$f^*/N$, fee $\\propto$ size", "speculator value, rule calibrated at size 1 (bp)",
          "speculator value, line through the sizes (bp)"],
          [[f"{r.size:g}", pct_up(r.fstar_flat_fee_over_N, 3), pct_up(r.fstar_prop_fee_over_N, 3), bp(r.v0_bp_proportional_rule, 1), bp(r.v0_bp_affine_rule, 1)] for r in sz.itertuples()])
    M("SizePropMaxResidual", bp(float(sz.v0_bp_proportional_rule.max()), 1)); M("SizeLineOverSmall", f"{float(sz.affine_over_fstar.max()):.0f}")
    M("SizePropFeeSpread", f"{float(sz.fstar_prop_fee_over_N.max() / sz.fstar_prop_fee_over_N.min() - 1) * 100:.1f}")
if has("extras_q_sweep.csv"):
    qs_ = R("extras_q_sweep.csv")
    table("tab_qsweep.tex", "rrr", ["$q$", "$f^*/N$", "speculator value without a forfeit (bp)"], [[f"{r.q:g}", pct_up(r.f_star, 3), bp(r.v0_without_forfeit_bp, 1)] for r in qs_.itertuples()])
    M("QfZero", pct_up(qs_[qs_.q == 0.0].f_star.iloc[0], 3)); M("QfSix", pct_up(qs_[qs_.q == 0.6].f_star.iloc[0], 3)); M("QfNine", pct_up(qs_[qs_.q == 0.9].f_star.iloc[0], 3))
if has("extras_honest_cost.csv"):
    hc = R("extras_honest_cost.csv")
    M("HonestMaxLossBp", f"{hc.honest_loss_bp.abs().max():.2f}"); pmax = float(hc.p_dropped.max())
    e10 = int(math.floor(math.log10(pmax)))
    M("HonestDropMax", f"${pmax / 10 ** e10:.1f}\\times10^{{{e10}}}$")

# ---- verification facts ----
vp = os.path.join(a.illustrative, "verification.json")
if os.path.exists(vp):
    V = json.load(open(vp))
    M("CentNash", V["centipede"]["nash"]); M("CentSPNE", V["centipede"]["spne"])
    M("BurnNash", V["protocol_burn"]["nash"]); M("BurnSPNE", V["protocol_burn"]["spne"]); M("BurnNonCred", V["protocol_burn"]["noncredible"])
    M("ProdNash", V["protocol_producer"]["nash"]); M("ProdSPNE", V["protocol_producer"]["spne"])
    M("HandV", f"{V['hand_v0']['solver']:.4f}"); M("PlainGainChecker", f"{V['plain_commit_reveal']['user_gain']:.2f}")
    M("ObsTwoRows", V["obs2"]["rows"]); M("ObsTwoDiff", sci(V["obs2"]["max_abs_diff"]))
else:
    raise SystemExit(f"missing {vp}: run scripts/run_course_core.py first")

with open(os.path.join(a.out, "numbers.tex"), "w") as fh:
    fh.write("% AUTO-GENERATED by scripts/make_report_assets.py: do not edit by hand\n")
    for k, val in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{val}}}\n")
for f in ("course_fig_q1_sweeps", "course_fig_feasibility_map", "course_fig_designer", "course_fig_extortion"):
    shutil.copy(os.path.join(a.results, f + ".png"), os.path.join(a.out, "figs", f + ".png"))
# data plotted directly by pgfplots in the report (design-feasibility figure)
shutil.copy(os.path.join(a.results, "extras_burn_range.csv"), os.path.join(a.out, "burn_range.csv"))
# design comparison of Q2 as a bar chart: one row per design, in the order of q2_routing_designs.csv
pd.DataFrame({"idx": range(len(R("q2_routing_designs.csv"))),
              "value": R("q2_routing_designs.csv").speculator_value_bp.round(1),
              "deterred": R("q2_routing_designs.csv").speculation_deterred.astype(int)}).to_csv(os.path.join(a.out, "q2_bars.csv"), index=False)
print(f"wrote {len(macros)} macros, 7 tables, 4 figures to {a.out}")
