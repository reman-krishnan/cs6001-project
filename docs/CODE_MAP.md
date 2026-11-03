# Code map, validation and scope

## Questions, code and output
Module names are in `dagame/`, scripts in `scripts/`, and output files in `results/main/` unless stated otherwise.

| Item | Code | Output |
|---|---|---|
| Game model, subgame-perfect equilibrium, deviation checker, Nash versus subgame perfect | `model.py`, `tree.py`, `solver.py`, `checker.py`, `normal_form.py` | `scripts/run_course_core.py` -> `results/illustrative/` |
| Closed-form checks of the structural results | `observations.py` | `scripts/run_course_core.py` |
| Q1 smallest deterring forfeit against window, volatility, size, fee | `course_results.q1_sweeps/feasibility_map`, `volatility.daily_volatility_ci`, `scripts/run_extras.py` | `q1_sweeps.csv`, `q1_feasibility_map.csv`, `q1_fstar_ci.csv`, `extras_fee_sensitivity.csv` |
| Q2 routing designs | `course_results.designs_table` | `q2_routing_designs.csv` |
| Q3 extortion, coalitions, impossibility, range of burned forfeits, window frontier, tolerance, mostly burned pool(n) | `strategic.py`, `course_results.extortion_table/coalition_table/impossibility_search`, `scripts/run_extras.py` | `q3_producer_incentives.csv`, `q3_coalition.csv`, `q3_impossibility.csv`, `extras_burn_range.csv`, `extras_frontier.csv`, `extras_tolerance.csv`, `extras_pool.csv`, `extras_honest_cost.csv` |
| Q3 rotating producers | `rotating.py`, `course_results.rotating_table` | `q3_rotating.csv` |
| Q4 bonded contest against race | `contest.py` | `q4_contest.csv`, `extras_contest_robustness.csv` |
| Bayesian versions, mixture of types and private type | `course_results.designer_table`, `bayes_private.py` | `bayes_designer.csv`, `bayes_private_*.csv`, `extras_bayes_*.csv` |
| Check against the binomial option price | `han_check.py` | `han_check.csv` |
| Real data, other assets, intraday fat tails, square-root-of-time check | `volatility.py`, `empirical.py`, `scripts/run_assets.sh`, `scripts/run_empirical.py` | `calibration.csv`, `results/assets/`, `results/intraday_1m/`, `results/intraday_12s/` |
| Report tables and the numbers quoted in the text | `scripts/make_report_assets.py` | `report/assets/` (every number in the report comes from `report/assets/numbers.tex`) |

`scripts/dev/` holds two development helpers that the report does not use: `run_week1.py`, the first checkpoint of the deviation checker, and `course_share.py`, which counts how much of the code implements syllabus concepts.

## Validation
The checks are hand-computed games and the lattice solver against a history-dependent game tree under each of the eight uniform tie-breaking rules (base, strategic and rotating models). A rule that lets an indifferent producer decide differently at each state is covered for the speculator's value by `solver.user_value_range`, which is checked against enumeration of all pure equilibria on small games. Further checks are exhaustive strategy enumeration (including the bonded contest), the centipede game, the all-pay equilibrium by indifference conditions and simulation, the binomial option price, and mutation tests (`tests/test_mutations.py`), in which twelve deliberately injected one-line bugs in the solvers, the payoffs, the lattice and the bond formula must each be caught.

## Scope and limits
* The contest mechanisms of Q4 are a formalization made for this project. Owner availability, attack cost, owner capital and the complete-information all-pay race are modelling choices, not measurements.
* The prior over private user types and the loss weights are illustrative. The private-type designer's loss is almost flat once moderately exposed users stop exploiting the delay, so its exact minimiser is unstable. The reported forfeit is the smallest one that captures 99% of the loss reduction (`bayes_private.eps_optimal_forfeit`), refined between coarse grid points. Its dependence on the forfeit grid, the number of lattice steps, the prior and the loss weights is in `extras_bayes_gridspacing.csv`, `extras_bayes_resolution.csv` and `extras_bayes_sensitivity.csv`. The steps experiment also changes the number of authorization attempts, so it is a joint sensitivity, not a pure refinement.
* The intraday interval comes from a block bootstrap with blocks twelve windows long (`--block-windows`). The 12-second sample covers only 14 calm days.
* Coalition gains are measured against the best equilibrium with uniform tie-breaking. Under a strictly credible fallback this is the honest equilibrium in which the user tries to authorize at every block.
* The impossibility result needs strict credibility, no contestation, a martingale price and positive volatility. The rotating-producer threshold is proved in the report.
* The model assumes that the producer's payoff does not depend on the price, honest producers are altruistic, censoring is independent across blocks, prices follow a fair binomial lattice, players are risk neutral, side contracts are enforceable, and the committed transaction stays executable throughout the window.
* The window, fee, publication cost, collateral limit and the 0.1 bp coalition-gain tolerance are chosen, not measured. The report includes sensitivity analyses for the fee and the tolerance, and the largest feasible window comes from evaluating every integer window from 2 to 400 blocks (`extras_frontier.csv`). Every minimum forfeit and bond in the report is rounded up, so a displayed value suffices, and interval ends are rounded outward.
* The literature survey was made from the references of the base paper and web searches, not from a citation database.
