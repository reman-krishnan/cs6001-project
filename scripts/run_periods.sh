#!/bin/bash
# Runs the whole pipeline on the full period and on four sub-periods; keeps each result folder in results/periods/.
set -eo pipefail
cd "$(dirname "$0")/.."                          # run from the repository root wherever it is called from
# data/historical_prices_FULL.csv is the canonical full series. It is created once and never overwritten from the working file,
# which holds a truncated sub-period while this script runs. The working file is restored on any exit, including a failure.
[ -f data/historical_prices_FULL.csv ] || cp data/historical_prices.csv data/historical_prices_FULL.csv
cp data/historical_prices_FULL.csv data/historical_prices.csv
trap 'cp data/historical_prices_FULL.csv data/historical_prices.csv' EXIT
run_period () {   # name, start (inclusive), end (exclusive), optional "skip" = fast mode
  python3 - "$1" "$2" "$3" << 'PY'
import sys, pandas as pd
name, a, b = sys.argv[1:4]
df = pd.read_csv("data/historical_prices_FULL.csv")
sub = df[(df.date >= a) & (df.date < b)]
sub.to_csv("data/historical_prices.csv", index=False)
print(f"{name}: {len(sub)} rows, {sub.date.iloc[0][:10]} to {sub.date.iloc[-1][:10]}")
PY
  rm -rf "results/periods/$1"
  python3 scripts/run_project.py ${4:+--skip-slow} --out "results/periods/$1" | tail -1
}
run_period all       2017-01-01 2100-01-01
run_period 2018_2019 2018-01-01 2020-01-01 skip
run_period 2020_2021 2020-01-01 2022-01-01 skip
run_period 2022_2023 2022-01-01 2024-01-01 skip
run_period 2024_2026 2024-01-01 2100-01-01 skip
cp data/historical_prices_FULL.csv data/historical_prices.csv
python3 - << 'PY'
import pandas as pd
print("\nPeriod comparison (daily vol, f* at k=100 / size 1, and speculator value of plain commit-reveal):")
for p in ["all","2018_2019","2020_2021","2022_2023","2024_2026"]:
    v = pd.read_csv(f"results/periods/{p}/calibration.csv").daily_vol.iloc[0]
    q = pd.read_csv(f"results/periods/{p}/q1_sweeps.csv")
    f = q[(q.factor=="k")&(q.value==100)].f_star_over_notional.iloc[0]
    d = pd.read_csv(f"results/periods/{p}/q2_routing_designs.csv").set_index("design").speculator_value_bp["plain commit-reveal (no forfeit)"]
    print(f"  {p:<10} daily vol {v:.4f}   f*/notional {f:.3%}   plain commit-reveal speculator value {d:.1f} bp")
PY
