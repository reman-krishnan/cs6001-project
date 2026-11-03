#!/bin/bash
# Multi-asset robustness: run the pipeline on the committed daily closes of each asset (fast mode).
# By default this reproduces the report from the frozen files in data/. Set REFRESH_DATA=1 to download the latest
# candles first; that overwrites data/*_1d.csv, so the results will then differ from the report.
set -eo pipefail
cd "$(dirname "$0")/.."                          # run from the repository root wherever it is called from
if [ "${REFRESH_DATA:-0}" = "1" ]; then           # only an explicit REFRESH_DATA=1 downloads; unset or 0 keeps the frozen data
  python3 scripts/download_binance.py --symbols ETHUSDT BTCUSDT SOLUSDT BNBUSDT --interval 1d --days 3400
fi
for s in ETHUSDT BTCUSDT SOLUSDT BNBUSDT; do
  python3 scripts/run_project.py --data data/${s}_1d.csv --out results/assets/${s} --skip-slow | tail -1
done
python3 - << 'PY'
import pandas as pd
print("\nAsset comparison (daily vol with 95% bootstrap interval, f* at k=100 / size 1):")
for s in ["ETHUSDT","BTCUSDT","SOLUSDT","BNBUSDT"]:
    c = pd.read_csv(f"results/assets/{s}/calibration.csv").iloc[0]
    q = pd.read_csv(f"results/assets/{s}/q1_sweeps.csv")
    f = q[(q.factor=="k")&(q.value==100)].f_star_over_notional.iloc[0]
    print(f"  {s:<8} daily vol {c.daily_vol:.4f} [{c.ci_low:.4f}, {c.ci_high:.4f}]   f*/notional {f:.3%}")
PY
