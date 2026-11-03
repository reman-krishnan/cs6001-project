#!/bin/bash
# Regenerates every result, table and figure of the report from the data in data/ (about 25 minutes).
# Set SKIP_TESTS=1 to skip the test suite. The data files are not downloaded again (see scripts/download_binance.py).
set -eo pipefail                                    # a failing step stops the run even when its output is piped to tail
cd "$(dirname "$0")"                                # run from the repository root
[ -n "$SKIP_TESTS" ] || python3 -m pytest -q
./scripts/run_periods.sh                          # sub-period comparison into results/periods/ (restores data/historical_prices.csv at the end)
python3 scripts/run_project.py                    # main calibration -> results/main
python3 scripts/run_course_core.py                # verification facts -> results/illustrative
python3 scripts/run_extras.py                     # sensitivity analyses -> results/main/extras_*.csv
for s in ETHUSDT BTCUSDT SOLUSDT BNBUSDT; do      # other assets
  python3 scripts/run_project.py --data data/${s}_1d.csv --out results/assets/${s} --skip-slow | tail -1
done
python3 scripts/run_empirical.py --data data/ETHUSDT_1m.csv
python3 scripts/run_empirical.py --data data/ETHUSDT_12s.csv --window-minutes 20 --vr-multiples 2 5 10 25 50 100 300 \
  --n-boot 1000 --block-windows 12 --seed 0 --out results/intraday_12s
python3 scripts/make_report_assets.py
(cd report && pdflatex -interaction=nonstopmode report.tex >/dev/null && pdflatex -interaction=nonstopmode report.tex >/dev/null)
echo "done: report/report.pdf"
