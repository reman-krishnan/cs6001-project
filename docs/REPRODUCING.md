# Reproducing the results

## Environment
Tested with Python 3.12 (the version used by the GitHub Actions workflow) and the packages in `requirements.txt`. Building the PDF needs a LaTeX distribution with `pdflatex` and the packages `amsmath`, `amssymb`, `amsthm`, `graphicx`, `xcolor`, `tikz`, `pgfplots` (1.18), `caption`, `hyperref`, `parskip`, `tabularx` and `makecell`. TeX Live 2023 with `texlive-latex-extra`, `texlive-pictures` and `texlive-science` is sufficient.

## One command
`./run_all.sh` regenerates everything in order from the frozen data in `data/` (about 25 minutes) and stops at the first failing step, even inside a pipe. Afterwards, `python scripts/compare_results.py` compares every regenerated CSV file with the committed one, staged or not. Its exit status is 0 if everything matches up to floating-point noise, 1 on a real difference (including a changed NaN or a sign change of an infinity), and 2 when there is no git checkout to compare against.

## Individual steps
All commands run from the repository root.
```
pip install -r requirements.txt
python3 -m pytest -q                                 # the test suite (about 2 min)
python3 scripts/run_project.py                       # main tables and figures -> results/main (about 2 min)
python3 scripts/run_extras.py                        # sensitivity analyses -> results/main/extras_*.csv
python3 scripts/run_course_core.py                   # verification facts (centipede, Nash vs subgame perfect, hand game) -> results/illustrative
./scripts/run_periods.sh                            # the same pipeline on five periods -> results/periods/
./scripts/run_assets.sh                             # BTC, SOL and BNB, from the frozen data -> results/assets/
python3 scripts/run_empirical.py --data data/ETHUSDT_1m.csv   # one-minute fat tails and variance ratios -> results/intraday_1m
python3 scripts/run_empirical.py --data data/ETHUSDT_12s.csv --window-minutes 20 --vr-multiples 2 5 10 25 50 100 300 \
  --n-boot 1000 --block-windows 12 --seed 0 --out results/intraday_12s
python3 scripts/make_report_assets.py                # runs the tests (stops if any fail), then writes report/assets
cd report && pdflatex report.tex && pdflatex report.tex
```

## Fresh data (changes the results)
The daily data end on 30 September 2026 and are a frozen snapshot. `REFRESH_DATA=1 ./scripts/run_assets.sh` downloads newer daily candles for the four assets first, which overwrites `data/*_1d.csv`, so the asset comparison will then differ from the report. Leaving `REFRESH_DATA` unset, or setting it to `0`, keeps the frozen files.

The 12-second analysis uses `data/ETHUSDT_12s.csv`, a derived series built from 14 days of 1-second closes (9 to 22 October 2026), carrying the last price over seconds without trades. The raw 1-second file is not committed. To rebuild it, run `python scripts/download_binance_daily.py --symbol ETHUSDT --interval 1s --days 14` and then `python scripts/run_empirical.py --data data/ETHUSDT_1s.csv --resample 12s --window-minutes 20 --vr-multiples 2 5 10 25 50 100 300 --out results/intraday_12s --save-resampled data/ETHUSDT_12s.csv`. The dates, and therefore the numbers, will differ.
