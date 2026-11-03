# Results

Every file here is produced by the scripts in `scripts/` and read by `scripts/make_report_assets.py` to build the report's tables and numbers.

| Folder | Content |
|---|---|
| `main/` | The main calibration used in the report: full ETH history 2017 to 2026, all four questions, the Bayesian designers and the sensitivity analyses (`extras_*.csv`) |
| `illustrative/` | The verification walk-through with illustrative parameters (hand-solved game, centipede, Nash versus subgame perfect). Its figures use illustrative units and must not be mixed with the calibrated ones |
| `periods/` | The same pipeline on the pooled period (`all/`) and four sub-periods |
| `assets/` | The pipeline on BTC, ETH, SOL and BNB daily data |
| `intraday_1m/`, `intraday_12s/` | Fat-tail, variance-ratio and empirical-forfeit checks on one-minute and 12-second ETH prices |

`run_info.json` in each folder records the parameters and data used.
