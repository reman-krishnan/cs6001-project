# Market data

All files are closing prices from Binance, downloaded with `scripts/download_binance.py` (daily and one-minute) or `scripts/download_binance_daily.py` (1-second bulk files). The daily series end on 30 September 2026 and are a frozen snapshot.

| File | Content |
|---|---|
| `ETHUSDT_1d.csv`, `BTCUSDT_1d.csv`, `SOLUSDT_1d.csv`, `BNBUSDT_1d.csv` | Daily closes for each asset |
| `historical_prices.csv` | Daily ETH/USDT closes from 17 August 2017 to 30 September 2026, used as the main calibration |
| `historical_prices_FULL.csv` | Backup of the full ETH series, restored by `scripts/run_periods.sh` after it slices sub-periods |
| `ETHUSDT_12s.csv` | Derived 12-second ETH/USDT closes from 9 to 22 October 2026, built from 1-second closes by carrying the last price over seconds without trades. The raw 1-second file is not committed (`.gitignore`). Used for the block-scale checks |
| `ETHUSDT_1m.csv` | One-minute ETH/USDT closes from 2 August to 1 October 2026, used only for the fat-tail and variance-ratio checks |

`scripts/run_project.py` reads `data/historical_prices.csv`. A file needs a date column (`date`, `timestamp`, `snapped_at` or `Open time`) and a price column (`close`, `price` or `Adj Close`). Block volatility is derived from the daily volatility by square-root-of-time scaling, sigma_block = sigma_daily * sqrt(12 / 86400).
