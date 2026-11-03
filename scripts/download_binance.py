"""Download public Binance klines (no account, no API key) into data/<SYMBOL>_<interval>.csv with columns date, close.

    python scripts/download_binance.py --symbols ETHUSDT BTCUSDT SOLUSDT --interval 1d --days 3000
    python scripts/download_binance.py --symbols ETHUSDT --interval 1m --days 30        # intraday: for the fat-tail and sqrt-time checks

Binance returns at most 1000 rows per call; this pages forward from the start date. If api.binance.com is blocked from your network
(HTTP 451/403) use --host https://api.binance.us. The last row is usually an unfinished candle and is dropped.
"""
import argparse
import os
import sys
import time

import pandas as pd
import requests

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ap = argparse.ArgumentParser()
ap.add_argument("--symbols", nargs="+", default=["ETHUSDT"])
ap.add_argument("--interval", default="1d")
ap.add_argument("--days", type=int, default=3000)
ap.add_argument("--host", default="https://api.binance.com")
ap.add_argument("--outdir", default=os.path.join(ROOT, "data"))
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
now_ms = int(time.time() * 1000)
for sym in a.symbols:
    start, rows = now_ms - a.days * 86_400_000, []
    while start < now_ms:
        for attempt in range(4):
            try:
                r = requests.get(f"{a.host}/api/v3/klines", params=dict(symbol=sym, interval=a.interval, startTime=start, limit=1000), timeout=30)
                break
            except requests.exceptions.RequestException as e:
                if attempt == 3:
                    raise
                print("retrying after", type(e).__name__, flush=True)
                time.sleep(3 * (attempt + 1))
        if r.status_code in (403, 451):
            sys.exit(f"HTTP {r.status_code} from {a.host}: try --host https://api.binance.us")
        r.raise_for_status()
        k = r.json()
        if not k:
            break
        rows += k
        start = k[-1][0] + 1
        time.sleep(0.15)
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "qav", "trades", "tbb", "tbq", "ignore"])
    df["date"] = pd.to_datetime(df.open_time, unit="ms", utc=True)
    df = df.drop_duplicates("open_time").iloc[:-1]            # drop the unfinished candle
    path = os.path.join(a.outdir, f"{sym}_{a.interval}.csv")
    df[["date", "close"]].to_csv(path, index=False)
    print(f"{sym}: {len(df)} rows {df.date.iloc[0]} -> {df.date.iloc[-1]}  ->  {path}")
