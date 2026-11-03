"""Download Binance spot klines from the public bulk files (data.binance.vision), one zip per day. Resumable, with progress.

    python scripts/download_binance_daily.py --symbol ETHUSDT --interval 1s --days 14
Writes data/<SYMBOL>_<interval>.csv with columns date, close. Zip files are cached in data/_cache/ (not committed), so an
interrupted run continues where it stopped. The last --lag-days days are skipped because the bulk files appear a day or more late.
Since 2025 the spot files use microsecond timestamps, earlier files use milliseconds, and some files carry a header row. All three
are handled by ``read_klines``.
"""
import argparse
import datetime as dt
import io
import os
import sys
import time
import zipfile

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BASE = "https://data.binance.vision/data/spot/daily/klines"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qav", "trades", "tbb", "tbq", "ignore"]


def read_klines(raw: bytes) -> pd.DataFrame:
    """Parse the bytes of one daily zip into a frame with columns date (UTC) and close."""
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = z.namelist()[0]
        with z.open(name) as fh:
            df = pd.read_csv(fh, header=None, names=COLS, dtype=str)
    if not df.empty and not str(df.open_time.iloc[0]).strip().isdigit():
        df = df.iloc[1:]                                              # header row
    t = pd.to_numeric(df.open_time)
    unit = "us" if float(t.iloc[0]) > 1e14 else "ms"
    return pd.DataFrame({"date": pd.to_datetime(t, unit=unit, utc=True), "close": pd.to_numeric(df.close)})


def fetch(url: str, tries: int = 6) -> bytes:
    import requests
    for i in range(tries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                raise FileNotFoundError(url)
            r.raise_for_status()
            return r.content
        except FileNotFoundError:
            raise
        except Exception as e:                                        # network hiccup: wait and retry
            print(f"    retry {i + 1}/{tries} after {type(e).__name__}", flush=True)
            time.sleep(min(60, 3 * 2 ** i))
    raise RuntimeError(f"giving up on {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--interval", default="1s")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--lag-days", type=int, default=3)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "data"))
    a = ap.parse_args()
    cache = os.path.join(a.outdir, "_cache")
    os.makedirs(cache, exist_ok=True)
    end = dt.date.today() - dt.timedelta(days=a.lag_days)
    days = [end - dt.timedelta(days=i) for i in range(a.days - 1, -1, -1)]
    frames = []
    for n, d in enumerate(days, 1):
        fn = f"{a.symbol}-{a.interval}-{d.isoformat()}.zip"
        path = os.path.join(cache, fn)
        if not os.path.exists(path):
            try:
                raw = fetch(f"{BASE}/{a.symbol}/{a.interval}/{fn}")
            except FileNotFoundError:
                print(f"[{n}/{len(days)}] {d} not published, skipped", flush=True)
                continue
            with open(path, "wb") as fh:
                fh.write(raw)
            state = "downloaded"
        else:
            state = "cached"
        frames.append(read_klines(open(path, "rb").read()))
        print(f"[{n}/{len(days)}] {d} {state}, {len(frames[-1]):,} rows", flush=True)
    if not frames:
        sys.exit("no files could be downloaded")
    df = pd.concat(frames).drop_duplicates("date").sort_values("date")
    out = os.path.join(a.outdir, f"{a.symbol}_{a.interval}.csv")
    df.to_csv(out, index=False)
    print(f"{a.symbol} {a.interval}: {len(df):,} rows {df.date.iloc[0]} -> {df.date.iloc[-1]} -> {out}")


if __name__ == "__main__":
    main()
