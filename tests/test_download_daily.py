"""The bulk-file parser must handle millisecond and microsecond timestamps and an optional header row."""
import importlib.util
import io
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("dl", os.path.join(ROOT, "scripts", "download_binance_daily.py"))
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)


def make_zip(rows, header=False):
    text = ("open_time,open,high,low,close,volume,close_time,qav,trades,tbb,tbq,ignore\n" if header else "")
    text += "".join(",".join(map(str, r)) + "\n" for r in rows)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.csv", text)
    return buf.getvalue()


def row(t, close):
    return [t, 1, 1, 1, close, 1, t + 999, 1, 1, 1, 1, 0]


def test_milliseconds_and_microseconds_give_the_same_dates():
    ms = dl.read_klines(make_zip([row(1767225600000, 2500.5), row(1767225601000, 2501.0)]))
    us = dl.read_klines(make_zip([row(1767225600000000, 2500.5), row(1767225601000000, 2501.0)]))
    assert list(ms.date) == list(us.date)
    assert ms.date.iloc[0].isoformat() == "2026-01-01T00:00:00+00:00"
    assert list(ms.close) == [2500.5, 2501.0]


def test_header_row_is_skipped():
    df = dl.read_klines(make_zip([row(1767225600000, 2500.5)], header=True))
    assert len(df) == 1 and df.close.iloc[0] == 2500.5
