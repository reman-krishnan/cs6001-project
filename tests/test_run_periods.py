"""scripts/run_periods.sh must keep the full price series intact when a run fails or is interrupted part-way."""
import os
import shutil
import subprocess
import textwrap

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_workspace(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "scripts").mkdir()
    days = pd.date_range("2017-01-01", "2026-09-30", freq="1D", tz="UTC")
    full = pd.DataFrame({"date": days.strftime("%Y-%m-%d %H:%M:%S+00:00"), "close": range(1, len(days) + 1)})
    full.to_csv(tmp_path / "data" / "historical_prices.csv", index=False)
    shutil.copy(os.path.join(ROOT, "scripts", "run_periods.sh"), tmp_path / "scripts" / "run_periods.sh")
    # a stub for the real pipeline: the first period succeeds, the second is interrupted (SIGTERM to the parent shell, like Ctrl+C)
    (tmp_path / "scripts" / "run_project.py").write_text(textwrap.dedent('''
        import os, signal, sys
        c = "calls.txt"
        n = int(open(c).read()) + 1 if os.path.exists(c) else 1
        open(c, "w").write(str(n))
        if n == 2:
            os.kill(os.getppid(), signal.SIGTERM)
            import time; time.sleep(5)
            sys.exit(1)
        os.makedirs(sys.argv[sys.argv.index("--out") + 1], exist_ok=True)
        '''))
    return len(full)


def test_a_failed_period_run_does_not_destroy_the_full_series(tmp_path):
    n = make_workspace(tmp_path)
    r = subprocess.run(["bash", "scripts/run_periods.sh"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode != 0                                                    # the second period was interrupted
    for f in ("historical_prices.csv", "historical_prices_FULL.csv"):
        assert len(pd.read_csv(tmp_path / "data" / f)) == n, f                  # both files hold the complete series afterwards
    # a rerun must not copy a truncated file over the backup
    (tmp_path / "calls.txt").write_text("5")                                    # later calls are neither interrupted nor failing... (n != 2)
    subprocess.run(["bash", "scripts/run_periods.sh"], cwd=tmp_path, capture_output=True, text=True)
    assert len(pd.read_csv(tmp_path / "data" / "historical_prices_FULL.csv")) == n
