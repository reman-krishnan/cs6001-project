"""After re-running the pipeline, compare the regenerated CSV files with the committed ones.

    python scripts/compare_results.py                 # compares the working tree (staged or not) with HEAD
    python scripts/compare_results.py --rtol 1e-8
Differences at the level of floating-point rounding (library versions, summation order) are reported as noise. A file whose values
differ by more than the tolerance, or whose columns changed, is listed as a real difference and the exit status is 1.
Without a git checkout and a HEAD commit there is nothing to compare against, and the exit status is 2.
PNG figures and the PDF always differ in bytes (rendering, timestamps) and are not compared.
"""
import argparse
import io
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--rtol", type=float, default=1e-8)
ap.add_argument("--atol", type=float, default=1e-10)
a = ap.parse_args()
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def sh(*c):
    """Run a git command in the repository root and stop with status 2 if it fails, so a missing repository or HEAD
    can never look like 'no differences'."""
    r = subprocess.run(c, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        print(f"git failed ({' '.join(c)}): {r.stderr.strip() or 'no output'}\n"
              "compare_results.py needs a git checkout with a HEAD commit to compare against. A plain archive can still "
              "reproduce the results, but this check cannot run there.", file=sys.stderr)
        sys.exit(2)
    return r.stdout


sh("git", "rev-parse", "--verify", "HEAD")
# "git diff HEAD" covers both staged and unstaged changes (plain "git diff" would miss staged files).
files = [f for f in sh("git", "diff", "HEAD", "--name-only").split() if f.endswith(".csv")]
real, noise, worst = [], 0, 0.0
for f in files:
    shown = subprocess.run(["git", "show", "HEAD:" + f], capture_output=True, text=True, cwd=ROOT)
    if shown.returncode != 0:
        real.append((f, "not in HEAD"))
        continue
    try:
        old = pd.read_csv(io.StringIO(shown.stdout))
    except Exception:
        real.append((f, "not readable at HEAD"))
        continue
    new = pd.read_csv(os.path.join(ROOT, f))
    if list(old.columns) != list(new.columns) or old.shape != new.shape:
        real.append((f, f"shape or columns changed {old.shape} -> {new.shape}"))
        continue
    bad = None
    for c in old.columns:
        x, y = old[c], new[c]
        if x.dtype.kind in "fi" and y.dtype.kind in "fi":
            u, v = x.to_numpy(float), y.to_numpy(float)
            # NaN, +inf and -inf must match position by position; a sign flip of an infinity is a real difference.
            if ((np.isnan(u) != np.isnan(v)).any() or (np.isposinf(u) != np.isposinf(v)).any()
                    or (np.isneginf(u) != np.isneginf(v)).any()):
                bad = f"{c}: NaN or infinity pattern differs"
                break
            ok = np.isfinite(u)
            d = np.abs(u[ok] - v[ok])
            tol = a.atol + a.rtol * np.maximum(np.abs(u[ok]), np.abs(v[ok]))
            if (d > tol).any():
                bad = f"{c}: largest difference {d.max():.3g}"
                break
            if len(d):
                worst = max(worst, float((d / np.maximum(1e-300, np.maximum(np.abs(u[ok]), np.abs(v[ok])))).max()))
        elif not x.astype(str).equals(y.astype(str)):
            bad = f"{c}: text differs"
            break
    if bad:
        real.append((f, bad))
    else:
        noise += 1
new = [f for f in sh("git", "ls-files", "--others", "--exclude-standard").split() if f.endswith(".csv")]
print(f"{len(files)} CSV files differ from HEAD: {noise} within tolerance (rounding noise), {len(real)} with real differences.")
if new:
    print(f"{len(new)} new CSV files not in HEAD (informational):", *new[:10], "..." if len(new) > 10 else "")
for f, why in real:
    print("  REAL", f, "-", why)
sys.exit(1 if real else 0)
