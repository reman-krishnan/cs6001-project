"""How much of the submitted implementation is course material? Counts executable lines (docstrings and comments excluded).

CORE modules implement syllabus concepts. EXTENSION modules (finance, calibration, coalition analysis, contestation) do not.
Extension lines living inside core files (coalition optimum, contestation branches) are counted as extension, found by the
function list and keyword pattern below, so the share is not flattered by file boundaries.
"""
import ast
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PKG = os.path.join(ROOT, "dagame")
CORE = ["numerics", "model", "tree", "solver", "strategic", "checker", "normal_form", "observations", "course_results", "contest", "rotating",
        "bayes_private"]
EXT_FILES = ["theory", "calibration", "benchmarks", "analysis", "experiments", "figures"]      # NOT in the submission
SUBMITTED_EXT = ["han_check", "volatility", "empirical"]      # promised by the proposal, not syllabus material: counted as extension
EXT_FUNCS = {"solver": {"joint_surplus", "_joint"}, "tree": {"max_joint"}}
EXT_PAT = re.compile(r"\b(beta|beta_eff|challenge_prob|challenge_cost|challenger|contestation|joint|coalition|coal|max_joint|joint_surplus)\b", re.I)


def code_lines(path):
    src = open(path).read()
    tree = ast.parse(src)
    doc = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body and isinstance(n.body[0], ast.Expr) \
                and isinstance(getattr(n.body[0], "value", None), ast.Constant) and isinstance(n.body[0].value.value, str):
            doc.update(range(n.body[0].lineno, n.body[0].end_lineno + 1))
    funcs = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef):
            funcs[n.name] = (n.lineno, n.end_lineno)
    out = {}
    for i, line in enumerate(src.splitlines(), 1):
        s = line.strip()
        if s and not s.startswith("#") and i not in doc:
            out[i] = line
    return out, funcs


def main():
    tot = {"core": 0, "ext_in_core": 0, "ext": 0}
    rows = []
    for m in CORE:
        lines, funcs = code_lines(os.path.join(PKG, m + ".py"))
        extr = set()
        for fn in EXT_FUNCS.get(m, ()):
            a, b = funcs[fn]
            extr.update(range(a, b + 1))
        n_ext = sum(1 for i, l in lines.items() if i in extr or EXT_PAT.search(l))
        rows.append((m, len(lines) - n_ext, n_ext))
        tot["core"] += len(lines) - n_ext
        tot["ext_in_core"] += n_ext
    for m in SUBMITTED_EXT:
        if not os.path.exists(os.path.join(PKG, m + ".py")):
            continue                      # e.g. empirical.py is absent from the core profile
        lines, _ = code_lines(os.path.join(PKG, m + ".py"))
        rows.append((m + " (proposal item)", 0, len(lines)))
        tot["ext_in_core"] += len(lines)
    ext_rows = []
    for m in EXT_FILES:
        if not os.path.exists(os.path.join(PKG, m + ".py")):
            continue                      # absent in the course-only submission
        lines, _ = code_lines(os.path.join(PKG, m + ".py"))
        ext_rows.append((m, len(lines)))
        tot["ext"] += len(lines)
    print(f"{'module':<16}{'course lines':>14}{'extension lines inside':>24}")
    for m, c, e in rows:
        print(f"{m:<16}{c:>14}{e:>24}")
    print("\nextension modules (not part of the course-aligned submission):")
    for m, n in ext_rows:
        print(f"  {m:<16}{n:>6}")
    sub_total = tot["core"] + tot["ext_in_core"]
    share_sub = tot["core"] / sub_total
    share_repo = tot["core"] / (sub_total + tot["ext"])
    print(f"\nSUBMISSION SET (core modules only): {tot['core']} course lines + {tot['ext_in_core']} extension lines "
          f"=> {share_sub:.1%} course")
    print(f"WHOLE REPOSITORY (incl. extension modules): {share_repo:.1%} course")
    return share_sub, share_repo


if __name__ == "__main__":
    s, r = main()
    sys.exit(0 if s >= 0.90 else 1)
