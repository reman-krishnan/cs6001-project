"""The course-aligned core must not depend on the finance / calibration / coalition-analysis extensions."""
import subprocess
import sys
import textwrap
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTENSIONS = ["dagame.theory", "dagame.calibration", "dagame.benchmarks", "dagame.analysis", "dagame.experiments",
              "dagame.figures"]


def test_core_imports_without_extensions():
    code = textwrap.dedent(f"""
        import sys
        class Block:
            def find_spec(self, name, path=None, target=None):
                if name in {EXTENSIONS!r}:
                    raise ImportError("blocked extension module " + name)
        sys.meta_path.insert(0, Block())
        sys.path.insert(0, {ROOT!r})
        import dagame, dagame.model, dagame.solver, dagame.tree, dagame.strategic, dagame.checker
        import dagame.normal_form, dagame.observations, dagame.course_results, dagame.contest, dagame.rotating, dagame.bayes_private
        print("ok")
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr
