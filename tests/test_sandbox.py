"""Pin the declared execution environment for generated scripts.

These tests guard published numbers. The script sandbox decides whether a
model-generated program counts as correct, so a change here silently moves
Figures 5 and 6. If one of these fails, that is the intended alarm -- do not
adjust the expectation without re-running the parity check in
``plans/that-is-a-yes-modular-orbit.md`` Phase 1.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from superarc.sandbox import (  # noqa: E402
    ALLOWED_MODULES,
    Executor,
    looks_unbounded,
    run_script,
    run_script_guarded,
)


class TestDeclaredEnvironment(unittest.TestCase):
    def test_stdlib_is_importable(self):
        self.assertEqual(run_script("import math;print(math.factorial(5))").stdout, "120")

    def test_sympy_is_blocked(self):
        # The four o1-Preview scripts of this exact shape are why Figure 5's
        # complexity-2 value is 20.00% and not 26.67%.
        result = run_script("import sympy;print(list(sympy.primerange(2,31)))")
        self.assertFalse(result.ok)
        self.assertIn("sympy", str(result.error))

    def test_numpy_is_blocked(self):
        result = run_script("import numpy as np;print(np.arange(3))")
        self.assertFalse(result.ok)

    def test_allowed_set_is_exactly_the_stdlib(self):
        # Admitting numpy was measured to change zero models; keep the strict
        # rule so the declaration in Methods stays simple and checkable.
        self.assertEqual(ALLOWED_MODULES, frozenset(sys.stdlib_module_names))

    def test_a_models_error_is_preserved_not_repaired(self):
        # An error made by a model is data. The sandbox must report it, never fix it.
        cell = Executor()("print(undefined_name)")
        self.assertTrue(cell.startswith("Error: "))


class TestNamespaceSemantics(unittest.TestCase):
    def test_shared_namespace_leaks_between_scripts(self):
        # Documents the published behaviour: this leakage is a real defect, but
        # reproducing the paper requires keeping it. See the module docstring.
        run = Executor(isolate=False)
        run("leaked_marker = 41")
        self.assertEqual(run("print(leaked_marker + 1)"), "42")

    def test_isolated_namespace_does_not_leak(self):
        run = Executor(isolate=True)
        run("leaked_marker = 41")
        self.assertTrue(run("print(leaked_marker + 1)").startswith("Error: "))

    def test_each_isolated_run_starts_clean(self):
        self.assertEqual(run_script("x = 7;print(x)").stdout, "7")
        self.assertFalse(run_script("print(x)").ok)


class TestNonTermination(unittest.TestCase):
    def test_unbounded_loop_is_detected(self):
        self.assertTrue(looks_unbounded("while True:\n    pass"))
        self.assertFalse(looks_unbounded("for i in range(3):\n    pass"))

    def test_unbounded_loop_is_killed_not_hung(self):
        result = run_script_guarded("while True:\n    pass", timeout=1.0)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.as_cell(), "Error: timeout")

    def test_guarded_run_still_returns_output(self):
        self.assertEqual(
            run_script_guarded("print('ok')", timeout=10.0).stdout, "ok"
        )


class TestSemicolonExpansion(unittest.TestCase):
    def test_semicolons_become_newlines(self):
        # Most generated scripts are written as one semicolon-separated line.
        self.assertEqual(run_script("a=1;b=2;print(a+b)").stdout, "3")


if __name__ == "__main__":
    unittest.main()
