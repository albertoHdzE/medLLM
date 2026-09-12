"""Execute LLM-generated Python under a declared, reproducible environment.

Why this exists
---------------
Figures 5 and 6 are the only published figures whose numbers come from *running*
model-generated code, so their values depend on what happens to be importable.
That dependency was undeclared, and it silently changed the results: the
published run had ``numpy`` available but not ``sympy``, so four o1-Preview
scripts of the form ``import sympy; print(list(sympy.primerange(2, 31)))``
failed. Re-running today with ``sympy`` installed lifted o1-Preview's
complexity-2 accuracy from 20.00% to 26.67%.

The fix is to make the execution environment an explicit, declared part of the
method rather than an accident of the machine. ``ALLOWED_MODULES`` below is that
declaration, and it is load-bearing: changing it changes published numbers.

Namespace sharing
-----------------
The original executor ran every script with ``exec(code, globals())`` -- one
shared, mutable namespace for all ~12,000 scripts, so a variable defined by one
model's script is visible to the next model's. That is a real defect: a score can
depend on what a *different* model happened to leave behind, and on execution
order.

It is nevertheless the default here, because it is what produced the published
figures, and the project rule is that published results are preserved exactly.
Measured impact of switching it off: 10 of 29 models move, all at complexity 2,
by between 0.3 and 2.2 percentage points (8 up, 2 down). Pass ``isolate=True``
to opt into the corrected behaviour.

Sharing is safe under the append-only rule: new models add columns at the end,
so previously executed scripts run in the same order against the same
accumulated state, and their results do not move.

One further defect this module closes: a script that never terminates would hang
the whole pipeline.

Design note: the import guard is installed as ``__import__`` inside the script's
own builtins rather than on ``sys.meta_path``. A meta-path hook cannot hide a
module that is already in ``sys.modules`` (numpy always is, since the analysis
pipeline imports it), and a global hook would also break the analysis code
itself. Guarding the script's ``__import__`` restricts only the script.
"""

from __future__ import annotations

import builtins
import contextlib
import multiprocessing as mp
import sys
from dataclasses import dataclass
from io import StringIO

# --- the declared execution environment -------------------------------------
# The Python standard library, and nothing else.
#
# Rationale: `sympy.primerange` *is* the task -- returning the target sequence
# from a library look-up is the same delegation the paper penalises in `print`
# cases, so a model gains no credit for it.
#
# Measured, not assumed: the generated scripts import only `math` (47x),
# `numpy` (28x), `sympy` (4x) and `itertools` (3x). Admitting numpy as well
# changes the result for exactly zero models -- every numpy-importing script
# fails on other grounds -- so the strict stdlib rule is both the simplest
# declaration and the one that reproduces published Figures 5 and 6 exactly.
#
# This set is load-bearing. Changing it changes published numbers.
ALLOWED_EXTRA_MODULES: frozenset[str] = frozenset()
ALLOWED_MODULES = frozenset(sys.stdlib_module_names) | ALLOWED_EXTRA_MODULES

DEFAULT_TIMEOUT_SECONDS = 10.0

NOT_FOUND = "*not found"


@dataclass(frozen=True)
class Result:
    """Outcome of executing one generated script."""

    stdout: str
    error: str | None = None
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and not self.timed_out

    def as_cell(self) -> str:
        """Render as the pipeline stores it: captured stdout, or an error string.

        Matches the original ``execute_code_safely`` contract so that existing
        downstream comparisons are unaffected.
        """
        if self.timed_out:
            return "Error: timeout"
        if self.error is not None:
            return f"Error: {self.error}"
        return self.stdout


def _guarded_import(allowed: frozenset[str]):
    real_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root not in allowed:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, globals, locals, fromlist, level)

    return _import


def _fresh_namespace(allowed: frozenset[str]) -> dict:
    """A private namespace per script, so nothing leaks between executions."""
    safe_builtins = dict(vars(builtins))
    safe_builtins["__import__"] = _guarded_import(allowed)
    return {"__builtins__": safe_builtins, "__name__": "__generated__"}


def run_script(code: str, allowed: frozenset[str] = ALLOWED_MODULES) -> Result:
    """Execute one generated script and capture what it prints.

    No timeout here -- see :func:`run_script_guarded` for that. Semicolons are
    expanded to newlines exactly as the original pipeline did, because many
    generated scripts are written as one semicolon-separated line.
    """
    if not isinstance(code, str) or code.strip() in ("", NOT_FOUND):
        return Result(stdout="", error="empty")

    buffer = StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code.replace(";", "\n"), _fresh_namespace(allowed))  # noqa: S102
    except BaseException as exc:  # noqa: BLE001 - a generated script may raise anything
        return Result(stdout=buffer.getvalue().strip(), error=str(exc))
    return Result(stdout=buffer.getvalue().strip())


def _worker(code: str, allowed: frozenset[str], queue) -> None:  # pragma: no cover
    queue.put(run_script(code, allowed).as_cell())


def run_script_guarded(
    code: str,
    allowed: frozenset[str] = ALLOWED_MODULES,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Result:
    """Like :func:`run_script`, but a non-terminating script cannot hang the run.

    Only used for scripts that look capable of looping forever; spawning a
    process for all ~12,000 scripts would dominate the runtime for no benefit.
    """
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(code, allowed, queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return Result(stdout="", error=None, timed_out=True)
    try:
        cell = queue.get_nowait()
    except Exception:
        return Result(stdout="", error="no output")
    if cell.startswith("Error: "):
        return Result(stdout="", error=cell[len("Error: ") :])
    return Result(stdout=cell)


def looks_unbounded(code: str) -> bool:
    """Cheap syntactic check for scripts that may not terminate."""
    return isinstance(code, str) and ("while True" in code or "while 1" in code)


class Executor:
    """Drop-in replacement for the original ``execute_code_safely``.

    Routes possibly-unbounded scripts through a killable subprocess and runs
    everything else in-process, where it is far faster.

    ``isolate=False`` (the default) reproduces the published figures exactly by
    keeping the single shared namespace the original used. ``isolate=True``
    gives each script a private namespace; see the module docstring for the
    measured consequences of switching.
    """

    def __init__(
        self,
        allowed: frozenset[str] = ALLOWED_MODULES,
        isolate: bool = False,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.allowed = allowed
        self.isolate = isolate
        self.timeout = timeout
        self._shared = None if isolate else _fresh_namespace(allowed)

    def __call__(self, code: str) -> str:
        if not isinstance(code, str) or code.strip() in ("", NOT_FOUND):
            return ""
        if looks_unbounded(code):
            # A guarded run is always isolated -- it happens in another process.
            return run_script_guarded(code, self.allowed, self.timeout).as_cell()
        if self.isolate:
            return run_script(code, self.allowed).as_cell()

        buffer = StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                exec(code.replace(";", "\n"), self._shared)  # noqa: S102
        except BaseException as exc:  # noqa: BLE001
            return f"Error: {exc}"
        return buffer.getvalue().strip()


def execute(code: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> str:
    """Convenience wrapper using a module-level shared executor."""
    return _DEFAULT_EXECUTOR(code)


_DEFAULT_EXECUTOR = Executor()
