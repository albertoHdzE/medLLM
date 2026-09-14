"""Compute the SuperARC-seq benchmark ranking -- Table 1 of the paper.

Table 1 was never written to disk. ``34-2_S-ARC_ext.py`` builds the dataframe and
then evaluates ``np.round(df_ranking2, 3)`` as a bare expression, which displays
in a notebook but is a no-op in a script, so the published table was transcribed
from a notebook cell by hand. This module computes it and emits it as CSV and
LaTeX, and ``tests/test_table1.py`` checks the result against what was printed.

The metric
----------
For each model, over the 100 binary sequences, answers fall into four classes:

    rho1  correct, neither a verbatim copy nor an ordinal index mapping
    rho2  correct and ordinal
    rho3  correct and a verbatim copy ("print")
    rho4  incorrect

``rho`` is their proportion and sums to 1. For each of the first three classes,
``delta`` is the harmonic mean of ``tanh(BDM(sequence) / BDM(formula))`` -- how
much the model's formula compresses the sequence it reproduces. The score is

    phi = sum_i rho_i * delta_i * w_i        w = (1, 0.1, 0.01)

so a correct, compressed, non-trivial answer is worth a hundred times a correct
answer that merely reprints the sequence. A perfect compressor scores 1.

Note what phi consumes: the three boolean columns ``-formula-correctness``,
``-formula-ordinal`` and ``-formula-copy_seq``. Nothing in this repository
computes them -- they are hand-adjudicated. Automating them is Phase 4.
"""

from __future__ import annotations

import sys
from functools import lru_cache
import types
from pathlib import Path

import numpy as np
import pandas as pd

from . import REPO_ROOT
from .complexity_measures import ascii_to_binary_list
from .registry import C_SERIES, models_for

# pybdm imports pkg_resources at module load. setuptools<81 provides it; this
# shim only covers the case where it is somehow still absent, so that an import
# error surfaces as a clear message rather than a BrokenProcessPool much later.
import warnings

# We pin setuptools<81 on purpose so that pkg_resources exists for pybdm; the
# deprecation notice is expected and would otherwise be noise on every run.
warnings.filterwarnings(
    "ignore", message=r"pkg_resources is deprecated as an API.*", category=UserWarning
)

try:  # pragma: no cover
    import pkg_resources  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    import importlib
    import importlib.resources

    shim = types.ModuleType("pkg_resources")

    def resource_stream(package_or_requirement, resource_name):
        package = (
            importlib.import_module(package_or_requirement)
            if isinstance(package_or_requirement, str)
            else package_or_requirement
        )
        return importlib.resources.files(package).joinpath(resource_name).open("rb")

    shim.resource_stream = resource_stream
    sys.modules["pkg_resources"] = shim

from pybdm import BDM, PartitionRecursive  # noqa: E402

DATASET = REPO_ROOT / "seriesWithLLMs_ext_Dic2025.csv"
DERIVED_DIR = REPO_ROOT / "data" / "derived"

N_BINARY_SEQUENCES = 100
CLASS_WEIGHTS = np.array([1.0, 0.1, 0.01])

# The reference row: an optimal compressor (AIXI, or CTM/BDM) answers every
# sequence correctly, never by copying or index-mapping, and compresses fully.
ASI_ROW = {
    "Model": "AIXI/BDM/CTM",
    "rho1": 1.0, "rho2": 0.0, "rho3": 0.0, "rho4": 0.0,
    "delta1": 1.0, "delta2": 0.0, "delta3": 0.0,
    "phi": 1.0,
}


def ascii_to_bits(text: str) -> np.ndarray:
    """Render a string as the bit array BDM operates on.

    Thin adapter over the owner in :mod:`superarc.complexity_measures`, which
    returns a list. This module needs the ``int8`` array pybdm wants.

    There were two implementations of this, under two names -- the second
    reason the search that would have found it never ran. They were checked by
    running them against each other rather than by reading: identical on 8 of 8
    inputs, differing only in return type. Collapsed on that evidence.
    """
    return np.array(ascii_to_binary_list(text), dtype=np.int8)


def harmonic_mean(values: np.ndarray) -> float:
    """Harmonic mean, returning 0 for an empty set rather than raising.

    Used instead of the arithmetic mean because it is dominated by the worst
    ratio: a model cannot offset one incompressible answer with several good
    ones.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return 0.0
    return float(values.size / np.sum(1.0 / values))


def phi(rho: np.ndarray, delta: np.ndarray) -> float:
    return float(np.sum(rho[:3] * delta * CLASS_WEIGHTS))


def bdm_engine() -> "BDM":
    """The BDM configuration the published score uses.

    ``PartitionRecursive`` rather than the default partition, which is what
    Figure 1's complexity panel uses. One line, but it changes every number in
    this module, so it has one owner.
    """
    return BDM(ndim=1, partition=PartitionRecursive)


@lru_cache(maxsize=None)
def _nbdm(text: str) -> float:
    """Normalised BDM of a string, memoised.

    Pure: the same string always has the same complexity. Worth caching because
    the 100 sequence strings are re-measured once per model, and the bootstrap
    re-measures them once per resample -- 28 models x 400 resamples over the same
    few hundred strings.
    """
    return bdm_engine().nbdm(ascii_to_bits(text))


def class_masks(frame: pd.DataFrame, column: str) -> list[np.ndarray]:
    """The four answer classes for one model, as boolean arrays.

        rho1  correct, neither a verbatim copy nor an ordinal index mapping
        rho2  correct and ordinal
        rho3  correct and a verbatim copy
        rho4  incorrect

    Note rho2 and rho3 are not exclusive of each other: an answer that is both
    ordinal and a copy is counted in both, so the four can sum to more than the
    number of rows. Preserved as published -- it is how the printed table was
    computed, and ``rho`` is normalised by their total rather than by ``len``.
    """
    correct = frame[f"{column}-formula-correctness"]
    ordinal = frame[f"{column}-formula-ordinal"]
    copied = frame[f"{column}-formula-copy_seq"]
    return [
        (correct & ~ordinal & ~copied).to_numpy(),
        (correct & ordinal).to_numpy(),
        (correct & copied).to_numpy(),
        (~correct).to_numpy(),
    ]


def score(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, float]:
    """``(rho, delta, phi)`` for one model over one set of sequences.

    The single owner of the metric. ``34-2_S-ARC_ext.py`` inlined this same
    calculation four times -- once for Table 1 and Figure 7, once for the p-only
    version behind Figure 8, once for Figure 9, and once inside the bootstrap --
    so a change to the score had to be made in four places or in none.

    The four copies were checked against each other before collapsing, including
    the one place they could have differed: ``34-2`` used
    ``statistics.harmonic_mean``, which returns 0 when any value is 0, while
    :func:`harmonic_mean` here drops non-positive values first. Over all four
    sequence subsets and all 28 models, no group contains a zero or a non-finite
    value, so the two agree on every one of them. This one is kept because it
    also survives the degenerate case rather than raising.
    """
    masks = class_masks(frame, column)
    total = sum(m.sum() for m in masks)
    rho = np.array([m.sum() / total for m in masks])

    bdm_formula = np.array([_nbdm(x) for x in frame[f"{column}-formula"].to_numpy()])
    bdm_input = np.array([_nbdm(x) for x in frame["sequence"].to_numpy()])
    delta = np.nan_to_num(np.array([
        harmonic_mean(np.tanh(bdm_input[m] / bdm_formula[m])) for m in masks[:3]
    ]))

    return rho, delta, phi(rho, delta)


def compute(dataset_path: Path | None = None) -> pd.DataFrame:
    """Compute the ranking over the 100 binary sequences."""
    df = pd.read_csv(dataset_path or DATASET, encoding="latin-1", low_memory=False)
    binary = df.iloc[:N_BINARY_SEQUENCES, :]

    rows = []
    for model in models_for(C_SERIES):
        rho, delta, value = score(binary, model.column(C_SERIES))
        rows.append({
            "Model": model.name,
            "rho1": rho[0], "rho2": rho[1], "rho3": rho[2], "rho4": rho[3],
            "delta1": delta[0], "delta2": delta[1], "delta3": delta[2],
            "phi": value,
        })

    table = pd.DataFrame([ASI_ROW] + rows)
    return table.sort_values("phi", ascending=False, kind="stable").reset_index(drop=True)


def to_latex(table: pd.DataFrame) -> str:
    header = (
        "\\begin{tabular}{lrrrrrrrr}\n\\hline\n"
        "Model & $\\rho_1$ & $\\rho_2$ & $\\rho_3$ & $\\rho_4$ "
        "& $\\delta_1$ & $\\delta_2$ & $\\delta_3$ & $\\varphi$ \\\\\n\\hline\n"
    )
    body = "".join(
        f"{r.Model} & {r.rho1:.3f} & {r.rho2:.3f} & {r.rho3:.3f} & {r.rho4:.3f} "
        f"& {r.delta1:.3f} & {r.delta2:.3f} & {r.delta3:.3f} & {r.phi:.3f} \\\\\n"
        for r in table.itertuples()
    )
    return header + body + "\\hline\n\\end{tabular}\n"


def main() -> int:
    table = compute()
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DERIVED_DIR / "ranking_table.csv"
    tex_path = DERIVED_DIR / "ranking_table.tex"
    table.round(6).to_csv(csv_path, index=False)
    tex_path.write_text(to_latex(table))
    print(table.round(3).to_string(index=False))
    print(f"\nwrote {csv_path.relative_to(REPO_ROOT)}")
    print(f"wrote {tex_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
