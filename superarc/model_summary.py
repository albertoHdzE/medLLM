"""Derive the per-model summary that the family-evolution figures consume.

``comparison_models_summary.csv`` feeds Supplementary Figures 5 and 6.

To be clear about what is and is not a problem here: those figures **do**
reproduce exactly as published -- running ``35_summary_statistics.py`` against
the committed ``comparison_models_summary.csv`` regenerates both parts pixel for
pixel, and the intermediate ``plots/multi_*/*.csv`` come out byte-identical.
That is checked by ``superarc.parity``.

What is missing is not reproduction but *extension*: nothing in this repository
writes ``comparison_models_summary.csv``, so adding a model updates every other
figure and leaves these two behind. The file is also hand-maintained in a way
that shows -- 39 model columns for 28 models, because the same model appears
under both the formula dataset's key and the script dataset's key
(``chatgpt-4.5`` and ``chatgpt_4.5``, ``claude-3-5-sonnet`` and ``claude_3.5``).

This module supplies the missing producer. The two experiment scripts emit the
measures they already compute, in a tidy long format keyed by canonical model
name, and :func:`build_wide` assembles them into the wide layout. Because both
sides come from the registry, the duplicate columns cannot recur.

:func:`compare_with_legacy` diffs the generated summary against the
hand-maintained file. They do not currently agree on most measures, and the
legacy values cannot be traced to any code, so switching Supplementary Figures 5
and 6 onto the generated summary would change them. That switch is a decision
for the authors and has not been made -- the figures still read the committed
file.

The seven measures, per case (formulae / script) and complexity:

    Accuracy         mean % of generated answers reproducing the target
    Equivalence      mean % pairwise agreement between a model's own answers
    Valid Instances  how many usable answers the model produced at all
    Known sequence   \\
    Pure math         |  type distribution of those answers
    Print             |  (Print applies to scripts only)
    Not found        /
"""

from __future__ import annotations

import pandas as pd

from . import REPO_ROOT

DERIVED_DIR = REPO_ROOT / "data" / "derived"
LEGACY_WIDE = REPO_ROOT / "comparison_models_summary.csv"

CASE_FORMULAE = "multiple formulae"
CASE_SCRIPT = "multiple script"

MEASURES = (
    "Accuracy",
    "Accuracy (per answer)",
    "Equivalence",
    "Valid Instances",
    "Known sequence",
    "Pure math",
    "Print",
    "Not found",
)


def long_path(case: str) -> "object":
    slug = "formulae" if "formulae" in case else "script"
    return DERIVED_DIR / f"summary_{slug}.csv"


def write_long(case: str, records: list[dict]) -> "object":
    """Persist one case's measures in long form.

    ``records`` entries are ``{model, complexity, measure, value}`` with ``model``
    already the canonical registry name.
    """
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(records)
    expected = {"model", "complexity", "measure", "value"}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"summary records missing columns: {sorted(missing)}")
    unknown = set(frame["measure"]) - set(MEASURES)
    if unknown:
        raise ValueError(f"unrecognised measure(s): {sorted(unknown)}")
    frame = frame.sort_values(["measure", "complexity", "model"], kind="stable")
    path = long_path(case)
    frame.to_csv(path, index=False)
    return path


def build_wide() -> pd.DataFrame:
    """Assemble the wide summary both cases share, as SI Figs 5/6 expect."""
    frames = []
    for case in (CASE_FORMULAE, CASE_SCRIPT):
        path = long_path(case)
        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} not found - run the {case} producer first so it can "
                "emit its measures."
            )
        part = pd.read_csv(path)
        part["case"] = case
        frames.append(part)

    tidy = pd.concat(frames, ignore_index=True)
    wide = (
        tidy.pivot_table(
            index=["case", "measure", "complexity"],
            columns="model",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"measure": "Measure"})
    )
    wide.columns.name = None
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    wide.to_csv(DERIVED_DIR / "model_summary.csv", index=False)
    return wide


def compare_with_legacy(wide: pd.DataFrame, tolerance: float = 0.01) -> pd.DataFrame:
    """Diff the generated summary against the hand-maintained file.

    The legacy file keys models by dataset column name, so it is matched through
    the registry. Returns one row per disagreement; an empty frame means the
    generated file reproduces the hand-maintained one.
    """
    from .registry import A_FORMULA, B_SCRIPT, by_column

    legacy = pd.read_csv(LEGACY_WIDE)
    rows = []
    for _, lrow in legacy.iterrows():
        case, measure, complexity = lrow["case"], lrow["Measure"], lrow["complexity"]
        dataset = A_FORMULA if "formulae" in case else B_SCRIPT
        target = wide[
            (wide["case"] == case)
            & (wide["Measure"] == measure)
            & (wide["complexity"] == complexity)
        ]
        if target.empty:
            rows.append({"case": case, "Measure": measure, "complexity": complexity,
                         "model": "*", "legacy": None, "generated": None,
                         "note": "row absent from generated summary"})
            continue
        target = target.iloc[0]
        for column, value in lrow.items():
            if column in ("case", "Measure", "complexity") or pd.isna(value):
                continue
            model = by_column(dataset, column)
            if model is None:
                continue  # retired or belongs to the other dataset's naming
            generated = target.get(model.name)
            if generated is None or pd.isna(generated):
                rows.append({"case": case, "Measure": measure, "complexity": complexity,
                             "model": model.name, "legacy": value, "generated": None,
                             "note": "missing in generated"})
            elif abs(float(generated) - float(value)) > tolerance:
                rows.append({"case": case, "Measure": measure, "complexity": complexity,
                             "model": model.name, "legacy": float(value),
                             "generated": float(generated), "note": "value differs"})
    return pd.DataFrame(rows)


def main() -> int:
    wide = build_wide()
    print(f"generated summary: {wide.shape[0]} rows x {wide.shape[1]} columns")
    print(f"wrote {(DERIVED_DIR / 'model_summary.csv').relative_to(REPO_ROOT)}")
    if LEGACY_WIDE.exists():
        diff = compare_with_legacy(wide)
        if diff.empty:
            print("\nmatches the hand-maintained comparison_models_summary.csv exactly")
        else:
            print(f"\n{len(diff)} disagreement(s) with the hand-maintained file:")
            print(diff.head(40).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
