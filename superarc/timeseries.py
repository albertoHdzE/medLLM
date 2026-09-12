"""The time-series forecasting experiment: Figure 1 (top, middle) and Figure 2.

This is a *closed* experiment. Chronos, TimeGPT-1 and lag-llama were run against
the sequence pool once; their predictions are the committed CSVs and are never
re-forecast. Only the figures are regenerated.

Three panels live here, all built in ``22_Timeseries_LLM_experiments.ipynb``:

    Fig 1 top     Success Rate by Model - Simple climbers          was saved
    Fig 1 middle  Success Rate by Model - Random Binary Sequences  was NOT saved
    Fig 2         Average similarity and Levenshtein vs Complexity was NOT saved

Figure 1's remaining panel -- BDM/Shannon/zip/lzw over complexity -- is in
``superarc.complexity_measures``.

Why this module exists
----------------------
Two of the three panels were drawn and discarded: the notebook calls ``plt.show``
and never ``savefig``, so nothing could be checked against anything. The code
they need lives in ``processing_answers.py``, which instantiates MOMENT-1-large
at import time and therefore cannot be imported without ``torch``. The functions
the figures actually need are pure, so they are owned here instead.

The one panel that *was* saved is the control: regenerating it with
``RESTYLED_PALETTE`` reproduces the committed ``plots/highResolution/figure01``
with 0.0000% pixel difference, which is what licenses trusting the other two.
That file is not what the article prints, though -- see ``PRINTED_PALETTE``.

Levenshtein
-----------
``processing_answers.py`` takes its edit distance from ``enchant.utils``, a
binding to the C library ``libenchant``. That is a heavy and fragile dependency
for a textbook algorithm, and it is not installed here. The implementation below
replaces it, and is *verified* rather than assumed: it reproduces all 1920
published ``levenshtein_*`` values in the four Chronos/TimeGPT CSVs exactly
(``tests/test_timeseries.py``).

Recovering those values required noticing that the two call sites disagree about
what they compare:

    compare_predictions     levenshtein(num_list_to_string(a), ...)  ->  "1 11 21"
    add_levenshtein_to_df   levenshtein(df["forecasted_10"], ...)    ->  "[1, 11, 21]"

The Chronos and TimeGPT columns came from the first, the lag-llama columns from
the second, so the published lag-llama distances are inflated by the brackets and
commas. Both forms are preserved here exactly as published -- see
:func:`add_levenshtein_to_df`. Correcting it is an author's decision, not a
refactor.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import REPO_ROOT, plots_dir

FORECASTERS = ("timeGPT-1", "chronos", "lag-llama")

# The three measures the figure averages, per prediction horizon.
HORIZONS = (10, 25, 75)
SERIES = {
    "sort_simi_percent": "sort_simi_percen_{}",
    "gral_simi_percent": "gral_simi_percen_{}",
    "levenshtein": "levenshtein_{}",
}

PREDICTION_FILES = (
    "timeGPT_original.csv",
    "timeGPT_extended.csv",
    "chronos_original.csv",
    "chronos_extended.csv",
)
LAG_LLAMA_FILES = tuple(f"lag-llama_c{i}.csv" for i in (1, 2, 3, 4))


def levenshtein(a: str, b: str) -> int:
    """Edit distance with unit insertion, deletion and substitution costs."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def num_list_to_string(numerical_list) -> str:
    return " ".join(str(e) for e in numerical_list)


def add_levenshtein_to_df(frame: pd.DataFrame, percentages) -> pd.DataFrame:
    """Add ``levenshtein_<pct>`` columns, as published.

    Deliberately compares the *stringified list* -- ``"[1, 11, 21]"`` -- because
    that is what produced the published lag-llama distances. See the module
    docstring.
    """
    for pct in percentages:
        expected = frame[f"to_predict_{pct}"].values
        predicted = frame[f"forecasted_{pct}"].values
        frame[f"levenshtein_{pct}"] = [
            levenshtein(str(p), str(e)) for p, e in zip(predicted, expected)
        ]
    return frame


def load_predictions(data_dir: Path | None = None) -> pd.DataFrame:
    """Assemble every forecaster's predictions into one frame.

    Mirrors cells 42-48 of ``22_Timeseries_LLM_experiments.ipynb``: the lag-llama
    runs are stored per complexity and without a ``seq_type``, so they are
    concatenated and labelled here.
    """
    root = data_dir or REPO_ROOT
    frames = [pd.read_csv(root / name) for name in PREDICTION_FILES]

    lag_llama = pd.concat([pd.read_csv(root / name) for name in LAG_LLAMA_FILES])
    lag_llama = add_levenshtein_to_df(lag_llama, [10, 25, 50, 75])
    lag_llama["seq_type"] = "original"

    return pd.concat(frames + [lag_llama], ignore_index=True)


def average_by_complexity(predictions: pd.DataFrame) -> pd.DataFrame:
    """Mean of each measure per forecaster, sequence type and complexity.

    Complexity levels 3 and 4 are swapped before level 4 is dropped. That is not
    a bug: the pool was generated with the two hardest tiers in the opposite
    order to the one the paper reports, and the published figure applies this
    swap. Done on a copy, because doing it in place -- as the notebook cell does
    -- silently swaps back when the cell is re-run.
    """
    frame = predictions.copy()
    frame["complexity"] = frame["complexity"].replace({3: 4, 4: 3})

    measures = [template.format(h) for template in SERIES.values() for h in HORIZONS]
    averages = (
        frame.groupby(["forecasting_method_name", "seq_type", "complexity"])[measures]
        .mean()
        .reset_index()
    )
    return averages[averages["complexity"].isin([1, 2, 3])]


def output_dir() -> Path:
    """Where figures are written. See ``superarc.plots_dir``."""
    return plots_dir()


def plot(averages: pd.DataFrame, out_dir: Path | None = None) -> list[Path]:
    """Draw one panel per forecaster and sequence type, and save each.

    One panel per file, exactly as the notebook draws them. The published article
    prints no such figure -- only its caption (see the module docstring in
    ``superarc.parity``) -- so there is no layout to match and none is invented
    here.
    """
    import matplotlib.pyplot as plt

    out_dir = out_dir or output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for seq_type in sorted(averages["seq_type"].unique()):
        for method in sorted(averages["forecasting_method_name"].unique()):
            panel = averages[
                (averages["seq_type"] == seq_type)
                & (averages["forecasting_method_name"] == method)
            ].sort_values("complexity")
            if panel.empty:
                continue  # lag-llama was never run on the extended pool

            plt.figure(figsize=(10, 6))
            plt.grid()
            for template in SERIES.values():
                columns = [template.format(h) for h in HORIZONS]
                plt.plot(panel["complexity"], panel[columns].mean(axis=1), marker="o")

            plt.xlabel("Complexity", fontsize=18)
            plt.ylabel("Average", fontsize=18)
            plt.title(
                f"Average similarity and Levenshtein vs Complexity - {method}",
                fontsize=18,
            )
            plt.legend(list(SERIES), fontsize=17)
            plt.xticks([1, 2, 3], fontsize=18)
            plt.yticks(fontsize=18)

            stem = f"figure02-{method}-{seq_type}"
            for suffix in (".pdf", ".svg", ".png"):
                path = out_dir / (stem + suffix)
                plt.savefig(path, bbox_inches="tight")
                written.append(path)
            plt.close()

    return written


# --------------------------------------------------------------------------
# Figure 1, top and middle panels: last-digit prediction on binary sequences.
#
# Both panels are the same bar chart over a different pool. The published
# article stacks them with the complexity panel from
# ``superarc.complexity_measures``; only the top one was ever saved by code.

BINARY_PANELS = {
    "figure01": (
        "Success Rate by Model - Simple climbers ",
        ("timeGPT", "chronos", "lag_llama"),
    ),
    "figure01-middle": (
        "Success Rate by Model - Random Binary Sequences",
        ("timeGPT_random", "chronos_random", "lag_llama_random"),
    ),
}

# The reference row: CTM/BDM identifies the generating rule, so it never misses.
REFERENCE_MODEL = "CTM/BDM"

# Two palettes exist for these two panels, and only one of them was printed.
#
# PRINTED is matplotlib's default cycle, and is what the article shows: sampling
# page 4 of the published PDF gives #1f78b4 / #ff800f / #29a02c / #d52728, those
# four with the PDF's colour shift. It is the default here, by author's ruling
# (2026-09-12): Figure 1 is the figure in the paper.
#
# RESTYLED is the per-model palette the rest of the paper's figures use. It is
# what ``plots/highResolution/figure01.pdf`` contains -- a later restyle that was
# never printed. Kept because it is the control that validated this module:
# regenerating with it reproduces that committed file at 0.0000% pixel
# difference, which is checked in ``tests/test_timeseries.py``.
#
# The bar heights are identical either way; no result depends on the choice.
PRINTED_PALETTE = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
RESTYLED_PALETTE = ("#5cb8e6", "#f77189", "#ef7d32", "#c69432")
PANEL_COLOURS = PRINTED_PALETTE


def binary_success_rates(pools) -> pd.DataFrame:
    """Percentage of correctly predicted final digits, per forecaster."""
    frame = pd.concat(
        [pd.read_csv(REPO_ROOT / f"bin_seq_last_digit_pred_{pool}.csv") for pool in pools]
    )
    rates = [
        (frame.loc[frame["model"] == model, "success"].sum()
         / len(frame[frame["model"] == model])) * 100
        for model in frame["model"].unique()
    ]
    return pd.DataFrame(
        {"Success Rate (%)": [100.0] + rates},
        index=[REFERENCE_MODEL] + list(frame["model"].unique()),
    )


def plot_binary_panels(
    out_dir: Path | None = None, colours: tuple[str, ...] = PANEL_COLOURS
) -> list[Path]:
    import matplotlib.pyplot as plt

    out_dir = out_dir or output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for stem, (title, pools) in BINARY_PANELS.items():
        rates = binary_success_rates(pools)
        plt.figure(figsize=(10, 6))
        plt.bar(rates.index, rates["Success Rate (%)"], color=colours)
        plt.title(title, fontsize=24)
        plt.xlabel("Model", fontsize=20)
        plt.ylabel("Success Rate (%)", fontsize=20)
        plt.xticks(fontsize=16, rotation=0)
        plt.yticks(fontsize=16)
        plt.ylim(0, 100)
        plt.tight_layout()
        for suffix in (".pdf", ".svg", ".png"):
            path = out_dir / (stem + suffix)
            plt.savefig(path, bbox_inches="tight")
            written.append(path)
        plt.close()

    return written


def main() -> int:
    predictions = load_predictions()
    averages = average_by_complexity(predictions)
    written = plot(averages) + plot_binary_panels()
    for path in written:
        if path.suffix == ".png":
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
