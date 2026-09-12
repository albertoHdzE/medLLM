"""Checks for the time-series panels of Figures 1 and 2.

The point of interest is the Levenshtein distance. ``processing_answers.py``
imported it from ``enchant.utils``, a binding to a C library that is not
installed here, so ``superarc.timeseries`` reimplements it. That replacement is
not taken on trust: the published CSVs carry the distances the original produced,
so the new implementation is held to all 1920 of them.
"""

from __future__ import annotations

import ast

import pandas as pd
import pytest

from superarc import REPO_ROOT
from superarc.timeseries import (
    BINARY_PANELS,
    PREDICTION_FILES,
    average_by_complexity,
    binary_success_rates,
    levenshtein,
    load_predictions,
    num_list_to_string,
)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("", "", 0),
        ("abc", "abc", 0),
        ("", "abc", 3),
        ("abc", "", 3),
        ("kitten", "sitting", 3),  # the textbook case
        ("flaw", "lawn", 2),
        ("a", "b", 1),
    ],
)
def test_levenshtein_known_cases(a, b, expected):
    assert levenshtein(a, b) == expected


def test_levenshtein_is_symmetric_and_bounded():
    a, b = "1 11 21 1211", "1 2 3 4 5"
    assert levenshtein(a, b) == levenshtein(b, a)
    assert levenshtein(a, b) <= max(len(a), len(b))


def test_levenshtein_reproduces_every_published_distance():
    """All 1920 published values, exactly.

    The string form matters and is easy to get wrong: the published distances
    came from ``num_list_to_string`` ("1 11 21"), not from the bracketed
    ``str(list)`` that ends up in the CSV. Comparing the bracketed form instead
    reproduces only 1773 of the 1920.
    """
    checked = 0
    for name in PREDICTION_FILES:
        frame = pd.read_csv(REPO_ROOT / name)
        for pct in (10, 25, 50, 75):
            for expected, predicted, published in zip(
                frame[f"to_predict_{pct}"],
                frame[f"forecasted_{pct}"],
                frame[f"levenshtein_{pct}"],
            ):
                recomputed = levenshtein(
                    num_list_to_string(ast.literal_eval(predicted)),
                    num_list_to_string(ast.literal_eval(expected)),
                )
                assert recomputed == published
                checked += 1
    assert checked == 1920


def test_complexity_three_and_four_are_swapped_then_four_dropped():
    averages = average_by_complexity(load_predictions())
    assert sorted(averages["complexity"].unique()) == [1, 2, 3]
    # Every forecaster and sequence type keeps all three levels.
    assert (averages.groupby(["forecasting_method_name", "seq_type"]).size() == 3).all()


def test_lag_llama_was_never_run_on_the_extended_pool():
    averages = average_by_complexity(load_predictions())
    combinations = set(zip(averages["forecasting_method_name"], averages["seq_type"]))
    assert ("lag-llama", "original") in combinations
    assert ("lag-llama", "extended") not in combinations


def test_published_success_rates_for_figure_one():
    """Pinned against the bars printed on page 4 of the article."""
    simple = binary_success_rates(BINARY_PANELS["figure01"][1])["Success Rate (%)"]
    assert simple["CTM/BDM"] == 100.0
    assert simple["timeGPT"] == pytest.approx(41.5, abs=0.5)
    assert simple["chronos"] == pytest.approx(45.8, abs=0.5)
    assert simple["lag-llama"] == pytest.approx(70.8, abs=0.5)

    random = binary_success_rates(BINARY_PANELS["figure01-middle"][1])["Success Rate (%)"]
    assert random["CTM/BDM"] == 100.0
    assert random["timeGPT"] == pytest.approx(57.0, abs=0.5)
    assert random["chronos"] == pytest.approx(53.0, abs=0.5)
    assert random["lag-llama"] == pytest.approx(58.0, abs=0.5)


def test_reference_model_beats_every_forecaster():
    """CTM/BDM identifies the generating rule, so it cannot be matched."""
    for _, (_, pools) in BINARY_PANELS.items():
        rates = binary_success_rates(pools)["Success Rate (%)"]
        assert (rates.drop("CTM/BDM") < rates["CTM/BDM"]).all()
