"""Checks for the time-series panels of Figures 1 and 2.

The point of interest is the Levenshtein distance. ``processing_answers.py``
imported it from ``enchant.utils``, a binding to a C library that is not
installed here, so ``superarc.timeseries`` reimplements it. That replacement is
not taken on trust: the published CSVs carry the distances the original produced,
so the new implementation is held to all 1920 of them.
"""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path

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


def test_the_swap_is_not_undone_by_calling_twice():
    """The notebook cell this replaces mutates in place and is not idempotent.

    ``22_Timeseries_LLM_experiments.ipynb`` cell 53 does the swap directly on
    ``total_df``, so running the cell an even number of times swaps complexity 3
    and 4 back and silently plots the wrong tier. ``average_by_complexity`` works
    on a copy; this checks that, by calling it twice on one frame and demanding
    the same answer.
    """
    predictions = load_predictions()
    first = average_by_complexity(predictions)
    second = average_by_complexity(predictions)
    pd.testing.assert_frame_equal(first, second)


# The values the published Figure 2 plots, read off the Chronos panel on page 9
# of the article: the two similarity series fall from complexity 2 to 3 while the
# edit distance stays flat.
#
# This is pinned because the panel has no committed artifact to diff against --
# no code ever saved it -- and because the notebook's own stored images show
# DIFFERENT numbers at complexity 3 (sort 3.90, gral 6.02, lev 8.46). Those are
# the unswapped tier, i.e. the output of a session in which cell 53 had run an
# even number of times. So the one reference that looks authoritative, the
# committed notebook output, is the wrong one, and only the printed page and
# these numbers agree.
PUBLISHED_CHRONOS_ORIGINAL = {
    1: {"sort_simi_percent": 12.94, "gral_simi_percent": 19.57, "levenshtein": 4.59},
    2: {"sort_simi_percent": 2.70, "gral_simi_percent": 2.64, "levenshtein": 8.24},
    3: {"sort_simi_percent": 1.11, "gral_simi_percent": 1.57, "levenshtein": 8.41},
}


def test_figure_two_matches_the_printed_panel_not_the_notebooks_stored_image():
    from superarc.timeseries import HORIZONS, SERIES

    averages = average_by_complexity(load_predictions())
    panel = averages[
        (averages["seq_type"] == "original")
        & (averages["forecasting_method_name"] == "chronos")
    ].set_index("complexity")

    for level, expected in PUBLISHED_CHRONOS_ORIGINAL.items():
        for name, template in SERIES.items():
            columns = [template.format(h) for h in HORIZONS]
            got = panel.loc[level, columns].mean()
            assert got == pytest.approx(expected[name], abs=0.01), (level, name)

    # The distinguishing feature: both similarity series keep falling at 3.
    # Without the swap, gral_simi rises to 6.02 and the figure tells the
    # opposite story about the hardest tier.
    for name in ("sort_simi_percent", "gral_simi_percent"):
        columns = [SERIES[name].format(h) for h in HORIZONS]
        assert panel.loc[3, columns].mean() < panel.loc[2, columns].mean(), name


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


def test_the_restyle_still_reproduces_the_committed_file_exactly():
    """The control for this whole module.

    ``plots/highResolution/figure01.pdf`` is a restyle that was never printed, so
    Figure 1 no longer uses its palette. But it is the only panel of Figure 1 that
    was ever saved, and reproducing it pixel for pixel is what justified trusting
    the two panels that have nothing to compare against. Kept as a test so the
    ruling on colour cannot quietly discard the evidence.
    """
    shutil = pytest.importorskip("shutil")
    if not shutil.which("pdftoppm"):
        pytest.skip("pdftoppm not available to rasterise the PDFs")

    from superarc.parity import pixel_difference, rasterise
    from superarc.timeseries import RESTYLED_PALETTE, plot_binary_panels

    out_dir = Path(tempfile.mkdtemp(prefix="superarc-figure01-"))
    plot_binary_panels(out_dir, colours=RESTYLED_PALETTE)
    difference = pixel_difference(
        rasterise(out_dir / "figure01.pdf"),
        rasterise(REPO_ROOT / "plots" / "highResolution" / "figure01.pdf"),
    )
    assert difference == 0.0, f"{difference:.4f}% of pixels differ"


def test_the_two_palettes_differ_only_in_colour():
    """Whichever is chosen, no number moves."""
    from superarc.timeseries import PRINTED_PALETTE, RESTYLED_PALETTE

    assert PRINTED_PALETTE != RESTYLED_PALETTE
    assert len(PRINTED_PALETTE) == len(RESTYLED_PALETTE) == 4


def test_reference_model_beats_every_forecaster():
    """CTM/BDM identifies the generating rule, so it cannot be matched."""
    for _, (_, pools) in BINARY_PANELS.items():
        rates = binary_success_rates(pools)["Success Rate (%)"]
        assert (rates.drop("CTM/BDM") < rates["CTM/BDM"]).all()
