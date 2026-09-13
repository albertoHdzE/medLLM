"""Checks for Supplementary Figure 1 -- complexity of the generated formulae.

Two things are worth pinning here. The figure's claim, which is that the answers
get *more* complex as the target does -- the opposite of compression. And the
placeholder draw, which is the one part of this figure that is not determined by
the data, so the tests state exactly how far it reaches.
"""

from __future__ import annotations

import pandas as pd
import pytest

from superarc.compression_metrics import (
    FORMULA_COLUMNS,
    PLACEHOLDER_EXPECTATION,
    PLOTTED_MODELS,
    compute,
    display_name,
    estimate_placeholder_expectation,
    load_formulas,
    metrics_for,
    per_string_measures,
)


@pytest.fixture(scope="module")
def table():
    return compute()


def test_every_plotted_model_has_a_column():
    frame = load_formulas()
    for model in PLOTTED_MODELS:
        assert model in FORMULA_COLUMNS
        assert FORMULA_COLUMNS[model] in frame.columns


def test_grok_and_gemini_1_5_are_computed_but_not_plotted():
    """The notebook comments them out of the model list; keep that visible."""
    assert "grok" in FORMULA_COLUMNS
    assert "gemini_1_5_advanced" in FORMULA_COLUMNS
    assert "grok" not in PLOTTED_MODELS
    assert "gemini_1_5_advanced" not in PLOTTED_MODELS


def test_grok_4_is_labelled_by_its_raw_key_as_published():
    """The display map's key is 'grok4', so 'grok_4' falls through.

    Printed that way in the Supplementary Information. Reproduced, not fixed --
    changing it is an author's call.
    """
    assert display_name("grok_4") == "grok_4"
    assert display_name("grok_3") == "Grok-3"


def test_no_answer_is_replaced_not_dropped():
    """'***' becomes a placeholder, so it scores as incompressible rather than
    silently vanishing from the average."""
    frame = load_formulas()
    for column in FORMULA_COLUMNS.values():
        assert not frame[column].isna().any()
        assert not (frame[column] == "***").any()


def test_the_figure_is_deterministic():
    """No seed appears in the result. Two runs must agree exactly."""
    a = compute(models=PLOTTED_MODELS[:4])
    b = compute(models=PLOTTED_MODELS[:4])
    merged = a.merge(b, on=["model", "complexity", "measure"], suffixes=("_a", "_b"))
    assert len(merged) == 4 * 3 * 6
    assert (merged["value_a"] == merged["value_b"]).all()


def test_placeholder_expectation_matches_a_fresh_estimate():
    """The pinned constants are reproducible, not asserted.

    Checked at 4 standard errors of a much smaller sample, which is loose enough
    not to be flaky and tight enough to catch a real change in the measure.
    """
    estimate = estimate_placeholder_expectation(draws=400, seed=7)
    for key, pinned in PLACEHOLDER_EXPECTATION.items():
        se = estimate.loc[key, "standard error"]
        got = estimate.loc[key, "expectation"]
        assert abs(got - pinned) <= 4 * se + 1e-9, f"{key}: {got} vs pinned {pinned}"


def test_lzw_length_of_a_placeholder_is_exactly_constant():
    """LZMA of any 45-character alphanumeric string base64s to 140 characters.

    This is why the Avg LZW panel carried no seed sensitivity at all, and it is
    worth pinning: if it ever stops holding, the expectation for that panel is
    no longer a constant and the docstring's claim is wrong.
    """
    estimate = estimate_placeholder_expectation(draws=200, seed=3)
    assert estimate.loc["avg_lzw", "sd"] == 0.0
    assert estimate.loc["avg_lzw", "expectation"] == 140.0


def test_per_string_decomposition_is_exact():
    """Each panel is a mean of per-answer values.

    The whole averaging argument rests on this: if a panel were not linear in
    its per-answer values, substituting the placeholder's expectation would not
    equal averaging the figure over draws.
    """
    frame = load_formulas(seed=5)
    column = FORMULA_COLUMNS[PLOTTED_MODELS[0]]
    answers = frame.loc[frame["complexity"] == 1, column].tolist()

    grouped = metrics_for(answers)
    decomposed = pd.DataFrame([per_string_measures(a) for a in answers]).mean()
    for key, value in grouped.items():
        assert value == pytest.approx(decomposed[key], abs=1e-9), key


def test_leaving_the_placeholder_literal_would_invert_the_measure():
    """'***' must not be scored as written.

    Three characters compress superbly, so a missing answer would rank as the
    best-compressing answer in the dataset -- the opposite of what it means.
    """
    missing = per_string_measures("***")
    assert missing["avg_bdm"] < PLACEHOLDER_EXPECTATION["avg_bdm"] / 2
    assert missing["avg_zip"] < PLACEHOLDER_EXPECTATION["avg_zip"] / 2
    assert missing["avg_lzw"] < PLACEHOLDER_EXPECTATION["avg_lzw"]


def test_answers_get_more_complex_as_the_target_does(table):
    """The claim the figure exists to support, across every measure.

    Compression would mean the opposite: a model that understood the rule should
    answer with something shorter than the sequence, and shorter still relative
    to a longer sequence.
    """
    means = table.groupby(["measure", "complexity"])["value"].mean().unstack()
    for measure, row in means.iterrows():
        assert row[1] < row[2] < row[3], f"{measure} is not increasing: {dict(row)}"


def test_compressed_length_panels_do_not_depend_on_the_draw():
    """LZW and ZIP lengths are exactly reproducible; the BDM panels are not.

    The compressed length of a random 45-character string is the same whichever
    characters were drawn, so these two panels carry no seed sensitivity at all.
    """
    a = compute(load_formulas(seed=1), models=PLOTTED_MODELS[:3])
    b = compute(load_formulas(seed=2), models=PLOTTED_MODELS[:3])
    merged = a.merge(b, on=["model", "complexity", "measure"], suffixes=("_a", "_b"))
    for measure in ("avg_lzw", "avg_zip"):
        rows = merged[merged["measure"] == measure]
        assert (rows["value_a"] == rows["value_b"]).all(), measure


def test_metrics_for_returns_every_panel():
    values = metrics_for(["1, 2, 3, 4", "n + 1"])
    assert set(values) == {
        "avg_bdm", "avg_shannon", "avg_lzw", "avg_zip", "avg_bdm_lzw", "avg_bdm_zip"
    }
    assert all(isinstance(v, float) and v > 0 for v in values.values())


def test_table_covers_every_model_complexity_and_measure(table):
    assert len(table) == len(PLOTTED_MODELS) * 3 * 6
    assert isinstance(table, pd.DataFrame)
    assert set(table["complexity"]) == {1, 2, 3}
