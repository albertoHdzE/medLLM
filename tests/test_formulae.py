"""Checks for the formulae-generation analysis.

The important one is the column resolution. Matching a model's columns by
prefix is the defect that put five models' figures partly on another model's
answers, and it is invisible in the output -- the numbers are plausible either
way. So it is pinned here rather than trusted.
"""

from __future__ import annotations

import pandas as pd
import pytest

from superarc.formulae import (
    CLASS_LABELS,
    MODEL_ORDER,
    _eval_cols,
    accuracy_table,
    add_measures,
    classify_content,
    compare_sequences,
    equivalence_table,
    get_model_display_name,
    load,
    process_formula_data,
)
from superarc.registry import A_FORMULA, models_for, prefix_collisions


@pytest.fixture(scope="module")
def measured():
    df = load()
    add_measures(df)
    return df


def test_model_order_covers_exactly_the_registry():
    """The figure's legend order is its own, but the *set* must not drift.

    MODEL_ORDER fixes each model's colour through tab20, so it cannot simply be
    the registry's ordering. Holding the two to the same set is what stops a
    model being added to one and missed by the other.
    """
    registry = {m.column(A_FORMULA) for m in models_for(A_FORMULA)}
    assert set(MODEL_ORDER) == registry
    assert len(MODEL_ORDER) == len(set(MODEL_ORDER)), "duplicate in MODEL_ORDER"


def test_display_names_reproduce_the_published_labels():
    """Checked elementwise against the script's map before it was removed:
    27 of 28 agreed, and the one that did not is the DeepSeek-R1-0525/-0528
    case the paper prints two ways."""
    assert get_model_display_name("deepseek_r1_0528") == "DeepSeek-R1-0525"
    assert get_model_display_name("gpt_4o") == "ChatGPT-4o"
    assert get_model_display_name("gpt_4o_mini") == "ChatGPT-4o-Mini"


def test_columns_are_matched_whole_not_by_prefix():
    """The correction behind both figures.

    `gpt_4o` and `gpt_4o_mini` are different models. `startswith("gpt_4o")` is
    true for both, so the published figures scored one on the other's answers.
    """
    df = load()
    for model in ("gpt_4o", "gemini", "mistral", "qwen", "deepseek"):
        resolved = _eval_cols(df, model)
        assert resolved, model
        for column in resolved:
            suffix = column[len(model):]
            assert suffix.startswith("_"), f"{column} is not a {model} column"
            # the part between the name and `_eval` must be a variant index only
            assert suffix.removesuffix("_eval").strip("_").isdigit(), column


def test_the_collision_would_have_been_real():
    """Names it, so the test fails loudly if someone reintroduces prefix
    matching and the report silently empties."""
    df = load()
    report = prefix_collisions(A_FORMULA, list(df.columns))
    assert set(report["model"]) == {"gpt_4o", "gemini", "mistral", "qwen", "deepseek"}
    assert (report["wrongly absorbed"] > 0).all()


def test_a_single_answer_cannot_agree_with_itself(measured):
    """Equivalence is pairwise, so one variant scores 0, not 100.

    Qwen produced a single formula. The published 42.22 at complexity 1 was
    Qwen-3's answers counted as Qwen's.
    """
    df = measured
    single = [m for m in MODEL_ORDER if len(_eval_cols(df, m)) == 1]
    assert single, "expected at least one single-variant model"
    for model in single:
        assert (df[f"equivalence_{model}"] == 0).all(), model


def test_equivalence_and_accuracy_are_percentages(measured):
    equiv = equivalence_table(measured)
    assert equiv["Equivalence %"].between(0, 100).all()
    assert set(equiv["Complexity"]) == {1, 2, 3}

    accuracy = accuracy_table(measured)
    assert accuracy.to_numpy().min() >= 0
    assert accuracy.to_numpy().max() <= 100


def test_both_measures_fall_as_complexity_rises(measured):
    """The result the figures exist to show, as an average over models."""
    equiv = equivalence_table(measured).groupby("Complexity")["Equivalence %"].mean()
    assert equiv[1] > equiv[3]

    accuracy = accuracy_table(measured).mean(axis=1)
    assert accuracy[1] > accuracy[2] > accuracy[3]


@pytest.mark.parametrize(
    "generated,target,expected",
    [
        ("1, 2, 3", "1,2,3,4,5", True),     # a prefix match counts
        ("1, 2, 4", "1,2,3,4,5", False),
        ("1, 2, 3", "*not found", False),   # unparseable is a miss, not an error
        ('"1, 2, 3"', "1,2,3", True),       # surrounding quotes are stripped
    ],
)
def test_compare_sequences(generated, target, expected):
    """Rescued from tests/test_multiFormula_analysis.py.

    That file was a placeholder written for this extraction and skipped because
    the package it imported never existed. Its specification was sound, so it is
    kept here and the placeholder is gone.
    """
    assert compare_sequences(generated, target) is expected


def test_accuracy_matches_the_values_pinned_before_the_extraction(measured):
    """Also rescued from the placeholder, and a real regression pin.

    These three models' accuracies were written down against the live dataset
    before any of this refactor happened, and they still hold -- including for
    Qwen, which the prefix-collision correction moved. Whoever wrote them had
    the correct column semantics in mind.
    """
    accuracy = accuracy_table(measured)
    expected = {
        "accuracy-percentage-qwen": [100.00, 46.67, 0.0],
        "accuracy-percentage-grok_3": [96.67, 45.00, 0.0],
        "accuracy-percentage-chatgpt_4.5": [78.89, 16.67, 0.0],
    }
    for column, values in expected.items():
        assert accuracy[column].round(2).tolist() == values, column


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Fibonacci sequence", "Known sequence"),
        ("the primes", "Known sequence"),
        ("a(n) = 3*n + 1", "Pure math"),
        ("*not found", "Not found"),
        (None, "Not found"),
    ],
)
def test_classification(text, expected):
    assert classify_content(text) == expected


def test_process_formula_data_covers_every_model_and_class(measured):
    volume, total_class, accurate_class = process_formula_data(measured)
    assert set(volume["Complexity"]) == {1, 2, 3}
    assert set(total_class["Classification"]) == set(CLASS_LABELS)
    assert len(volume) == len(MODEL_ORDER) * 3
    assert isinstance(volume, pd.DataFrame)
