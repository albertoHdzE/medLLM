"""Checks for the script-generation analysis (published Figures 5 and 6).

Three of these are not really unit tests but pinned measurements. They exist
because this dataset carries two defects that are invisible in the output -- the
numbers look plausible either way -- and a defect nobody can see is a defect that
comes back. Pinning the measured value means any change to it has to be
deliberate:

* ``test_the_substring_collision_would_have_been_real`` -- four model names are
  substrings of a longer model's column name here, which is how published
  Figure 5 scored Grok-3 partly on Grok-4's programs;
* ``test_a_missing_answer_is_shredded_into_one_character_programs`` -- 216 cells
  hold the quoted string ``"*not found*"`` instead of a list, and are indexed
  character by character;
* ``test_the_hand_set_count_is_the_only_one`` -- one count in Figure 6 was typed
  over its computed value, and this records what was typed over.
"""

from __future__ import annotations

import ast

import pandas as pd
import pytest

from superarc import scripts as S
from superarc.registry import (
    B_SCRIPT,
    RETIRED_COLUMNS,
    columns_for,
    models_for,
    prefix_collisions,
    resolve_column,
)


@pytest.fixture(scope="module")
def measured():
    """The dataset with every derived column. Runs ~12,000 programs; ~10 s."""
    return S.add_measures(S.load())


# ---- identity ----------------------------------------------------------------


def test_model_order_covers_exactly_the_registry_plus_the_retired_column():
    """The figure's marker order is its own, but the *set* must not drift.

    ``MODEL_ORDER`` also holds the one column the registry declares retired,
    because the published figures iterated it and filtered on the display-name
    lookup. Keeping it listed is what makes the exclusion visible instead of
    implicit.
    """
    registry = set(columns_for(B_SCRIPT))
    retired = set(RETIRED_COLUMNS[B_SCRIPT])
    assert set(S.MODEL_ORDER) == registry | retired
    assert len(S.MODEL_ORDER) == len(set(S.MODEL_ORDER)), "duplicate in MODEL_ORDER"


def test_registered_excludes_only_the_declared_retired_column():
    excluded = set(S.MODEL_ORDER) - set(S.registered())
    assert excluded == set(RETIRED_COLUMNS[B_SCRIPT])


@pytest.mark.parametrize(
    "column,label",
    [
        # The two maps the script carried disagreed about exactly this row.
        ("gemini-2.5-pro", "Gemini-2.5-Pro"),
        ("gemini-thinking", "Gemini"),
        # This dataset's column names lag the model versions by one.
        ("grok", "Grok-3"),
        ("grok-3", "Grok-4"),
        # Swapped between datasets: the script data calls it 0525.
        ("deepseek_r1_0525", "DeepSeek-R1-0528"),
        ("chatgpt-o1", "o1-Preview"),
    ],
)
def test_display_names(column, label):
    assert S.get_model_display_name(column) == label


def test_the_retired_column_comes_back_unchanged():
    """Which is how `registered` filters it out, and the only way it can be."""
    assert S.get_model_display_name("gemini") == "gemini"


# ---- column resolution -------------------------------------------------------


def test_the_registry_resolves_the_derived_columns():
    """The derived grammar has to live with the dataset grammar, not beside it."""
    for column, expected in [
        ("mistral_script_1", ("mistral", "_script_1")),
        ("mistral_large2405_script_1", ("mistral_large2405", "_script_1")),
        ("grok-3_script_2_accuracy", ("grok-3", "_script_2_accuracy")),
        ("qwen_script_1_result_formatted", ("qwen", "_script_1_result_formatted")),
        ("qwen3_equivalence_percentage", ("qwen3", "_equivalence_percentage")),
    ]:
        assert resolve_column(B_SCRIPT, column) == expected, column


def test_columns_are_matched_whole_not_by_substring(measured):
    """The correction behind Figure 5's accuracy panel.

    `mistral` is a different model from `mistral_large2405`, but
    `"mistral" in "mistral_large2405_script_1_accuracy"` is true, so the
    published panel averaged Mistral over sixteen columns instead of two.
    """
    df = measured
    for model in ("mistral", "grok", "qwen", "deepseek", "chatgpt-4o"):
        whole = S._script_cols(df, model, "_accuracy")
        assert whole, model
        for column in whole:
            assert resolve_column(B_SCRIPT, column)[0] == model, column


def test_the_substring_collision_would_have_been_real(measured):
    """Named, so the test fails loudly if substring matching comes back."""
    df = measured
    accuracy_cols = [c for c in df.columns if c.endswith("_accuracy")]
    inflated = {}
    for model in ("mistral", "grok", "qwen", "deepseek"):
        by_substring = [c for c in accuracy_cols if model.lower() in c.lower()]
        whole = S._script_cols(df, model, "_accuracy")
        inflated[model] = (len(whole), len(by_substring))
    assert inflated == {
        "mistral": (2, 16),
        "grok": (2, 7),
        "qwen": (2, 13),
        "deepseek": (2, 13),
    }


def test_prefix_matching_would_also_have_been_wrong(measured):
    """`startswith` is not a safe weakening of the substring test either."""
    report = prefix_collisions(B_SCRIPT, list(measured.columns))
    assert set(report["model"]) >= {"mistral", "qwen", "deepseek"}
    assert (report["wrongly absorbed"] > 0).all()


# ---- the per-answer functions ------------------------------------------------


@pytest.mark.parametrize(
    "generated,target,expected",
    [
        ("1, 2, 3", "1, 2, 3", True),
        ("1,2,3", "1, 2, 3", True),        # whitespace is stripped per element
        ("1, 2, 3", "1, 2, 3, 4", False),  # unlike the formulae case, no prefix match
        ("Error: boom", "1, 2, 3", False),
    ],
)
def test_compare_sequences(generated, target, expected):
    """Note the contrast with :func:`superarc.formulae.compare_sequences`.

    That one counts a prefix of the target as a match; this one demands the whole
    sequence. Two different definitions of "right", one per experiment, both as
    published. Recorded here so the difference is a decision and not a surprise.
    """
    assert S.compare_sequences(generated, target) is expected


@pytest.mark.parametrize(
    "script,expected",
    [
        ("print([1, 2, 3])", "Print"),
        # Precedence matters and is as published: the Print test runs first, so a
        # bare print of the word still counts as copying rather than as knowing
        # the sequence. That is the right way round for this benchmark.
        ("print('fibonacci')", "Print"),
        ("for n in fibonacci(10): print(n)", "Known sequence"),
        ("for i in range(10): print(i)", "Pure math"),
        ("*not found", "Not found"),
        ("Error: SyntaxError", "Not found"),
        ("", "Not found"),
        # A single asterisk is a mathematical operator, so the shredded first
        # character of `"*not found*"` classifies as Pure math. See
        # test_a_missing_answer_is_shredded_into_one_character_programs.
        ("*", "Pure math"),
        ("n", "Not found"),
    ],
)
def test_classify_script(script, expected):
    assert S.classify_script(script) == expected


@pytest.mark.parametrize(
    "result,expected",
    [
        ("[1, 2, 3]", "1, 2, 3"),
        ("1\n2\n3", "1, 2, 3"),
        ("Error: boom", "Error: boom"),   # errors pass through untouched
    ],
)
def test_clean_result(result, expected):
    assert S.clean_result(result) == expected


def test_literal_eval_parses_every_cell_of_the_dataset():
    """The published parser was ``eval``, which executes what it reads.

    Switching to ``ast.literal_eval`` was verified to give the identical object
    on all 2495 populated cells. This holds that: a cell that only ``eval`` can
    read is a cell that would run code out of a data file.
    """
    raw = pd.read_csv(S.DATASET)
    parsed = 0
    for column in [c for c in raw.columns if c not in ("sequence", "Complexity")]:
        for cell in raw[column]:
            if pd.isna(cell) or cell == "" or cell == "*not found":
                continue
            ast.literal_eval(cell)
            parsed += 1
    assert parsed == 2495


# ---- aggregation -------------------------------------------------------------


def test_equivalence_needs_two_programs_to_compare(measured):
    """One valid program scores 0, not 100 -- it agrees with nothing."""
    df = measured
    for model in S.registered():
        if len(S._script_cols(df, model, "_result_formatted")) == 1:
            assert (df[f"{model}_equivalence_percentage"] == 0).all(), model


def test_both_measures_are_percentages(measured):
    equiv = S.equivalence_table(measured)
    assert equiv["Equivalence %"].between(0, 100).all()
    assert set(equiv["Complexity"]) == set(S.COMPLEXITIES)

    accuracy = S.accuracy_table(measured)
    assert accuracy["Accuracy"].between(0, 100).all()


def test_both_measures_fall_as_complexity_rises(measured):
    """The result the figures exist to show, as an average over models."""
    accuracy = S.accuracy_table(measured).groupby("Complexity")["Accuracy"].mean()
    assert accuracy[1] > accuracy[2] > accuracy[3]

    equiv = S.equivalence_table(measured).groupby("Complexity")["Equivalence %"].mean()
    assert equiv[1] > equiv[3]


def test_the_equivalence_summary_is_fully_populated(measured):
    """The defect the consolidation fixed, held down.

    The published pivot took its values from the first model's column, so 84 of
    87 cells were NaN and "highest average equivalence" always printed
    ChatGPT-4o. Console output only, but wrong is wrong.
    """
    pivot = S.equivalence_table(measured).pivot(
        index="Model", columns="Complexity", values="Equivalence %")
    assert pivot.notna().all().all()
    assert pivot.shape == (len(S.MODEL_ORDER), len(S.COMPLEXITIES))


def test_process_model_data_covers_every_registered_model_and_class(measured):
    volume, total_class, accurate_class = S.process_model_data(measured)
    models = S.registered()
    assert set(volume["Complexity"]) == set(S.COMPLEXITIES)
    assert len(volume) == len(models) * len(S.COMPLEXITIES)
    assert set(total_class["Classification"]) == set(S.CLASS_LABELS)
    assert len(total_class) == len(models) * len(S.COMPLEXITIES) * len(S.CLASS_LABELS)
    assert len(accurate_class) == len(total_class)


# ---- the two pinned defects --------------------------------------------------


def test_a_missing_answer_is_shredded_into_one_character_programs():
    """216 cells hold a quoted string, not a list, and are indexed per character.

    ``"*not found*"`` is eleven characters, so it becomes eleven one-character
    programs -- and because ``max_scripts`` is the longest value in the column,
    it also sets the column count for the whole model. Seven models therefore
    have eleven program columns where they wrote at most three or four, which is
    the denominator of their accuracy in Figure 5 and the height of their
    Not-found and Pure-math bars in Figure 6.

    Preserved, not fixed: it changes two published figures and is the authors'
    call. Pinned here so the extent is on the record and cannot drift unnoticed.
    """
    raw = pd.read_csv(S.DATASET)
    shredded = {}
    for column in [c for c in raw.columns if c not in ("sequence", "Complexity")]:
        def parse(x):
            if pd.isna(x) or x == "" or x == "*not found":
                return ["*not found"]
            return ast.literal_eval(x)
        parsed = raw[column].apply(parse)
        count = int(parsed.apply(lambda v: isinstance(v, str)).sum())
        if count:
            real = max(len(v) for v in parsed if isinstance(v, list))
            shredded[column] = (count, real, int(parsed.str.len().max()))

    assert shredded == {
        # column: (cells shredded, longest real answer, columns created)
        "gpt-4o-mini": (61, 3, 11),
        "gemini-2.5-pro": (60, 3, 11),
        "claude-3.7": (16, 4, 11),
        "deepseek_r1_0525": (16, 3, 11),
        "llama_4_scout": (21, 3, 11),
        "qwen3": (21, 3, 11),
        "mistral_large2405": (21, 3, 11),
    }
    assert sum(v[0] for v in shredded.values()) == 216


def test_the_hand_set_count_is_the_only_one(measured):
    """Figure 6 shows 1 valid ChatGPT-4o-Mini script at complexity 3; it computes 11.

    The published script assigned that cell by hand under the comment "Fix for
    ChatGPT-4o-Mini at complexity 3". The computed 11 is itself an artifact --
    all 30 of its complexity-3 cells are shredded ``"*not found*"`` strings, and
    the counting rule calls a column valid when no row in it is NaN or exactly
    ``'*not found'``, which a column of ``'*'`` satisfies. So neither number is
    the truth, which is 0.

    Kept as published. This test states what was typed over, and asserts that it
    is the only such override.
    """
    df = measured
    complexity_3 = df[df["Complexity"] == 3]
    columns = S._script_cols(df, "gpt-4o-mini")
    computed = sum(
        1 for c in columns
        if not complexity_3[c].isna().any() and not (complexity_3[c] == "*not found").any()
    )
    assert computed == 11, "the override no longer covers what it used to"

    volume, _, _ = S.process_model_data(df)
    published = volume[(volume["Model"] == "gpt-4o-mini") & (volume["Complexity"] == 3)]
    assert published["Count"].iloc[0] == 1

    # And nothing else is overridden: every other cell equals its computation.
    overrides = 0
    for complexity in S.COMPLEXITIES:
        subset = df[df["Complexity"] == complexity]
        for model in S.registered():
            cols = S._script_cols(df, model)
            expected = sum(
                1 for c in cols
                if not subset[c].isna().any() and not (subset[c] == "*not found").any()
            )
            actual = volume[(volume["Model"] == model)
                            & (volume["Complexity"] == complexity)]["Count"].iloc[0]
            if actual != expected:
                overrides += 1
    assert overrides == 1, f"{overrides} hand-set counts, expected exactly 1"


# ---- the emitted summary -----------------------------------------------------


def test_the_emitted_measures_cover_every_registered_model(measured, tmp_path):
    """The file Supplementary Figures 5 and 6 read."""
    from superarc.model_summary import CASE_SCRIPT, long_path

    volume, total_class, _ = S.process_model_data(measured)
    S.emit_summary_measures(measured, volume, total_class)
    emitted = pd.read_csv(long_path(CASE_SCRIPT))
    expected = {S.get_model_display_name(m) for m in S.registered()}
    assert set(emitted["model"]) == expected
    assert set(emitted["complexity"]) == set(S.COMPLEXITIES)
    assert {"Accuracy", "Equivalence", "Valid Instances"} <= set(emitted["measure"])


def test_every_registered_model_appears_in_the_figures():
    """The silent-drop guard, at the level the figures actually use.

    Every published producer filtered with ``if display_name != model``, so a
    column missing from the label map vanished from the figure without a word.
    That is how one Gemini column disappeared from each of Figures 5 and 6.
    """
    assert len(S.registered()) == len(models_for(B_SCRIPT))
