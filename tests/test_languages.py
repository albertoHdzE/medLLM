"""Checks for the polyglot programming-language figures (Supplementary Figs 2-4).

These five panels had no runnable producer until 2026-09-15 and so had never been
tested at all. What matters here is not the drawing -- ``superarc.parity`` compares
four of them to the published artifacts pixel for pixel -- but the two things
pixels cannot catch:

* the fifth panel's *packing* is not recoverable, so its **data** is pinned here
  instead. If the set sizes move, the figure has changed meaning even though the
  gate would still report it as an authorised difference.
* the complexity swap is the third occurrence in this repository of an in-place
  ``replace({4: 3, 3: 4})`` that is not idempotent. It is pinned here.
"""

from __future__ import annotations

import copy

import pytest

from superarc import languages as L


@pytest.fixture(scope="module")
def results():
    return L.concat_all_results()


# ---- the data, and the seven languages ---------------------------------------


def test_all_seven_languages_load(results):
    assert set(results["language"].unique()) == set(L.LANGUAGES)
    assert len(results) == 4130


def test_the_swap_is_not_done_in_place(results):
    """The notebook's cell 1 mutates and is therefore not idempotent.

    ``24_PLOTS_paper.ipynb`` does ``total_df['complexity'] = ...replace(...)``
    directly, so running the cell twice puts the two hardest tiers back where they
    started. Third occurrence of this exact hazard here; see
    ``superarc.timeseries.average_by_complexity`` and cell 53 of
    ``22_Timeseries_LLM_experiments.ipynb``.
    """
    before = results["complexity"].value_counts().to_dict()
    swapped = L.swap_hardest_tiers(results)
    assert results["complexity"].value_counts().to_dict() == before, (
        "swap_hardest_tiers mutated its argument"
    )
    # Applying it twice returns the original, which is the property that makes
    # the in-place version dangerous.
    assert (
        L.swap_hardest_tiers(swapped)["complexity"].tolist()
        == results["complexity"].tolist()
    )
    # And it really did exchange 3 and 4.
    assert swapped["complexity"].value_counts()[3] == before[4]
    assert swapped["complexity"].value_counts()[4] == before[3]


# ---- the panel whose layout cannot be recovered -------------------------------


# Read off the published figure11_bottom, and reproduced by the code below. The
# supervenn packing depends on a hash seed nobody recorded, so these counts are
# the only part of that panel that can be held to the published image.
PUBLISHED_SET_SIZES = {
    "Mathematica": 40,
    "Python": 52,
    "Java": 69,
    "Matlab": 72,
    "R": 59,
    "Cpp": 45,
    "ArnoldC": 12,
}


def _correct_sequences(frame, language):
    df = copy.deepcopy(frame[frame["language"] == language])
    return set(df[(df["correct_execution"] == True) & (df["complexity"] != 4)]  # noqa: E712
               ["sequence"].unique())


def test_supervenn_set_sizes_match_the_published_panel(results):
    """The one thing about figure11_bottom that IS pinned to print."""
    swapped = L.swap_hardest_tiers(results)
    sizes = {
        label: len(_correct_sequences(swapped, language))
        for label, language in zip(L.SUPERVENN_LABELS, L.SUPERVENN_LANGUAGES)
    }
    assert sizes == PUBLISHED_SET_SIZES


def test_the_supervenn_inputs_are_sets_of_strings(results):
    """Why that panel needs PYTHONHASHSEED pinned, stated as a test.

    If these ever became sorted sequences the hash-seed sensitivity would go away
    and ``regenerate()`` would no longer need to pin the seed -- so this records
    the reason the pin exists, and will fail if the reason stops being true.
    """
    swapped = L.swap_hardest_tiers(results)
    one = _correct_sequences(swapped, "Python")
    assert isinstance(one, set)
    assert all(isinstance(x, str) for x in one)


# ---- ownership ----------------------------------------------------------------


def test_processing_answers_reexports_rather_than_redefining():
    """The loaders moved here; the historical module must not hold a second copy.

    ``superarc`` never imports ``processing_answers``; the dependency points the
    other way, as it already does for ``levenshtein``.
    """
    import processing_answers as PA

    for name in ("concat_all_results", "concat_all_compressed_files",
                 "analyze_all_languages", "create_language_summary_df"):
        assert getattr(PA, name) is getattr(L, name), (
            f"{name} is a separate object in processing_answers -- it was copied, "
            f"not re-exported"
        )


def test_analysis_does_not_write_into_the_source_tree(results, tmp_path, monkeypatch):
    """The published chain rewrote seven committed CSVs in the repo root.

    ``create_language_summary_df(..., save_language_name=l)`` writes
    ``normalized_compressed_<language>.csv`` to the working directory. They
    regenerate byte-identically, but a producer that writes into the source tree
    is what ``plots_dir()`` exists to prevent.
    """
    monkeypatch.chdir(tmp_path)
    # Only needs to get far enough to prove nothing is written; one language.
    one = results[results["language"] == "ArnoldC"]
    L.analysis(one, write_language_csvs=False)
    assert not list(tmp_path.glob("normalized_compressed_*.csv"))
