"""Checks for the family-evolution figures (Supplementary Figs 5 and 6).

Until 2026-09-16 these two panels were regenerated on every parity run and then
compared against nothing: the gate held their two input CSVs and not the figures,
so a change in the plotting code would have passed silently. They are compared as
images now, and what remains for a test is the part pixels cannot see.

Two things, specifically:

* ``FAMILY_ORDER`` existed twice, in ``35_summary_statistics.py`` and in the
  registry, and the two disagreed. ``tests/test_structure.py`` refused to collapse
  them without an elementwise diff; this file is the diff, kept as an assertion so
  the collapse cannot quietly come apart.
* nine families are in the data and eight are drawn. That is a property of the
  published figure, not a defect to fix, and it is invisible in a pixel count.
"""

from __future__ import annotations

import pandas as pd
import pytest

from superarc import REPO_ROOT
from superarc import evolution as E
from superarc import registry as R


# ---- the ordering that was collapsed -----------------------------------------


# Verbatim from 35_summary_statistics.py before the collapse. Kept as the record
# of what the hand-written dict said, so the claims below are checked against it
# rather than against a memory of it.
SCRIPT_FAMILY_ORDER = {
    'OpenAI': ['ChatGPT-4o', 'ChatGPT-4o-Mini', 'ChatGPT-4.5',
               'o1-Preview', 'o1-Mini', 'ChatGPT-5', 'ChatGPT-5.2'],
    'Grok': ['Grok', 'Grok-3', 'Grok-4', 'Grok-4.1'],
    'Claude': ['Claude-3.5', 'Claude-3.7', 'Claude-Sonnet-4', 'Claude-Opus-4',
               'Claude-4.5'],
    'Gemini': ['Gemini', 'Gemini-2.5-Pro', 'Gemini-3-Pro'],
    'Mistral': ['Mistral', 'Mistral-Large-2405', 'Mistral-Large-3'],
    'Qwen': ['Qwen', 'Qwen-3'],
    'Deepseek': ['DeepSeek', 'DeepSeek-R1-0528', 'DeepSeek-R1-0525'],
    'Meta': ['Meta', 'Llama-4-Scout'],
}

# The only names the hand-written dict held that no dataset contains. Both were
# inert: ``list.index`` shifts every later rank by one, uniformly, so the DRAWN
# order is unchanged either way. Declared so that if either ever becomes a real
# model, this test fails and the ordering gets looked at again.
PHANTOM_ENTRIES = {"Grok", "DeepSeek-R1-0525"}


def test_the_two_orderings_agree_once_the_phantoms_are_removed():
    """The elementwise diff that licensed collapsing them."""
    known = {m.name for m in R.MODELS}
    disagreements = {}
    for family, order in SCRIPT_FAMILY_ORDER.items():
        key = "DeepSeek" if family == "Deepseek" else family
        theirs = [n for n in order if n in known]
        ours = E.FAMILY_ORDER.get(key)
        assert ours is not None, f"{key} vanished from the collapsed ordering"
        if theirs != ours:
            disagreements[family] = (theirs, ours)
    assert not disagreements, disagreements


def test_the_phantom_entries_really_are_phantoms():
    """Zero matches would make the test above pass for the wrong reason."""
    known = {m.name for m in R.MODELS}
    for name in PHANTOM_ENTRIES:
        assert name not in known, (
            f"{name!r} is now a real model. It was treated as an inert "
            f"placeholder when the two orderings were collapsed; that reasoning "
            f"no longer holds and the within-family order must be re-derived."
        )
    # And the entries the diff relies on ARE present, so the search works.
    assert "Grok-3" in known and "DeepSeek-R1-0528" in known


def test_the_drawn_family_label_is_the_published_spelling():
    """The registry says DeepSeek; the Supplementary Information prints Deepseek.

    Taking the registry spelling to the canvas moved 0.0937% of Part 2's pixels
    -- two subplot titles, one character each -- for no reason but internal
    consistency. This pins the printed string at the point of drawing while the
    key stays the registry's everywhere it is a key.
    """
    assert "DeepSeek" in E.FAMILY_ORDER, "the KEY must be the registry's"
    assert E.PUBLISHED_FAMILY_LABEL["DeepSeek"] == "Deepseek"
    assert E.get_family("deepseek_r1") == "DeepSeek", (
        "get_family and FAMILY_ORDER must use the same spelling, or the whole "
        "family drops out of both figures"
    )


# ---- nine families in the data, eight in the figures ---------------------------


def test_cursor_is_in_the_table_and_not_in_the_figure():
    """A published omission, recorded rather than corrected.

    ``cursor_small`` is a column of ``comparison_models_summary.csv``, resolves
    through the registry, and appears in both output CSVs. It is drawn in
    neither figure, because the published script splits the eight families four
    and four and Cursor is in neither half. The printed captions name eight
    families, so this stays exactly as published -- but a family that is in the
    table and not in the figure is invisible unless something says so.
    """
    raw = pd.read_csv(REPO_ROOT / "comparison_models_summary.csv", nrows=1)
    assert "cursor_small" in raw.columns

    assert any(m.family == "Cursor" for m in R.MODELS)
    assert "Cursor" in R.FAMILY_ORDER
    assert "Cursor" not in E.PUBLISHED_FAMILIES
    assert "Cursor" not in E.FAMILY_ORDER

    # It is in the committed CSVs the figures are drawn from.
    for path in ("plots/multi_script/multiple_script_values.csv",
                 "plots/multi_formula/multiple_formulae_values.csv"):
        frame = pd.read_csv(REPO_ROOT / path)
        assert (frame["model"] == "Cursor-Small").sum() == 3, path


def test_the_two_halves_are_the_families_the_captions_name():
    families = list(E.FAMILY_ORDER)
    assert len(families) == 8
    # SI Fig 5: "OpenAI, Grok, Google, Claude".  SI Fig 6: "Mistral, DeepSeek,
    # Meta, Qwen".  The captions list them in a different order than the figure
    # lays them out, so this checks membership, not sequence.
    assert set(families[:4]) == {"OpenAI", "Grok", "Claude", "Gemini"}
    assert set(families[4:]) == {"Mistral", "Qwen", "DeepSeek", "Meta"}


# ---- ownership -----------------------------------------------------------------


def test_the_script_reexports_rather_than_redefining():
    """The producer keeps its published name and forwards; it holds no second copy."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "s35", REPO_ROOT / "35_summary_statistics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name in ("FAMILY_ORDER", "get_family", "aggregate_data",
                 "plot_evolution_subset", "process_data_and_generate_csvs"):
        assert getattr(module, name) is getattr(E, name), (
            f"{name} is a separate object in 35_summary_statistics.py -- it was "
            f"copied, not re-exported"
        )


@pytest.mark.slow
def test_the_csvs_regenerate_byte_identically(tmp_path, monkeypatch):
    """The two files the figures are drawn from, against the committed ones."""
    monkeypatch.setenv("SUPERARC_PLOTS_DIR", str(tmp_path))
    dfs = E.process_data_and_generate_csvs()
    assert dfs is not None

    for name, published in (
        ("multi_script/multiple_script_values.csv",
         "plots/multi_script/multiple_script_values.csv"),
        ("multi_formula/multiple_formulae_values.csv",
         "plots/multi_formula/multiple_formulae_values.csv"),
    ):
        produced = tmp_path / name
        assert produced.exists(), f"{name} not produced"
        if name.startswith("multi_formula"):
            # The formulae side is untouched by every correction so far.
            assert produced.read_bytes() == (REPO_ROOT / published).read_bytes()
        else:
            # The script side carries the Gemini-2.5-Pro correction, which is
            # recorded in superarc.parity.DATA_AUTHORISED. It must still parse
            # and still cover every model.
            a = pd.read_csv(produced)
            b = pd.read_csv(REPO_ROOT / published)
            assert set(a["model"]) == set(b["model"])
