"""Checks for the SuperARC-seq figures (published Figures 7 to 10).

The score itself is tested in ``tests/test_table1.py``, against the printed
table. These check the things that sit between the score and the four figures,
which is where this producer's risk actually lives:

* ``MODEL_ORDER`` is not cosmetic here. The by-type figures reshape a flat array
  by model count and then *assign* that list as the index, so a reordering would
  relabel every point without changing a single number. ``test_reshape_recovers``
  pins the reshape against a direct computation rather than trusting it.
* the same score was inlined four times in the published script; these confirm
  the one call reproduces each of the four.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superarc import superarc_seq as S
from superarc import table1
from superarc.registry import C_SERIES, columns_for


@pytest.fixture(scope="module")
def df():
    return S.load()


# ---- identity and order ------------------------------------------------------


def test_model_order_covers_exactly_the_registry():
    assert set(S.MODEL_ORDER) == set(columns_for(C_SERIES))
    assert len(S.MODEL_ORDER) == len(set(S.MODEL_ORDER)), "duplicate in MODEL_ORDER"


def test_display_names_come_from_the_registry():
    """The script's own 28-entry map was checked against these before deletion."""
    from superarc.registry import display_name

    for column in S.MODEL_ORDER:
        assert S.get_model_display_name(column) == display_name(C_SERIES, column)


def test_the_sequence_sets_partition_the_dataset(df):
    """100 binary sequences and three sets of 30 integer sequences, no overlap."""
    sizes = {name: len(S.subset(df, name)) for name in S.SEQUENCE_SETS}
    assert sizes == {"Binary": 100, "Integers Type 1": 30,
                     "Integers Type 2": 30, "Integers Type 3": 30}
    assert sum(sizes.values()) == len(df)


# ---- the score, and the four copies it replaced ------------------------------


def test_the_ranking_matches_table_1(df):
    """Figure 7 and Table 1 are the same numbers under different column names.

    They were computed by two separate inlined copies in the published script,
    so this is the check that collapsing them changed nothing.
    """
    figure = S.ranking(df, reference=False).set_index("Model")["tst"]
    table = table1.compute().set_index("Model")["phi"]
    shared = [m for m in figure.index if m in table.index]
    assert len(shared) == len(S.MODEL_ORDER)
    for model in shared:
        assert figure[model] == pytest.approx(table[model], abs=1e-12), model


def test_the_phi_by_type_figure_agrees_with_the_ranking_on_the_binary_set(df):
    """The third inlined copy of the score, held to the first.

    ``scores_by_type`` recomputes phi over each sequence set. On the Binary set
    it must return exactly what the ranking does. It does -- this figure's
    reshape uses the default C order and is sound.
    """
    ranking = S.ranking(df, reference=False).set_index("Model")
    phi = S.scores_by_type(df)
    phi_binary = phi[phi["c2"] == "Binary"].set_index("c1")["values"]
    for model in ranking.index:
        assert phi_binary[model] == pytest.approx(ranking.loc[model, "tst"], abs=1e-12)


def _rho_in_published_loop_order(df):
    """rho[p] for every (sequence set, model), set-major -- the script's order."""
    flat = {p: [] for p in range(4)}
    for name in S.SEQUENCE_SETS:
        part = S.subset(df, name)
        for column in S.MODEL_ORDER:
            rho, _delta, _phi = table1.score(part, column)
            for p in range(4):
                flat[p].append(rho[p])
    return flat


def test_every_point_of_the_p_by_type_figure_is_its_own_models(df):
    """The correction, checked point by point against a direct computation.

    ``proportions_by_type`` flattens a column of 112 values (4 sequence sets x 28
    models, set-major), reshapes it, transposes and then *assigns*
    ``MODEL_ORDER`` as the index. The reshape decides which model each row
    belongs to, and nothing downstream can catch a mistake in it -- the figure
    plots either way, and every number in it is a real number that some model
    really scored.

    This test is the only thing standing between a correct figure and a plausible
    one, so it checks all 448 points rather than a sample.
    """
    published = S.proportions_by_type(df)
    for name in S.SEQUENCE_SETS:
        part = S.subset(df, name)
        rows = published[published["c2"] == name]
        for column in S.MODEL_ORDER:
            rho, _delta, _phi = table1.score(part, column)
            label = S.get_model_display_name(column)
            for i, prob in enumerate(S.PROB_LABELS):
                got = rows[(rows["c1"] == label) & (rows["Prob"] == prob)]["values"]
                assert got.iloc[0] == pytest.approx(rho[i], abs=1e-12), (name, column, prob)


def test_the_published_reshape_is_the_one_that_was_corrected(df):
    """What the published figure drew, kept on the record.

    ``order='F'`` fills column by column, so the point drawn for model ``j`` and
    sequence set ``i`` was ``flat[i + 4*j]`` -- four CONSECUTIVE entries of a
    set-major array, which is four different models' values for one set. 173 of
    the 448 points showed a number belonging to a different model or a different
    sequence set; the rest coincided, because many class proportions are zero or
    repeat across models.

    This reconstructs it so the size of the correction stays measurable, and so
    that anyone comparing against the printed figure can see why it differs.
    """
    flat = _rho_in_published_loop_order(df)
    n = len(S.MODEL_ORDER)
    misplaced = sum(
        abs(flat[p][i + 4 * j] - flat[p][i * n + j]) > 1e-12
        for p in range(4)
        for j in range(n)
        for i in range(len(S.SEQUENCE_SETS))
    )
    assert misplaced == 173, f"{misplaced} points would have been misplaced, was 173"
    assert S.RESHAPE_ORDER == "C"


def test_the_row_ordering_of_both_by_type_figures_is_a_no_op(df):
    """Both figures sort their y-axis by a quantity that is the same for everyone.

    ``plot_probabilities`` sorts models by ``sum(values)``, and
    ``plot_phi_by_type`` reuses that ordering. But the values being summed are
    the four class proportions, which sum to 1 for every model on every sequence
    set -- so the key is exactly 4.0 for all 28, scrambled or not. The rows
    therefore appear in data order, and the apparent top-to-bottom ranking in
    those two figures is not a ranking.

    Not a defect in the numbers, and not something to change without an author
    saying so, since giving it a real key would reorder both figures. Recorded
    because a sort that silently does nothing reads exactly like a sort that
    works.

    What ordered the published rows was floating-point noise: summing a
    differently ordered set of addends gave 3.99999999999999956 for some models
    and 4.0 for others, a spread of 4.4e-16. With the reshape corrected the tie
    is *exact*, so the order is stable instead of accidental. This asserts
    exactness, not approximate equality -- an approximate assertion would have
    passed on the published version too.
    """
    as_drawn = S.proportions_by_type(df).groupby("c1")["values"].sum()
    assert as_drawn.nunique() == 1, (
        f"the sort key is no longer an exact tie: spread "
        f"{as_drawn.max() - as_drawn.min():.2e}. The row order is then decided by "
        f"rounding, which is how the published figure was ordered."
    )
    assert as_drawn.iloc[0] == 4.0


# ---- properties of the measure -----------------------------------------------


def test_the_class_proportions_sum_to_one(df):
    for name in S.SEQUENCE_SETS:
        part = S.subset(df, name)
        for column in S.MODEL_ORDER:
            rho, _delta, _phi = table1.score(part, column)
            assert rho.sum() == pytest.approx(1.0), (name, column)


def test_no_model_reaches_the_reference_score(df):
    """A perfect compressor scores 1. The point of the paper is that nothing does."""
    ranking = S.ranking(df, reference=True)
    reference = ranking[ranking["Model"] == "ASI"]["tst"].iloc[0]
    assert reference == 1
    best = ranking[ranking["Model"] != "ASI"]["tst"].max()
    assert best < 0.1, f"best model scores {best}, within 10% of a perfect compressor"


def test_the_ranking_is_sorted_and_the_reference_is_first(df):
    ranking = S.ranking(df)
    assert ranking["tst"].is_monotonic_decreasing
    assert ranking.iloc[0]["Model"] == "ASI"


def test_the_reference_row_is_labelled_differently_in_the_table(df):
    """One row, two labels in print. Recorded rather than quietly unified."""
    assert S.REFERENCE_ROW["Model"] == "ASI"
    assert table1.ASI_ROW["Model"] == "AIXI/BDM/CTM"
    assert S.REFERENCE_ROW["tst"] == table1.ASI_ROW["phi"] == 1


# ---- the bootstrap -----------------------------------------------------------


def test_the_bootstrap_is_reproducible(df):
    """The published one was not: np.random.default_rng() ignores np.random.seed.

    Small sizes here -- the full run is 400 resamples over 28 models and belongs
    in the parity gate, not a unit test.
    """
    a = S.bootstrap(df, seed=42, sizes=(25,), repeats=3)
    b = S.bootstrap(df, seed=42, sizes=(25,), repeats=3)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))

    c = S.bootstrap(df, seed=7, sizes=(25,), repeats=3)
    assert not np.allclose(a["Test Score"].to_numpy(), c["Test Score"].to_numpy()), (
        "a different seed gave an identical draw; the seed is not reaching the RNG"
    )


def test_the_bootstrap_covers_every_model(df):
    frame = S.bootstrap(df, seed=42, sizes=(25,), repeats=2)
    expected = {S.get_model_display_name(m) for m in S.MODEL_ORDER}
    assert set(frame["Model"]) == expected
    assert len(frame) == len(expected) * 2


def test_the_tiers_partition_the_models(df):
    frame = S.bootstrap(df, seed=42, sizes=(25, 50), repeats=4)
    tiered, means = S.assign_tiers(frame)
    assert set(tiered["Broad_Tier"]) <= {"High", "Medium", "Low"}
    # Every model lands in exactly one tier.
    per_model = tiered.groupby("Model")["Broad_Tier"].nunique()
    assert (per_model == 1).all()
    # And the boundaries are the ones the figure's titles claim.
    for model, tier in tiered.groupby("Model")["Broad_Tier"].first().items():
        mean = means[model]
        expected = "High" if mean >= 0.030 else ("Medium" if mean >= 0.007 else "Low")
        assert tier == expected, model


# ---- the memoised BDM --------------------------------------------------------


def test_the_bdm_cache_returns_what_an_uncached_engine_returns():
    """The score got 4.5x faster by memoising BDM. This checks it stayed the same.

    BDM of a string is a pure function, so caching is safe -- but "is safe" is
    not the same as "is the same", and the cache sits under every number in
    Table 1 and Figures 7 to 10.
    """
    engine = table1.bdm_engine()
    for text in ("0,1,0,1,0,1", "1,1,2,3,5,8", "a(n) = n**2", "*not found"):
        assert table1._nbdm(text) == engine.nbdm(table1.ascii_to_bits(text)), text


def test_the_score_is_unchanged_by_a_cold_cache(df):
    """Clearing the cache mid-run must not move a value."""
    binary = S.subset(df, "Binary")
    column = S.MODEL_ORDER[0]
    warm = table1.score(binary, column)
    table1._nbdm.cache_clear()
    cold = table1.score(binary, column)
    assert warm[2] == cold[2]
    assert np.array_equal(warm[0], cold[0])
    assert np.array_equal(warm[1], cold[1])
